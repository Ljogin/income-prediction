import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, pull, finalize_model

st.set_page_config(page_title="Income Prediction App", layout="centered")

st.title("🏦 Predykcja dochodu > 50K USD")
st.write("Wgraj plik CSV, wybierz kolumne celu i naucz model. Potem podaj dane do predykcji.")

# ---- STEP 1: Upload CSV ----
uploaded_file = st.file_uploader("📂 Wgraj plik CSV do trenowania modelu", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Podglad danych:")
    st.dataframe(data.head())

    # ---- STEP 2: Select target column ----
    target_col = st.selectbox("🎯 Wybierz kolumne celu (czy zarobki >50K)", data.columns)

    if st.button("⚙️ Trenuj model"):
        with st.spinner("Trenowanie modelu..."):
            setup(data=data, target=target_col, silent=True, preprocess=True)
            
            best_model = compare_models()
            best_model = finalize_model(best_model)

        st.success("✅ Model nauczony!")

        # Save session state model
        st.session_state["model"] = best_model
        
        # Feature importance
        st.subheader("📊 Najwazniejsze cechy")
        fi = pull()
        st.dataframe(fi)

# ---- STEP 3: Prediction form ----
if "model" in st.session_state:
    st.subheader("🔮 Predykcja dochodu")

    # Dynamic form based on dataset columns (except target)
    input_cols = [c for c in data.columns if c != target_col]
    user_input = {}

    for col in input_cols:
        if data[col].dtype == 'object':
            user_input[col] = st.selectbox(f"{col}", options=data[col].dropna().unique())
        else:
            user_input[col] = st.number_input(f"{col}", value=float(data[col].median()))

    input_df = pd.DataFrame([user_input])

    if st.button("👉 Przewiduj"):
        pred = predict_model(st.session_state["model"], data=input_df)
        result = pred["prediction_label"].iloc[0]

        if result == 1 or result == ">50K":
            st.success("✅ Model przewiduje dochod **> 50K USD**")
        else:
            st.warning("❌ Model przewiduje dochod **≤ 50K USD**")

        st.write("Dane wejsciowe:")
        st.dataframe(input_df)
