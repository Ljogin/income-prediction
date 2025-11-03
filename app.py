import streamlit as st
import pandas as pd
from pycaret.classification import ClassificationExperiment

st.set_page_config(page_title="Predykcja dochodu >50K", layout="centered")

st.title("🏦 Predykcja dochodu > 50K USD rocznie")
st.write("Wgraj plik CSV, wybierz kolumne celu, naucz model i wykonaj predykcje.")

# ---------------------- UPLOAD CSV ----------------------
uploaded_file = st.file_uploader("📂 Wgraj plik CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Podglad danych:")
    st.dataframe(data.head())

    target_col = st.selectbox("🎯 Wybierz kolumne celu", data.columns)

    if st.button("⚙️ Trenuj model"):
        with st.spinner("Trenowanie modelu... prosze czekac ⏳"):

            # Tworzymy eksperyment pycaret
            exp = ClassificationExperiment()
            exp.setup(
                data=data,
                target=target_col,
                session_id=42,
                verbose=False,
                silent=True
            )

            # Wybieramy najlepszy model
            best_model = exp.compare_models()
            final_model = exp.finalize_model(best_model)

            # Zapamiętujemy eksperyment i model
            st.session_state["exp"] = exp
            st.session_state["model"] = final_model
            st.session_state["feature_df"] = exp.pull()  # tabela ważności cech

        st.success("✅ Model wytrenowany!")

        st.subheader("📈 Najwazniejsze cechy modelu")
        st.dataframe(st.session_state["feature_df"])

# ---------------------- PREDICT ----------------------
if "model" in st.session_state:

    st.subheader("🔮 Predykcja nowej osoby")

    exp = st.session_state["exp"]
    model = st.session_state["model"]

    # Tworzymy formularz predykcji
    input_cols = [c for c in exp.X.columns if c != exp.target]
    user_input = {}

    st.write("Wprowadz dane:")

    for col in input_cols:
        if exp.X[col].dtype == "object":
            user_input[col] = st.selectbox(col, options=exp.X[col].dropna().unique())
        else:
            user_input[col] = st.number_input(col, value=float(exp.X[col].median()))

    input_df = pd.DataFrame([user_input])

    if st.button("👉 Przewiduj"):
        prediction = exp.predict_model(model, data=input_df)
        result = prediction["prediction_label"].iloc[0]

        if str(result) in ["1", ">50K", "True"]:
            st.success("✅ Model przewiduje dochod **> 50K USD**")
        else:
            st.warning("❌ Model przewiduje dochod **≤ 50K USD**")

        st.write("📋 Dane wejsciowe:")
        st.dataframe(input_df)
