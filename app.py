import streamlit as st
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.datasets import get_data

st.set_page_config(page_title="Predykcja dochodu >50K", layout="centered")

st.title("🏦 Predykcja dochodu > 50K USD rocznie")
st.write("Model jest trenowany na wbudowanym datasetcie 'income'. Podaj dane, aby otrzymac predykcje.")

# ---------------------- LOAD DATA (cache ok) ----------------------
@st.cache_data
def load_income():
    return get_data("income", verbose=False)

df = load_income()

# ---------------------- TRAIN MODEL IF NOT EXISTS ----------------------
if "exp" not in st.session_state or "model" not in st.session_state:
    with st.spinner("Trenowanie modelu..."):
        exp = ClassificationExperiment()
        exp.setup(data=df, target="income", session_id=42, verbose=False, silent=True)

        best_model = exp.compare_models()
        final_model = exp.finalize_model(best_model)

        st.session_state["exp"] = exp
        st.session_state["model"] = final_model

    st.success("✅ Model wytrenowany na datasetcie 'income'!")

exp = st.session_state["exp"]
model = st.session_state["model"]

# ---------------------- FEATURE IMPORTANCE ----------------------
fi = exp.pull()
st.subheader("📊 Najwazniejsze cechy modelu")
st.dataframe(fi)

# ---------------------- PREDICTION FORM ----------------------
st.subheader("🔮 Predykcja nowej osoby")

input_cols = [c for c in df.columns if c != "income"]
user_input = {}

st.write("Wprowadz dane:")

for col in input_cols:
    if df[col].dtype == "object":
        user_input[col] = st.selectbox(col, options=df[col].dropna().unique())
    else:
        median_value = float(df[col].median())
        user_input[col] = st.number_input(col, value=median_value)

input_df = pd.DataFrame([user_input])

if st.button("👉 Przewiduj"):
    prediction = exp.predict_model(model, data=input_df)
    result = prediction["prediction_label"].iloc[0]

    if str(result).lower() in [">50k", "1", "true"]:
        st.success("✅ Model przewiduje dochod **> 50K USD**")
    else:
        st.warning("❌ Model przewiduje dochod **≤ 50K USD**")

    st.write("📋 Dane wejsciowe:")
    st.dataframe(input_df)
