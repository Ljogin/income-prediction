import streamlit as st
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.datasets import get_data

st.set_page_config(page_title="Predykcja dochodu >50K", layout="centered")

st.title("🏦 Predykcja dochodu > 50K USD rocznie")
st.write("Model zostal wytrenowany automatycznie na datasetcie 'income'. Podaj dane ponizej aby otrzymac predykcje.")

# ---------------------- LOAD INCOME DATASET ----------------------
@st.cache_resource
def train_model():
    df = get_data("income", verbose=False)

    exp = ClassificationExperiment()
    exp.setup(
        data=df,
        target="income",
        session_id=42,
        verbose=False,
        silent=True,
    )

    best_model = exp.compare_models()
    final_model = exp.finalize_model(best_model)

    # Save columns for input form
    return exp, final_model, df

exp, model, df = train_model()

st.success("✅ Model zostal wytrenowany na datasetcie 'income'.")

# ---------------------- SHOW FEATURE IMPORTANCE ----------------------
fi = exp.pull()  # feature importance table
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
