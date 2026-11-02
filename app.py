import pickle
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Starbucks Review Rating Predictor",
    page_icon="☕",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "decision_tree_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open(BASE_DIR / "tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

st.title("☕ Starbucks Review Rating Predictor")
st.write(
    "Enter a customer review to predict its rating (1 to 5 stars) "
    "using the trained Decision Tree model."
)

try:
    model, vectorizer = load_artifacts()
    st.success("Model and TF-IDF vectorizer loaded successfully.")
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.info(
        "Make sure decision_tree_model.pkl and tfidf_vectorizer.pkl "
        "are in the same GitHub repository folder as app.py."
    )
    st.stop()

review = st.text_area(
    "Customer review",
    placeholder="Example: The coffee was great and the staff were friendly...",
    height=150,
)

if st.button("Predict rating", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            features = vectorizer.transform([review.strip()])
            prediction = model.predict(features)[0]
            st.subheader("Predicted rating")
            st.markdown(f"## {prediction} ⭐")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
