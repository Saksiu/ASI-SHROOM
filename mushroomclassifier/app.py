import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("data/06_models/best_model.pkl")
preprocessor = joblib.load("data/06_models/preprocessor.pkl")

# Cecha: kolumny danych (bez "class")
FEATURES = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

# Przykładowe możliwe wartości (można je automatycznie wyciągnąć z danych)
OPTIONS = {
    'cap-shape': ['b', 'c', 'x', 'f', 'k', 's'],
    'cap-surface': ['f', 'g', 'y', 's'],
    'cap-color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    'bruises': ['t', 'f'],
    'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    'gill-attachment': ['a', 'd', 'f', 'n'],
    'gill-spacing': ['c', 'w', 'd'],
    'gill-size': ['b', 'n'],
    'gill-color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    'stalk-shape': ['e', 't'],
    'stalk-root': ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    'stalk-surface-above-ring': ['f', 'y', 'k', 's'],
    'stalk-surface-below-ring': ['f', 'y', 'k', 's'],
    'stalk-color-above-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    'stalk-color-below-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    'veil-color': ['n', 'o', 'w', 'y'],
    'ring-number': ['n', 'o', 't'],
    'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
    'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
    'population': ['a', 'c', 'n', 's', 'v', 'y'],
    'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd']
}

st.title("🍄 Klasyfikator grzybów – Jadalny czy Trujący?")
st.markdown("Wprowadź cechy grzyba, a model spróbuje przewidzieć, czy jest jadalny (`e`) czy trujący (`p`).")

# Formularz cech
with st.form("mushroom_form"):
    inputs = {}
    for feature in FEATURES:
        inputs[feature] = st.selectbox(feature, OPTIONS[feature])
    submitted = st.form_submit_button("Sprawdź jadalność")

if submitted:
    input_df = pd.DataFrame([inputs])
    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]
    label = "JADALNY 🍽️" if prediction == 'e' else "TRUJĄCY ☠️"

    st.subheader("📢 Wynik predykcji:")
    st.success(f"Grzyb jest: **{label}** (klasa `{prediction}`)")
