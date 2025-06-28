import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# ----------------------
# Mapowania cech (ludzkie opisy <-> kody literowe)
# ----------------------
feature_maps = {
    "cap-shape": {
        "b": "stożkowy", "c": "wypukły", "x": "wypukło-płaski", "f": "płaski", "k": "guzkowaty", "s": "wduszony"
    },
    "cap-surface": {
        "f": "włóknista", "g": "rowkowana", "y": "łuskowata", "s": "gładka"
    },
    "cap-color": {
        "n": "brązowy", "b": "buff", "c": "cynamonowy", "g": "szary", "r": "zielony", "p": "różowy", "u": "fioletowy", "e": "czerwony", "w": "biały", "y": "żółty"
    },
    "bruises": {
        "t": "tak", "f": "nie"
    },
    "odor": {
        "a": "migdałowy", "l": "anizowy", "c": "creozot", "y": "rybi", "f": "nieprzyjemny", "m": "stęchły", "n": "brak", "p": "ostry", "s": "korzenny"
    },
    "gill-attachment": {
        "a": "wolne", "d": "przyczepione"
    },
    "gill-spacing": {
        "c": "blisko", "w": "szeroko"
    },
    "gill-size": {
        "b": "szerokie", "n": "wąskie"
    },
    "gill-color": {
        "k": "czarny", "n": "brązowy", "b": "buff", "h": "czekoladowy", "g": "szary", "r": "zielony", "o": "pomarańczowy", "p": "różowy", "u": "fioletowy", "e": "czerwony", "w": "biały", "y": "żółty"
    },
    "stalk-shape": {
        "e": "poszerzający się", "t": "zwężający się"
    },
    "stalk-root": {
        "b": "bulwa", "c": "maczugowaty", "u": "bez podstawy", "e": "równy", "?": "brak danych"
    },
    "stalk-surface-above-ring": {
        "f": "włóknista", "y": "łuskowata", "k": "jedwabista", "s": "gładka"
    },
    "stalk-surface-below-ring": {
        "f": "włóknista", "y": "łuskowata", "k": "jedwabista", "s": "gładka"
    },
    "stalk-color-above-ring": {
        "n": "brązowy", "b": "buff", "c": "cynamonowy", "g": "szary", "o": "pomarańczowy", "p": "różowy", "e": "czerwony", "w": "biały", "y": "żółty"
    },
    "stalk-color-below-ring": {
        "n": "brązowy", "b": "buff", "c": "cynamonowy", "g": "szary", "o": "pomarańczowy", "p": "różowy", "e": "czerwony", "w": "biały", "y": "żółty"
    },
    "veil-type": {
        "p": "częściowy"
    },
    "veil-color": {
        "n": "brązowy", "o": "pomarańczowy", "w": "biały", "y": "żółty"
    },
    "ring-number": {
        "n": "brak", "o": "jeden", "t": "dwa"
    },
    "ring-type": {
        "c": "zwężający się", "e": "rozszerzający się", "f": "flarowaty", "l": "duży", "n": "brak", "p": "wiszący", "s": "powierzchniowy", "z": "na zewnątrz"
    },
    "spore-print-color": {
        "k": "czarny", "n": "brązowy", "b": "buff", "h": "czekoladowy", "r": "zielony", "o": "pomarańczowy", "u": "fioletowy", "w": "biały", "y": "żółty"
    },
    "population": {
        "a": "abundant", "c": "clustery", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"
    },
    "habitat": {
        "g": "trawa", "l": "łąka", "m": "moczary", "p": "ścieżka", "u": "miejski", "w": "las", "d": "odpady"
    }
}

model = load_model("mushroomclassifier/data/06_models/best_model")

st.title("🌳 Klasyfikator grzybów")
st.markdown("Wybierz cechy grzyba, a model oceni, czy jest **jadalny** czy **trujący**.")

inputs = {}

with st.form("grzyb_form"):
    for feature, value_map in feature_maps.items():
        label = feature.replace("-", " ").capitalize()
        reverse_map = {v: k for k, v in value_map.items()}
        selection = st.selectbox(label, list(value_map.values()), key=feature)
        inputs[feature] = reverse_map[selection]

    submitted = st.form_submit_button("Sprawdź")

if submitted:
    input_df = pd.DataFrame([inputs])
    result = predict_model(model, data=input_df)
    prediction = result.loc[0, 'prediction_label']
    label = "JADALNY 🍽️" if prediction == 'e' else "TRUJĄCY ☠️"

    st.markdown("### Wynik:")
    st.success(label)
