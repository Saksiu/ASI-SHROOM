# Dockerfile do uruchomienia aplikacji Streamlit (Mushroom Classifier)

FROM python:3.10-slim

# Ustawienia katalogu roboczego
WORKDIR /app

# Kopiuj pliki projektu
COPY . /app

# Instalacja zależności
RUN pip install --upgrade pip \
    && pip install -r mushroomclassifier/requirements.txt

# Ustaw port domyślny Streamlit
EXPOSE 8501

# Uruchom aplikację Streamlit
CMD ["streamlit", "run", "mushroomclassifier/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
