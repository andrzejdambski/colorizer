# Image de base (Docker utlise sa propre version pas celle qu'on a en local sur l'ordi)
FROM python:3.12-slim

# Pas de .pyc + logs non bufferisés
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dossier de travail dans le conteneur
WORKDIR /app

# Dépendances système utiles pour Pillow / scikit-image / opencv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier tout le projet dans l'image
COPY . .

# Port exposé (Cloud Run utilisera la variable PORT)
EXPOSE 8000

# Commande de démarrage FastAPI
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
