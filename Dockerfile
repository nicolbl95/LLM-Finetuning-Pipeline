# Image de base Python 3.12 slim pour reduire la taille
FROM python:3.12-slim

# Definir le repertoire de travail
WORKDIR /app

# Copier le fichier des dependances pour le deploiement
# Note : requirements-deploy.txt contient uniquement les dependances necessaires
# pour l'API FastAPI en mode mock (pas de torch, transformers, pywin32, etc.)
COPY requirements-deploy.txt .

# Installer les dependances Python
# --no-cache-dir : ne pas garder le cache pip pour reduire la taille
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copier tout le projet dans le conteneur
COPY . .

# Exposer le port 8000 (FastAPI)
EXPOSE 8000

# Lancer l'API FastAPI avec uvicorn
# --host 0.0.0.0 : ecouter sur toutes les interfaces (necessaire pour Docker)
# --port ${PORT:-8000} : utiliser la variable PORT de Render ou 8000 par defaut
CMD uvicorn api.serve:app --host 0.0.0.0 --port ${PORT:-8000}
