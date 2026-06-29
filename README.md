# LLM Finance Fine-Tuning Pipeline

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Cloud_Deploy-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Transformers](https://img.shields.io/badge/Transformers-4.46-FFB000?style=for-the-badge&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA_QLoRA-B91C1C?style=for-the-badge&logoColor=white)
![TRL](https://img.shields.io/badge/TRL-SFTTrainer-5B21B6?style=for-the-badge&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DeepEval](https://img.shields.io/badge/DeepEval-LLM_Evaluation-6B46C1?style=for-the-badge&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=for-the-badge&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-Embeddings-2E7D32?style=for-the-badge&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-Experiment_Tracking-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![Chainlit](https://img.shields.io/badge/Chainlit-Chat_Interface-1C1C1C?style=for-the-badge&logoColor=white)
![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-Dashboard-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI_Server-111827?style=for-the-badge&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Data_Validation-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
Pipeline complet de fine-tuning d'un modèle de langage (Mistral 7B) avec LoRA/QLoRA sur le domaine financier, incluant préparation des données, entraînement, évaluation automatisée avec DeepEval, recherche sémantique avec Pinecone, et déploiement d'une API REST FastAPI.

## Description

Ce projet démontre un pipeline end-to-end de fine-tuning LLM pour créer un assistant financier spécialisé. Il utilise **Parameter-Efficient Fine-Tuning (PEFT)** avec LoRA pour adapter Mistral 7B au dataset finance-alpaca, tout en maintenant une empreinte mémoire réduite. Le projet inclut des outils d'évaluation automatisée (DeepEval), de recherche sémantique (Pinecone), de tracking d'expériences (Weights & Biases), et une API de production déployable.

**Objectif** : Créer un assistant conversationnel capable de répondre à des questions financières complexes (actions, obligations, investissements, etc.) avec précision et clarté.

## Démo en Ligne

**Démo publique (mode mock)** : [https://llm-finetuning-pipeline.onrender.com/](https://llm-finetuning-pipeline.onrender.com/)

**Documentation API interactive** : [https://llm-finetuning-pipeline.onrender.com/docs](https://llm-finetuning-pipeline.onrender.com/docs)

**Code source** : [https://github.com/nicolbl95/LLM-Finetuning-Pipeline](https://github.com/nicolbl95/LLM-Finetuning-Pipeline)

> **Note importante** : La démo publique fonctionne actuellement en **mode mock** pour permettre aux recruteurs de tester l'architecture de l'API sans nécessiter de GPU. Les réponses sont simulées pour démontrer le flux de l'API. Le vrai modèle Mistral 7B fine-tuné avec LoRA sera disponible après entraînement complet sur GPU (Kaggle ou infrastructure locale).

### Comment tester la démo

1. **Accéder à la landing page** : [https://llm-finetuning-pipeline.onrender.com/](https://llm-finetuning-pipeline.onrender.com/)
2. **Cliquer sur "Try the Interactive API Demo"** pour accéder à l'interface Swagger
3. **Tester l'endpoint POST /generate** :
   - Cliquer sur "Try it out"
   - Utiliser l'exemple JSON fourni avec `"use_mock": true`
   - Cliquer sur "Execute" pour voir la réponse simulée

## Fonctionnalités Clés

### Pipeline Complet de Fine-Tuning
- **Préparation automatisée des données** : Téléchargement et formatage du dataset finance-alpaca (1000 exemples)
- **Fine-tuning LoRA/QLoRA** : Adaptation efficace de Mistral 7B avec PEFT
- **Configuration flexible** : Hyperparamètres LoRA ajustables (rang, alpha, dropout, target modules)
- **Support multi-plateforme** : LoRA sur Windows, QLoRA sur Linux/Kaggle

### Évaluation et Observabilité
- **Évaluation automatisée** : DeepEval avec métriques Answer Relevancy et Finance Correctness personnalisée
- **Recherche sémantique** : Pinecone + Sentence Transformers pour trouver des questions similaires
- **Tracking d'expériences** : Intégration Weights & Biases pour suivre la loss, les métriques, et l'utilisation GPU
- **Comparaison avant/après** : Mesure quantitative de l'amélioration apportée par le fine-tuning

### API et Interfaces
- **API REST FastAPI** : Endpoints `/health`, `/info`, `/generate` avec documentation Swagger automatique
- **Mode mock** : Test de l'API sans charger le modèle (idéal pour démo et développement)
- **Chargement lazy** : Le modèle n'est chargé qu'au premier appel réel (économie de ressources)
- **Interface chat Chainlit** : Interface conversationnelle intuitive pour interagir avec l'assistant
- **Dashboard Plotly Dash** : Visualisation des résultats (KPIs, comparaison scores, training loss)

### Déploiement et Production
- **Déploiement Render** : Configuration Docker prête pour déploiement cloud
- **Deux fichiers requirements** : `requirements.txt` (complet) et `requirements-deploy.txt` (léger pour Render)
- **Upload Hugging Face Hub** : Script pour partager le modèle fine-tuné publiquement
- **Gestion d'erreurs robuste** : Messages clairs si le modèle n'est pas disponible ou si la mémoire est insuffisante

## Architecture du Système

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE LLM FINE-TUNING                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Hugging Face   │─────▶│  Préparation     │─────▶│  Dataset    │
│  finance-alpaca │      │  train.jsonl     │      │  JSONL      │
│  (1000 ex.)     │      │  eval.jsonl      │      │  formaté    │
└─────────────────┘      └──────────────────┘      └─────────────┘
                                                           │
                                                           ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Mistral 7B     │─────▶│  Fine-Tuning     │─────▶│  Adaptateurs│
│  (base model)   │      │  LoRA/QLoRA      │      │  LoRA       │
│                 │      │  PEFT + TRL      │      │  (outputs/) │
└─────────────────┘      └──────────────────┘      └─────────────┘
                                                           │
                                                           ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  DeepEval       │◀─────│  Évaluation      │◀─────│  Modèle     │
│  Metrics        │      │  Automatisée     │      │  Fine-tuné  │
│  (Relevancy +   │      │  (base vs tuned) │      │             │
│   Correctness)  │      └──────────────────┘      └─────────────┘
└─────────────────┘                                        │
                                                           ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Pinecone       │◀─────│  Recherche       │      │  FastAPI    │
│  Vector DB      │      │  Sémantique      │      │  REST API   │
│  (embeddings)   │      │  (questions)     │      │  /generate  │
└─────────────────┘      └──────────────────┘      └─────────────┘
                                                           │
                                                           ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Weights &      │◀─────│  Tracking        │      │  Déploiement│
│  Biases         │      │  Expériences     │      │  Render     │
│  (monitoring)   │      │  (loss, metrics) │      │  (Docker)   │
└─────────────────┘      └──────────────────┘      └─────────────┘
```

## Flux de Traitement

### 1. Préparation des Données
```bash
python -m training.prepare_dataset
```
- Télécharge le dataset `gbharti/finance-alpaca` depuis Hugging Face
- Convertit 1000 exemples au format JSONL attendu
- Crée `data/processed/train.jsonl` (900 exemples) et `data/eval/eval.jsonl` (100 exemples)
- Format Alpaca : instruction, input (optionnel), output

### 2. Configuration LoRA
```python
# training/config.py
LoRAConfig(
    r=16,                    # Rang de la décomposition
    lora_alpha=32,           # Facteur de scaling (2x le rang)
    lora_dropout=0.1,        # Dropout pour régularisation
    target_modules=["q_proj", "v_proj"]  # Couches à adapter
)
```

### 3. Fine-Tuning
```bash
# Dry-run (vérification sans entraînement)
python -m training.train --dry-run

# Entraînement local (LoRA sur Windows)
python -m training.train

# Entraînement Kaggle (QLoRA avec GPU T4)
# Voir section "Entraînement sur Kaggle" ci-dessous
```

### 4. Évaluation
```bash
# Évaluer le modèle de base (10 questions)
python -m evaluation.evaluate --model-type base --limit 10

# Évaluer le modèle fine-tuné
python -m evaluation.evaluate --model-type finetuned --limit 10
```

### 5. Recherche Sémantique
```bash
# Indexer les questions d'évaluation dans Pinecone
python -m evaluation.metrics --index

# Rechercher des questions similaires
python -m evaluation.metrics --search "What is a stock?" --top-k 3
```

### 6. Déploiement API
```bash
# Lancer l'API localement
uvicorn api.serve:app --reload --port 8000

# Tester en mode mock
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is a stock?", "use_mock": true}'
```

## Technologies Utilisées

### LLM et Fine-Tuning
- **Transformers** (Hugging Face) : Chargement et manipulation de Mistral 7B
- **PEFT** : Parameter-Efficient Fine-Tuning avec LoRA/QLoRA
- **TRL** : Supervised Fine-Tuning Trainer pour l'entraînement
- **bitsandbytes** : Quantization 4-bit (QLoRA) pour économiser la mémoire
- **Accelerate** : Optimisation multi-GPU et mixed precision

### Évaluation et Observabilité
- **DeepEval** : Framework d'évaluation LLM avec métriques personnalisées
- **Pinecone** : Base de données vectorielle pour recherche sémantique
- **Sentence Transformers** : Génération d'embeddings (all-MiniLM-L6-v2)
- **Weights & Biases** : Tracking d'expériences et monitoring

### API et Interfaces
- **FastAPI** : Framework web moderne pour l'API REST
- **Pydantic** : Validation des données et schémas
- **Chainlit** : Interface de chat conversationnelle
- **Plotly Dash** : Dashboard de visualisation des résultats

### Déploiement
- **Docker** : Containerisation de l'application
- **Render** : Plateforme de déploiement cloud
- **Uvicorn** : Serveur ASGI pour FastAPI

## Structure du Projet

```
LLM-Finetuning-Pipeline/
├── api/
│   └── serve.py                    # API REST FastAPI
├── data/
│   ├── processed/                  # Données d'entraînement (train.jsonl)
│   ├── eval/                       # Données d'évaluation (eval.jsonl, eval_questions.json)
│   └── raw/                        # Données brutes (optionnel)
├── evaluation/
│   ├── evaluate.py                 # Script d'évaluation DeepEval
│   └── metrics.py                  # Métriques personnalisées + Pinecone
├── interfaces/
│   ├── chainlit_app.py             # Interface chat Chainlit
│   └── dash_app.py                 # Dashboard Plotly Dash
├── training/
│   ├── config.py                   # Configuration LoRA et entraînement
│   ├── dataset.py                  # Chargement et préparation des données
│   ├── prepare_dataset.py          # Script de préparation du dataset
│   ├── train.py                    # Script d'entraînement principal
│   └── upload_to_hub.py            # Upload vers Hugging Face Hub
├── outputs/                        # Modèles fine-tunés et résultats
├── .env                            # Variables d'environnement (non versionné)
├── .gitignore
├── Dockerfile                      # Image Docker pour déploiement
├── render.yaml                     # Configuration Render
├── requirements.txt                # Dépendances complètes (développement local)
├── requirements-deploy.txt         # Dépendances minimales (Render)
└── README.md
```

## Installation Locale

### Prérequis
- Python 3.12+
- Git
- (Optionnel) GPU NVIDIA avec CUDA pour entraînement local
- (Optionnel) Compte Kaggle pour entraînement GPU gratuit

### Cloner le Repository
```bash
git clone https://github.com/nicolbl95/LLM-Finetuning-Pipeline.git
cd LLM-Finetuning-Pipeline
```

### Installer les Dépendances

**Pour le développement local complet** (fine-tuning, évaluation, interfaces) :
```bash
pip install -r requirements.txt
```

**Pour tester uniquement l'API en mode mock** (sans dépendances lourdes) :
```bash
pip install -r requirements-deploy.txt
```

> **Note** : `requirements.txt` inclut torch, transformers, peft, deepeval, pinecone, etc. (plusieurs GB). `requirements-deploy.txt` inclut uniquement FastAPI et ses dépendances (quelques MB).

## Configuration des Variables d'Environnement

Créez un fichier `.env` à la racine du projet :

```env
# Pinecone (recherche sémantique)
PINECONE_API_KEY=votre_cle_api_pinecone

# OpenAI (évaluation DeepEval)
OPENAI_API_KEY=votre_cle_api_openai

# Hugging Face (upload modèle)
HF_TOKEN=votre_token_huggingface

# Weights & Biases (tracking expériences - optionnel)
WANDB_API_KEY=votre_cle_api_wandb

# Mode API (mock ou production)
APP_MODE=mock  # ou "production" pour charger le vrai modèle
```

> **Important** : Ne commitez JAMAIS le fichier `.env` dans Git. Il est déjà dans `.gitignore`.

### Obtenir les Clés API

- **Pinecone** : [https://www.pinecone.io/](https://www.pinecone.io/) (gratuit jusqu'à 1 index)
- **OpenAI** : [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) (payant, ~0.01-0.05$ par question évaluée)
- **Hugging Face** : [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (gratuit, permissions "write")
- **Weights & Biases** : [https://wandb.ai/authorize](https://wandb.ai/authorize) (gratuit pour projets publics)

## Lancement Local

### 1. Préparer le Dataset
```bash
python -m training.prepare_dataset
```
Crée `data/processed/train.jsonl` et `data/eval/eval.jsonl` depuis le dataset finance-alpaca.

### 2. Vérifier la Configuration (Dry-Run)
```bash
python -m training.train --dry-run
```
Vérifie que le dataset est correctement formaté sans lancer l'entraînement.

### 3. Entraîner le Modèle

**Sur Windows (LoRA uniquement)** :
```bash
# Dans training/config.py, assurez-vous que use_4bit=False
python -m training.train
```

**Sur Kaggle (QLoRA avec GPU T4 gratuit)** :
1. Créer un notebook Kaggle : [https://www.kaggle.com/code](https://www.kaggle.com/code)
2. Activer le GPU : Settings → Accelerator → GPU T4 x2
3. Installer les dépendances :
   ```python
   !pip install -q transformers datasets peft trl bitsandbytes accelerate wandb
   ```
4. Cloner le repo ou uploader le code
5. Configurer pour QLoRA :
   ```python
   from training.config import TrainingConfig, LoRAConfig
   cfg = TrainingConfig()
   cfg.use_4bit = True  # Activer QLoRA
   cfg.output_dir = "/kaggle/working/mistral-finance-lora"
   ```
6. Lancer l'entraînement :
   ```python
   from training.train import train
   train(cfg, LoRAConfig())
   ```

**Temps estimé** : 30-45 minutes sur GPU T4 (Kaggle) pour 1000 exemples, 3 epochs.

### 4. Évaluer le Modèle
```bash
# Évaluer le modèle de base (10 questions)
python -m evaluation.evaluate --model-type base --limit 10

# Évaluer le modèle fine-tuné (10 questions)
python -m evaluation.evaluate --model-type finetuned --limit 10
```

Les résultats sont sauvegardés dans `outputs/evaluation/` avec timestamp.

### 5. Indexer les Questions (Pinecone)
```bash
python -m evaluation.metrics --index
```
Crée un index Pinecone avec les embeddings des questions d'évaluation.

### 6. Lancer l'API
```bash
uvicorn api.serve:app --reload --port 8000
```
Accédez à la documentation Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. Lancer l'Interface Chat (Optionnel)
```bash
# Terminal 1 : API
uvicorn api.serve:app --reload --port 8000

# Terminal 2 : Chainlit
chainlit run interfaces/chainlit_app.py --port 8080
```
Accédez à l'interface : [http://localhost:8080](http://localhost:8080)

### 8. Lancer le Dashboard (Optionnel)
```bash
python interfaces/dash_app.py
```
Accédez au dashboard : [http://localhost:8050](http://localhost:8050)

## Déploiement Render

Le projet est configuré pour un déploiement automatique sur Render via Docker.

### Fichiers de Configuration
- **Dockerfile** : Image Docker avec Python 3.12 slim
- **render.yaml** : Configuration du service web Render
- **.dockerignore** : Exclusion des fichiers inutiles (data/, outputs/, notebooks/)

### Variables d'Environnement Render
Dans le dashboard Render, configurez :
- `APP_MODE=mock` : Force le mode mock (pas de chargement du modèle)
- `PYTHONUNBUFFERED=1` : Logs en temps réel

### Déployer sur Render
1. Connectez votre repository GitHub à Render
2. Render détectera automatiquement le `render.yaml`
3. Le service sera déployé avec le plan gratuit
4. L'API sera accessible via l'URL fournie par Render (ex: `https://llm-finetuning-pipeline.onrender.com/`)

### Architecture du Déploiement Render
- Utilise `requirements-deploy.txt` (léger, sans torch/transformers/peft)
- Les imports lourds sont faits uniquement dans `load_model_and_tokenizer()`
- En mode mock, ces imports ne sont jamais exécutés
- L'API démarre en quelques secondes sans charger Mistral 7B
- Les endpoints `/health` et `/info` fonctionnent sans dépendances lourdes
- Le endpoint `/generate` avec `use_mock=true` retourne des réponses simulées
- Le endpoint `/generate` avec `use_mock=false` retourne une erreur HTTP 503 claire

### Pourquoi Deux Fichiers Requirements ?
- **requirements.txt** : Environnement complet pour développement local (inclut torch, transformers, peft, deepeval, pinecone, etc.)
- **requirements-deploy.txt** : Dépendances minimales pour Render (FastAPI uniquement, pas de pywin32 ni bibliothèques lourdes)

Cela permet de :
- Déployer rapidement sur Render sans installer des GB de dépendances
- Éviter les erreurs de compatibilité (pywin32 sur Linux)
- Démontrer l'architecture de l'API sans nécessiter de GPU

## Évaluation et Observabilité

### Métriques DeepEval

Le projet utilise deux métriques principales :

1. **Answer Relevancy** (seuil : 0.7)
   - Mesure si la réponse est pertinente par rapport à la question
   - Score de 0 à 1 (1 = parfaitement pertinent)

2. **Finance Correctness** (métrique personnalisée, seuil : 0.7)
   - Évalue l'exactitude financière des informations
   - Vérifie la clarté de la réponse
   - Détecte les contradictions avec la réponse attendue
   - Score de 0 à 1 (1 = parfaitement correct)

### Recherche Sémantique avec Pinecone

Permet de trouver des questions similaires dans le dataset d'évaluation :

```bash
# Indexer les questions
python -m evaluation.metrics --index

# Rechercher des questions similaires
python -m evaluation.metrics --search "What is a stock?" --top-k 3
```

**Cas d'usage** :
- Trouver des questions d'évaluation existantes avant d'en créer de nouvelles
- Identifier des doublons dans le dataset
- Explorer le contenu du dataset d'évaluation
- Tester la couverture thématique du dataset

### Tracking avec Weights & Biases

Pour suivre vos expériences d'entraînement :

```python
# Dans training/config.py
cfg = TrainingConfig()
cfg.report_to_wandb = True
cfg.wandb_project = "mistral-finance-finetuning"
```

Puis dans votre notebook Kaggle ou terminal local :
```python
import wandb
wandb.login()  # Entrez votre clé API W&B
```

Vous pourrez suivre en temps réel :
- La loss d'entraînement
- L'utilisation de la mémoire GPU
- Le temps par epoch
- Les gradients et poids du modèle

Accédez à votre dashboard : [https://wandb.ai/](https://wandb.ai/)

## Exemples de Requêtes

### Tester l'API en Mode Mock

**Avec curl** :
```bash
curl -X POST "https://llm-finetuning-pipeline.onrender.com/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Quelle est la difference entre une action et une obligation ?",
       "max_tokens": 128,
       "use_mock": true
     }'
```

**Avec Python** :
```python
import requests

response = requests.post(
    "https://llm-finetuning-pipeline.onrender.com/generate",
    json={
        "question": "Quelle est la difference entre une action et une obligation ?",
        "max_tokens": 128,
        "use_mock": True
    }
)
print(response.json())
```

**Réponse attendue** :
```json
{
  "answer": "[MODE DEMO] Reponse simulee pour la question : 'Quelle est la difference entre une action et une obligation ?'. Cette API est une demonstration du pipeline LLM fine-tuning. Le modele Mistral 7B fine-tune avec LoRA sur le dataset finance-alpaca sera disponible apres entrainement complet. Stack technique : FastAPI + Transformers + PEFT + DeepEval + Pinecone + W&B. Consultez /info pour plus de details sur le projet.",
  "model": "mock-model",
  "mode": "mock"
}
```

### Tester l'API en Mode Réel (Après Entraînement)

**Avec curl** :
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is a stock?",
       "max_tokens": 256,
       "use_mock": false
     }'
```

**Avec Python** :
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "question": "What is a stock?",
        "max_tokens": 256,
        "use_mock": False
    }
)
print(response.json())
```

> **Note** : Le mode réel nécessite 16+ GB de RAM ou un GPU. Sur Render, `use_mock=false` retournera une erreur HTTP 503 car les dépendances lourdes ne sont pas installées.

## Limites Actuelles

### Démo Publique en Mode Mock
- La démo Render fonctionne actuellement en **mode mock** uniquement
- Les réponses sont simulées pour démontrer l'architecture de l'API
- Le vrai modèle Mistral 7B fine-tuné sera disponible après entraînement complet sur GPU

### Ressources Matérielles
- **Entraînement local** : Nécessite 16+ GB de VRAM (GPU) ou beaucoup de RAM (CPU lent)
- **QLoRA sur Windows** : Non supporté (bitsandbytes nécessite Linux)
- **Déploiement production** : Nécessite un service avec GPU pour le mode réel

### Coûts
- **OpenAI API** : Évaluation DeepEval consomme des tokens (~0.01-0.05$ par question)
- **Pinecone** : Gratuit jusqu'à 1 index, puis payant
- **Render** : Plan gratuit limité (750h/mois), puis payant pour GPU

### Dataset
- **Taille limitée** : 1000 exemples du dataset finance-alpaca
- **Domaine spécifique** : Finance uniquement (actions, obligations, investissements)
- **Langue** : Principalement anglais

## Améliorations Possibles

### Court Terme
- [ ] Entraîner le modèle sur GPU et déployer les adaptateurs LoRA
- [ ] Ajouter plus d'exemples au dataset (5000-10000)
- [ ] Créer des métriques d'évaluation supplémentaires (Faithfulness, Contextual Relevancy)
- [ ] Implémenter un cache Redis pour les réponses fréquentes

### Moyen Terme
- [ ] Ajouter un système de RAG (Retrieval-Augmented Generation) avec Pinecone
- [ ] Implémenter un fine-tuning multi-domaines (finance + legal + medical)
- [ ] Créer une interface web React pour remplacer Chainlit
- [ ] Ajouter des tests unitaires et d'intégration

### Long Terme
- [ ] Déployer sur AWS/GCP avec GPU pour le mode production
- [ ] Implémenter un système de feedback utilisateur pour améliorer le modèle
- [ ] Créer un pipeline CI/CD pour automatiser l'entraînement et le déploiement
- [ ] Ajouter un système de monitoring et d'alertes (Prometheus + Grafana)

## Auteur

**Nicolas Blondeau**
- GitHub : [nicolbl95](https://github.com/nicolbl95)
- LinkedIn : [Nicolas Blondeau](https://www.linkedin.com/in/nicolas-blondeau-data/)
- Email : nicolas.blondeau@example.com

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## Résumé

Ce projet démontre un pipeline complet de fine-tuning LLM pour créer un assistant financier spécialisé. Il couvre toutes les étapes du cycle de vie d'un modèle de langage : préparation des données, fine-tuning avec LoRA/QLoRA, évaluation automatisée avec DeepEval, recherche sémantique avec Pinecone, tracking avec Weights & Biases, et déploiement d'une API REST FastAPI.

**Points forts** :
- Architecture modulaire et extensible
- Configuration flexible des hyperparamètres LoRA
- Évaluation automatisée avec métriques personnalisées
- Recherche sémantique pour explorer le dataset
- API REST avec mode mock pour démo sans GPU
- Déploiement Render prêt pour production
- Documentation complète et exemples de code

**Idéal pour** :
- Portfolio de LLM Engineer / ML Engineer
- Démonstration de compétences en fine-tuning LLM
- Base de départ pour des projets de fine-tuning personnalisés
- Apprentissage des bonnes pratiques LLMOps

**Technologies clés** : Transformers, PEFT, LoRA, QLoRA, DeepEval, Pinecone, FastAPI, Docker, Render, Weights & Biases.
