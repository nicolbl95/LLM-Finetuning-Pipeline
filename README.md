# Projet de Fine-Tuning LoRA

Ce projet permet de fine-tuner un modèle de langage avec LoRA sur le dataset finance-alpaca.

## Installation

### Environnement de développement local complet

Pour le développement local avec toutes les fonctionnalités (fine-tuning, évaluation, etc.) :

```bash
pip install -r requirements.txt
```

**Note** : `requirements.txt` contient toutes les dépendances nécessaires pour :
- Fine-tuning LoRA/QLoRA (transformers, peft, bitsandbytes)
- Évaluation avec DeepEval
- Recherche sémantique avec Pinecone
- Interfaces Dash et Chainlit
- API FastAPI complète

### Déploiement Render (mode mock uniquement)

Pour le déploiement sur Render, un fichier `requirements-deploy.txt` allégé est utilisé :

```bash
pip install -r requirements-deploy.txt
```

**Note** : `requirements-deploy.txt` contient uniquement les dépendances minimales pour l'API FastAPI en mode mock :
- fastapi
- uvicorn[standard]
- pydantic
- python-dotenv

Ce fichier **exclut volontairement** :
- Les dépendances Windows (pywin32) qui causent des erreurs sur Linux
- Les bibliothèques lourdes (torch, transformers, peft) non nécessaires en mode mock
- Les outils d'évaluation (deepeval, sentence-transformers, pinecone)

**Architecture technique** :
- Les imports lourds (transformers, peft, torch) sont faits **à l'intérieur** de `load_model_and_tokenizer()` uniquement
- En mode mock (`APP_MODE=mock`), ces imports ne sont jamais exécutés
- L'API démarre rapidement sans charger Mistral 7B
- Les endpoints `/health` et `/info` fonctionnent sans dépendances lourdes
- Le endpoint `/generate` retourne des réponses mock sans charger le modèle

Le déploiement Render utilise `APP_MODE=mock` pour éviter de charger le modèle Mistral 7B et fonctionne sans GPU ni clés API.

**Si quelqu'un essaie `use_mock=false` sur Render** :
- L'API retourne une erreur HTTP 503 claire
- Le message explique que cette instance est en mode demo uniquement
- Le message recommande d'utiliser `use_mock=true` ou de déployer localement avec `requirements.txt` complet

## Préparation du Dataset

Avant de commencer l'entraînement, vous devez préparer le dataset :

```bash
python -m training.prepare_dataset
```

Cette commande va :
- Télécharger le dataset `gbharti/finance-alpaca` depuis Hugging Face
- Convertir 1000 exemples au format JSONL attendu
- Créer `data/processed/train.jsonl` (900 exemples)
- Créer `data/eval/eval.jsonl` (100 exemples)

Les dossiers nécessaires seront créés automatiquement.

## Configuration des Hyperparamètres LoRA

Le fichier `training/config.py` contient deux dataclasses principales :

### LoRAConfig - Paramètres LoRA

```python
@dataclass
class LoRAConfig:
    r: int = 16                    # Rang de la décomposition (8, 16, 32, 64)
    lora_alpha: int = 32           # Facteur de scaling (généralement 2x le rang)
    lora_dropout: float = 0.1      # Dropout pour régularisation
    target_modules: List[str]      # Modules à adapter (ex: ["q_proj", "v_proj"])
```

**Explications :**
- **r (rang)** : Plus le rang est élevé, plus le modèle peut apprendre de variations, mais plus il consomme de mémoire. Commencez avec 16.
- **lora_alpha** : Contrôle l'importance des adaptations LoRA. Règle générale : `lora_alpha = 2 * r`.
- **lora_dropout** : Aide à éviter le surapprentissage. Valeurs typiques : 0.05 à 0.1.
- **target_modules** : Couches du modèle à adapter. Pour Mistral/Llama : `["q_proj", "v_proj"]` est un bon début.

### TrainingConfig - Paramètres d'entraînement

Les paramètres clés :
- `model_name` : Modèle Hugging Face à fine-tuner (ex: "mistralai/Mistral-7B-v0.1")
- `num_epochs` : Nombre de passages sur le dataset (3 est généralement suffisant)
- `batch_size` : Taille du batch par GPU (réduire si manque de mémoire)
- `learning_rate` : Taux d'apprentissage (2e-4 est une bonne valeur pour LoRA)
- `use_4bit` : Quantization 4-bit (QLoRA) - **ATTENTION : Ne fonctionne PAS sur Windows !**

### QLoRA vs LoRA

- **LoRA** (`use_4bit=False`) : Compatible Windows, utilise plus de mémoire
- **QLoRA** (`use_4bit=True`) : Nécessite Linux/Kaggle, économise beaucoup de mémoire

**Sur Windows, laissez `use_4bit=False`**. Sur Kaggle/Linux, vous pouvez activer `use_4bit=True` pour réduire l'utilisation mémoire.

### Vérifier la configuration

```bash
python -m training.config
```

Cette commande affiche tous les paramètres configurés.

## Entraînement

### Vérification avant entraînement (Dry-Run)

Avant de lancer un entraînement complet, vous pouvez vérifier que tout est correctement configuré :

```bash
python -m training.train --dry-run
```

Cette commande va :
- Charger la configuration
- Charger le tokenizer
- Charger le dataset
- Afficher un exemple formaté
- **Ne PAS lancer l'entraînement**

C'est utile pour vérifier que vos données sont au bon format sans consommer de ressources GPU.

### Entraînement local (Windows - LoRA uniquement)

Sur Windows, la quantization 4-bit (QLoRA) ne fonctionne pas. Utilisez LoRA classique :

1. Dans `training/config.py`, assurez-vous que `use_4bit=False`
2. Lancez l'entraînement :

```bash
python -m training.train
```

**Attention** : L'entraînement d'un modèle 7B comme Mistral nécessite beaucoup de mémoire GPU (16+ GB). Sur Windows sans GPU puissant, préférez Kaggle.

### Entraînement sur Kaggle (Recommandé)

Kaggle offre des GPUs gratuits (T4 avec 16GB de VRAM) parfaits pour le fine-tuning.

**Étapes :**

1. **Créer un nouveau notebook Kaggle**
   - Allez sur [kaggle.com/code](https://www.kaggle.com/code)
   - Cliquez sur "New Notebook"
   - Activez le GPU : Settings → Accelerator → GPU T4 x2

2. **Uploader votre code**
   - Créez un dataset Kaggle avec votre code :
     - Compressez votre dossier `training/` en ZIP
     - Uploadez sur Kaggle Datasets
   - Ou clonez depuis GitHub directement dans le notebook

3. **Installer les dépendances**
   ```python
   !pip install -q transformers datasets peft trl bitsandbytes accelerate wandb
   ```

4. **Configurer pour QLoRA**
   Dans votre notebook, modifiez la config :
   ```python
   from training.config import TrainingConfig, LoRAConfig
   
   cfg = TrainingConfig()
   cfg.use_4bit = True  # Activer QLoRA sur Kaggle
   cfg.num_epochs = 3
   cfg.output_dir = "/kaggle/working/mistral-finance-lora"
   
   lora_cfg = LoRAConfig()
   ```

5. **Lancer l'entraînement**
   ```python
   from training.train import train
   train(cfg, lora_cfg)
   ```

6. **Télécharger le modèle**
   - Le modèle sera sauvegardé dans `/kaggle/working/mistral-finance-lora`
   - Téléchargez les fichiers depuis l'interface Kaggle
   - Ou sauvegardez directement sur Kaggle Datasets

**Astuce W&B** : Pour suivre vos expériences avec Weights & Biases sur Kaggle :
```python
import wandb
wandb.login()  # Entrez votre clé API W&B

cfg.report_to_wandb = True
cfg.wandb_project = "mistral-finance-finetuning"
```

### Monitoring de l'entraînement

Si vous avez activé Weights & Biases (`report_to_wandb=True`), vous pouvez suivre :
- La loss d'entraînement en temps réel
- L'utilisation de la mémoire GPU
- Le temps par epoch
- Les gradients et poids du modèle

Accédez à votre dashboard W&B : https://wandb.ai/

### Temps d'entraînement estimé

Sur GPU T4 (Kaggle) avec QLoRA :
- 1000 exemples, 3 epochs : ~30-45 minutes
- 5000 exemples, 3 epochs : ~2-3 heures

Sur CPU (déconseillé) : plusieurs jours

## Évaluation

L'évaluation automatisée utilise DeepEval pour mesurer la qualité des réponses générées par le modèle.

### Recherche Sémantique avec Pinecone

Le projet inclut une fonctionnalité de recherche sémantique pour trouver des questions similaires dans le dataset d'évaluation.

#### Configuration Pinecone

1. **Créer un compte Pinecone** : https://www.pinecone.io/
2. **Obtenir votre clé API** depuis le dashboard Pinecone
3. **Créer un fichier `.env`** à la racine du projet :
```
PINECONE_API_KEY=votre_cle_api_pinecone
OPENAI_API_KEY=votre_cle_api_openai
```

#### Indexer les questions d'évaluation

Avant de pouvoir rechercher des questions similaires, vous devez indexer le dataset :

```bash
python -m evaluation.metrics --index
```

Cette commande va :
- Charger les questions depuis `data/eval/eval_questions.json`
- Générer les embeddings avec Sentence Transformers (all-MiniLM-L6-v2)
- Créer un index Pinecone nommé `eval-index`
- Indexer toutes les questions avec leurs métadonnées

L'indexation prend environ 1-2 minutes pour 100 questions.

#### Rechercher des questions similaires

Une fois l'index créé, vous pouvez rechercher des questions similaires :

```bash
# Rechercher les 3 questions les plus similaires
python -m evaluation.metrics --search "What is a stock?"

# Rechercher les 5 questions les plus similaires
python -m evaluation.metrics --search "How to invest in bonds?" --top-k 5
```

La recherche retourne :
- Les questions similaires trouvées
- Leurs réponses attendues
- Un score de similarité (0 à 1, 1 = identique)

**Cas d'usage** :
- Trouver des questions d'évaluation existantes avant d'en créer de nouvelles
- Identifier des doublons dans le dataset
- Explorer le contenu du dataset d'évaluation
- Tester la couverture thématique du dataset

### Prérequis

1. **Installer DeepEval** :
```bash
pip install deepeval
```

2. **Configurer OpenAI API** :
DeepEval utilise OpenAI pour évaluer les réponses. Créez un fichier `.env` à la racine du projet :
```
OPENAI_API_KEY=votre_cle_api_openai
```

3. **Fichier de questions** :
Le fichier `data/eval/eval_questions.json` doit contenir vos questions de test au format :
```json
[
  {
    "question": "What is a stock?",
    "expected_answer": "A stock represents ownership in a company..."
  }
]
```

### Vérification rapide (Dry-Run)

Avant de lancer une évaluation complète, vérifiez que tout est correctement configuré :

```bash
python -m evaluation.evaluate --dry-run
```

Cette commande :
- Lit les 2 premières questions du fichier d'évaluation
- Affiche les prompts qui seront utilisés
- Vérifie que le fichier JSON est au bon format
- **Ne charge PAS le modèle Mistral 7B** (rapide, pas de GPU nécessaire)

### Évaluer le modèle de base

Pour évaluer le modèle **avant** le fine-tuning :

```bash
# Évaluer sur 10 questions (recommandé pour un test rapide)
python -m evaluation.evaluate --model-type base --limit 10

# Évaluer sur toutes les questions
python -m evaluation.evaluate --model-type base
```

**Attention** : Le chargement de Mistral 7B nécessite beaucoup de mémoire (16+ GB de RAM ou GPU).

### Évaluer le modèle fine-tuné

Après avoir entraîné votre modèle avec `python -m training.train`, évaluez-le :

```bash
# Évaluer le modèle fine-tuné (10 questions)
python -m evaluation.evaluate --model-type finetuned --limit 10

# Évaluer avec un chemin d'adaptateur personnalisé
python -m evaluation.evaluate --model-type finetuned --adapter-path mon/chemin/custom

# Évaluer sur toutes les questions
python -m evaluation.evaluate --model-type finetuned
```

### Métriques utilisées

DeepEval évalue les réponses selon deux métriques :

1. **Answer Relevancy** (seuil : 0.7)
   - Mesure si la réponse est pertinente par rapport à la question
   - Score de 0 à 1 (1 = parfaitement pertinent)

2. **Finance Correctness** (seuil : 0.7)
   - Métrique personnalisée qui évalue :
     - L'exactitude financière des informations
     - La clarté de la réponse
     - L'absence de contradictions avec la réponse attendue
   - Score de 0 à 1 (1 = parfaitement correct)

### Résultats

Les résultats sont sauvegardés dans `outputs/evaluation/` avec un timestamp :
- `base_model_YYYYMMDD_HHMMSS_results.json` : Résultats du modèle de base
- `finetuned_model_YYYYMMDD_HHMMSS_results.json` : Résultats du modèle fine-tuné

Vous pouvez comparer les scores pour mesurer l'amélioration apportée par le fine-tuning.

### Conseils

- **Commencez petit** : Utilisez `--limit 10` pour tester rapidement
- **GPU recommandé** : L'évaluation est beaucoup plus rapide avec un GPU
- **Coût OpenAI** : Chaque évaluation consomme des tokens OpenAI (environ 0.01-0.05$ par question)
- **Kaggle** : Vous pouvez aussi évaluer sur Kaggle avec un GPU gratuit

## Live Demo - FastAPI on Render

Une démo en ligne de l'API est disponible sur Render pour les recruteurs et testeurs.

### Accès à la démo

**URL principale de la démo** : `https://votre-app.onrender.com/`

Cette page d'accueil présente le projet de manière claire et professionnelle, avec :
- Objectif du projet et stack technique
- Explication du mode mock actuel
- Liens vers tous les endpoints disponibles
- Exemple d'utilisation de l'API

**URL de la documentation API interactive (Swagger UI)** : `https://votre-app.onrender.com/docs`

Cette interface permet de tester tous les endpoints de l'API directement depuis le navigateur.

### Endpoints disponibles

#### GET /
Page d'accueil de la démo avec présentation du projet, stack technique, et guide d'utilisation.

**Exemple de réponse** : Page HTML professionnelle avec toutes les informations pour les recruteurs.

#### GET /health
Vérifie que l'API fonctionne correctement.

**Exemple de réponse** :
```json
{
  "status": "healthy",
  "model_loaded": false,
  "model_name": "not loaded yet"
}
```

#### GET /info
Informations détaillées sur l'API, le modèle, et la stack technique.

**Exemple de réponse** :
```json
{
  "api_version": "1.0.0",
  "deployment": "Render",
  "api_framework": "FastAPI",
  "app_mode": "mock",
  "base_model": "mistralai/Mistral-7B-v0.1",
  "llmops_stack": {
    "evaluation": "DeepEval",
    "vector_db": "Pinecone",
    "experiment_tracking": "Weights & Biases"
  },
  "note": "Le modele Mistral 7B fine-tune sera disponible apres entrainement LoRA. Mode mock actif pour demo."
}
```

#### POST /generate
Génère une réponse à partir d'une question financière.

**Corps de la requête** :
```json
{
  "question": "Quelle est la difference entre une action et une obligation ?",
  "max_tokens": 128,
  "use_mock": true
}
```

**Exemple de réponse** :
```json
{
  "answer": "[MODE DEMO] Reponse simulee pour la question : 'Quelle est la difference entre une action et une obligation ?'. Cette API est une demonstration du pipeline LLM fine-tuning. Le modele Mistral 7B fine-tune avec LoRA sur le dataset finance-alpaca sera disponible apres entrainement complet. Stack technique : FastAPI + Transformers + PEFT + DeepEval + Pinecone + W&B. Consultez /info pour plus de details sur le projet.",
  "model": "mock-model",
  "mode": "mock"
}
```

### Mode actuel : Mock

La démo sur Render fonctionne actuellement en **mode mock** :
- Aucun modèle Mistral 7B n'est chargé (économie de ressources)
- Les réponses sont simulées pour démontrer l'architecture de l'API
- Le mode mock permet de tester l'API sans GPU ni modèle lourd
- Utilise `requirements-deploy.txt` (dépendances minimales) au lieu de `requirements.txt` (complet)

**Pourquoi deux fichiers requirements ?**
- `requirements.txt` : Environnement complet pour développement local (inclut torch, transformers, peft, etc.)
- `requirements-deploy.txt` : Dépendances minimales pour Render (FastAPI uniquement, pas de pywin32 ni bibliothèques lourdes)

**Après l'entraînement LoRA** :
- Le modèle Mistral 7B fine-tuné sera disponible
- Le mode `use_mock=false` permettra d'obtenir de vraies réponses générées par le modèle
- Les adaptateurs LoRA seront chargés automatiquement
- Il faudra alors utiliser `requirements.txt` complet et un service avec GPU

### Tester l'API avec curl

```bash
# Verifier la sante de l'API
curl https://votre-app.onrender.com/health

# Obtenir les informations
curl https://votre-app.onrender.com/info

# Generer une reponse (mode mock - RECOMMANDE sur Render)
curl -X POST "https://votre-app.onrender.com/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Quelle est la difference entre une action et une obligation ?",
       "max_tokens": 128,
       "use_mock": true
     }'

# Note : use_mock=false ne fonctionnera PAS sur Render car transformers/peft/torch
# ne sont pas installes (requirements-deploy.txt est volontairement leger).
# Pour utiliser le modele reel, deployez localement avec requirements.txt complet.
```

### Tester l'API avec Python

```python
import requests

BASE_URL = "https://votre-app.onrender.com"

# Verifier la sante
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Obtenir les informations
response = requests.get(f"{BASE_URL}/info")
print(response.json())

# Generer une reponse (mode mock - RECOMMANDE sur Render)
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "question": "Quelle est la difference entre une action et une obligation ?",
        "max_tokens": 128,
        "use_mock": True  # IMPORTANT : Toujours True sur Render
    }
)
print(response.json())

# Note : use_mock=False retournera une erreur HTTP 503 sur Render
# car les dependances lourdes ne sont pas installees.
```

### Déploiement sur Render

Le projet est configuré pour un déploiement automatique sur Render via Docker.

**Fichiers de configuration** :
- `Dockerfile` : Image Docker avec Python 3.12 slim
- `render.yaml` : Configuration du service web Render
- `.dockerignore` : Exclusion des fichiers inutiles

**Variables d'environnement sur Render** :
- `APP_MODE=mock` : Force le mode mock (pas de chargement du modele)
- `PYTHONUNBUFFERED=1` : Logs en temps reel

**Architecture technique du deploiement Render** :
- Utilise `requirements-deploy.txt` (leger, sans transformers/peft/torch)
- Les imports lourds sont faits uniquement dans `load_model_and_tokenizer()`
- En mode mock, ces imports ne sont jamais executes
- L'API demarre en quelques secondes sans charger Mistral 7B
- Les endpoints `/health` et `/info` fonctionnent sans dependances lourdes
- Le endpoint `/generate` avec `use_mock=true` retourne des reponses demo
- Le endpoint `/generate` avec `use_mock=false` retourne une erreur HTTP 503 claire

**Pour deployer** :
1. Connectez votre repo GitHub a Render
2. Render detectera automatiquement le `render.yaml`
3. Le service sera deploye avec le plan gratuit
4. L'API sera accessible via l'URL fournie par Render
5. Testez avec `use_mock=true` uniquement

### Stack technique démontrée

- **API** : FastAPI avec documentation Swagger automatique
- **LLM** : Mistral 7B (base) + LoRA fine-tuning
- **Fine-tuning** : PEFT (Parameter-Efficient Fine-Tuning)
- **Dataset** : finance-alpaca (1000 exemples)
- **Évaluation** : DeepEval avec métriques personnalisées
- **Vector DB** : Pinecone pour recherche sémantique
- **Tracking** : Weights & Biases pour suivi des expériences
- **Déploiement** : Docker + Render

## Dashboard de Visualisation

Un dashboard Plotly Dash est disponible pour visualiser les résultats du fine-tuning.

### Lancer le dashboard

```bash
python interfaces/dash_app.py
```

Puis ouvrez dans votre navigateur : http://localhost:8050

### Fonctionnalités du dashboard

Le dashboard affiche :
- **4 KPI Cards** : Amélioration des métriques, pourcentage de paramètres entraînés, taille des adaptateurs LoRA
- **Graphique de comparaison** : Scores avant/après fine-tuning (Answer Relevancy et Finance Correctness)
- **Courbe de training loss** : Évolution de la loss pendant l'entraînement
- **Informations sur le projet** : Modèle, méthode, domaine, dataset, configuration LoRA

**Note importante** : Les scores affichés sont des **données mock** tant que l'entraînement réel n'est pas terminé. Une fois que vous aurez lancé `python -m training.train` et `python -m evaluation.evaluate`, le dashboard pourra être mis à jour pour afficher les vrais résultats.

### Utilité pour un portfolio

Ce dashboard est conçu pour être présenté à des recruteurs :
- Interface professionnelle et claire
- Visualisations pertinentes pour comprendre l'impact du fine-tuning
- Métriques techniques (LoRA) et business (amélioration des scores)
- Fonctionne immédiatement sans dépendances lourdes (pas besoin de charger Mistral 7B)

## API REST

Une API FastAPI est disponible pour exposer le modèle fine-tuné et générer des réponses.

### Lancer l'API

```bash
uvicorn api.serve:app --reload --port 8000
```

Puis ouvrez dans votre navigateur : http://localhost:8000/docs

## Interface Chat Chainlit

Une interface de chat conviviale est disponible pour interagir avec l'assistant financier.

### Lancer l'interface Chainlit

**Prérequis** : L'API FastAPI doit être démarrée (voir section précédente).

1. **Démarrer l'API FastAPI** (dans un premier terminal) :
```bash
uvicorn api.serve:app --reload --port 8000
```

2. **Lancer Chainlit** (dans un second terminal) :
```bash
chainlit run interfaces/chainlit_app.py --port 8080
```

3. **Ouvrir l'interface** dans votre navigateur : http://localhost:8080

### Fonctionnalités

- Interface de chat intuitive et professionnelle
- Streaming des réponses en temps réel
- Historique de conversation par session
- Gestion d'erreurs claire si l'API n'est pas disponible
- Mode mock par défaut (pas besoin de charger Mistral 7B pour tester)

### Mode Mock vs Mode Réel

Par défaut, l'interface utilise `use_mock=True` pour tester sans charger le modèle Mistral 7B.

Pour utiliser le vrai modèle fine-tuné, modifiez `interfaces/chainlit_app.py` :
```python
payload = {
    "question": message.content,
    "max_tokens": 512,
    "use_mock": False  # Utiliser le vrai modèle
}
```

**Attention** : Le mode réel nécessite 16+ GB de RAM ou un GPU.

### Endpoints disponibles

- **GET /health** : Vérifier que l'API fonctionne
- **GET /info** : Informations sur le modèle et sa disponibilité
- **POST /generate** : Générer une réponse à partir d'une question

### Mode Mock (sans charger le modèle)

Pour tester l'API sans charger le modèle Mistral 7B (utile sur Windows ou machines avec peu de RAM) :

```json
{
  "question": "What is a stock?",
  "use_mock": true
}
```

L'API retournera une réponse factice mais structurée, sans charger le modèle.

### Mode Réel (avec le modèle)

Pour utiliser le vrai modèle Mistral fine-tuné :

```json
{
  "question": "What is a stock?",
  "use_mock": false,
  "max_tokens": 256
}
```

**Attention** : Le modèle Mistral 7B nécessite beaucoup de mémoire (16+ GB de RAM ou GPU). Si le modèle n'est pas disponible ou trop lourd, l'API retournera une erreur claire et recommandera d'utiliser `use_mock=true`.

### Chargement Lazy

Le modèle n'est **pas chargé au démarrage** de l'API. Il est chargé seulement au premier appel `/generate` avec `use_mock=false`. Cela permet de :
- Démarrer l'API rapidement
- Tester les endpoints `/health` et `/info` sans consommer de mémoire
- Charger le modèle uniquement quand nécessaire

### Exemple avec curl

```bash
# Mode mock
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is a stock?", "use_mock": true}'

# Mode réel
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is a stock?", "use_mock": false, "max_tokens": 256}'
```

### Exemple avec Python

```python
import requests

# Mode mock
response = requests.post(
    "http://localhost:8000/generate",
    json={"question": "What is a stock?", "use_mock": True}
)
print(response.json())

# Mode réel
response = requests.post(
    "http://localhost:8000/generate",
    json={"question": "What is a stock?", "use_mock": False, "max_tokens": 256}
)
print(response.json())
```

## Upload vers Hugging Face Hub

Une fois votre modèle fine-tuné, vous pouvez le partager sur Hugging Face Hub pour le rendre accessible à d'autres utilisateurs.

### Prérequis

1. **Créer un compte Hugging Face** : https://huggingface.co/join
2. **Obtenir un token d'accès** :
   - Allez sur https://huggingface.co/settings/tokens
   - Cliquez sur "New token"
   - Donnez un nom au token (ex: "upload-lora-models")
   - Sélectionnez les permissions "write"
   - Copiez le token généré

3. **Configurer le token dans .env** :
```
HF_TOKEN=votre_token_huggingface_ici
```

**Important** : Ne commitez JAMAIS votre token dans Git. Le fichier `.env` est déjà dans `.gitignore`.

### Vérifier la configuration

Avant d'uploader, vérifiez que tout est correctement configuré :

```bash
python -m training.upload_to_hub --check-only
```

Cette commande vérifie :
- Que le token HF est présent dans `.env`
- Que le dossier `outputs/mistral-finance` existe
- Que le dossier contient des fichiers

Si tout est OK, vous verrez :
```
[OK] Token Hugging Face trouve
[OK] Dossier du modele trouve: outputs/mistral-finance
     Nombre de fichiers: X
[MODE CHECK-ONLY] Verification terminee avec succes!
```

### Uploader le modèle

Une fois l'entraînement terminé et la vérification passée :

```bash
# Upload public (par défaut)
python -m training.upload_to_hub

# Upload privé
python -m training.upload_to_hub --private

# Upload avec des paramètres personnalisés
python -m training.upload_to_hub --folder-path mon/dossier --repo-id mon-user/mon-modele
```

Le script va :
1. Créer le repo sur Hugging Face (s'il n'existe pas)
2. Uploader tous les fichiers du dossier
3. Afficher l'URL finale : https://huggingface.co/nicolbl95/mistral-7b-finance-finetuned

### Utiliser le modèle uploadé

Une fois uploadé, n'importe qui peut charger votre modèle :

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Charger le modèle de base
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Charger vos adaptateurs LoRA depuis Hugging Face
model = PeftModel.from_pretrained(base_model, "nicolbl95/mistral-7b-finance-finetuned")
```

### Erreurs courantes

**Token non trouvé** :
```
[ERREUR] Token Hugging Face non trouve!
```
→ Ajoutez `HF_TOKEN=...` dans votre fichier `.env`

**Dossier non trouvé** :
```
[ERREUR] Le dossier 'outputs/mistral-finance' n'existe pas!
```
→ Lancez d'abord l'entraînement avec `python -m training.train`

**Dossier vide** :
```
[ERREUR] Le dossier 'outputs/mistral-finance' est vide!
```
→ L'entraînement ne s'est pas terminé correctement, relancez-le

## Structure du Projet

```
├── data/
│   ├── processed/     # Données d'entraînement
│   ├── eval/          # Données d'évaluation
│   └── raw/           # Données brutes (optionnel)
├── training/          # Scripts d'entraînement
├── evaluation/        # Scripts d'évaluation
├── interfaces/        # Interfaces utilisateur
└── api/              # API de service
    └── serve.py      # API REST FastAPI
```
