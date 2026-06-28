# Projet de Fine-Tuning LoRA

Ce projet permet de fine-tuner un modèle de langage avec LoRA sur le dataset finance-alpaca.

## Installation

```bash
pip install -r requirements.txt
```

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

(Instructions à venir)

## Évaluation

(Instructions à venir)

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
```
