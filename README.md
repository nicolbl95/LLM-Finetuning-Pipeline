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
