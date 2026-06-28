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
