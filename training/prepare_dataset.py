"""
Script pour préparer le dataset finance-alpaca.

Ce script télécharge le dataset depuis Hugging Face et le convertit
au format JSONL attendu par le pipeline d'entraînement.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset


def format_example(example):
    """
    Convertit un exemple du dataset au format attendu.
    
    Args:
        example: Dictionnaire avec les clés 'instruction', 'input', 'output'
    
    Returns:
        Dictionnaire avec les clés 'prompt' et 'response'
    """
    instruction = example['instruction']
    input_text = example.get('input', '').strip()
    output_text = example['output']
    
    # Si input est vide, le prompt est seulement l'instruction
    if not input_text:
        prompt = instruction
    else:
        # Sinon, on combine instruction et input
        prompt = f"{instruction}\n\nInput: {input_text}"
    
    return {
        'prompt': prompt,
        'response': output_text
    }


def save_jsonl(data, file_path):
    """
    Sauvegarde une liste de dictionnaires au format JSONL.
    
    Args:
        data: Liste de dictionnaires à sauvegarder
        file_path: Chemin du fichier de sortie
    """
    # Créer le dossier parent s'il n'existe pas
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Écrire chaque exemple sur une ligne
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Sauvegardé {len(data)} exemples dans {file_path}")


def main():
    """
    Fonction principale pour préparer le dataset.
    """
    print("Téléchargement du dataset finance-alpaca...")
    
    # Télécharger le dataset depuis Hugging Face
    dataset = load_dataset("gbharti/finance-alpaca")
    
    # Prendre seulement les 1000 premiers exemples
    max_examples = 1000
    train_size = 900
    eval_size = 100
    
    # Extraire les exemples du dataset
    all_examples = dataset['train'].select(range(min(max_examples, len(dataset['train']))))
    
    print(f"Formatage de {len(all_examples)} exemples...")
    
    # Convertir au format attendu
    formatted_examples = [format_example(ex) for ex in all_examples]
    
    # Séparer en train et eval
    train_data = formatted_examples[:train_size]
    eval_data = formatted_examples[train_size:train_size + eval_size]
    
    # Sauvegarder les fichiers
    print("\nSauvegarde des fichiers...")
    save_jsonl(train_data, 'data/processed/train.jsonl')
    save_jsonl(eval_data, 'data/eval/eval.jsonl')
    
    print("\n✓ Préparation du dataset terminée avec succès!")
    print(f"  - Entraînement: {len(train_data)} exemples")
    print(f"  - Évaluation: {len(eval_data)} exemples")


if __name__ == "__main__":
    main()
