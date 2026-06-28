"""
Chargement et préparation du dataset pour le fine-tuning.

Ce module gère la lecture des données JSONL et leur transformation
en format compatible avec l'entraînement de modèles de langage.
"""

import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Charge un fichier JSONL (JSON Lines).
    
    Args:
        file_path: Chemin vers le fichier JSONL
        
    Returns:
        Liste de dictionnaires, un par ligne du fichier
        
    Format attendu du JSONL :
        {"prompt": "Question ou instruction", "response": "Réponse attendue"}
        {"prompt": "Autre question", "response": "Autre réponse"}
    """
    data = []
    
    # Vérifier que le fichier existe
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Le fichier '{file_path}' n'existe pas.\n"
            f"Créez ce fichier avec vos données d'entraînement au format JSONL."
        )
    
    # Lire le fichier ligne par ligne
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Ignorer les lignes vides
                continue
            
            try:
                # Parser le JSON de chaque ligne
                item = json.loads(line)
                
                # Vérifier que les champs requis sont présents
                if 'prompt' not in item or 'response' not in item:
                    print(f"Attention : ligne {line_num} ignorée (manque 'prompt' ou 'response')")
                    continue
                
                data.append(item)
                
            except json.JSONDecodeError as e:
                print(f"Erreur de parsing JSON à la ligne {line_num}: {e}")
                continue
    
    if not data:
        raise ValueError(
            f"Aucune donnée valide trouvée dans '{file_path}'.\n"
            f"Vérifiez le format de votre fichier JSONL."
        )
    
    print(f"✓ {len(data)} exemples chargés depuis {file_path}")
    return data


def format_prompt_response(example: Dict) -> str:
    """
    Formate un exemple prompt/response en texte pour l'entraînement.
    
    Args:
        example: Dictionnaire avec les clés 'prompt' et 'response'
        
    Returns:
        Texte formaté pour l'entraînement
        
    Format de sortie :
        ### Instruction:
        {prompt}
        
        ### Response:
        {response}
    """
    return (
        f"### Instruction:\n{example['prompt']}\n\n"
        f"### Response:\n{example['response']}"
    )


def prepare_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Prépare le dataset pour l'entraînement.
    
    Cette fonction :
    1. Charge les données depuis le fichier JSONL
    2. Formate chaque exemple (prompt + response)
    3. Tokenize le texte
    4. Crée les labels pour l'entraînement
    
    Args:
        data_path: Chemin vers le fichier JSONL
        tokenizer: Tokenizer du modèle
        max_length: Longueur maximale des séquences en tokens
        
    Returns:
        Dataset Hugging Face prêt pour l'entraînement
    """
    # Charger les données brutes
    raw_data = load_jsonl(data_path)
    
    # Formater chaque exemple
    formatted_texts = [format_prompt_response(example) for example in raw_data]
    
    # Créer un dataset Hugging Face
    dataset = Dataset.from_dict({"text": formatted_texts})
    
    def tokenize_function(examples):
        """
        Tokenize les textes et prépare les labels.
        
        Pour l'entraînement causal language modeling :
        - input_ids : tokens d'entrée
        - labels : mêmes tokens (le modèle apprend à prédire le token suivant)
        """
        # Tokenizer avec padding et troncature
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None  # Retourner des listes Python
        )
        
        # Pour le causal LM, les labels sont identiques aux input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Appliquer la tokenization à tout le dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,  # Supprimer la colonne 'text' originale
        desc="Tokenization des données"
    )
    
    print(f"✓ Dataset préparé : {len(tokenized_dataset)} exemples tokenizés")
    return tokenized_dataset
