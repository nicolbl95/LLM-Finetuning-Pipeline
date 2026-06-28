"""
Chargement et préparation du dataset pour le fine-tuning.

Ce module gère la lecture des données JSONL et leur transformation
en format compatible avec l'entraînement de modèles de langage.

Supporte deux formats :
1. Format Alpaca : {"instruction": "...", "input": "...", "output": "..."}
2. Format simple : {"prompt": "...", "response": "..."}
"""

import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer


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


def format_instruction(example: Dict) -> str:
    """
    Formate un exemple en texte pour le fine-tuning instructionnel.
    
    Supporte deux formats :
    
    1. Format Alpaca (avec instruction/input/output) :
       ### Instruction:
       {instruction}
       
       ### Input:
       {input}
       
       ### Response:
       {output}
    
    2. Format simple (avec prompt/response) :
       ### Instruction:
       {prompt}
       
       ### Response:
       {response}
    
    Args:
        example: Dictionnaire contenant soit :
                 - 'instruction', 'input' (optionnel), 'output'
                 - 'prompt', 'response'
        
    Returns:
        Texte formaté selon le template Alpaca
    """
    # Format Alpaca : instruction/input/output
    if 'instruction' in example and 'output' in example:
        instruction = example['instruction'].strip()
        input_text = example.get('input', '').strip()
        output = example['output'].strip()
        
        # Si input est présent et non vide, l'inclure
        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            # Pas d'input, juste instruction et output
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
    
    # Format simple : prompt/response
    elif 'prompt' in example and 'response' in example:
        prompt = example['prompt'].strip()
        response = example['response'].strip()
        
        return (
            f"### Instruction:\n{prompt}\n\n"
            f"### Response:\n{response}"
        )
    
    else:
        raise ValueError(
            f"Format d'exemple non reconnu. Attendu : "
            f"'instruction'/'output' ou 'prompt'/'response'. "
            f"Reçu : {list(example.keys())}"
        )


def format_prompt_response(example: Dict) -> str:
    """
    Fonction de compatibilité - utilise format_instruction().
    
    Args:
        example: Dictionnaire avec les clés 'prompt' et 'response'
        
    Returns:
        Texte formaté pour l'entraînement
    """
    return format_instruction(example)


def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dict:
    """
    Tokenize un texte avec gestion automatique du pad_token.
    
    Cette fonction :
    1. Vérifie que le tokenizer a un pad_token (sinon utilise eos_token)
    2. Tokenize le texte avec troncature et padding
    3. Crée les labels pour l'entraînement (identiques aux input_ids)
    
    Args:
        text: Texte à tokenizer
        tokenizer: Tokenizer du modèle
        max_length: Longueur maximale en tokens
        
    Returns:
        Dictionnaire avec 'input_ids', 'attention_mask', 'labels'
    """
    # S'assurer que le tokenizer a un pad_token
    # Si pas de pad_token, utiliser eos_token (fin de séquence)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenizer le texte
    # truncation=True : couper si trop long
    # max_length : longueur maximale autorisée
    # padding="max_length" : remplir avec pad_token jusqu'à max_length
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None  # Retourner des listes Python (pas de tensors)
    )
    
    # Pour le causal language modeling (prédire le token suivant) :
    # Les labels sont identiques aux input_ids
    # Le modèle apprend à prédire chaque token à partir des précédents
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def prepare_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Prépare le dataset pour l'entraînement.
    
    Cette fonction :
    1. Charge les données depuis le fichier JSONL
    2. Formate chaque exemple selon le template Alpaca
    3. Tokenize le texte
    4. Crée les labels pour l'entraînement
    
    Args:
        data_path: Chemin vers le fichier JSONL
        tokenizer: Tokenizer du modèle
        max_length: Longueur maximale des séquences en tokens
        
    Returns:
        Dataset Hugging Face prêt pour l'entraînement
    """
    # Charger les données brutes depuis le fichier JSONL
    raw_data = load_jsonl(data_path)
    
    # Formater chaque exemple avec le template Alpaca
    # Supporte automatiquement les formats instruction/input/output et prompt/response
    formatted_texts = [format_instruction(example) for example in raw_data]
    
    # Créer un dataset Hugging Face à partir des textes formatés
    dataset = Dataset.from_dict({"text": formatted_texts})
    
    def tokenize_function(examples):
        """
        Fonction de tokenization appliquée à chaque batch.
        
        Args:
            examples: Batch d'exemples avec la clé 'text'
            
        Returns:
            Dictionnaire avec input_ids, attention_mask, labels
        """
        # S'assurer que le tokenizer a un pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenizer tous les textes du batch
        tokenized = tokenizer(
            examples["text"],
            truncation=True,  # Couper si trop long
            max_length=max_length,  # Longueur max
            padding="max_length",  # Remplir jusqu'à max_length
            return_tensors=None  # Listes Python
        )
        
        # Créer les labels (identiques aux input_ids pour causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Appliquer la tokenization à tout le dataset en mode batch
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,  # Traiter par batch pour plus d'efficacité
        remove_columns=dataset.column_names,  # Supprimer la colonne 'text' originale
        desc="Tokenization des données"
    )
    
    print(f"✓ Dataset préparé : {len(tokenized_dataset)} exemples tokenizés")
    return tokenized_dataset


if __name__ == "__main__":
    """
    Test simple du formatage et de la tokenization.
    
    Affiche un exemple formaté et tokenisé sans télécharger de gros modèle.
    """
    print("=" * 60)
    print("TEST DU MODULE DATASET")
    print("=" * 60)
    
    # Exemple 1 : Format Alpaca complet (avec input)
    print("\n1. Format Alpaca avec input :")
    print("-" * 60)
    example_alpaca = {
        "instruction": "Traduire le texte suivant en anglais",
        "input": "Bonjour, comment allez-vous ?",
        "output": "Hello, how are you?"
    }
    formatted_alpaca = format_instruction(example_alpaca)
    print(formatted_alpaca)
    
    # Exemple 2 : Format Alpaca sans input
    print("\n2. Format Alpaca sans input :")
    print("-" * 60)
    example_alpaca_no_input = {
        "instruction": "Expliquer ce qu'est le machine learning",
        "output": "Le machine learning est une branche de l'intelligence artificielle..."
    }
    formatted_alpaca_no_input = format_instruction(example_alpaca_no_input)
    print(formatted_alpaca_no_input)
    
    # Exemple 3 : Format simple prompt/response
    print("\n3. Format simple prompt/response :")
    print("-" * 60)
    example_simple = {
        "prompt": "Quelle est la capitale de la France ?",
        "response": "La capitale de la France est Paris."
    }
    formatted_simple = format_instruction(example_simple)
    print(formatted_simple)
    
    # Test de tokenization avec GPT-2 (petit modèle, rapide à charger)
    print("\n4. Test de tokenization (GPT-2) :")
    print("-" * 60)
    try:
        # Charger un petit tokenizer pour le test
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Tokenizer l'exemple simple
        tokenized = tokenize_text(formatted_simple, tokenizer, max_length=128)
        
        print(f"✓ Tokenizer chargé : {tokenizer.__class__.__name__}")
        print(f"  - Pad token : {tokenizer.pad_token}")
        print(f"  - Nombre de tokens : {len(tokenized['input_ids'])}")
        print(f"  - Premiers tokens : {tokenized['input_ids'][:20]}")
        
        # Décoder pour vérifier
        decoded = tokenizer.decode(tokenized['input_ids'], skip_special_tokens=False)
        print(f"\n  Texte décodé (premiers 200 caractères) :")
        print(f"  {decoded[:200]}...")
        
    except Exception as e:
        print(f"⚠ Impossible de charger le tokenizer (normal si pas de connexion) : {e}")
    
    print("\n" + "=" * 60)
    print("TEST TERMINÉ")
    print("=" * 60)
