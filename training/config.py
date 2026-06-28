"""
Configuration pour le fine-tuning LoRA d'un modèle de langage.

Ce fichier contient tous les paramètres nécessaires pour l'entraînement.
Modifiez les valeurs ici plutôt que directement dans le code d'entraînement.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration complète pour le fine-tuning LoRA.
    
    Utilisez cette classe pour centraliser tous les hyperparamètres
    et chemins de fichiers nécessaires à l'entraînement.
    """
    
    # === Modèle de base ===
    # Nom du modèle pré-entraîné depuis Hugging Face Hub
    # Exemples : "gpt2", "microsoft/phi-2", "meta-llama/Llama-2-7b-hf"
    model_name: str = "gpt2"
    
    # === Chemins des données ===
    # Chemin vers le fichier JSONL d'entraînement
    train_data_path: str = "data/processed/train.jsonl"
    
    # Chemin où sauvegarder le modèle fine-tuné
    output_dir: str = "models/lora_finetuned"
    
    # === Paramètres LoRA ===
    # Rang de la décomposition LoRA (plus petit = moins de paramètres entraînables)
    # Valeurs typiques : 8, 16, 32, 64
    lora_r: int = 8
    
    # Alpha LoRA (facteur de scaling, généralement 2x le rang)
    lora_alpha: int = 16
    
    # Dropout pour la régularisation LoRA
    lora_dropout: float = 0.05
    
    # Modules cibles pour appliquer LoRA
    # Pour GPT-2 : ["c_attn"] ou ["c_attn", "c_proj"]
    # Pour Llama : ["q_proj", "v_proj"] ou ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules: list = None  # None = auto-détection
    
    # === Hyperparamètres d'entraînement ===
    # Nombre d'époques (passages complets sur le dataset)
    num_epochs: int = 3
    
    # Taille du batch par GPU
    per_device_train_batch_size: int = 4
    
    # Nombre de steps de gradient accumulation (simule des batchs plus grands)
    gradient_accumulation_steps: int = 4
    
    # Taux d'apprentissage (learning rate)
    learning_rate: float = 2e-4
    
    # Longueur maximale des séquences (en tokens)
    max_length: int = 512
    
    # === Optimisation et performance ===
    # Utiliser le gradient checkpointing pour économiser la mémoire
    gradient_checkpointing: bool = True
    
    # Utiliser la précision mixte (fp16) pour accélérer l'entraînement
    fp16: bool = True
    
    # === Logging et sauvegarde ===
    # Fréquence de logging (tous les N steps)
    logging_steps: int = 10
    
    # Fréquence de sauvegarde (tous les N steps)
    save_steps: int = 100
    
    # Nombre maximum de checkpoints à garder
    save_total_limit: int = 2
    
    # Activer le logging vers Weights & Biases
    # Mettez à False si vous ne voulez pas utiliser W&B
    use_wandb: bool = False
    
    # Nom du projet W&B (si use_wandb=True)
    wandb_project: str = "lora-finetuning"
    
    # === Seed pour la reproductibilité ===
    seed: int = 42
    
    def __post_init__(self):
        """Validation et création des répertoires nécessaires."""
        # Créer le répertoire de sortie s'il n'existe pas
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Vérifier que le fichier de données existe
        if not Path(self.train_data_path).exists():
            raise FileNotFoundError(
                f"Le fichier de données '{self.train_data_path}' n'existe pas.\n"
                f"Veuillez créer ce fichier JSONL avec vos données d'entraînement.\n"
                f"Format attendu : chaque ligne doit être un JSON avec les clés 'prompt' et 'response'."
            )
