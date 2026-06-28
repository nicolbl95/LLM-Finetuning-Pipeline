"""
Configuration pour le fine-tuning LoRA/QLoRA d'un modèle de langage.

Ce fichier contient tous les paramètres nécessaires pour l'entraînement.
Modifiez les valeurs ici plutôt que directement dans le code d'entraînement.

Structure :
- LoRAConfig : Paramètres spécifiques à LoRA (rang, alpha, dropout, modules cibles)
- TrainingConfig : Paramètres d'entraînement (modèle, dataset, hyperparamètres, logging)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LoRAConfig:
    """
    Configuration des paramètres LoRA (Low-Rank Adaptation).
    
    LoRA permet de fine-tuner un modèle en n'entraînant qu'un petit nombre
    de paramètres additionnels, ce qui réduit drastiquement les besoins en mémoire.
    
    Paramètres clés :
    - r : Rang de la décomposition (plus petit = moins de paramètres)
    - lora_alpha : Facteur de scaling (généralement 2x le rang)
    - lora_dropout : Dropout pour éviter le surapprentissage
    - target_modules : Couches du modèle à adapter avec LoRA
    """
    
    # Rang de la décomposition LoRA
    # Plus le rang est élevé, plus le modèle peut apprendre de variations
    # Valeurs typiques : 8, 16, 32, 64
    # Recommandé : 16 pour un bon équilibre performance/mémoire
    r: int = 16
    
    # Alpha LoRA (facteur de scaling)
    # Contrôle l'importance des adaptations LoRA
    # Règle générale : lora_alpha = 2 * r
    lora_alpha: int = 32
    
    # Dropout pour la régularisation LoRA
    # Aide à éviter le surapprentissage
    # Valeurs typiques : 0.05 à 0.1
    lora_dropout: float = 0.1
    
    # Modules cibles pour appliquer LoRA
    # Pour Mistral/Llama : ["q_proj", "v_proj"] est un bon début
    # Pour plus de capacité : ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Pour GPT-2 : ["c_attn"] ou ["c_attn", "c_proj"]
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """
    Configuration complète pour le fine-tuning LoRA/QLoRA.
    
    Cette classe centralise tous les hyperparamètres et chemins de fichiers
    nécessaires à l'entraînement d'un modèle de langage.
    """
    
    # ========================================
    # MODÈLE ET DATASET
    # ========================================
    
    # Nom du modèle pré-entraîné depuis Hugging Face Hub
    # Exemples : "gpt2", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf"
    model_name: str = "mistralai/Mistral-7B-v0.1"
    
    # Nom du dataset Hugging Face (pour référence)
    dataset_name: str = "gbharti/finance-alpaca"
    
    # Chemin vers le fichier JSONL d'entraînement
    train_file: str = "data/processed/train.jsonl"
    
    # Chemin vers le fichier JSONL d'évaluation
    eval_file: str = "data/eval/eval.jsonl"
    
    # Répertoire où sauvegarder le modèle fine-tuné
    output_dir: str = "./outputs/mistral-finance"
    
    # ========================================
    # HYPERPARAMÈTRES D'ENTRAÎNEMENT
    # ========================================
    
    # Nombre d'époques (passages complets sur le dataset)
    # 3 époques est généralement suffisant pour du fine-tuning
    num_epochs: int = 3
    
    # Taille du batch par GPU/CPU
    # Réduire si vous manquez de mémoire (essayez 2 ou 1)
    batch_size: int = 4
    
    # Nombre de steps de gradient accumulation
    # Simule des batchs plus grands : batch_effectif = batch_size * gradient_accumulation
    # Exemple : 4 * 4 = 16 exemples par mise à jour des poids
    gradient_accumulation: int = 4
    
    # Taux d'apprentissage (learning rate)
    # Pour LoRA, 2e-4 est une bonne valeur de départ
    # Réduire (1e-4) si le modèle diverge, augmenter (3e-4) s'il apprend trop lentement
    learning_rate: float = 2e-4
    
    # Longueur maximale des séquences (en tokens)
    # Les séquences plus longues seront tronquées
    # 512 est un bon compromis pour la plupart des tâches
    max_length: int = 512
    
    # Ratio de warmup (échauffement du learning rate)
    # 0.03 = 3% des steps avec learning rate croissant
    # Aide à stabiliser l'entraînement au début
    warmup_ratio: float = 0.03
    
    # ========================================
    # QUANTIZATION 4-BIT (QLoRA)
    # ========================================
    
    # Utiliser la quantization 4-bit pour réduire l'utilisation mémoire
    # ATTENTION : Nécessite bitsandbytes, qui ne fonctionne PAS sur Windows
    # Mettre à True uniquement sur Linux/Kaggle pour économiser de la mémoire
    # Sur Windows, laisser à False et utiliser LoRA classique
    use_4bit: bool = False
    
    # Type de calcul pour la quantization 4-bit
    # "float16" ou "bfloat16" (bfloat16 recommandé si supporté par votre GPU)
    bnb_4bit_compute_dtype: str = "float16"
    
    # ========================================
    # LOGGING ET SUIVI (WEIGHTS & BIASES)
    # ========================================
    
    # Nom du projet Weights & Biases
    # Permet de suivre vos expériences et comparer les résultats
    wandb_project: str = "llm-finetuning-mistral"
    
    # Nom de l'exécution W&B (identifie cette expérience spécifique)
    wandb_run_name: str = "mistral-finance-lora-v1"
    
    # Activer le logging vers Weights & Biases
    # Mettre à False si vous ne voulez pas utiliser W&B
    report_to_wandb: bool = True
    
    # ========================================
    # PARAMÈTRES AVANCÉS
    # ========================================
    
    # Fréquence de logging (tous les N steps)
    logging_steps: int = 10
    
    # Fréquence de sauvegarde (tous les N steps)
    save_steps: int = 100
    
    # Nombre maximum de checkpoints à garder
    save_total_limit: int = 2
    
    # Utiliser le gradient checkpointing pour économiser la mémoire
    # Recommandé : True (ralentit un peu l'entraînement mais économise beaucoup de mémoire)
    gradient_checkpointing: bool = True
    
    # Utiliser la précision mixte (fp16) pour accélérer l'entraînement
    # Recommandé : True si votre GPU le supporte
    fp16: bool = True
    
    # Seed pour la reproductibilité
    seed: int = 42
    
    # ========================================
    # CONFIGURATION LORA
    # ========================================
    
    # Configuration LoRA (voir LoRAConfig pour les détails)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    def __post_init__(self):
        """Validation et création des répertoires nécessaires."""
        # Créer le répertoire de sortie s'il n'existe pas
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Vérifier que le fichier de données d'entraînement existe
        if not Path(self.train_file).exists():
            raise FileNotFoundError(
                f"Le fichier de données '{self.train_file}' n'existe pas.\n"
                f"Veuillez d'abord exécuter : python -m training.prepare_dataset\n"
                f"Format attendu : chaque ligne doit être un JSON avec les clés 'prompt' et 'response'."
            )
        
        # Avertissement si use_4bit est activé sur Windows
        import platform
        if self.use_4bit and platform.system() == "Windows":
            print("⚠️  ATTENTION : use_4bit=True ne fonctionne pas sur Windows !")
            print("   bitsandbytes n'est pas compatible avec Windows.")
            print("   Mettez use_4bit=False ou utilisez Linux/Kaggle pour QLoRA.")
    
    # ========================================
    # PROPRIÉTÉS DE COMPATIBILITÉ
    # ========================================
    # Ces propriétés assurent la compatibilité avec l'ancien code
    
    @property
    def train_data_path(self) -> str:
        """Alias pour train_file (compatibilité)."""
        return self.train_file
    
    @property
    def per_device_train_batch_size(self) -> int:
        """Alias pour batch_size (compatibilité)."""
        return self.batch_size
    
    @property
    def gradient_accumulation_steps(self) -> int:
        """Alias pour gradient_accumulation (compatibilité)."""
        return self.gradient_accumulation
    
    @property
    def lora_r(self) -> int:
        """Rang LoRA (compatibilité)."""
        return self.lora.r
    
    @property
    def lora_alpha(self) -> int:
        """Alpha LoRA (compatibilité)."""
        return self.lora.lora_alpha
    
    @property
    def lora_dropout(self) -> float:
        """Dropout LoRA (compatibilité)."""
        return self.lora.lora_dropout
    
    @property
    def target_modules(self) -> Optional[List[str]]:
        """Modules cibles LoRA (compatibilité)."""
        return self.lora.target_modules
    
    @property
    def use_wandb(self) -> bool:
        """Alias pour report_to_wandb (compatibilité)."""
        return self.report_to_wandb


if __name__ == "__main__":
    """
    Test de chargement de la configuration.
    
    Exécutez ce fichier directement pour vérifier que la configuration
    se charge correctement et afficher tous les paramètres.
    """
    print("=" * 70)
    print("CONFIGURATION DU FINE-TUNING LORA/QLORA")
    print("=" * 70)
    
    # Créer une configuration par défaut
    config = TrainingConfig()
    
    print("\n📋 MODÈLE ET DATASET")
    print("-" * 70)
    print(f"  Modèle              : {config.model_name}")
    print(f"  Dataset             : {config.dataset_name}")
    print(f"  Fichier train       : {config.train_file}")
    print(f"  Fichier eval        : {config.eval_file}")
    print(f"  Répertoire sortie   : {config.output_dir}")
    
    print("\n🎯 HYPERPARAMÈTRES D'ENTRAÎNEMENT")
    print("-" * 70)
    print(f"  Époques             : {config.num_epochs}")
    print(f"  Batch size          : {config.batch_size}")
    print(f"  Gradient accum.     : {config.gradient_accumulation}")
    print(f"  Batch effectif      : {config.batch_size * config.gradient_accumulation}")
    print(f"  Learning rate       : {config.learning_rate}")
    print(f"  Max length          : {config.max_length} tokens")
    print(f"  Warmup ratio        : {config.warmup_ratio}")
    
    print("\n🔧 PARAMÈTRES LORA")
    print("-" * 70)
    print(f"  Rang (r)            : {config.lora.r}")
    print(f"  Alpha               : {config.lora.lora_alpha}")
    print(f"  Dropout             : {config.lora.lora_dropout}")
    print(f"  Modules cibles      : {config.lora.target_modules}")
    
    print("\n💾 QUANTIZATION 4-BIT (QLoRA)")
    print("-" * 70)
    print(f"  Activée             : {config.use_4bit}")
    if config.use_4bit:
        print(f"  Type de calcul      : {config.bnb_4bit_compute_dtype}")
        print("  ⚠️  Nécessite Linux/Kaggle (pas compatible Windows)")
    else:
        print("  ℹ️  LoRA classique (compatible Windows)")
    
    print("\n📊 LOGGING ET SUIVI")
    print("-" * 70)
    print(f"  Weights & Biases    : {'✓ Activé' if config.report_to_wandb else '✗ Désactivé'}")
    if config.report_to_wandb:
        print(f"  Projet W&B          : {config.wandb_project}")
        print(f"  Nom de l'exécution  : {config.wandb_run_name}")
    print(f"  Logging steps       : {config.logging_steps}")
    print(f"  Save steps          : {config.save_steps}")
    
    print("\n⚙️  OPTIMISATION")
    print("-" * 70)
    print(f"  Gradient checkpoint : {config.gradient_checkpointing}")
    print(f"  FP16                : {config.fp16}")
    print(f"  Seed                : {config.seed}")
    
    print("\n" + "=" * 70)
    print("✓ Configuration chargée avec succès !")
    print("=" * 70)
    
    # Afficher un avertissement si le fichier de données n'existe pas
    if not Path(config.train_file).exists():
        print("\n⚠️  ATTENTION : Le fichier de données n'existe pas encore.")
        print(f"   Exécutez : python -m training.prepare_dataset")
        print("=" * 70)
