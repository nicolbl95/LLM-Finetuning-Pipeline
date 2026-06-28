"""
Script principal pour le fine-tuning LoRA d'un modèle de langage.

Ce script orchestre tout le processus d'entraînement :
1. Chargement de la configuration
2. Chargement du modèle et du tokenizer
3. Application de LoRA avec PEFT
4. Préparation du dataset
5. Entraînement avec Hugging Face Trainer
6. Sauvegarde du modèle fine-tuné
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from training.config import TrainingConfig
from training.dataset import prepare_dataset


def main():
    """Fonction principale d'entraînement."""
    
    # ========================================
    # 1. CHARGEMENT DE LA CONFIGURATION
    # ========================================
    print("=" * 50)
    print("FINE-TUNING LORA - DÉMARRAGE")
    print("=" * 50)
    
    config = TrainingConfig()
    print(f"\n✓ Configuration chargée")
    print(f"  - Modèle : {config.model_name}")
    print(f"  - Données : {config.train_data_path}")
    print(f"  - Sortie : {config.output_dir}")
    
    # Définir le seed pour la reproductibilité
    set_seed(config.seed)
    
    # ========================================
    # 2. CHARGEMENT DU TOKENIZER
    # ========================================
    print(f"\n📝 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Ajouter un token de padding si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  ⚠ Pas de pad_token défini, utilisation de eos_token")
    
    print(f"✓ Tokenizer chargé")
    
    # ========================================
    # 3. CHARGEMENT DU MODÈLE
    # ========================================
    print(f"\n🤖 Chargement du modèle {config.model_name}...")
    
    # Charger le modèle en précision mixte si demandé
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",  # Répartition automatique sur les GPUs disponibles
        trust_remote_code=True  # Nécessaire pour certains modèles
    )
    
    print(f"✓ Modèle chargé")
    print(f"  - Paramètres totaux : {model.num_parameters():,}")
    
    # ========================================
    # 4. CONFIGURATION ET APPLICATION DE LORA
    # ========================================
    print(f"\n🔧 Configuration de LoRA...")
    
    # Préparer le modèle pour l'entraînement (gradient checkpointing, etc.)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    
    # Configuration LoRA
    lora_config = LoraConfig(
        r=config.lora_r,  # Rang de la décomposition
        lora_alpha=config.lora_alpha,  # Facteur de scaling
        target_modules=config.target_modules,  # Modules à adapter (None = auto)
        lora_dropout=config.lora_dropout,  # Dropout pour régularisation
        bias="none",  # Ne pas adapter les biais
        task_type="CAUSAL_LM"  # Type de tâche
    )
    
    # Appliquer LoRA au modèle
    model = get_peft_model(model, lora_config)
    
    # Afficher les statistiques des paramètres
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ LoRA appliqué")
    print(f"  - Paramètres entraînables : {trainable_params:,}")
    print(f"  - Paramètres totaux : {total_params:,}")
    print(f"  - Pourcentage entraînable : {100 * trainable_params / total_params:.2f}%")
    
    # ========================================
    # 5. PRÉPARATION DU DATASET
    # ========================================
    print(f"\n📊 Préparation du dataset...")
    
    train_dataset = prepare_dataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    # ========================================
    # 6. CONFIGURATION DE L'ENTRAÎNEMENT
    # ========================================
    print(f"\n⚙️ Configuration de l'entraînement...")
    
    # Déterminer le report_to pour le logging
    report_to = "wandb" if config.use_wandb else "none"
    
    # Si W&B est activé, configurer le projet
    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        print(f"  - Logging W&B activé (projet: {config.wandb_project})")
    
    training_args = TrainingArguments(
        # Répertoire de sortie
        output_dir=config.output_dir,
        
        # Hyperparamètres d'entraînement
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        
        # Optimisation
        fp16=config.fp16,
        optim="adamw_torch",  # Optimiseur AdamW
        
        # Logging et sauvegarde
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=report_to,
        
        # Autres paramètres
        seed=config.seed,
        remove_unused_columns=False,  # Garder toutes les colonnes
        load_best_model_at_end=False,  # Pas d'évaluation donc pas de "meilleur modèle"
    )
    
    print(f"✓ Arguments d'entraînement configurés")
    
    # ========================================
    # 7. CRÉATION DU TRAINER
    # ========================================
    print(f"\n🏋️ Création du Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    print(f"✓ Trainer créé")
    
    # ========================================
    # 8. LANCEMENT DE L'ENTRAÎNEMENT
    # ========================================
    print(f"\n🚀 Démarrage de l'entraînement...")
    print("=" * 50)
    
    # Entraîner le modèle
    trainer.train()
    
    print("=" * 50)
    print(f"✓ Entraînement terminé !")
    
    # ========================================
    # 9. SAUVEGARDE DU MODÈLE
    # ========================================
    print(f"\n💾 Sauvegarde du modèle...")
    
    # Sauvegarder le modèle LoRA
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"✓ Modèle sauvegardé dans : {config.output_dir}")
    
    # ========================================
    # 10. RÉSUMÉ FINAL
    # ========================================
    print("\n" + "=" * 50)
    print("FINE-TUNING TERMINÉ AVEC SUCCÈS !")
    print("=" * 50)
    print(f"\nPour utiliser votre modèle fine-tuné :")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  ")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{config.model_name}')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{config.output_dir}')")
    print("=" * 50)


if __name__ == "__main__":
    main()
