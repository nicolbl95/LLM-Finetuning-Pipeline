"""
Script principal pour le fine-tuning LoRA/QLoRA d'un modele de langage.

Ce script orchestre tout le processus d'entrainement :
1. Chargement de la configuration
2. Chargement du tokenizer
3. Chargement du modele (avec ou sans quantization 4-bit)
4. Application de LoRA avec PEFT
5. Preparation du dataset
6. Entrainement avec SFTTrainer (TRL)
7. Sauvegarde du modele fine-tune

Usage:
    # Dry-run (verification sans entrainement)
    python -m training.train --dry-run
    
    # Entrainement complet
    python -m training.train
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig

from training.config import TrainingConfig, LoRAConfig
from training.dataset import format_instruction


def load_tokenizer(cfg: TrainingConfig) -> AutoTokenizer:
    """
    Charge le tokenizer du modele.
    
    Args:
        cfg: Configuration d'entrainement
        
    Returns:
        Tokenizer pret a l'emploi avec pad_token configure
    """
    print(f"\n[1/6] Chargement du tokenizer...")
    print(f"  Modele: {cfg.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True
    )
    
    # Configurer le pad_token si necessaire
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Info: pad_token defini comme eos_token")
    
    print(f"  OK: Tokenizer charge")
    return tokenizer


def load_model(cfg: TrainingConfig) -> AutoModelForCausalLM:
    """
    Charge le modele de langage avec ou sans quantization 4-bit.
    
    Args:
        cfg: Configuration d'entrainement
        
    Returns:
        Modele pret pour l'application de LoRA
    """
    print(f"\n[2/6] Chargement du modele...")
    print(f"  Modele: {cfg.model_name}")
    print(f"  Quantization 4-bit: {cfg.use_4bit}")
    
    # Configuration de la quantization 4-bit (QLoRA)
    if cfg.use_4bit:
        print(f"  Mode: QLoRA (quantization 4-bit)")
        
        # Verifier que bitsandbytes est disponible
        try:
            import bitsandbytes
        except ImportError:
            raise ImportError(
                "bitsandbytes n'est pas installe.\n"
                "Sur Linux/Kaggle: pip install bitsandbytes\n"
                "Sur Windows: Mettez use_4bit=False dans la config"
            )
        
        # Configuration de la quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Charger le modele avec quantization
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Preparer le modele pour l'entrainement quantize
        model = prepare_model_for_kbit_training(model)
        
    else:
        print(f"  Mode: LoRA classique (sans quantization)")
        
        # Charger le modele normalement
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Activer le gradient checkpointing si demande
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"  Info: Gradient checkpointing active")
    
    # Afficher les statistiques du modele
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK: Modele charge ({total_params:,} parametres)")
    
    return model


def apply_lora(model: AutoModelForCausalLM, lora_cfg: LoRAConfig) -> AutoModelForCausalLM:
    """
    Applique LoRA au modele.
    
    Args:
        model: Modele de base
        lora_cfg: Configuration LoRA
        
    Returns:
        Modele avec LoRA applique
    """
    print(f"\n[3/6] Application de LoRA...")
    print(f"  Rang (r): {lora_cfg.r}")
    print(f"  Alpha: {lora_cfg.lora_alpha}")
    print(f"  Dropout: {lora_cfg.lora_dropout}")
    print(f"  Modules cibles: {lora_cfg.target_modules}")
    
    # Configuration LoRA
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Appliquer LoRA
    model = get_peft_model(model, peft_config)
    
    # Statistiques des parametres
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    
    print(f"  OK: LoRA applique")
    print(f"  Parametres entrainables: {trainable_params:,}")
    print(f"  Parametres totaux: {total_params:,}")
    print(f"  Pourcentage entrainable: {trainable_percent:.2f}%")
    
    return model


def load_training_dataset(cfg: TrainingConfig, tokenizer: AutoTokenizer) -> Dataset:
    """
    Charge le dataset d'entrainement.
    
    Essaie d'abord de charger depuis le fichier local JSONL.
    Si le fichier n'existe pas, charge depuis Hugging Face Hub.
    
    Args:
        cfg: Configuration d'entrainement
        tokenizer: Tokenizer pour la preparation des donnees
        
    Returns:
        Dataset pret pour l'entrainement
    """
    print(f"\n[4/6] Chargement du dataset...")
    
    train_file = Path(cfg.train_file)
    
    # Essayer de charger depuis le fichier local
    if train_file.exists():
        print(f"  Source: Fichier local {cfg.train_file}")
        
        # Charger le fichier JSONL
        data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        print(f"  Exemples charges: {len(data)}")
        
        # Formater les exemples
        formatted_texts = [format_instruction(example) for example in data]
        dataset = Dataset.from_dict({"text": formatted_texts})
        
    else:
        print(f"  Source: Hugging Face Hub ({cfg.dataset_name})")
        print(f"  Attention: Fichier local {cfg.train_file} non trouve")
        
        # Charger depuis Hugging Face
        dataset = load_dataset(cfg.dataset_name, split="train")
        
        # Formater les exemples
        def format_example(example):
            return {"text": format_instruction(example)}
        
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        print(f"  Exemples charges: {len(dataset)}")
    
    print(f"  OK: Dataset pret")
    return dataset


def train(cfg: TrainingConfig, lora_cfg: LoRAConfig, dry_run: bool = False):
    """
    Fonction principale d'entrainement.
    
    Args:
        cfg: Configuration d'entrainement
        lora_cfg: Configuration LoRA
        dry_run: Si True, verifie la config sans lancer l'entrainement
    """
    print("=" * 70)
    print("FINE-TUNING LORA/QLORA")
    print("=" * 70)
    
    if dry_run:
        print("\nMODE DRY-RUN: Verification sans entrainement")
    
    # Definir le seed pour la reproductibilite
    set_seed(cfg.seed)
    
    # Charger le tokenizer
    tokenizer = load_tokenizer(cfg)
    
    # Charger le dataset
    dataset = load_training_dataset(cfg, tokenizer)
    
    if dry_run:
        print("\n[DRY-RUN] Affichage d'un exemple formate:")
        print("-" * 70)
        print(dataset[0]["text"][:500])
        print("...")
        print("-" * 70)
        print("\n[DRY-RUN] Verification terminee avec succes!")
        print("Pour lancer l'entrainement: python -m training.train")
        return
    
    # Charger le modele
    model = load_model(cfg)
    
    # Appliquer LoRA
    model = apply_lora(model, lora_cfg)
    
    # Configuration de l'entrainement
    print(f"\n[5/6] Configuration de l'entrainement...")
    
    # Configurer W&B si active
    if cfg.report_to_wandb:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        print(f"  W&B: Active (projet: {cfg.wandb_project})")
        report_to = "wandb"
    else:
        print(f"  W&B: Desactive")
        report_to = "none"
    
    # Creer le repertoire de sortie
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration SFT (Supervised Fine-Tuning)
    sft_config = SFTConfig(
        # Repertoire de sortie
        output_dir=cfg.output_dir,
        
        # Hyperparametres d'entrainement
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        
        # Optimisation
        fp16=cfg.fp16,
        optim="adamw_torch",
        
        # Logging et sauvegarde
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=report_to,
        
        # Parametres SFT
        max_seq_length=cfg.max_length,
        dataset_text_field="text",
        packing=False,
        
        # Autres
        seed=cfg.seed,
    )
    
    print(f"  OK: Configuration prete")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Gradient accumulation: {cfg.gradient_accumulation}")
    print(f"  Learning rate: {cfg.learning_rate}")
    
    # Creer le trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Lancer l'entrainement
    print(f"\n[6/6] Lancement de l'entrainement...")
    print("=" * 70)
    
    trainer.train()
    
    print("=" * 70)
    print("ENTRAINEMENT TERMINE!")
    
    # Sauvegarder le modele
    print(f"\nSauvegarde du modele...")
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"  OK: Modele sauvegarde dans {cfg.output_dir}")
    
    # Instructions d'utilisation
    print("\n" + "=" * 70)
    print("UTILISATION DU MODELE FINE-TUNE")
    print("=" * 70)
    print(f"\nfrom peft import PeftModel")
    print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"")
    print(f"# Charger le modele de base")
    print(f"base_model = AutoModelForCausalLM.from_pretrained('{cfg.model_name}')")
    print(f"")
    print(f"# Charger les adaptateurs LoRA")
    print(f"model = PeftModel.from_pretrained(base_model, '{cfg.output_dir}')")
    print(f"")
    print(f"# Charger le tokenizer")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{cfg.output_dir}')")
    print("=" * 70)


if __name__ == "__main__":
    # Verifier si mode dry-run
    dry_run = "--dry-run" in sys.argv
    
    # Charger les configurations
    cfg = TrainingConfig()
    lora_cfg = LoRAConfig()
    
    # Lancer l'entrainement
    train(cfg, lora_cfg, dry_run=dry_run)
