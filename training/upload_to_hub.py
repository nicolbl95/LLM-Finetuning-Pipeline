"""
Script pour uploader les adaptateurs LoRA fine-tunes vers Hugging Face Hub.

Ce script permet de partager votre modele fine-tune sur Hugging Face.
Il verifie que le token HF est configure et que le dossier du modele existe.

Usage:
    # Verifier la configuration sans uploader
    python -m training.upload_to_hub --check-only
    
    # Uploader le modele (par defaut: outputs/mistral-finance)
    python -m training.upload_to_hub
    
    # Uploader avec des parametres personnalises
    python -m training.upload_to_hub --folder-path mon/dossier --repo-id mon-user/mon-modele --private
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi


def check_hf_token() -> str:
    """
    Verifie que le token Hugging Face est configure dans .env
    
    Returns:
        str: Le token HF
        
    Raises:
        ValueError: Si le token n'est pas trouve
    """
    # Charger les variables d'environnement depuis .env
    load_dotenv()
    
    token = os.getenv("HF_TOKEN")
    
    if not token:
        raise ValueError(
            "\n[ERREUR] Token Hugging Face non trouve!\n"
            "Veuillez ajouter votre token dans le fichier .env a la racine du projet:\n"
            "  HF_TOKEN=votre_token_huggingface\n\n"
            "Pour obtenir votre token:\n"
            "  1. Allez sur https://huggingface.co/settings/tokens\n"
            "  2. Creez un nouveau token avec les permissions 'write'\n"
            "  3. Copiez le token dans votre fichier .env\n"
        )
    
    print("[OK] Token Hugging Face trouve")
    return token


def check_model_folder(folder_path: str) -> Path:
    """
    Verifie que le dossier du modele existe et contient des fichiers.
    
    Args:
        folder_path: Chemin vers le dossier du modele
        
    Returns:
        Path: Le chemin valide du dossier
        
    Raises:
        FileNotFoundError: Si le dossier n'existe pas ou est vide
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(
            f"\n[ERREUR] Le dossier '{folder_path}' n'existe pas!\n"
            "Vous devez d'abord entrainer le modele avec:\n"
            "  python -m training.train\n\n"
            "Le modele sera sauvegarde dans outputs/mistral-finance/\n"
        )
    
    # Verifier que le dossier contient des fichiers
    files = list(folder.glob("*"))
    if not files:
        raise FileNotFoundError(
            f"\n[ERREUR] Le dossier '{folder_path}' est vide!\n"
            "Assurez-vous que l'entrainement s'est termine correctement.\n"
        )
    
    print(f"[OK] Dossier du modele trouve: {folder_path}")
    print(f"     Nombre de fichiers: {len(files)}")
    
    return folder


def upload_to_hub(
    folder_path: str,
    repo_id: str,
    token: str,
    private: bool = False
):
    """
    Upload le modele vers Hugging Face Hub.
    
    Args:
        folder_path: Chemin vers le dossier du modele
        repo_id: ID du repo Hugging Face (format: username/repo-name)
        token: Token Hugging Face
        private: Si True, cree un repo prive
    """
    print(f"\n[UPLOAD] Preparation de l'upload vers {repo_id}...")
    
    # Initialiser l'API Hugging Face
    api = HfApi()
    
    # Creer le repo s'il n'existe pas
    print(f"[UPLOAD] Creation/verification du repo...")
    try:
        api.create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True  # Ne pas echouer si le repo existe deja
        )
        print(f"[OK] Repo pret: {repo_id}")
    except Exception as e:
        print(f"[ERREUR] Impossible de creer le repo: {e}")
        raise
    
    # Uploader le dossier complet
    print(f"[UPLOAD] Upload des fichiers en cours...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            token=token,
            repo_type="model"
        )
        print(f"[OK] Upload termine avec succes!")
    except Exception as e:
        print(f"[ERREUR] Echec de l'upload: {e}")
        raise
    
    # Afficher l'URL finale
    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"\n[SUCCES] Modele disponible sur:")
    print(f"  {repo_url}")
    print(f"\nVous pouvez maintenant:")
    print(f"  - Partager ce lien avec d'autres")
    print(f"  - Charger le modele avec: PeftModel.from_pretrained('{repo_id}')")


def main():
    """Point d'entree principal du script."""
    parser = argparse.ArgumentParser(
        description="Upload des adaptateurs LoRA vers Hugging Face Hub"
    )
    
    parser.add_argument(
        "--folder-path",
        type=str,
        default="outputs/mistral-finance",
        help="Chemin vers le dossier du modele fine-tune (defaut: outputs/mistral-finance)"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        default="nicolbl95/mistral-7b-finance-finetuned",
        help="ID du repo Hugging Face (defaut: nicolbl95/mistral-7b-finance-finetuned)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Creer un repo prive (defaut: public)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verifier la configuration sans uploader"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Upload vers Hugging Face Hub")
    print("=" * 60)
    
    try:
        # Etape 1: Verifier le token HF
        print("\n[1/3] Verification du token Hugging Face...")
        token = check_hf_token()
        
        # Etape 2: Verifier le dossier du modele
        print(f"\n[2/3] Verification du dossier du modele...")
        folder = check_model_folder(args.folder_path)
        
        # Etape 3: Upload (sauf si --check-only)
        if args.check_only:
            print("\n[MODE CHECK-ONLY] Verification terminee avec succes!")
            print("Pour uploader le modele, relancez sans --check-only:")
            print(f"  python -m training.upload_to_hub")
        else:
            print(f"\n[3/3] Upload vers Hugging Face Hub...")
            upload_to_hub(
                folder_path=str(folder),
                repo_id=args.repo_id,
                token=token,
                private=args.private
            )
        
        print("\n" + "=" * 60)
        
    except (ValueError, FileNotFoundError) as e:
        print(f"\n{e}")
        print("=" * 60)
        exit(1)
    except Exception as e:
        print(f"\n[ERREUR INATTENDUE] {e}")
        print("=" * 60)
        exit(1)


if __name__ == "__main__":
    main()
