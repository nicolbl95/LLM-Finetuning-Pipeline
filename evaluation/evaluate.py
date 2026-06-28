"""
Script d'évaluation automatisée avec DeepEval.

Ce script permet d'évaluer la qualité des réponses générées par :
- Le modèle de base (avant fine-tuning)
- Le modèle fine-tuné (après entraînement)

Utilisation :
    # Vérification rapide (dry-run)
    python -m evaluation.evaluate --dry-run
    
    # Évaluer le modèle de base
    python -m evaluation.evaluate --model-type base --limit 10
    
    # Évaluer le modèle fine-tuné
    python -m evaluation.evaluate --model-type finetuned --adapter-path outputs/mistral-finance

Prérequis :
    - Fichier data/eval/eval_questions.json avec les questions de test
    - Variable d'environnement OPENAI_API_KEY pour DeepEval
    - Modèle fine-tuné dans outputs/mistral-finance (pour --model-type finetuned)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from training.config import TrainingConfig
from evaluation.metrics import get_finance_correctness_metric


def load_eval_questions(eval_file: str) -> List[Dict[str, str]]:
    """
    Charge les questions d'évaluation depuis un fichier JSON.
    
    Le fichier doit contenir une liste d'objets avec les clés :
    - "question" : La question à poser au modèle
    - "expected_answer" : La réponse attendue
    
    Args:
        eval_file: Chemin vers le fichier JSON
        
    Returns:
        Liste de dictionnaires contenant les questions et réponses attendues
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        json.JSONDecodeError: Si le fichier n'est pas un JSON valide
    """
    eval_path = Path(eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Le fichier d'évaluation '{eval_file}' n'existe pas.\n"
            f"Assurez-vous que le fichier existe avec le format :\n"
            f'[{{"question": "...", "expected_answer": "..."}}]'
        )
    
    # Utiliser utf-8-sig pour gérer le BOM (Byte Order Mark) UTF-8 sur Windows
    # Cela permet de lire correctement les fichiers avec ou sans BOM
    with open(eval_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    # Vérifier que c'est bien une liste
    if not isinstance(data, list):
        raise ValueError(f"Le fichier {eval_file} doit contenir une liste JSON")
    
    # Vérifier que chaque élément a les bonnes clés
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"L'élément {i} n'est pas un dictionnaire")
        if "question" not in item or "expected_answer" not in item:
            raise ValueError(
                f"L'élément {i} doit contenir les clés 'question' et 'expected_answer'"
            )
    
    return data


def build_prompt(question: str) -> str:
    """
    Construit le prompt au format Alpaca pour une question donnée.
    
    Le format utilisé est :
    ### Instruction:
    {question}
    
    ### Response:
    
    Args:
        question: La question à poser au modèle
        
    Returns:
        Le prompt formaté
    """
    return f"""### Instruction:
{question}

### Response:
"""


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 256
) -> str:
    """
    Génère une réponse à partir d'une question en utilisant le modèle.
    
    Args:
        model: Le modèle de langage à utiliser
        tokenizer: Le tokenizer correspondant au modèle
        question: La question à poser
        max_new_tokens: Nombre maximum de tokens à générer
        
    Returns:
        La réponse générée par le modèle
    """
    # Construire le prompt
    prompt = build_prompt(question)
    
    # Tokenizer le prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Déplacer sur le même device que le modèle
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Générer la réponse
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Décoder la réponse complète
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire uniquement la partie après "### Response:"
    if "### Response:" in full_response:
        response = full_response.split("### Response:")[-1].strip()
    else:
        response = full_response.strip()
    
    return response


def load_base_model(cfg: TrainingConfig):
    """
    Charge le modèle de base (avant fine-tuning).
    
    Args:
        cfg: Configuration contenant le nom du modèle
        
    Returns:
        Tuple (model, tokenizer)
    """
    print(f"\n[CHARGEMENT] Modele de base : {cfg.model_name}")
    print("-" * 70)
    
    # Charger le tokenizer
    print("  Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    # Configurer le pad_token si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle
    print("  Chargement du modele...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Mettre en mode évaluation
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Modele charge sur : {device}")
    print("-" * 70)
    
    return model, tokenizer


def load_finetuned_model(cfg: TrainingConfig, adapter_path: str):
    """
    Charge le modèle fine-tuné avec les adaptateurs LoRA.
    
    Args:
        cfg: Configuration contenant le nom du modèle de base
        adapter_path: Chemin vers les adaptateurs LoRA
        
    Returns:
        Tuple (model, tokenizer)
        
    Raises:
        FileNotFoundError: Si le chemin des adaptateurs n'existe pas
    """
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        raise FileNotFoundError(
            f"Le chemin des adaptateurs '{adapter_path}' n'existe pas.\n"
            f"Assurez-vous d'avoir d'abord entraîné le modèle avec :\n"
            f"  python -m training.train"
        )
    
    print(f"\n[CHARGEMENT] Modele fine-tune : {cfg.model_name}")
    print(f"             Adaptateurs LoRA : {adapter_path}")
    print("-" * 70)
    
    # Charger le tokenizer
    print("  Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    # Configurer le pad_token si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle de base
    print("  Chargement du modele de base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Charger les adaptateurs LoRA
    print("  Chargement des adaptateurs LoRA...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Fusionner les adaptateurs pour accélérer l'inférence
    print("  Fusion des adaptateurs...")
    model = model.merge_and_unload()
    
    # Mettre en mode évaluation
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Modele charge sur : {device}")
    print("-" * 70)
    
    return model, tokenizer


def build_test_cases(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, str]],
    limit: Optional[int] = None
) -> List[LLMTestCase]:
    """
    Construit les cas de test DeepEval en générant les réponses du modèle.
    
    Args:
        model: Le modèle à évaluer
        tokenizer: Le tokenizer correspondant
        eval_data: Liste des questions et réponses attendues
        limit: Nombre maximum de questions à évaluer (None = toutes)
        
    Returns:
        Liste de LLMTestCase pour DeepEval
    """
    # Limiter le nombre de questions si demandé
    if limit is not None:
        eval_data = eval_data[:limit]
    
    print(f"\n[GENERATION] Generation des reponses pour {len(eval_data)} questions...")
    print("-" * 70)
    
    test_cases = []
    
    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        
        print(f"  [{i}/{len(eval_data)}] Generation en cours...", end="\r")
        
        # Générer la réponse
        actual_output = generate_response(model, tokenizer, question)
        
        # Créer le cas de test DeepEval
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_answer
        )
        
        test_cases.append(test_case)
    
    print(f"  [{len(eval_data)}/{len(eval_data)}] Generation terminee !{' ' * 20}")
    print("-" * 70)
    
    return test_cases


def run_deepeval(test_cases: List[LLMTestCase], run_name: str) -> dict:
    """
    Exécute l'évaluation DeepEval avec les métriques configurées.
    
    Args:
        test_cases: Liste des cas de test à évaluer
        run_name: Nom de l'exécution pour identification
        
    Returns:
        Dictionnaire contenant les résultats de l'évaluation
    """
    print(f"\n[EVALUATION] Execution de DeepEval : {run_name}")
    print("-" * 70)
    
    # Configurer les métriques
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        get_finance_correctness_metric(threshold=0.7)
    ]
    
    print(f"  Metriques configurees :")
    print(f"    - Answer Relevancy (seuil: 0.7)")
    print(f"    - Finance Correctness (seuil: 0.7)")
    print(f"  Nombre de cas de test : {len(test_cases)}")
    print()
    print("  ATTENTION : DeepEval utilise OpenAI pour l'evaluation.")
    print("  Assurez-vous que OPENAI_API_KEY est definie dans votre environnement.")
    print("-" * 70)
    
    # Exécuter l'évaluation
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        run_async=False  # Exécution séquentielle pour plus de stabilité
    )
    
    print("\n[RESULTATS] Evaluation terminee !")
    print("-" * 70)
    
    return results


def save_results(results: dict, output_path: str):
    """
    Sauvegarde les résultats de l'évaluation dans un fichier JSON.
    
    Args:
        results: Résultats de l'évaluation DeepEval
        output_path: Chemin du fichier de sortie
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Préparer les résultats pour la sérialisation JSON
    serializable_results = {
        "timestamp": datetime.now().isoformat(),
        "results": str(results)  # DeepEval results peuvent ne pas être directement sérialisables
    }
    
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAUVEGARDE] Resultats sauvegardes dans : {output_path}")


def dry_run(eval_file: str):
    """
    Mode dry-run : vérifie que le fichier d'évaluation est lisible sans charger le modèle.
    
    Args:
        eval_file: Chemin vers le fichier d'évaluation
    """
    print("\n" + "=" * 70)
    print("MODE DRY-RUN : VERIFICATION DU FICHIER D'EVALUATION")
    print("=" * 70)
    
    try:
        # Charger les questions
        eval_data = load_eval_questions(eval_file)
        
        print(f"\n[OK] Fichier charge avec succes : {eval_file}")
        print(f"     Nombre total de questions : {len(eval_data)}")
        
        # Afficher les 2 premières questions
        print("\n[APERCU] Affichage des 2 premieres questions :")
        print("-" * 70)
        
        for i, item in enumerate(eval_data[:2], 1):
            question = item["question"]
            expected_answer = item["expected_answer"]
            
            print(f"\n--- Question {i} ---")
            print(f"Question : {question}")
            print(f"\nReponse attendue : {expected_answer}")
            
            # Afficher le prompt qui serait utilisé
            prompt = build_prompt(question)
            print(f"\nPrompt genere :")
            print(prompt)
            print("-" * 70)
        
        print("\n[OK] Dry-run termine avec succes !")
        print("     Le fichier est au bon format et pret pour l'evaluation.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERREUR] Probleme detecte : {e}")
        print("=" * 70)
        raise


def main():
    """Point d'entrée principal du script d'évaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluation automatisee avec DeepEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :

  # Verification rapide (dry-run)
  python -m evaluation.evaluate --dry-run

  # Evaluer le modele de base (10 premieres questions)
  python -m evaluation.evaluate --model-type base --limit 10

  # Evaluer le modele fine-tune (toutes les questions)
  python -m evaluation.evaluate --model-type finetuned

  # Evaluer avec un chemin d'adaptateur personnalise
  python -m evaluation.evaluate --model-type finetuned --adapter-path mon/chemin/custom
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode verification : lit 2 questions et affiche les prompts sans charger le modele"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["base", "finetuned"],
        default="base",
        help="Type de modele a evaluer : 'base' (avant fine-tuning) ou 'finetuned' (apres)"
    )
    
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="outputs/mistral-finance",
        help="Chemin vers les adaptateurs LoRA (pour --model-type finetuned)"
    )
    
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval/eval_questions.json",
        help="Chemin vers le fichier JSON contenant les questions d'evaluation"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximum de questions a evaluer (None = toutes)"
    )
    
    args = parser.parse_args()
    
    # Mode dry-run
    if args.dry_run:
        dry_run(args.eval_file)
        return
    
    # Charger la configuration
    cfg = TrainingConfig()
    
    # Charger les questions d'évaluation
    eval_data = load_eval_questions(args.eval_file)
    print(f"\n[INFO] {len(eval_data)} questions chargees depuis {args.eval_file}")
    
    if args.limit:
        print(f"[INFO] Limitation a {args.limit} questions")
    
    # Charger le modèle approprié
    if args.model_type == "base":
        model, tokenizer = load_base_model(cfg)
        run_name = f"base_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        model, tokenizer = load_finetuned_model(cfg, args.adapter_path)
        run_name = f"finetuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Construire les cas de test
    test_cases = build_test_cases(model, tokenizer, eval_data, args.limit)
    
    # Exécuter l'évaluation
    results = run_deepeval(test_cases, run_name)
    
    # Sauvegarder les résultats
    output_path = f"outputs/evaluation/{run_name}_results.json"
    save_results(results, output_path)
    
    print("\n" + "=" * 70)
    print("[TERMINE] Evaluation completee avec succes !")
    print("=" * 70)


if __name__ == "__main__":
    main()
