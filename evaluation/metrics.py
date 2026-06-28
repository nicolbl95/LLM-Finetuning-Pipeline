"""
Métriques personnalisées pour l'évaluation avec DeepEval.

Ce fichier contient les définitions des métriques utilisées pour évaluer
la qualité des réponses générées par le modèle fine-tuné.

Il inclut également des fonctions pour la recherche sémantique avec Pinecone :
- Indexation des questions d'évaluation
- Recherche de questions similaires
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Charger les variables d'environnement depuis .env
load_dotenv()


def get_finance_correctness_metric(threshold: float = 0.7) -> GEval:
    """
    Crée une métrique personnalisée pour évaluer la qualité des réponses financières.
    
    Cette métrique utilise G-Eval pour comparer la réponse générée avec la réponse
    attendue selon trois critères :
    1. Exactitude financière (les informations sont-elles correctes ?)
    2. Clarté (la réponse est-elle facile à comprendre ?)
    3. Cohérence (pas de contradiction avec la réponse attendue)
    
    Args:
        threshold: Seuil minimum pour considérer la réponse comme acceptable (0.0 à 1.0)
        
    Returns:
        GEval configurée pour l'évaluation financière
    """
    return GEval(
        name="Finance Correctness",
        criteria=(
            "Determine whether the actual output is factually correct, clear, "
            "and consistent with the expected answer in a financial context. "
            "Consider: (1) financial accuracy of information, "
            "(2) clarity and comprehensibility, "
            "(3) absence of contradictions with the expected answer."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=threshold,
        # DeepEval utilisera automatiquement OPENAI_API_KEY depuis l'environnement
    )


# ============================================================================
# FONCTIONS POUR LA RECHERCHE SEMANTIQUE AVEC PINECONE
# ============================================================================

def load_json_eval_data(eval_file: str) -> List[Dict[str, str]]:
    """
    Charge les données d'évaluation depuis un fichier JSON.
    
    Le fichier doit contenir une liste d'objets avec les clés :
    - "question" : La question
    - "expected_answer" : La réponse attendue
    
    Args:
        eval_file: Chemin vers le fichier JSON
        
    Returns:
        Liste de dictionnaires contenant les questions et réponses
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        json.JSONDecodeError: Si le fichier n'est pas un JSON valide
    """
    eval_path = Path(eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(f"Le fichier '{eval_file}' n'existe pas")
    
    # Utiliser utf-8-sig pour gérer le BOM UTF-8 sur Windows
    with open(eval_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Le fichier {eval_file} doit contenir une liste JSON")
    
    return data


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Charge le modèle d'embedding Sentence Transformers.
    
    Le modèle par défaut (all-MiniLM-L6-v2) génère des embeddings de dimension 384.
    C'est un bon compromis entre performance et vitesse.
    
    Args:
        model_name: Nom du modèle Sentence Transformers à utiliser
        
    Returns:
        Modèle SentenceTransformer chargé
    """
    print(f"[EMBEDDER] Chargement du modele : {model_name}")
    embedder = SentenceTransformer(model_name)
    print(f"[EMBEDDER] Modele charge (dimension: {embedder.get_sentence_embedding_dimension()})")
    return embedder


def get_pinecone_client() -> Pinecone:
    """
    Crée un client Pinecone en utilisant la clé API depuis l'environnement.
    
    La clé API doit être définie dans :
    - Variable d'environnement PINECONE_API_KEY
    - Ou fichier .env à la racine du projet
    
    Returns:
        Client Pinecone initialisé
        
    Raises:
        ValueError: Si PINECONE_API_KEY n'est pas définie
    """
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY n'est pas definie.\n"
            "Veuillez creer un fichier .env a la racine du projet avec :\n"
            "PINECONE_API_KEY=votre_cle_api_pinecone\n"
            "\n"
            "Obtenez votre cle API sur : https://www.pinecone.io/"
        )
    
    print("[PINECONE] Connexion au client Pinecone...")
    pc = Pinecone(api_key=api_key)
    print("[PINECONE] Client connecte avec succes")
    return pc


def ensure_pinecone_index(
    index_name: str = "eval-index",
    dimension: int = 384
) -> None:
    """
    Crée l'index Pinecone s'il n'existe pas déjà.
    
    L'index est créé avec :
    - Dimension : 384 (pour all-MiniLM-L6-v2)
    - Métrique : cosine (similarité cosinus)
    - Serverless : AWS us-east-1 (gratuit)
    
    Args:
        index_name: Nom de l'index à créer
        dimension: Dimension des embeddings (doit correspondre au modèle)
    """
    pc = get_pinecone_client()
    
    # Vérifier si l'index existe déjà
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"[PINECONE] L'index '{index_name}' existe deja")
        return
    
    # Créer l'index avec ServerlessSpec
    print(f"[PINECONE] Creation de l'index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"[PINECONE] Index '{index_name}' cree avec succes")


def index_eval_dataset(
    eval_file: str,
    index_name: str = "eval-index"
) -> None:
    """
    Indexe toutes les questions d'évaluation dans Pinecone.
    
    Cette fonction :
    1. Charge les questions depuis le fichier JSON
    2. Génère les embeddings avec Sentence Transformers
    3. Stocke les embeddings dans Pinecone avec les métadonnées
    
    Args:
        eval_file: Chemin vers le fichier JSON contenant les questions
        index_name: Nom de l'index Pinecone à utiliser
    """
    # Charger les données
    print(f"[INDEXATION] Chargement des donnees depuis : {eval_file}")
    eval_data = load_json_eval_data(eval_file)
    print(f"[INDEXATION] {len(eval_data)} questions chargees")
    
    # Charger l'embedder
    embedder = get_embedder()
    
    # S'assurer que l'index existe
    ensure_pinecone_index(index_name, dimension=embedder.get_sentence_embedding_dimension())
    
    # Connexion à l'index
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    # Préparer les données pour l'indexation
    print("[INDEXATION] Generation des embeddings...")
    vectors_to_upsert = []
    
    for i, item in enumerate(eval_data):
        question = item["question"]
        expected_answer = item["expected_answer"]
        
        # Générer l'embedding de la question
        embedding = embedder.encode(question).tolist()
        
        # Préparer le vecteur avec métadonnées
        vector = {
            "id": f"q_{i}",
            "values": embedding,
            "metadata": {
                "question": question,
                "expected_answer": expected_answer
            }
        }
        vectors_to_upsert.append(vector)
    
    # Indexer par batch de 100 (limite Pinecone)
    print(f"[INDEXATION] Indexation de {len(vectors_to_upsert)} vecteurs...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"[INDEXATION] Batch {i // batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1) // batch_size} indexe")
    
    print(f"[INDEXATION] Indexation terminee ! {len(vectors_to_upsert)} questions indexees dans '{index_name}'")


def find_similar_questions(
    query: str,
    top_k: int = 3,
    index_name: str = "eval-index"
) -> List[Dict]:
    """
    Trouve les questions les plus similaires à une requête donnée.
    
    Utilise la recherche sémantique pour trouver les questions d'évaluation
    les plus proches de la requête en termes de sens.
    
    Args:
        query: Question à rechercher
        top_k: Nombre de résultats à retourner
        index_name: Nom de l'index Pinecone à interroger
        
    Returns:
        Liste de dictionnaires contenant :
        - question : La question similaire trouvée
        - expected_answer : La réponse attendue
        - score : Score de similarité (0 à 1, 1 = identique)
    """
    # Charger l'embedder
    embedder = get_embedder()
    
    # Générer l'embedding de la requête
    print(f"[RECHERCHE] Recherche de questions similaires a : '{query}'")
    query_embedding = embedder.encode(query).tolist()
    
    # Connexion à l'index
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    # Rechercher les vecteurs similaires
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Formater les résultats
    similar_questions = []
    for match in results.matches:
        similar_questions.append({
            "question": match.metadata["question"],
            "expected_answer": match.metadata["expected_answer"],
            "score": match.score
        })
    
    print(f"[RECHERCHE] {len(similar_questions)} questions similaires trouvees")
    return similar_questions


# ============================================================================
# BLOC CLI POUR TESTER LES FONCTIONS
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gestion de l'index Pinecone pour les questions d'evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :

  # Indexer les questions d'evaluation
  python -m evaluation.metrics --index

  # Rechercher des questions similaires
  python -m evaluation.metrics --search "What is a stock?"

  # Rechercher avec plus de resultats
  python -m evaluation.metrics --search "How to invest?" --top-k 5
        """
    )
    
    parser.add_argument(
        "--index",
        action="store_true",
        help="Indexer les questions d'evaluation dans Pinecone"
    )
    
    parser.add_argument(
        "--search",
        type=str,
        help="Rechercher des questions similaires"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de resultats a retourner (defaut: 3)"
    )
    
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval/eval_questions.json",
        help="Chemin vers le fichier JSON d'evaluation"
    )
    
    parser.add_argument(
        "--index-name",
        type=str,
        default="eval-index",
        help="Nom de l'index Pinecone (defaut: eval-index)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.index:
            # Indexer les questions
            print("\n" + "=" * 70)
            print("INDEXATION DES QUESTIONS D'EVALUATION")
            print("=" * 70)
            index_eval_dataset(args.eval_file, args.index_name)
            print("=" * 70)
            
        elif args.search:
            # Rechercher des questions similaires
            print("\n" + "=" * 70)
            print("RECHERCHE DE QUESTIONS SIMILAIRES")
            print("=" * 70)
            results = find_similar_questions(args.search, args.top_k, args.index_name)
            
            print(f"\nRequete : {args.search}")
            print(f"Nombre de resultats : {len(results)}")
            print("-" * 70)
            
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] Score de similarite : {result['score']:.4f}")
                print(f"Question : {result['question']}")
                print(f"Reponse attendue : {result['expected_answer']}")
                print("-" * 70)
            
            print("=" * 70)
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        exit(1)
