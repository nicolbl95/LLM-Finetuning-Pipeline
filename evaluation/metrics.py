"""
Métriques personnalisées pour l'évaluation avec DeepEval.

Ce fichier contient les définitions des métriques utilisées pour évaluer
la qualité des réponses générées par le modèle fine-tuné.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


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
