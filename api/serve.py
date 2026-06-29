"""
API REST FastAPI pour exposer le modele fine-tune.

Cette API permet de generer des reponses a partir du modele Mistral fine-tune
avec LoRA sur le dataset finance-alpaca.

Fonctionnalites :
- Mode mock pour tester l'API sans charger le modele (use_mock=True)
- Chargement lazy du modele (charge seulement au premier appel)
- Support des adaptateurs LoRA si disponibles
- Endpoints : /health, /info, /generate

Usage :
    uvicorn api.serve:app --reload --port 8000
    
Puis ouvrir : http://localhost:8000/docs
"""

from pathlib import Path
from typing import Optional
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

from training.config import TrainingConfig


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mode de l'application (mock ou model)
# En production sur Render, APP_MODE=mock par defaut
APP_MODE = os.getenv("APP_MODE", "mock")
logger.info(f"APP_MODE: {APP_MODE}")


# ========================================
# MODELES PYDANTIC
# ========================================

class GenerationRequest(BaseModel):
    """
    Requete de generation de texte.
    
    Attributes:
        question: Question ou instruction a soumettre au modele
        max_tokens: Nombre maximum de tokens a generer (defaut: 512)
        use_mock: Si True, retourne une reponse factice sans charger le modele
    """
    question: str = Field(..., description="Question ou instruction", min_length=1)
    max_tokens: int = Field(512, description="Nombre max de tokens a generer", ge=1, le=2048)
    use_mock: bool = Field(False, description="Utiliser le mode mock (pas de modele)")


class GenerationResponse(BaseModel):
    """
    Reponse de generation de texte.
    
    Attributes:
        answer: Reponse generee par le modele
        model: Nom du modele utilise
        mode: Mode d'execution (mock ou real)
    """
    answer: str = Field(..., description="Reponse generee")
    model: str = Field(..., description="Nom du modele")
    mode: str = Field(..., description="Mode d'execution (mock/real)")


# ========================================
# APPLICATION FASTAPI
# ========================================

app = FastAPI(
    title="API Mistral Finance Fine-Tuned",
    description="API REST pour generer des reponses financieres avec Mistral 7B fine-tune avec LoRA",
    version="1.0.0"
)


# ========================================
# VARIABLES GLOBALES (CHARGEMENT LAZY)
# ========================================

# Ces variables seront initialisees au premier appel /generate avec use_mock=False
_model_pipeline = None
_model_name = None
_is_finetuned = False


# ========================================
# FONCTIONS UTILITAIRES
# ========================================

def build_prompt(question: str) -> str:
    """
    Construit le prompt au format Alpaca pour le modele.
    
    Format :
        ### Instruction:
        {question}
        
        ### Response:
    
    Args:
        question: Question ou instruction de l'utilisateur
        
    Returns:
        Prompt formate pret pour le modele
    """
    return f"""### Instruction:
{question}

### Response:
"""


def load_model_and_tokenizer(cfg: TrainingConfig):
    """
    Charge le modele et le tokenizer avec support des adaptateurs LoRA.
    
    Cette fonction :
    1. Charge le tokenizer du modele de base
    2. Charge le modele de base (Mistral 7B)
    3. Si des adaptateurs LoRA existent dans output_dir, les charge
    4. Cree un pipeline text-generation
    
    Args:
        cfg: Configuration d'entrainement (contient model_name et output_dir)
        
    Returns:
        tuple: (pipeline, model_name, is_finetuned)
            - pipeline: Pipeline Hugging Face pour la generation
            - model_name: Nom du modele charge
            - is_finetuned: True si adaptateurs LoRA charges
            
    Raises:
        FileNotFoundError: Si le modele de base n'existe pas
        RuntimeError: Si erreur lors du chargement
    """
    global _model_pipeline, _model_name, _is_finetuned
    
    logger.info(f"Chargement du tokenizer : {cfg.model_name}")
    
    try:
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        
        # S'assurer que le tokenizer a un pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("pad_token defini comme eos_token")
        
        logger.info(f"Chargement du modele de base : {cfg.model_name}")
        logger.info("ATTENTION : Cela peut prendre plusieurs minutes et necessite beaucoup de memoire")
        
        # Charger le modele de base
        # device_map="auto" : repartit automatiquement le modele sur GPU/CPU
        # torch_dtype=torch.float16 : utilise float16 pour economiser la memoire
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Verifier si des adaptateurs LoRA existent
        adapter_path = Path(cfg.output_dir)
        is_finetuned = False
        
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            logger.info(f"Adaptateurs LoRA trouves dans : {adapter_path}")
            logger.info("Chargement des adaptateurs LoRA...")
            
            try:
                # Charger les adaptateurs LoRA par-dessus le modele de base
                model = PeftModel.from_pretrained(model, str(adapter_path))
                is_finetuned = True
                logger.info("Adaptateurs LoRA charges avec succes")
            except Exception as e:
                logger.warning(f"Impossible de charger les adaptateurs LoRA : {e}")
                logger.warning("Utilisation du modele de base sans fine-tuning")
        else:
            logger.info("Aucun adaptateur LoRA trouve")
            logger.info(f"Pour utiliser le modele fine-tune, assurez-vous que {adapter_path} existe")
            logger.info("Utilisation du modele de base")
        
        # Creer le pipeline de generation
        logger.info("Creation du pipeline de generation...")
        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        model_name = cfg.model_name
        if is_finetuned:
            model_name += " (fine-tuned with LoRA)"
        
        logger.info("Modele charge avec succes")
        
        return gen_pipeline, model_name, is_finetuned
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modele : {e}")
        raise RuntimeError(f"Impossible de charger le modele : {e}")


def get_model_pipeline():
    """
    Recupere le pipeline du modele (chargement lazy).
    
    Si le modele n'est pas encore charge, le charge maintenant.
    
    Returns:
        tuple: (pipeline, model_name, is_finetuned)
        
    Raises:
        RuntimeError: Si erreur lors du chargement
    """
    global _model_pipeline, _model_name, _is_finetuned
    
    # Si deja charge, retourner directement
    if _model_pipeline is not None:
        return _model_pipeline, _model_name, _is_finetuned
    
    # Sinon, charger maintenant
    logger.info("Premier appel : chargement du modele...")
    cfg = TrainingConfig()
    _model_pipeline, _model_name, _is_finetuned = load_model_and_tokenizer(cfg)
    
    return _model_pipeline, _model_name, _is_finetuned


# ========================================
# ENDPOINTS
# ========================================

@app.get("/health")
async def health_check():
    """
    Endpoint de sante de l'API.
    
    Retourne le statut de l'API et si le modele est charge.
    
    Returns:
        dict: Statut de l'API
    """
    return {
        "status": "healthy",
        "model_loaded": _model_pipeline is not None,
        "model_name": _model_name if _model_name else "not loaded yet"
    }


@app.get("/info")
async def info():
    """
    Informations sur l'API et le modele.
    
    Returns:
        dict: Informations detaillees
    """
    cfg = TrainingConfig()
    
    adapter_path = Path(cfg.output_dir)
    adapter_exists = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
    
    return {
        "api_version": "1.0.0",
        "deployment": "Render",
        "api_framework": "FastAPI",
        "app_mode": APP_MODE,
        "base_model": cfg.model_name,
        "adapter_path": str(adapter_path),
        "adapter_available": adapter_exists,
        "model_loaded": _model_pipeline is not None,
        "is_finetuned": _is_finetuned if _model_pipeline else None,
        "mock_mode_available": True,
        "llmops_stack": {
            "evaluation": "DeepEval",
            "vector_db": "Pinecone",
            "experiment_tracking": "Weights & Biases"
        },
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "generate": "POST /generate"
        },
        "note": "Le modele Mistral 7B fine-tune sera disponible apres entrainement LoRA. Mode mock actif pour demo."
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Genere une reponse a partir d'une question.
    
    Deux modes disponibles :
    1. Mode mock (use_mock=True) : Retourne une reponse factice sans charger le modele
    2. Mode reel (use_mock=False) : Utilise le modele Mistral fine-tune
    
    En environnement Render (APP_MODE=mock), le mode mock est force par defaut.
    
    Args:
        request: Requete contenant la question et les parametres
        
    Returns:
        GenerationResponse: Reponse generee
        
    Raises:
        HTTPException: Si erreur lors de la generation
    """
    # Si APP_MODE=mock, forcer le mode mock (environnement Render)
    use_mock = request.use_mock or (APP_MODE == "mock")
    
    # Mode mock : reponse factice sans charger le modele
    if use_mock:
        logger.info(f"Mode mock : question = '{request.question[:50]}...'")
        
        mock_answer = (
            f"[MODE DEMO] Reponse simulee pour la question : '{request.question}'. "
            f"Cette API est une demonstration du pipeline LLM fine-tuning. "
            f"Le modele Mistral 7B fine-tune avec LoRA sur le dataset finance-alpaca "
            f"sera disponible apres entrainement complet. "
            f"Stack technique : FastAPI + Transformers + PEFT + DeepEval + Pinecone + W&B. "
            f"Consultez /info pour plus de details sur le projet."
        )
        
        return GenerationResponse(
            answer=mock_answer,
            model="mock-model",
            mode="mock"
        )
    
    # Mode reel : utiliser le modele
    try:
        logger.info(f"Mode reel : chargement/utilisation du modele...")
        logger.info(f"Question : {request.question[:100]}...")
        
        # Charger le modele si necessaire (lazy loading)
        gen_pipeline, model_name, is_finetuned = get_model_pipeline()
        
        # Construire le prompt au format Alpaca
        prompt = build_prompt(request.question)
        
        # Generer la reponse
        logger.info("Generation en cours...")
        outputs = gen_pipeline(
            prompt,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Extraire la reponse generee
        generated_text = outputs[0]["generated_text"]
        
        # Extraire seulement la partie apres "### Response:"
        if "### Response:" in generated_text:
            answer = generated_text.split("### Response:")[-1].strip()
        else:
            answer = generated_text.strip()
        
        logger.info(f"Reponse generee : {answer[:100]}...")
        
        return GenerationResponse(
            answer=answer,
            model=model_name,
            mode="real"
        )
        
    except RuntimeError as e:
        logger.error(f"Erreur lors du chargement du modele : {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Impossible de charger le modele : {str(e)}. "
                f"Le modele Mistral 7B est peut-etre trop lourd pour votre machine. "
                f"Essayez avec use_mock=true pour tester l'API sans charger le modele."
            )
        )
    except Exception as e:
        logger.error(f"Erreur lors de la generation : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la generation : {str(e)}"
        )


# ========================================
# POINT D'ENTREE
# ========================================

if __name__ == "__main__":
    """
    Point d'entree pour lancer l'API en mode developpement.
    
    Pour lancer l'API :
        uvicorn api.serve:app --reload --port 8000
    
    Puis ouvrir dans votre navigateur :
        http://localhost:8000/docs
    
    Endpoints disponibles :
        - GET  /health  : Verifier que l'API fonctionne
        - GET  /info    : Informations sur le modele
        - POST /generate : Generer une reponse
    
    Exemple de requete (mode mock) :
        curl -X POST "http://localhost:8000/generate" \
             -H "Content-Type: application/json" \
             -d '{"question": "What is a stock?", "use_mock": true}'
    
    Exemple de requete (mode reel) :
        curl -X POST "http://localhost:8000/generate" \
             -H "Content-Type: application/json" \
             -d '{"question": "What is a stock?", "use_mock": false, "max_tokens": 256}'
    """
    import uvicorn
    
    print("=" * 70)
    print("API MISTRAL FINANCE FINE-TUNED")
    print("=" * 70)
    print()
    print("Pour lancer l'API, executez :")
    print("  uvicorn api.serve:app --reload --port 8000")
    print()
    print("Puis ouvrez dans votre navigateur :")
    print("  http://localhost:8000/docs")
    print()
    print("Endpoints disponibles :")
    print("  - GET  /health   : Verifier que l'API fonctionne")
    print("  - GET  /info     : Informations sur le modele")
    print("  - POST /generate : Generer une reponse")
    print()
    print("Mode mock (sans charger le modele) :")
    print('  {"question": "What is a stock?", "use_mock": true}')
    print()
    print("Mode reel (avec le modele Mistral) :")
    print('  {"question": "What is a stock?", "use_mock": false}')
    print()
    print("=" * 70)
