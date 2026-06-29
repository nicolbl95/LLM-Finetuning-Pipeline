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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# NOTE: Les imports lourds (transformers, peft, torch) sont deplaces
# dans load_model_and_tokenizer() pour permettre le mode mock sans ces dependances


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
    title="LLM Finance Fine-Tuning Pipeline - API Demo",
    description=(
        "Pipeline complet de fine-tuning LLM pour le domaine financier. "
        "Cette API expose un modele Mistral 7B fine-tune avec LoRA sur le dataset finance-alpaca. "
        "Mode demo actuel : mock (sans GPU). Le modele reel sera disponible apres entrainement. "
        "Stack : FastAPI + Docker + Render + Mistral 7B + LoRA/QLoRA + DeepEval + Pinecone + W&B + Chainlit + Plotly Dash."
    ),
    version="1.0.0",
    contact={
        "name": "Nicolas Blondeau",
        "url": "https://github.com/nicolbl95"
    }
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


def load_model_and_tokenizer(cfg):
    """
    Charge le modele et le tokenizer avec support des adaptateurs LoRA.
    
    Cette fonction :
    1. Importe les dependances lourdes (transformers, peft, torch)
    2. Charge le tokenizer du modele de base
    3. Charge le modele de base (Mistral 7B)
    4. Si des adaptateurs LoRA existent dans output_dir, les charge
    5. Cree un pipeline text-generation
    
    NOTE: Les imports sont faits ici pour permettre le mode mock sans ces dependances.
    
    Args:
        cfg: Configuration d'entrainement (contient model_name et output_dir)
        
    Returns:
        tuple: (pipeline, model_name, is_finetuned)
            - pipeline: Pipeline Hugging Face pour la generation
            - model_name: Nom du modele charge
            - is_finetuned: True si adaptateurs LoRA charges
            
    Raises:
        ImportError: Si transformers/peft/torch ne sont pas installes
        FileNotFoundError: Si le modele de base n'existe pas
        RuntimeError: Si erreur lors du chargement
    """
    global _model_pipeline, _model_name, _is_finetuned
    
    # Importer les dependances lourdes seulement quand necessaire
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from peft import PeftModel
        import torch
    except ImportError as e:
        logger.error(f"Dependances manquantes : {e}")
        raise ImportError(
            "Les dependances transformers/peft/torch ne sont pas installees. "
            "Cette instance Render utilise requirements-deploy.txt (mode mock uniquement). "
            "Pour utiliser le modele reel, installez requirements.txt complet dans un environnement local ou Kaggle."
        )
    
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
    
    NOTE: Cette fonction importe TrainingConfig seulement quand necessaire
    pour eviter les imports lourds en mode mock.
    
    Returns:
        tuple: (pipeline, model_name, is_finetuned)
        
    Raises:
        ImportError: Si les dependances ne sont pas installees
        RuntimeError: Si erreur lors du chargement
    """
    global _model_pipeline, _model_name, _is_finetuned
    
    # Si deja charge, retourner directement
    if _model_pipeline is not None:
        return _model_pipeline, _model_name, _is_finetuned
    
    # Sinon, charger maintenant
    logger.info("Premier appel : chargement du modele...")
    
    # Importer TrainingConfig seulement maintenant
    try:
        from training.config import TrainingConfig
    except ImportError as e:
        logger.error(f"Impossible d'importer TrainingConfig : {e}")
        raise ImportError(
            "Le module training.config n'est pas disponible. "
            "Cette instance Render est en mode mock uniquement."
        )
    
    cfg = TrainingConfig()
    _model_pipeline, _model_name, _is_finetuned = load_model_and_tokenizer(cfg)
    
    return _model_pipeline, _model_name, _is_finetuned


# ========================================
# ENDPOINTS
# ========================================

@app.get("/")
async def home():
    """
    Page d'accueil de la demo pour les recruteurs.
    
    Presente le projet, la stack technique, et guide vers les endpoints disponibles.
    
    Returns:
        HTMLResponse: Page HTML professionnelle
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Finance Fine-Tuning Pipeline - Demo</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2em;
                margin-bottom: 10px;
                font-weight: 700;
            }
            .header p {
                font-size: 1.1em;
                opacity: 0.95;
            }
            .content {
                padding: 40px 30px;
            }
            .section {
                margin-bottom: 35px;
            }
            .section h2 {
                color: #667eea;
                font-size: 1.5em;
                margin-bottom: 15px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 8px;
            }
            .section p {
                margin-bottom: 12px;
                color: #555;
            }
            .tech-stack {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 15px;
            }
            .tech-badge {
                background: #f0f0f0;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                color: #667eea;
                font-weight: 600;
            }
            .buttons {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-top: 20px;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 600;
                transition: all 0.3s ease;
                text-align: center;
            }
            .btn:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            .btn-secondary {
                background: #764ba2;
            }
            .btn-secondary:hover {
                background: #653a8a;
            }
            .code-block {
                background: #f5f5f5;
                border-left: 4px solid #667eea;
                padding: 20px;
                border-radius: 6px;
                overflow-x: auto;
                margin-top: 15px;
            }
            .code-block pre {
                margin: 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #333;
            }
            .highlight {
                background: #fff3cd;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #ffc107;
                margin-top: 15px;
            }
            .highlight strong {
                color: #856404;
            }
            .footer {
                background: #f8f9fa;
                padding: 20px 30px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }
            @media (max-width: 600px) {
                .header h1 {
                    font-size: 1.5em;
                }
                .content {
                    padding: 30px 20px;
                }
                .buttons {
                    flex-direction: column;
                }
                .btn {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>LLM Finance Fine-Tuning Pipeline</h1>
                <p>Pipeline complet de fine-tuning LLM pour le domaine financier</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>Objectif du Projet</h2>
                    <p>
                        Ce projet presente un <strong>pipeline complet de fine-tuning LLM</strong> pour le domaine financier.
                        Il couvre toutes les etapes : preparation des donnees, fine-tuning LoRA/QLoRA, evaluation automatisee,
                        recherche semantique, et deploiement API.
                    </p>
                </div>

                <div class="section">
                    <h2>Demo Actuelle</h2>
                    <p>
                        Cette instance Render est une <strong>demo publique en mode mock</strong>. L'API fonctionne sans charger
                        le modele Mistral 7B (7 milliards de parametres) pour economiser les ressources.
                    </p>
                    <div class="highlight">
                        <strong>Pourquoi mode mock ?</strong> Le vrai modele Mistral 7B fine-tune avec LoRA necessite un GPU
                        et sera branche apres l'entrainement complet. Cette demo permet de tester l'architecture de l'API
                        sans infrastructure lourde.
                    </div>
                </div>

                <div class="section">
                    <h2>Stack Technique</h2>
                    <div class="tech-stack">
                        <span class="tech-badge">FastAPI</span>
                        <span class="tech-badge">Docker</span>
                        <span class="tech-badge">Render</span>
                        <span class="tech-badge">Mistral 7B</span>
                        <span class="tech-badge">LoRA/QLoRA</span>
                        <span class="tech-badge">PEFT</span>
                        <span class="tech-badge">DeepEval</span>
                        <span class="tech-badge">Pinecone</span>
                        <span class="tech-badge">Weights & Biases</span>
                        <span class="tech-badge">Chainlit</span>
                        <span class="tech-badge">Plotly Dash</span>
                    </div>
                </div>

                <div class="section">
                    <h2>Endpoints Disponibles</h2>
                    <div class="buttons">
                        <a href="/docs" class="btn">Open API Docs (Swagger)</a>
                        <a href="/health" class="btn btn-secondary">Health Check</a>
                        <a href="/info" class="btn btn-secondary">Project Info</a>
                    </div>
                </div>

                <div class="section">
                    <h2>Exemple d'Utilisation</h2>
                    <p>Testez l'endpoint de generation avec cet exemple :</p>
                    <div class="code-block">
                        <pre>POST /generate

{
  "question": "Quelle est la difference entre une action et une obligation ?",
  "max_tokens": 128,
  "use_mock": true
}</pre>
                    </div>
                    <p style="margin-top: 15px;">
                        <strong>Note :</strong> Utilisez <code>use_mock: true</code> pour obtenir une reponse demo sans charger le modele.
                        Le mode <code>use_mock: false</code> sera disponible apres l'entrainement GPU complet.
                    </p>
                </div>

                <div class="section">
                    <h2>Fonctionnalites Completes</h2>
                    <p>Le projet complet inclut :</p>
                    <ul style="margin-left: 20px; color: #555;">
                        <li>Fine-tuning LoRA/QLoRA sur dataset finance-alpaca (1000 exemples)</li>
                        <li>Evaluation automatisee avec DeepEval (metriques personnalisees)</li>
                        <li>Recherche semantique avec Pinecone (embeddings + vector DB)</li>
                        <li>Tracking des experiences avec Weights & Biases</li>
                        <li>Interface chat Chainlit pour interaction utilisateur</li>
                        <li>Dashboard Plotly Dash pour visualisation des resultats</li>
                        <li>API REST FastAPI avec documentation Swagger</li>
                        <li>Deploiement Docker sur Render</li>
                    </ul>
                </div>

                <div class="section">
                    <h2>Code Source</h2>
                    <p>
                        Le code complet est disponible sur GitHub avec documentation detaillee, notebooks d'exploration,
                        et scripts d'entrainement/evaluation.
                    </p>
                    <div class="buttons">
                        <a href="https://github.com/nicolbl95" class="btn" target="_blank">Voir sur GitHub</a>
                    </div>
                </div>
            </div>

            <div class="footer">
                <p>LLM Finance Fine-Tuning Pipeline v1.0.0 | Nicolas Blondeau | 2024</p>
                <p style="margin-top: 5px;">Construit avec FastAPI, Mistral 7B, LoRA, DeepEval, Pinecone, W&B</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """
    Endpoint de sante de l'API.
    
    Verifie que l'API fonctionne correctement et indique si le modele est charge en memoire.
    Utile pour les health checks automatises et le monitoring.
    
    Returns:
        dict: Statut de l'API avec informations sur le modele
    """
    return {
        "status": "healthy",
        "model_loaded": _model_pipeline is not None,
        "model_name": _model_name if _model_name else "not loaded yet"
    }


@app.get("/info")
async def info():
    """
    Informations detaillees sur l'API, le modele, et la stack technique.
    
    Retourne des informations sur :
    - La version de l'API et le mode de deploiement
    - Le modele de base et les adaptateurs LoRA
    - La stack technique complete (evaluation, vector DB, tracking)
    - Les endpoints disponibles
    
    NOTE: En mode mock (Render), certaines informations ne sont pas disponibles
    car TrainingConfig necessite des dependances lourdes.
    
    Returns:
        dict: Informations detaillees sur le projet
    """
    # En mode mock, retourner des infos limitees sans charger TrainingConfig
    if APP_MODE == "mock":
        return {
            "api_version": "1.0.0",
            "deployment": "Render",
            "api_framework": "FastAPI",
            "app_mode": "mock",
            "base_model": "mistralai/Mistral-7B-v0.1",
            "model_loaded": False,
            "mock_mode_available": True,
            "real_mode_available": False,
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
            "note": "Cette instance Render est en mode demo (mock). Le modele Mistral 7B reel necessite requirements.txt complet et un environnement avec GPU. Utilisez use_mock=true dans vos requetes /generate."
        }
    
    # Mode reel : charger TrainingConfig
    try:
        from training.config import TrainingConfig
        cfg = TrainingConfig()
        
        adapter_path = Path(cfg.output_dir)
        adapter_exists = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
        
        return {
            "api_version": "1.0.0",
            "deployment": "Local/Kaggle",
            "api_framework": "FastAPI",
            "app_mode": "real",
            "base_model": cfg.model_name,
            "adapter_path": str(adapter_path),
            "adapter_available": adapter_exists,
            "model_loaded": _model_pipeline is not None,
            "is_finetuned": _is_finetuned if _model_pipeline else None,
            "mock_mode_available": True,
            "real_mode_available": True,
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
            "note": "Le modele Mistral 7B fine-tune sera disponible apres entrainement LoRA. Mode mock et reel disponibles."
        }
    except ImportError:
        # Si TrainingConfig n'est pas disponible, retourner infos limitees
        return {
            "api_version": "1.0.0",
            "deployment": "Unknown",
            "api_framework": "FastAPI",
            "app_mode": APP_MODE,
            "model_loaded": _model_pipeline is not None,
            "mock_mode_available": True,
            "real_mode_available": False,
            "note": "Configuration incomplete. Mode mock uniquement disponible."
        }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Genere une reponse financiere a partir d'une question.
    
    Deux modes disponibles :
    1. **Mode mock** (use_mock=True) : Retourne une reponse demo sans charger le modele.
       Ideal pour tester l'API sans GPU ni infrastructure lourde.
    
    2. **Mode reel** (use_mock=False) : Utilise le modele Mistral 7B fine-tune avec LoRA.
       Necessite un GPU et le modele entraine. Sera disponible apres entrainement complet.
    
    En environnement Render (APP_MODE=mock), le mode mock est force par defaut pour economiser les ressources.
    
    Args:
        request: Requete contenant la question financiere et les parametres de generation
        
    Returns:
        GenerationResponse: Reponse generee avec le modele utilise et le mode d'execution
        
    Raises:
        HTTPException: Si erreur lors de la generation ou si les dependances ne sont pas installees
        
    Example:
        ```json
        {
          "question": "Quelle est la difference entre une action et une obligation ?",
          "max_tokens": 128,
          "use_mock": true
        }
        ```
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
    
    except ImportError as e:
        logger.error(f"Dependances manquantes : {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Le mode reel n'est pas disponible sur cette instance. "
                "Les dependances transformers/peft/torch ne sont pas installees. "
                "Cette instance Render utilise requirements-deploy.txt (mode mock uniquement). "
                "Pour utiliser le modele Mistral 7B reel, deployez avec requirements.txt complet "
                "dans un environnement local ou Kaggle avec GPU. "
                "Utilisez use_mock=true pour tester l'API en mode demo."
            )
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
