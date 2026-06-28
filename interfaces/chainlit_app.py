"""
Interface Chainlit pour l'assistant financier.

Cette interface permet de discuter avec le modele Mistral fine-tune
via l'API FastAPI. Elle utilise le mode mock par defaut pour tester
sans charger le modele complet.

Pour lancer :
1. Demarrer l'API : uvicorn api.serve:app --reload --port 8000
2. Lancer Chainlit : chainlit run interfaces/chainlit_app.py --port 8080
3. Ouvrir http://localhost:8080
"""

import chainlit as cl
import requests
from typing import List, Dict

# Configuration de l'API
API_URL = "http://localhost:8000"
API_TIMEOUT = 30  # Timeout en secondes pour eviter les blocages

# Message d'accueil professionnel
WELCOME_MESSAGE = """Bienvenue sur l'assistant financier IA.

Je suis un assistant specialise dans les questions financieres, entraine sur des donnees du domaine de la finance.

Vous pouvez me poser des questions sur :
- Les actions et obligations
- Les investissements
- Les concepts financiers
- La gestion de portefeuille
- Et bien plus encore

Comment puis-je vous aider aujourd'hui ?"""


@cl.on_chat_start
async def start():
    """
    Fonction appelee au demarrage du chat.
    Initialise l'historique et affiche le message d'accueil.
    """
    # Initialiser l'historique de conversation dans la session utilisateur
    cl.user_session.set("history", [])
    
    # Afficher le message d'accueil
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Fonction appelee a chaque message utilisateur.
    Envoie la question a l'API FastAPI et streame la reponse.
    
    Args:
        message: Message envoye par l'utilisateur
    """
    # Recuperer l'historique de la session
    history: List[Dict] = cl.user_session.get("history")
    
    # Ajouter le message utilisateur a l'historique
    history.append({"role": "user", "content": message.content})
    
    # Preparer la requete pour l'API
    payload = {
        "question": message.content,
        "max_tokens": 512,
        "use_mock": True  # Mode mock par defaut pour tester sans Mistral 7B
    }
    
    # Creer un message vide pour streamer la reponse
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Envoyer la requete a l'API FastAPI
        response = requests.post(
            f"{API_URL}/generate",
            json=payload,
            timeout=API_TIMEOUT
        )
        
        # Verifier que la requete a reussi
        response.raise_for_status()
        
        # Extraire la reponse JSON
        data = response.json()
        answer = data.get("answer", "Aucune reponse generee.")
        
        # Streamer la reponse mot par mot pour un effet naturel
        words = answer.split()
        for i, word in enumerate(words):
            # Ajouter un espace sauf pour le premier mot
            token = word if i == 0 else f" {word}"
            await msg.stream_token(token)
        
        # Finaliser le message
        await msg.update()
        
        # Ajouter la reponse a l'historique
        history.append({"role": "assistant", "content": answer})
        
    except requests.exceptions.ConnectionError:
        # L'API n'est pas demarree
        error_msg = (
            "Erreur : L'API FastAPI n'est pas demarree.\n\n"
            "Pour demarrer l'API, executez :\n"
            "uvicorn api.serve:app --reload --port 8000"
        )
        await msg.stream_token(error_msg)
        await msg.update()
        
    except requests.exceptions.Timeout:
        # L'API a mis trop de temps a repondre
        error_msg = (
            f"Erreur : L'API n'a pas repondu dans les {API_TIMEOUT} secondes.\n\n"
            "Le modele est peut-etre en train de se charger. Reessayez dans quelques instants."
        )
        await msg.stream_token(error_msg)
        await msg.update()
        
    except requests.exceptions.HTTPError as e:
        # L'API a retourne une erreur HTTP
        error_msg = f"Erreur HTTP {response.status_code} : {str(e)}"
        await msg.stream_token(error_msg)
        await msg.update()
        
    except Exception as e:
        # Autre erreur inattendue
        error_msg = f"Erreur inattendue : {str(e)}"
        await msg.stream_token(error_msg)
        await msg.update()
    
    # Mettre a jour l'historique dans la session
    cl.user_session.set("history", history)
