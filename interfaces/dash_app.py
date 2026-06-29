"""
Dashboard Plotly Dash pour visualiser le fine-tuning LoRA de Mistral 7B.

Ce dashboard affiche :
- KPIs d'amelioration des metriques
- Comparaison avant/apres fine-tuning
- Courbe de training loss
- Informations sur le modele et la methode

Les donnees sont mock tant que l'entrainement reel n'est pas termine.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from pathlib import Path
import json
from datetime import datetime

# Initialiser l'application Dash
app = Dash(__name__)
app.title = "LLM Fine-tuning Dashboard"

# ============================================================================
# DONNEES MOCK
# ============================================================================
# Ces donnees seront remplacees par les vrais resultats d'evaluation
# une fois l'entrainement termine

def get_mock_evaluation_data():
    """
    Genere des donnees mock pour l'evaluation.
    
    Returns:
        dict: Scores avant et apres fine-tuning
    """
    return {
        "base_model": {
            "answer_relevancy": 0.62,
            "finance_correctness": 0.58,
            "num_questions": 100
        },
        "finetuned_model": {
            "answer_relevancy": 0.84,
            "finance_correctness": 0.89,
            "num_questions": 100
        }
    }

def get_mock_training_loss():
    """
    Genere une courbe de training loss simulee.
    
    Returns:
        pd.DataFrame: Steps et loss values
    """
    # Simuler une courbe de loss decroissante
    steps = list(range(0, 1001, 50))
    # Loss qui diminue de 2.5 a 0.8 avec un peu de bruit
    import numpy as np
    base_loss = [2.5 - (2.5 - 0.8) * (s / 1000) for s in steps]
    noise = np.random.normal(0, 0.05, len(steps))
    loss = [max(0.5, b + n) for b, n in zip(base_loss, noise)]
    
    return pd.DataFrame({
        "step": steps,
        "loss": loss
    })

def get_lora_config():
    """
    Retourne la configuration LoRA du projet.
    
    Returns:
        dict: Parametres LoRA
    """
    return {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "total_params": 7_000_000_000,  # Mistral 7B
    }

# ============================================================================
# CALCULS DES METRIQUES
# ============================================================================

def calculate_improvement(base_score, finetuned_score):
    """
    Calcule l'amelioration en pourcentage.
    
    Args:
        base_score: Score du modele de base
        finetuned_score: Score du modele fine-tune
        
    Returns:
        float: Amelioration en pourcentage
    """
    if base_score == 0:
        return 0
    return ((finetuned_score - base_score) / base_score) * 100

def calculate_trainable_params_percentage(lora_r, total_params):
    """
    Estime le pourcentage de parametres entrainables avec LoRA.
    
    Args:
        lora_r: Rang LoRA
        total_params: Nombre total de parametres du modele
        
    Returns:
        float: Pourcentage de parametres entrainables
    """
    # Estimation simplifiee : LoRA ajoute 2 * r * d parametres par module
    # Pour Mistral 7B avec 2 modules (q_proj, v_proj) et d=4096
    d = 4096  # Dimension cachee de Mistral 7B
    num_layers = 32  # Nombre de couches
    num_modules = 2  # q_proj et v_proj
    
    lora_params = num_layers * num_modules * 2 * lora_r * d
    percentage = (lora_params / total_params) * 100
    
    return percentage

def estimate_adapter_size_mb(lora_r):
    """
    Estime la taille des adaptateurs LoRA en MB.
    
    Args:
        lora_r: Rang LoRA
        
    Returns:
        float: Taille estimee en MB
    """
    # Estimation : ~4 bytes par parametre (float32)
    d = 4096
    num_layers = 32
    num_modules = 2
    
    lora_params = num_layers * num_modules * 2 * lora_r * d
    size_mb = (lora_params * 4) / (1024 * 1024)
    
    return size_mb

# ============================================================================
# COMPOSANTS DU DASHBOARD
# ============================================================================

def create_kpi_card(title, value, subtitle="", color="#1f77b4"):
    """
    Cree une carte KPI.
    
    Args:
        title: Titre de la metrique
        value: Valeur a afficher
        subtitle: Texte supplementaire
        color: Couleur de la carte
        
    Returns:
        html.Div: Composant Dash
    """
    return html.Div([
        html.H4(title, style={
            "margin": "0",
            "color": "#666",
            "fontSize": "14px",
            "fontWeight": "normal"
        }),
        html.H2(value, style={
            "margin": "10px 0",
            "color": color,
            "fontSize": "32px",
            "fontWeight": "bold"
        }),
        html.P(subtitle, style={
            "margin": "0",
            "color": "#999",
            "fontSize": "12px"
        })
    ], style={
        "backgroundColor": "white",
        "padding": "20px",
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "textAlign": "center"
    })

def create_comparison_chart(eval_data):
    """
    Cree un graphique de comparaison avant/apres.
    
    Args:
        eval_data: Donnees d'evaluation
        
    Returns:
        dcc.Graph: Graphique Plotly
    """
    # Preparer les donnees
    metrics = ["Answer Relevancy", "Finance Correctness"]
    base_scores = [
        eval_data["base_model"]["answer_relevancy"],
        eval_data["base_model"]["finance_correctness"]
    ]
    finetuned_scores = [
        eval_data["finetuned_model"]["answer_relevancy"],
        eval_data["finetuned_model"]["finance_correctness"]
    ]
    
    df = pd.DataFrame({
        "Metric": metrics * 2,
        "Score": base_scores + finetuned_scores,
        "Model": ["Base Model"] * 2 + ["Fine-tuned Model"] * 2
    })
    
    # Creer le graphique
    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Comparaison des Scores : Modele de Base vs Fine-tune",
        color_discrete_map={
            "Base Model": "#ff7f0e",
            "Fine-tuned Model": "#2ca02c"
        }
    )
    
    fig.update_layout(
        yaxis_title="Score",
        yaxis_range=[0, 1],
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(gridcolor="#f0f0f0")
    
    return dcc.Graph(figure=fig)

def create_training_loss_chart(loss_data):
    """
    Cree un graphique de training loss.
    
    Args:
        loss_data: DataFrame avec steps et loss
        
    Returns:
        dcc.Graph: Graphique Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=loss_data["step"],
        y=loss_data["loss"],
        mode="lines",
        name="Training Loss",
        line=dict(color="#1f77b4", width=2)
    ))
    
    fig.update_layout(
        title="Evolution de la Training Loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(gridcolor="#f0f0f0")
    
    return dcc.Graph(figure=fig)

def create_info_section():
    """
    Cree la section d'informations sur le projet.
    
    Returns:
        html.Div: Composant Dash
    """
    return html.Div([
        html.H3("Informations sur le Projet", style={
            "marginBottom": "20px",
            "color": "#333"
        }),
        html.Div([
            html.Div([
                html.Strong("Modele : "),
                html.Span("Mistral 7B")
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Strong("Methode : "),
                html.Span("LoRA (Low-Rank Adaptation) / QLoRA (4-bit)")
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Strong("Domaine : "),
                html.Span("Finance")
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Strong("Dataset : "),
                html.Span("finance-alpaca (1000 exemples)")
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Strong("Configuration LoRA : "),
                html.Span("r=16, alpha=32, target_modules=[q_proj, v_proj]")
            ], style={"marginBottom": "10px"}),
        ])
    ], style={
        "backgroundColor": "white",
        "padding": "20px",
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "marginTop": "20px"
    })

# ============================================================================
# LAYOUT DU DASHBOARD
# ============================================================================

# Charger les donnees
eval_data = get_mock_evaluation_data()
loss_data = get_mock_training_loss()
lora_config = get_lora_config()

# Calculer les metriques
relevancy_improvement = calculate_improvement(
    eval_data["base_model"]["answer_relevancy"],
    eval_data["finetuned_model"]["answer_relevancy"]
)

correctness_improvement = calculate_improvement(
    eval_data["base_model"]["finance_correctness"],
    eval_data["finetuned_model"]["finance_correctness"]
)

trainable_params_pct = calculate_trainable_params_percentage(
    lora_config["r"],
    lora_config["total_params"]
)

adapter_size = estimate_adapter_size_mb(lora_config["r"])

# Definir le layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("LLM Fine-tuning Dashboard", style={
            "margin": "0",
            "color": "white",
            "fontSize": "28px"
        }),
        html.P("Mistral 7B - Fine-tuning LoRA sur le domaine Finance", style={
            "margin": "5px 0 0 0",
            "color": "#e0e0e0",
            "fontSize": "14px"
        })
    ], style={
        "backgroundColor": "#1f77b4",
        "padding": "30px",
        "marginBottom": "30px"
    }),
    
    # Container principal
    html.Div([
        # Note sur les donnees mock
        html.Div([
            html.P([
                html.Strong("Note : "),
                "Les scores affiches sont des donnees mock. "
                "Ils seront remplaces par les vrais resultats d'evaluation "
                "une fois l'entrainement termine."
            ], style={
                "margin": "0",
                "color": "#856404",
                "fontSize": "14px"
            })
        ], style={
            "backgroundColor": "#fff3cd",
            "border": "1px solid #ffeaa7",
            "borderRadius": "4px",
            "padding": "15px",
            "marginBottom": "30px"
        }),
        
        # KPI Cards
        html.Div([
            html.Div([
                create_kpi_card(
                    "Amelioration Answer Relevancy",
                    f"+{relevancy_improvement:.1f}%",
                    f"De {eval_data['base_model']['answer_relevancy']:.2f} a {eval_data['finetuned_model']['answer_relevancy']:.2f}",
                    color="#2ca02c"
                )
            ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}),
            
            html.Div([
                create_kpi_card(
                    "Amelioration Finance Correctness",
                    f"+{correctness_improvement:.1f}%",
                    f"De {eval_data['base_model']['finance_correctness']:.2f} a {eval_data['finetuned_model']['finance_correctness']:.2f}",
                    color="#2ca02c"
                )
            ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}),
            
            html.Div([
                create_kpi_card(
                    "Parametres Entrainables",
                    f"{trainable_params_pct:.2f}%",
                    f"LoRA r={lora_config['r']}",
                    color="#ff7f0e"
                )
            ], style={"width": "23%", "display": "inline-block", "marginRight": "2%"}),
            
            html.Div([
                create_kpi_card(
                    "Taille Adaptateurs LoRA",
                    f"{adapter_size:.0f} MB",
                    "vs 14 GB modele complet",
                    color="#9467bd"
                )
            ], style={"width": "23%", "display": "inline-block"})
        ], style={"marginBottom": "30px"}),
        
        # Graphiques
        html.Div([
            # Graphique de comparaison
            html.Div([
                create_comparison_chart(eval_data)
            ], style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                "width": "48%",
                "display": "inline-block",
                "marginRight": "2%",
                "verticalAlign": "top"
            }),
            
            # Graphique de training loss
            html.Div([
                create_training_loss_chart(loss_data)
            ], style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                "width": "48%",
                "display": "inline-block",
                "verticalAlign": "top"
            })
        ], style={"marginBottom": "30px"}),
        
        # Section d'informations
        create_info_section()
        
    ], style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "0 20px"
    })
], style={
    "backgroundColor": "#f5f5f5",
    "minHeight": "100vh",
    "fontFamily": "Arial, sans-serif"
})

# ============================================================================
# POINT D'ENTREE
# ============================================================================

if __name__ == "__main__":
    print("Demarrage du dashboard LLM Fine-tuning...")
    print("Ouvrez votre navigateur a l'adresse : http://localhost:8050")
    app.run(debug=True, port=8050)
