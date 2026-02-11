"""
============================================================
CONFIGURATION GLOBALE - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Styles CSS, constantes et configurations partagées
============================================================
"""

# Mapping pays ISO3
PAYS_ISO3_MAP = {
    "Niger": "ner",
    "Burkina Faso": "bfa",
    "Mali": "mli",
    "Mauritanie": "mrt"
}

# CSS personnalisé unifié (branding MSF)
CUSTOM_CSS = """
<style>
    /* Variables de couleurs MSF */
    :root {
        --msf-red: #E43535;
        --msf-dark: #1a1a1a;
        --msf-gray: #f0f2f6;
        --msf-blue: #4A90E2;
        --msf-green: #27AE60;
    }

    /* Bannière principale */
    .main-header {
        background: linear-gradient(135deg, var(--msf-red) 0%, #c92a2a 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Cartes d'information */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--msf-red);
        margin-bottom: 1rem;
    }

    /* Métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Boutons */
    .stButton>button {
        background-color: var(--msf-red);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #c92a2a;
        box-shadow: 0 4px 8px rgba(228, 53, 53, 0.3);
    }

    /* Alertes */
    .alert-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        background-color: #f0f2f6;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--msf-red);
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }

    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }

    .dataframe th {
        background-color: var(--msf-red) !important;
        color: white !important;
        font-weight: 600;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
"""

# Configuration page Streamlit par défaut
PAGE_CONFIG = {
    "page_title": "MSF - Surveillance Épidémiologique",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
