"""
============================================================
APPLICATION PRINCIPALE - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Version 4.0 - Architecture modulaire
Développée pour Médecins Sans Frontières (MSF)
============================================================
"""

import streamlit as st
import os
from config import CUSTOM_CSS, PAGE_CONFIG

# Configuration de la page (DOIT être la première commande Streamlit)
st.set_page_config(**PAGE_CONFIG)

# Application du CSS personnalisé
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# INITIALISATION DE L'ÉTAT
# ============================================================
if 'page_choice' not in st.session_state:
    st.session_state.page_choice = "Accueil"

# ============================================================
# NAVIGATION SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="main-header"><h2>🏥 MSF</h2><p>Surveillance Épidémiologique</p></div>', 
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧭 Navigation")

    page = st.selectbox(
        "Choisir une application",
        ["Accueil", "Paludisme", "Rougeole", "Manuel"],
        index=["Accueil", "Paludisme", "Rougeole", "Manuel"].index(st.session_state.page_choice)
    )

    if page != st.session_state.page_choice:
        st.session_state.page_choice = page
        st.rerun()

    st.markdown("---")

    # Informations dans la sidebar
    with st.expander("ℹ️ À propos"):
        st.markdown("""
        **Plateforme de Surveillance Épidémiologique**

        Version 4.0 - Architecture modulaire

        Développée par **Youssoupha MBODJI**

        © 2026 MSF - Open Source
        """)

# ============================================================
# FONCTION POUR CHARGER LES APPLICATIONS
# ============================================================
def load_app(filename):
    """Charge et exécute une application Python"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()

            # Supprimer st.set_page_config et imports de config s'ils existent
            lines = code.split('\n')
            cleaned_lines = []
            skip_next = 0

            for i, line in enumerate(lines):
                if skip_next > 0:
                    skip_next -= 1
                    if ')' in line:
                        skip_next = 0
                    continue

                # Skip st.set_page_config
                if 'st.set_page_config' in line:
                    if ')' not in line:
                        skip_next = 10  # Max 10 lignes
                    continue

                # Skip duplicate CSS if already applied
                if 'st.markdown("""<style>' in line or 'st.markdown(\'\'\'<style>' in line:
                    # Skip jusqu'à </style>
                    skip_next = 50
                    continue

                if skip_next == 0:
                    cleaned_lines.append(line)

            cleaned_code = '\n'.join(cleaned_lines)

            # Injection du CSS dans le contexte global
            globals()['CUSTOM_CSS'] = CUSTOM_CSS

            exec(cleaned_code, globals())
        else:
            st.error(f"❌ Fichier '{filename}' introuvable")
            st.warning(f"Assurez-vous que '{filename}' est dans le même dossier que main_app.py")

            if st.button("🏠 Retour à l'accueil"):
                st.session_state.page_choice = "Accueil"
                st.rerun()

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de {filename}")
        st.code(str(e))

        with st.expander("📋 Détails de l'erreur"):
            import traceback
            st.code(traceback.format_exc())

        if st.button("🏠 Retour à l'accueil"):
            st.session_state.page_choice = "Accueil"
            st.rerun()

# ============================================================
# ROUTAGE
# ============================================================
if st.session_state.page_choice == "Paludisme":
    load_app("app_paludisme.py")

elif st.session_state.page_choice == "Rougeole":
    load_app("app_rougeole.py")

elif st.session_state.page_choice == "Manuel":
    load_app("app_manuel.py")

else:
    # ========================================================
    # PAGE D'ACCUEIL
    # ========================================================

    st.markdown('<div class="main-header"><h1>🏥 MSF - Surveillance Épidémiologique</h1><p>Plateforme intégrée de surveillance et prédiction des maladies infectieuses</p></div>', 
                unsafe_allow_html=True)

    st.markdown("---")

    # Message de bienvenue
    st.markdown("""
    ### 👋 Bienvenue sur la plateforme MSF de surveillance épidémiologique

    Cette plateforme combine des outils avancés d'analyse spatiale, de modélisation prédictive 
    et de visualisation interactive pour la surveillance du **paludisme** et de la **rougeole**.
    """)

    st.info("🧭 **Navigation :** Utilisez le menu dans la barre latérale ou cliquez sur les boutons ci-dessous")

    st.markdown("---")

    # Applications disponibles
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🦟 Paludisme")
        st.markdown("""
        Application spécialisée dans l'analyse spatiotemporelle du paludisme avec intégration 
        de données climatiques et environnementales.

        **Fonctionnalités clés :**
        - 📍 Cartographie interactive des cas
        - 🌦️ Intégration données climatiques (NASA POWER)
        - 👥 Données démographiques (WorldPop)
        - 🤖 Prédiction par machine learning
        - 📊 Analyses de corrélation environnementales
        - 🎯 Identification zones à risque
        """)

        if st.button("🦟 Ouvrir Paludisme", key="btn_palu", use_container_width=True):
            st.session_state.page_choice = "Paludisme"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🦠 Rougeole")
        st.markdown("""
        Application spécialisée dans la surveillance de la rougeole avec analyse de couverture 
        vaccinale et détection de clusters épidémiques.

        **Fonctionnalités clés :**
        - 📍 Cartographie des cas et clusters
        - 💉 Analyse couverture vaccinale
        - 🎯 Détection seuils épidémiques OMS
        - 📈 Prédiction temporelle avancée
        - 👶 Analyse populations vulnérables
        - 🚨 Alertes précoces automatisées
        """)

        if st.button("🦠 Ouvrir Rougeole", key="btn_roug", use_container_width=True):
            st.session_state.page_choice = "Rougeole"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📚 Manuel d'utilisation")
        st.markdown("""
        Documentation complète avec guides pas-à-pas, méthodologies et bonnes pratiques.

        **Contenu :**
        - 📖 Guides détaillés pas-à-pas
        - 🔬 Méthodologies scientifiques
        - 📊 Interprétation des résultats
        - 🧮 Glossaire des variables
        - 💡 Conseils d'optimisation
        - ❓ FAQ et support technique
        """)

        if st.button("📚 Consulter le Manuel", key="btn_manuel", use_container_width=True):
            st.session_state.page_choice = "Manuel"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Caractéristiques techniques
    st.markdown("### 🔧 Caractéristiques techniques")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        **📊 Sources de données :**
        - 🗺️ Données géographiques (Shapefiles, GeoJSON)
        - 🌦️ Climat : NASA POWER API (gratuit, global)
        - 👥 Démographie : WorldPop via Google Earth Engine
        - 🌊 Données environnementales (inondations, rivières)
        """)

    with col_t2:
        st.markdown("""
        **🤖 Technologies :**
        - 🐍 Python 3.9+
        - 📊 Streamlit (interface web)
        - 🗺️ GeoPandas, Folium (cartographie)
        - 🧠 Scikit-learn (machine learning)
        - 🌍 Google Earth Engine (données satellites)
        """)

    st.markdown("---")

    # Architecture modulaire (nouvelle section)
    st.markdown("### 🏗️ Architecture Version 4.0")

    st.success("""
    ✨ **Nouveauté** : Architecture modulaire avec données partagées !

    Les données géographiques, climatiques et démographiques sont maintenant centralisées 
    et utilisables par les deux applications sans duplication.
    """)

    with st.expander("📂 Structure des modules"):
        st.code("""
📁 Plateforme MSF
├── 📄 main_app.py          # Application principale
├── 📄 config.py            # Configuration et CSS
├── 📄 data_loader.py       # Chargement données centralisé
├── 📄 utils.py             # Fonctions utilitaires
├── 📄 app_paludisme.py     # Module Paludisme
├── 📄 app_rougeole.py      # Module Rougeole
├── 📄 app_manuel.py        # Documentation
└── 📁 data/
    └── ao_hlthArea.zip     # Aires de santé (Afrique de l'Ouest)
        """, language="text")

    st.markdown("---")

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("""
    ---

    **📧 Contact Support Technique**

    Email : youssoupha.mbodji@example.com

    **Version 4.0** | Développé par **Youssoupha MBODJI** | © 2026 MSF

    Licence Open Source MIT | Python • Streamlit • GeoPandas • Scikit-learn
    """)
    st.markdown('</div>', unsafe_allow_html=True)
