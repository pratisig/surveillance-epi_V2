"""
============================================================
APPLICATION PRINCIPALE - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Développée pour Médecins Sans Frontières (MSF)
Version 4.0 - Architecture modulaire
============================================================
"""

import streamlit as st
import os
import sys

# Ajouter le dossier modules au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Imports des modules partagés
from modules.ui_components import apply_msf_branding, msf_header, msf_footer
from modules.data_loader import DataManager

# Configuration de la page (DOIT être la première commande Streamlit)
st.set_page_config(
    page_title="MSF - Surveillance Épidémiologique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer le branding MSF
apply_msf_branding()

# ============================================================
# INITIALISATION DU GESTIONNAIRE DE DONNÉES
# ============================================================
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

dm = st.session_state.data_manager

# ============================================================
# INITIALISATION DE L'ÉTAT
# ============================================================
if 'page_choice' not in st.session_state:
    st.session_state.page_choice = "Accueil"

# Navigation dans la sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    page = st.selectbox(
        "Choisir une application",
        ["Accueil", "Paludisme", "Rougeole", "Manuel"],
        index=["Accueil", "Paludisme", "Rougeole", "Manuel"].index(st.session_state.page_choice)
    )
    
    if page != st.session_state.page_choice:
        st.session_state.page_choice = page
        st.rerun()
    
    # Afficher le statut des données chargées
    st.markdown("---")
    st.markdown("### 📊 Données Chargées")
    
    summary = dm.get_summary()
    status_icons = {True: "✅", False: "⬜"}
    
    st.markdown(f"{status_icons[summary['geodata']]} Aires de santé")
    st.markdown(f"{status_icons[summary['climate']]} Données climatiques")
    st.markdown(f"{status_icons[summary['worldpop']]} WorldPop (GEE)")
    st.markdown(f"{status_icons[summary['epidemio_paludisme']]} Linelists Paludisme")
    st.markdown(f"{status_icons[summary['epidemio_rougeole']]} Linelists Rougeole")
    st.markdown(f"{status_icons[summary['vaccination']]} Couverture vaccinale")
    
    if summary['last_update']:
        st.caption(f"⏱️ MAJ : {summary['last_update'].strftime('%H:%M:%S')}")
    
    # Bouton pour réinitialiser toutes les données
    st.markdown("---")
    if st.button("🔄 Réinitialiser toutes les données", help="Efface toutes les données chargées"):
        dm.clear_all()
        st.success("✅ Données réinitialisées")
        st.rerun()

# ============================================================
# FONCTION POUR CHARGER LES APPLICATIONS
# ============================================================
def load_app(filename):
    """Charge et exécute une application Python"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
                # Supprimer st.set_page_config s'il existe
                lines = code.split('\n')
                cleaned_lines = []
                skip_next = 0
                for i, line in enumerate(lines):
                    if skip_next > 0:
                        skip_next -= 1
                        if ')' in line:
                            skip_next = 0
                        continue
                    if 'st.set_page_config' in line:
                        if ')' not in line:
                            skip_next = 10
                        continue
                    cleaned_lines.append(line)
                
                cleaned_code = '\n'.join(cleaned_lines)
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

else:  # Page d'accueil
    
    msf_header(
        "Plateforme de Surveillance Épidémiologique",
        "Outils d'analyse, cartographie et prédiction pour le paludisme et la rougeole"
    )
    
    st.markdown("""
    <div style="text-align:center; margin:1.5rem 0;">
        <h2 style="color:#E4032E; font-size:1.8rem;">Choisissez votre module d'analyse</h2>
        <p style="font-size:1.1rem; color:#58595B;">
            Cliquez sur les boutons ci-dessous ou utilisez le menu dans la barre latérale
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🦟 Paludisme</h3>
            <h4 style="color:#58595B; font-weight:normal; margin-top:0.3rem;">Outil d'analyse et de prédiction avancée</h4>
            <p>
                Cette application combine cartographie interactive, données environnementales et climatiques 
                pour identifier les zones à risque de transmission du paludisme.
            </p>
            <p><strong>Fonctionnalités clés :</strong></p>
            <ul style="list-style:none; padding-left:0; line-height:1.7; color:#555;">
                <li>• <strong>Cartographie dynamique</strong> : Répartition spatiale des cas avec popups enrichis</li>
                <li>• <strong>Données démographiques</strong> : Intégration WorldPop pour taux d'incidence précis</li>
                <li>• <strong>Analyse climatique</strong> : NASA POWER API (température, précipitations, humidité)</li>
                <li>• <strong>Environnement</strong> : Zones inondables, altitude, distance aux cours d'eau</li>
                <li>• <strong>Prédiction ML</strong> : Modèles avec validation croisée temporelle (2-12 mois)</li>
                <li>• <strong>Clustering géographique</strong> : Identification des zones homogènes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🦟 Accéder à l'application Paludisme", key="btn_palu_home"):
            st.session_state.page_choice = "Paludisme"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🦠 Rougeole</h3>
            <h4 style="color:#58595B; font-weight:normal; margin-top:0.3rem;">Surveillance et prédiction par semaines épidémiologiques</h4>
            <p>
                Application spécialisée dans l'analyse des épidémies de rougeole avec suivi temporel précis 
                et évaluation des couvertures vaccinales.
            </p>
            <p><strong>Fonctionnalités clés :</strong></p>
            <ul style="list-style:none; padding-left:0; line-height:1.7; color:#555;">
                <li>• <strong>Suivi hebdomadaire</strong> : Analyse par semaines épidémiologiques</li>
                <li>• <strong>Couverture vaccinale</strong> : Identification des poches de susceptibilité</li>
                <li>• <strong>Données démographiques</strong> : Population par tranches d'âge via WorldPop</li>
                <li>• <strong>Prédiction avancée</strong> : Gradient Boosting et Random Forest optimisés</li>
                <li>• <strong>Alertes précoces</strong> : Seuils épidémiques automatiques</li>
                <li>• <strong>Multi-pays</strong> : Niger, Burkina Faso, Mali, Mauritanie</li>
                <li>• <strong>Pyramide des âges</strong> : Visualisation de la structure démographique</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🦠 Accéder à l'application Rougeole", key="btn_rougeole_home"):
            st.session_state.page_choice = "Rougeole"
            st.rerun()
    
    st.markdown("""
    <div style="background:#F8F9FA; padding:2rem; border-radius:12px; margin:2rem 0; border-left:4px solid #E4032E;">
        <h2 style="color:#E4032E; text-align:center; margin-bottom:1rem;">📚 Documentation et Ressources</h2>
        <p style="text-align:center; font-size:1rem; color:#58595B;">
            Guides complets, méthodologies et bonnes pratiques
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📖 Manuel d'utilisation</h3>
            <p>Guide détaillé pas-à-pas pour utiliser chaque module, interpréter les résultats et optimiser vos analyses.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📖 Consulter le manuel", key="btn_manuel_home"):
            st.session_state.page_choice = "Manuel"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Méthodologie</h3>
            <p>Explication des algorithmes de machine learning, validation croisée temporelle et feature engineering.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔬 Voir la méthodologie", key="btn_methodo_home"):
            st.session_state.page_choice = "Manuel"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>💡 Glossaire</h3>
            <p>Définitions des variables (lags, moyennes mobiles, ACP, clustering spatial, etc.) et concepts clés.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("💡 Accéder au glossaire", key="btn_glossaire_home"):
            st.session_state.page_choice = "Manuel"
            st.rerun()
    
    st.markdown("""
    <div style="text-align:center; margin:2.5rem 0 1.5rem 0;">
        <h2 style="color:#E4032E; font-size:1.8rem;">⚙️ Caractéristiques Techniques</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🗺️ Cartographie</h3>
            <ul style="list-style:none; padding:0; color:#58595B;">
                <li>✓ Folium interactif</li>
                <li>✓ Popups enrichis</li>
                <li>✓ Couches multiples</li>
                <li>✓ Export GeoJSON</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Machine Learning</h3>
            <ul style="list-style:none; padding:0; color:#58595B;">
                <li>✓ Gradient Boosting</li>
                <li>✓ Random Forest</li>
                <li>✓ Validation temporelle</li>
                <li>✓ R² > 0.80 typique</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Sources Données</h3>
            <ul style="list-style:none; padding:0; color:#58595B;">
                <li>✓ NASA POWER API</li>
                <li>✓ WorldPop (GEE)</li>
                <li>✓ Rasters environnement</li>
                <li>✓ Linelists épidémio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Nouveauté : Section Architecture Modulaire
    st.markdown("""
    <div style="background:#E8F5E9; padding:1.5rem; border-radius:10px; margin:2rem 0; border-left:4px solid #4CAF50;">
        <h3 style="color:#2E7D32; margin-top:0;">✨ Nouvelle Architecture v4.0</h3>
        <p style="color:#1B5E20;">
            <strong>Données partagées intelligemment :</strong> Chargez vos données géographiques, 
            climatiques et démographiques une seule fois, puis utilisez-les librement dans les 
            applications Paludisme ET Rougeole sans rechargement !
        </p>
        <ul style="color:#1B5E20; line-height:1.8;">
            <li>🔄 <strong>Cache intelligent</strong> : Les données NASA POWER et WorldPop sont mises en cache</li>
            <li>💾 <strong>Économie de temps</strong> : Plus besoin de re-télécharger les mêmes données</li>
            <li>🎯 <strong>Cohérence garantie</strong> : Même source de données pour toutes les analyses</li>
            <li>📊 <strong>Statut en temps réel</strong> : Voir les données chargées dans la barre latérale</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    msf_footer()
