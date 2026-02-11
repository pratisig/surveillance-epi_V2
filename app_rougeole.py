"""
============================================================
VERSION 4.0 - APPLICATION ROUGEOLE COMPLÈTE
Garde toutes les fonctionnalités + modules partagés
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import json
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile
import tempfile
import os
from shapely.geometry import shape
import warnings
import sys

warnings.filterwarnings('ignore')

# Ajouter le dossier modules au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Imports des modules partagés
try:
    from modules.ui_components import apply_msf_branding, msf_header, msf_footer
    from modules.data_loader import DataManager
    from modules.geo_loader import GeoLoader
    from modules.climate_loader import ClimateLoader
    from modules.worldpop_loader import WorldPopLoader
    from modules.utils import safe_int, safe_float, format_large_number
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    st.warning("⚠️ Modules partagés non disponibles. Fonctionnement en mode autonome.")

# ============================================================
# APPLIQUER LE BRANDING MSF
# ============================================================
if MODULES_AVAILABLE:
    apply_msf_branding()
else:
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #E4032E;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# CSS spécifique à l'app Rougeole
st.markdown("""
<style>
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        font-weight: bold;
        padding: 5px;
        border-radius: 3px;
    }
    
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 5px;
        border-radius: 3px;
    }
    
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 5px;
        border-radius: 3px;
    }
    
    .model-hint {
        background-color: #fff9c4;
        padding: 8px;
        border-radius: 5px;
        font-size: 0.9em;
        margin: 5px 0;
    }
    
    .weight-box {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
if MODULES_AVAILABLE:
    msf_header(
        "🦠 Dashboard de Surveillance et Prédiction - Rougeole",
        "Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques"
    )
else:
    st.markdown('<h1 class="main-header">🦠 Dashboard de Surveillance et Prédiction - Rougeole</h1>', unsafe_allow_html=True)
    st.markdown("**Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques**")

# ============================================================
# MAPPING PAYS (UNIFIÉ)
# ============================================================
PAYS_ISO3_MAP = {
    "Niger": "ner",
    "Burkina Faso": "bfa",
    "Mali": "mli",
    "Mauritanie": "mrt"
}

# ============================================================
# INITIALISATION DU GESTIONNAIRE DE DONNÉES
# ============================================================
if MODULES_AVAILABLE:
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    dm = st.session_state.data_manager
else:
    dm = None

# ============================================================
# INITIALISATION SESSION STATE
# ============================================================
if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def safe_int(value):
    """Convertit en int en gérant les NaN"""
    if pd.isna(value) or value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def safe_float(value, default=0.0):
    """Convertit en float en gérant les NaN"""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def normalize_colonnes(dataframe, mapping):
    """Renommer les colonnes du dataframe selon le mapping standardisé"""
    rename_dict = {}
    for col_standard, col_possibles in mapping.items():
        for col_possible in col_possibles:
            if col_possible in dataframe.columns and col_possible != col_standard:
                rename_dict[col_possible] = col_standard
                break
    if rename_dict:
        dataframe = dataframe.rename(columns=rename_dict)
    return dataframe

def generate_dummy_linelists(sa_gdf, n=500, start=None, end=None):
    """Génère des données fictives de rougeole"""
    np.random.seed(42)
    
    if start is None:
        start = datetime(2024, 1, 1)
    if end is None:
        end = datetime.today()
    
    delta_days = (end - start).days
    
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.random.exponential(scale=delta_days/3, size=n).clip(0, delta_days).astype(int),
        unit='D'
    )
    
    df = pd.DataFrame({
        'ID_Cas': range(1, n+1),
        'Date_Debut_Eruption': dates,
        'Date_Notification': dates + pd.to_timedelta(np.random.poisson(3, n), unit='D'),
        'Aire_Sante': np.random.choice(sa_gdf['health_area'].unique(), n),
        'Age_Mois': np.random.gamma(shape=2, scale=30, size=n).clip(6, 180).astype(int),
        'Statut_Vaccinal': np.random.choice(['Oui', 'Non'], n, p=[0.55, 0.45]),
        'Sexe': np.random.choice(['M', 'F'], n),
        'Issue': np.random.choice(['Guéri', 'Décédé', 'Inconnu'], n, p=[0.92, 0.03, 0.05])
    })
    
    return df

def generate_dummy_vaccination(sa_gdf):
    """Génère des données fictives de vaccination"""
    np.random.seed(42)
    return pd.DataFrame({
        'health_area': sa_gdf['health_area'],
        'Taux_Vaccination': np.random.beta(a=8, b=2, size=len(sa_gdf)) * 100
    })

# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

st.sidebar.header("📂 Configuration de l'Analyse")

# ============================================================
# Section 1 : Mode d'utilisation
# ============================================================
st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives pour tester l'application"
)

# ============================================================
# Section 2 : Aires de santé (UNIFIÉ)
# ============================================================
st.sidebar.subheader("🗺️ Aires de Santé")

sa_gdf = None

# Vérifier si des données géographiques sont déjà chargées via DataManager
if MODULES_AVAILABLE and dm and dm.has_geodata():
    gdf_info = GeoLoader.get_geodata_info(dm.get_geodata())
    st.sidebar.success(f"✅ {gdf_info['n_features']} aires chargées (réutilisées)")
    
    if st.sidebar.button("🔄 Recharger de nouvelles aires"):
        dm.clear_by_type('geodata')
        st.session_state.sa_gdf_cache = None
        st.rerun()
    
    sa_gdf = dm.get_geodata()

else:
    option_aire = st.sidebar.radio(
        "Source des données géographiques",
        ["Fichier local (ao_hlthArea.zip)", "Upload personnalisé"],
        key='option_aire_rougeole'
    )
    
    pays_selectionne = None
    iso3_pays = None
    
    # OPTION 1 : Fichier local
    if option_aire == "Fichier local (ao_hlthArea.zip)":
        pays_selectionne = st.sidebar.selectbox(
            "🌍 Sélectionner le pays",
            list(PAYS_ISO3_MAP.keys()),
            key='pays_select_rougeole'
        )
        
        iso3_pays = PAYS_ISO3_MAP[pays_selectionne]
        
        # Vérifier si changement de pays
        pays_change = st.session_state.pays_precedent != pays_selectionne
        
        if pays_change:
            st.session_state.pays_precedent = pays_selectionne
            st.session_state.sa_gdf_cache = None
        
        # Utiliser le cache si disponible
        if st.session_state.sa_gdf_cache is not None and not pays_change:
            sa_gdf = st.session_state.sa_gdf_cache
            st.sidebar.success(f"✅ {len(sa_gdf)} aires de santé chargées (cache)")
        else:
            if st.sidebar.button("📥 Charger les aires") or pays_change:
                with st.spinner(f"⏳ Chargement des aires de {pays_selectionne}..."):
                    if MODULES_AVAILABLE:
                        sa_gdf = GeoLoader.load_local_ao_hltharea(iso3_pays)
                    else:
                        # Fallback sans modules
                        try:
                            zip_path = os.path.join("data", "ao_hlthArea.zip")
                            if not os.path.exists(zip_path):
                                zip_path = "ao_hlthArea.zip"
                            
                            with tempfile.TemporaryDirectory() as tmpdir:
                                with zipfile.ZipFile(zip_path, 'r') as z:
                                    z.extractall(tmpdir)
                                shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                                if shp_files:
                                    gdf_full = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                                    
                                    # Filtrer par iso3
                                    iso3_col = None
                                    for col in ['iso3', 'ISO3', 'iso_code', 'ISOCODE']:
                                        if col in gdf_full.columns:
                                            iso3_col = col
                                            break
                                    
                                    if iso3_col:
                                        sa_gdf = gdf_full[gdf_full[iso3_col].str.lower() == iso3_pays.lower()].copy()
                                    else:
                                        sa_gdf = gdf_full
                        except Exception as e:
                            st.sidebar.error(f"❌ Erreur : {e}")
                            sa_gdf = None
                    
                    if sa_gdf is not None and not sa_gdf.empty:
                        # Normaliser health_area
                        if 'health_area' not in sa_gdf.columns:
                            for col in ['healtharea', 'HEALTHAREA', 'name_fr', 'NAME', 'nom', 'NOM', 'aire_sante']:
                                if col in sa_gdf.columns:
                                    sa_gdf['health_area'] = sa_gdf[col]
                                    break
                        
                        # Valider géométrie
                        sa_gdf = sa_gdf[sa_gdf.geometry.is_valid].copy()
                        
                        # WGS84
                        if sa_gdf.crs is None:
                            sa_gdf.set_crs('EPSG:4326', inplace=True)
                        elif sa_gdf.crs.to_epsg() != 4326:
                            sa_gdf = sa_gdf.to_crs('EPSG:4326')
                        
                        st.session_state.sa_gdf_cache = sa_gdf
                        
                        if MODULES_AVAILABLE and dm:
                            dm.set_geodata(sa_gdf, source=f"local_{iso3_pays}")
                        
                        st.sidebar.success(f"✅ {len(sa_gdf)} aires de santé chargées")
                    else:
                        st.sidebar.error(f"❌ Impossible de charger les données pour {pays_selectionne}")
    
    # OPTION 2 : Upload personnalisé
    else:
        upload_file = st.sidebar.file_uploader(
            "Charger un fichier géographique",
            type=["shp", "geojson", "zip"],
            help="Format : Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'",
            key='upload_geo_rougeole'
        )
        
        if upload_file is not None:
            if MODULES_AVAILABLE:
                sa_gdf = GeoLoader.load_from_file(upload_file)
            else:
                # Fallback sans modules
                try:
                    if upload_file.name.endswith('.geojson'):
                        sa_gdf = gpd.read_file(upload_file)
                    elif upload_file.name.endswith('.zip'):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            zip_path = os.path.join(tmpdir, 'upload.zip')
                            with open(zip_path, 'wb') as f:
                                f.write(upload_file.getvalue())
                            
                            with zipfile.ZipFile(zip_path, 'r') as z:
                                z.extractall(tmpdir)
                            
                            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                            if shp_files:
                                sa_gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                except Exception as e:
                    st.sidebar.error(f"❌ Erreur : {e}")
                    sa_gdf = None
            
            if sa_gdf is not None:
                # Normaliser
                if 'health_area' not in sa_gdf.columns:
                    for col in ['healtharea', 'HEALTHAREA', 'name_fr', 'NAME', 'nom']:
                        if col in sa_gdf.columns:
                            sa_gdf['health_area'] = sa_gdf[col]
                            break
                
                # WGS84
                if sa_gdf.crs is None:
                    sa_gdf.set_crs('EPSG:4326', inplace=True)
                elif sa_gdf.crs.to_epsg() != 4326:
                    sa_gdf = sa_gdf.to_crs('EPSG:4326')
                
                if MODULES_AVAILABLE and dm:
                    dm.set_geodata(sa_gdf, source="upload")
                
                st.sidebar.success(f"✅ {len(sa_gdf)} aires de santé chargées")
        else:
            st.sidebar.info("👆 Uploadez un fichier pour commencer")

if sa_gdf is None or sa_gdf.empty:
    st.error("❌ Aucune aire de santé chargée. Configurez dans la sidebar.")
    st.stop()

# ============================================================
# Section 3 : Données épidémiologiques
# ============================================================
st.sidebar.subheader("📊 Données Épidémiologiques")

df = None
vaccination_df = None

if mode_demo == "🧪 Mode démo (données simulées)":
    linelist_file = None
    vaccination_file = None
    st.sidebar.info("📊 Mode démo activé - Données simulées")
else:
    # Vérifier si déjà chargées
    if MODULES_AVAILABLE and dm and dm.has_epidemio_data('rougeole'):
        epidemio_info = dm.get_summary()['source_info'].get('epidemio_rougeole', {})
        st.sidebar.success(f"✅ {epidemio_info.get('n_records', 0)} cas (réutilisés)")
        
        if st.sidebar.button("🔄 Recharger les données rougeole"):
            dm.clear_by_type('epidemio')
            st.rerun()
        
        df = dm.get_epidemio_data('rougeole')
    
    else:
        linelist_file = st.sidebar.file_uploader(
            "📋 Linelists rougeole (CSV)",
            type=["csv"],
            help="Format : health_area, Semaine_Epi, Cas_Total OU Date_Debut_Eruption, Aire_Sante...",
            key='upload_cases_rougeole'
        )
    
    # Vaccination
    if MODULES_AVAILABLE and dm and dm.has_vaccination_data():
        st.sidebar.success("✅ Couverture vaccinale (réutilisée)")
        vaccination_df = dm.get_vaccination_data()
    else:
        vaccination_file = st.sidebar.file_uploader(
            "💉 Couverture vaccinale (CSV - optionnel)",
            type=["csv"],
            help="Format : health_area, Taux_Vaccination (en %)",
            key='upload_vacc_rougeole'
        )

# ============================================================
# Section 4 : Période d'analyse
# ============================================================
st.sidebar.subheader("📅 Période d'Analyse")

col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Date début",
        value=datetime(2024, 1, 1),
        key='start_date_rougeole'
    )

with col2:
    end_date = st.date_input(
        "Date fin",
        value=datetime.today(),
        key='end_date_rougeole'
    )

# ============================================================
# Section 5 : Paramètres de prédiction
# ============================================================
st.sidebar.subheader("🔮 Paramètres de Prédiction")

pred_mois = st.sidebar.slider(
    "Période de prédiction (mois)",
    min_value=1,
    max_value=12,
    value=3,
    help="Nombre de mois à prédire après la dernière semaine de données"
)

n_weeks_pred = pred_mois * 4
st.sidebar.info(f"📆 Prédiction sur **{n_weeks_pred} semaines épidémiologiques** (~{pred_mois} mois)")

# ============================================================
# Section 6 : Choix du modèle
# ============================================================
st.sidebar.subheader("🤖 Modèle de Prédiction")

modele_choisi = st.sidebar.selectbox(
    "Choisissez votre algorithme",
    [
        "GradientBoosting (Recommandé)",
        "RandomForest",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree"
    ],
    help="Sélectionnez l'algorithme de machine learning pour la prédiction"
)

# Hints pour chaque modèle
model_hints = {
    "GradientBoosting (Recommandé)": "🎯 **Gradient Boosting** : Très performant pour les séries temporelles. Combine plusieurs modèles faibles pour créer un modèle fort. Excellent pour capturer les relations non-linéaires. Recommandé pour la surveillance épidémiologique.",
    "RandomForest": "🌳 **Random Forest** : Ensemble d'arbres de décision. Robuste aux valeurs aberrantes et aux données manquantes. Bon pour les interactions complexes entre variables.",
    "Ridge Regression": "📊 **Ridge Regression** : Régression linéaire avec régularisation L2. Simple et rapide. Idéal pour relations linéaires. Moins performant sur données non-linéaires.",
    "Lasso Regression": "🎯 **Lasso Regression** : Régularisation L1 avec sélection automatique des variables. Utile quand beaucoup de variables peu importantes. Simplifie le modèle.",
    "Decision Tree": "🌲 **Decision Tree** : Arbre de décision unique. Simple à interpréter mais risque de sur-apprentissage. Moins robuste que les méthodes d'ensemble."
}

st.sidebar.markdown(f'<div class="model-hint">{model_hints[modele_choisi]}</div>', unsafe_allow_html=True)

# ============================================================
# Section 7 : Importance des variables (CONSERVÉ)
# ============================================================
st.sidebar.subheader("⚖️ Importance des Variables")

mode_importance = st.sidebar.radio(
    "Mode de pondération",
    ["Automatique (ML)", "Manuel (Expert)"],
    help="Automatique : calcul par le modèle ML | Manuel : poids définis par expertise épidémiologique"
)

poids_manuels = {}
poids_normalises = {}

if mode_importance == "Manuel (Expert)":
    with st.sidebar.expander("⚙️ Configurer les poids", expanded=True):
        st.markdown("Définissez l'importance de chaque groupe de variables")
        st.caption("Les poids seront automatiquement normalisés pour totaliser 100%")
        
        poids_manuels['Historique_Cas'] = st.slider(
            "📈 Historique des cas (lags)",
            min_value=0, max_value=100, value=40, step=5,
            help="Importance des cas passés (4 dernières semaines)"
        )
        
        poids_manuels['Vaccination'] = st.slider(
            "💉 Couverture vaccinale",
            min_value=0, max_value=100, value=35, step=5,
            help="Importance du taux de vaccination et non-vaccinés"
        )
        
        poids_manuels['Demographie'] = st.slider(
            "👥 Démographie",
            min_value=0, max_value=100, value=15, step=5,
            help="Importance de la population et densité"
        )
        
        poids_manuels['Urbanisation'] = st.slider(
            "🏙️ Urbanisation",
            min_value=0, max_value=100, value=8, step=2,
            help="Importance du type d'habitat (urbain/rural)"
        )
        
        poids_manuels['Climat'] = st.slider(
            "🌡️ Facteurs climatiques",
            min_value=0, max_value=100, value=2, step=1,
            help="Importance de la température, humidité, saison"
        )
        
        # Normaliser
        total_poids = sum(poids_manuels.values())
        if total_poids > 0:
            for key in poids_manuels:
                poids_normalises[key] = poids_manuels[key] / total_poids
        
        st.markdown("---")
        st.markdown("**Répartition normalisée:**")
        for key, value in poids_normalises.items():
            st.markdown(f"- {key}: {value*100:.1f}%")
        
        if abs(total_poids - 100) > 5:
            st.info(f"💡 Total brut: {total_poids}% → Normalisé: 100%")
else:
    st.sidebar.info("Le modèle ML calculera automatiquement l'importance optimale de chaque variable")

# ============================================================
# Section 8 : Seuils d'alerte
# ============================================================
st.sidebar.subheader("🚨 Seuils d'Alerte")

with st.sidebar.expander("⚙️ Configurer les seuils", expanded=False):
    seuil_baisse = st.slider(
        "📉 Seuil de baisse significative (%)",
        min_value=10, max_value=90, value=75, step=5,
        help="Afficher les aires avec baisse >= X% par rapport à la moyenne"
    )
    
    seuil_hausse = st.slider(
        "📈 Seuil de hausse significative (%)",
        min_value=10, max_value=200, value=50, step=10,
        help="Afficher les aires avec hausse >= X% par rapport à la moyenne"
    )
    
    seuil_alerte_epidemique = st.number_input(
        "⚠️ Seuil d'alerte épidémique (cas/semaine)",
        min_value=1, max_value=100, value=5,
        help="Nombre de cas par semaine déclenchant une alerte"
    )

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

st.markdown("---")
st.header("📥 Chargement des Données")

with st.spinner("⏳ Chargement des données de cas..."):
    
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf, n=500, start=start_date, end=end_date)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
        st.sidebar.info(f"✅ {len(df)} cas simulés générés")
    
    else:
        # Chargement des données réelles
        if not MODULES_AVAILABLE or not dm or not dm.has_epidemio_data('rougeole'):
            if linelist_file is not None:
                try:
                    df_raw = pd.read_csv(linelist_file)
                    
                    # MAPPING DES COLONNES
                    COLONNES_MAPPING = {
                        'Aire_Sante': ['Aire_Sante', 'aire_sante', 'health_area', 'HEALTHAREA', 'name_fr', 'NAME', 'nom', 'NOM'],
                        'Date_Debut_Eruption': ['Date_Debut_Eruption', 'datedebuteruption', 'DateDebut', 'dateonset', 'DateOnset', 'symptom_onset'],
                        'Date_Notification': ['Date_Notification', 'datenotification', 'DateNotif', 'datenotif', 'notification_date'],
                        'ID_Cas': ['ID_Cas', 'idcas', 'ID', 'id', 'CaseID', 'caseid'],
                        'Age_Mois': ['Age_Mois', 'agemois', 'Age', 'age', 'AGE', 'AgeMonths', 'age_months'],
                        'Statut_Vaccinal': ['Statut_Vaccinal', 'statutvaccinal', 'Vaccin', 'vaccin', 'VaccinationStatus', 'vaccination_status'],
                        'Sexe': ['Sexe', 'sexe', 'Sex', 'sex', 'Gender', 'gender'],
                        'Issue': ['Issue', 'issue', 'Outcome', 'outcome', 'OUTCOME']
                    }
                    
                    # Format agrégé ou linelist ?
                    if 'Semaine_Epi' in df_raw.columns and 'Cas_Total' in df_raw.columns:
                        # Format agrégé → désagréger
                        expanded_rows = []
                        
                        for _, row in df_raw.iterrows():
                            # Trouver colonne aire
                            aire = None
                            for col in ['health_area', 'Aire_Sante', 'name_fr', 'NAME']:
                                if col in row and not pd.isna(row.get(col)):
                                    aire = row[col]
                                    break
                            
                            semaine = int(row['Semaine_Epi'])
                            cas_total = int(row['Cas_Total'])
                            annee = row.get('Annee', 2024)
                            
                            base_date = datetime.strptime(f'{annee}-W{semaine:02d}-1', "%Y-W%W-%w")
                            
                            for i in range(cas_total):
                                expanded_rows.append({
                                    'ID_Cas': len(expanded_rows) + 1,
                                    'Date_Debut_Eruption': base_date + timedelta(days=np.random.randint(0, 7)),
                                    'Date_Notification': base_date + timedelta(days=np.random.randint(0, 10)),
                                    'Aire_Sante': aire,
                                    'Age_Mois': 0,
                                    'Statut_Vaccinal': 'Inconnu',
                                    'Sexe': 'Inconnu',
                                    'Issue': 'Inconnu'
                                })
                        
                        df = pd.DataFrame(expanded_rows)
                    
                    elif 'Date_Debut_Eruption' in df_raw.columns or any(col in df_raw.columns for col in COLONNES_MAPPING['Date_Debut_Eruption']):
                        # Format linelist
                        df = normalize_colonnes(df_raw, COLONNES_MAPPING)
                        
                        # Convertir dates
                        for col in ['Date_Debut_Eruption', 'Date_Notification']:
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    else:
                        st.error("❌ Format CSV non reconnu")
                        st.info(f"📋 Colonnes détectées : {', '.join(df_raw.columns)}")
                        st.stop()
                    
                    if MODULES_AVAILABLE and dm:
                        dm.set_epidemio_data(df, disease='rougeole')
                    
                    st.sidebar.success(f"✅ {len(df)} cas chargés")
                    
                except Exception as e:
                    st.error(f"❌ Erreur CSV : {e}")
                    st.stop()
            else:
                st.error("❌ Veuillez uploader un fichier CSV de lineliste")
                st.stop()
        
        # Vaccination
        if not MODULES_AVAILABLE or not dm or not dm.has_vaccination_data():
            if vaccination_file is not None:
                try:
                    vaccination_df = pd.read_csv(vaccination_file)
                    
                    # Normaliser
                    if 'health_area' not in vaccination_df.columns:
                        for col in ['Aire_Sante', 'aire_sante', 'name_fr']:
                            if col in vaccination_df.columns:
                                vaccination_df['health_area'] = vaccination_df[col]
                                break
                    
                    if MODULES_AVAILABLE and dm:
                        dm.set_vaccination_data(vaccination_df)
                    
                    st.sidebar.success(f"✅ Couverture vaccinale chargée ({len(vaccination_df)} aires)")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Erreur vaccination CSV : {e}")
                    vaccination_df = None
            else:
                # Extraire de la linelist si disponible
                if df is not None and 'Statut_Vaccinal' in df.columns:
                    vacc_by_area = df.groupby('Aire_Sante').agg({
                        'Statut_Vaccinal': lambda x: (x == 'Oui').sum() / len(x) * 100 if len(x) > 0 else 0
                    }).reset_index()
                    
                    vacc_by_area.columns = ['health_area', 'Taux_Vaccination']
                    vaccination_df = vacc_by_area
                    st.sidebar.info("💉 Taux vaccination extrait de la linelist")
                else:
                    vaccination_df = None
                    st.sidebar.info("ℹ️ Pas de données de vaccination")

# Normalisation finale des colonnes
if df is not None:
    if 'Aire_Sante' in df.columns and 'health_area' not in df.columns:
        df['health_area'] = df['Aire_Sante']
    
    if 'Date_Debut_Eruption' in df.columns:
        df['Date_Debut_Eruption'] = pd.to_datetime(df['Date_Debut_Eruption'], errors='coerce')
    
    # Filtrer par période
    df = df[
        (df['Date_Debut_Eruption'] >= pd.to_datetime(start_date)) &
        (df['Date_Debut_Eruption'] <= pd.to_datetime(end_date))
    ].copy()
    
    if len(df) == 0:
        st.warning("⚠️ Aucun cas dans la période sélectionnée")
        st.stop()
    
    # Ajouter semaine épidémiologique
    df['Semaine_Epi'] = df['Date_Debut_Eruption'].dt.isocalendar().week
    df['Annee'] = df['Date_Debut_Eruption'].dt.year

# ============================================================
# AFFICHAGE DES STATISTIQUES GÉNÉRALES
# ============================================================

st.markdown("---")
st.header("📊 Vue d'Ensemble")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🗺️ Aires de santé", len(sa_gdf))

with col2:
    st.metric("🦠 Cas totaux", len(df))

with col3:
    n_weeks = df['Semaine_Epi'].nunique()
    st.metric("📅 Semaines", n_weeks)

with col4:
    if vaccination_df is not None:
        couv_moy = vaccination_df['Taux_Vaccination'].mean()
        st.metric("💉 Couverture moy.", f"{couv_moy:.1f}%")
    else:
        st.metric("💉 Couverture", "N/A")

# Afficher aperçu
with st.expander("👀 Aperçu des données"):
    st.dataframe(df.head(20))


# ============================================================
# SECTION CARTOGRAPHIE INTERACTIVE
# ============================================================

st.markdown("---")
st.header("🗺️ Cartographie des Cas")

# Agrégation par aire de santé
cases_by_area = df.groupby('health_area').size().reset_index(name='cas_total')

# Fusion avec la géométrie
gdf_cases = sa_gdf.merge(cases_by_area, on='health_area', how='left')
gdf_cases['cas_total'] = gdf_cases['cas_total'].fillna(0)

# Fusion avec vaccination si disponible
if vaccination_df is not None:
    gdf_cases = gdf_cases.merge(vaccination_df[['health_area', 'Taux_Vaccination']], on='health_area', how='left')

# Créer la carte
center_lat = gdf_cases.geometry.centroid.y.mean()
center_lon = gdf_cases.geometry.centroid.x.mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles='CartoDB positron'
)

# Choroplèthe - Nombre de cas
folium.Choropleth(
    geo_data=gdf_cases,
    data=gdf_cases,
    columns=['health_area', 'cas_total'],
    key_on='feature.properties.health_area',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Nombre de cas de rougeole',
    name='Cas de rougeole'
).add_to(m)

# Ajouter des popups enrichis
for idx, row in gdf_cases.iterrows():
    popup_html = f"""
    <div style="width:300px; font-family:Arial; font-size:12px;">
        <h4 style="color:#E4032E; margin:0;">{row['health_area']}</h4>
        <hr style="margin:5px 0;">
        <table style="width:100%;">
            <tr><td><b>🦠 Cas:</b></td><td>{int(row['cas_total'])}</td></tr>
    """
    
    if 'Taux_Vaccination' in row and not pd.isna(row['Taux_Vaccination']):
        taux = row['Taux_Vaccination']
        color = '#4caf50' if taux >= 95 else ('#ff9800' if taux >= 80 else '#f44336')
        popup_html += f"<tr style='background:{color}20;'><td><b>💉 Vaccination:</b></td><td>{taux:.1f}%</td></tr>"
    
    popup_html += """
        </table>
    </div>
    """
    
    # Taille du marker proportionnelle au nombre de cas
    radius = min(5 + row['cas_total'] / 5, 20)
    
    folium.CircleMarker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=300),
        color='#E4032E',
        fill=True,
        fillColor='#E4032E',
        fillOpacity=0.6
    ).add_to(m)

# Si vaccination disponible, ajouter couche
if vaccination_df is not None:
    folium.Choropleth(
        geo_data=gdf_cases,
        data=gdf_cases,
        columns=['health_area', 'Taux_Vaccination'],
        key_on='feature.properties.health_area',
        fill_color='RdYlGn',
        fill_opacity=0.5,
        line_opacity=0.2,
        legend_name='Couverture vaccinale (%)',
        name='Couverture vaccinale',
        show=False
    ).add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, width=1200, height=600, key='rougeole_map')

# Légende vaccination
if vaccination_df is not None:
    st.markdown("""
    <div style="background:#f0f2f6; padding:1rem; border-radius:8px; margin-top:1rem;">
        <b>💉 Légende Couverture Vaccinale:</b><br>
        🟢 ≥95% : Objectif OMS atteint (immunité collective)<br>
        🟡 80-94% : Insuffisant (risque flambées localisées)<br>
        🔴 <80% : Très insuffisant (risque épidémie majeure)
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION ANALYSE TEMPORELLE ET ÉPIDÉMIOLOGIQUE
# ============================================================

st.markdown("---")
st.header("📈 Analyse Épidémiologique")

# Agrégation hebdomadaire
weekly_cases = df.groupby(['Annee', 'Semaine_Epi']).size().reset_index(name='cas')
weekly_cases['date'] = pd.to_datetime(
    weekly_cases['Annee'].astype(str) + '-W' + weekly_cases['Semaine_Epi'].astype(str).str.zfill(2) + '-1',
    format='%Y-W%W-%w'
)
weekly_cases = weekly_cases.sort_values('date')

# Calculer seuil épidémique (Moyenne + 2 SD)
mean_cases = weekly_cases['cas'].mean()
std_cases = weekly_cases['cas'].std()
seuil_epidemique_calc = mean_cases + 2 * std_cases

# Courbe épidémique
fig_epi = go.Figure()

fig_epi.add_trace(go.Scatter(
    x=weekly_cases['date'],
    y=weekly_cases['cas'],
    mode='lines+markers',
    name='Cas hebdomadaires',
    line=dict(color='#E4032E', width=2),
    marker=dict(size=6),
    fill='tozeroy',
    fillcolor='rgba(228, 3, 46, 0.1)'
))

# Ligne seuil épidémique
fig_epi.add_trace(go.Scatter(
    x=[weekly_cases['date'].min(), weekly_cases['date'].max()],
    y=[seuil_epidemique_calc, seuil_epidemique_calc],
    mode='lines',
    name=f'Seuil épidémique (M+2SD = {seuil_epidemique_calc:.0f})',
    line=dict(color='red', width=2, dash='dash')
))

# Ligne moyenne
fig_epi.add_trace(go.Scatter(
    x=[weekly_cases['date'].min(), weekly_cases['date'].max()],
    y=[mean_cases, mean_cases],
    mode='lines',
    name=f'Moyenne ({mean_cases:.0f})',
    line=dict(color='gray', width=1, dash='dot')
))

fig_epi.update_layout(
    title='Courbe Épidémique - Rougeole',
    xaxis_title='Date (Semaine épidémiologique)',
    yaxis_title='Nombre de cas',
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_epi, use_container_width=True)

# Alertes épidémiques
semaines_alerte = weekly_cases[weekly_cases['cas'] > seuil_epidemique_calc]

if len(semaines_alerte) > 0:
    st.error(f"🚨 **ALERTE ÉPIDÉMIQUE** : {len(semaines_alerte)} semaines au-dessus du seuil !")
    
    with st.expander("📋 Détails des semaines en alerte"):
        st.dataframe(semaines_alerte[['Annee', 'Semaine_Epi', 'cas', 'date']])
else:
    st.success("✅ Aucune semaine au-dessus du seuil épidémique")

# Analyse par aire de santé
st.subheader("📊 Analyse par Aire de Santé")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔝 Top 10 Aires Touchées**")
    top10 = gdf_cases.nlargest(10, 'cas_total')[['health_area', 'cas_total']]
    
    if 'Taux_Vaccination' in gdf_cases.columns:
        top10 = gdf_cases.nlargest(10, 'cas_total')[['health_area', 'cas_total', 'Taux_Vaccination']]
    
    st.dataframe(top10, use_container_width=True)

with col2:
    st.markdown("**📉 Distribution des Cas**")
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Box(
        y=gdf_cases['cas_total'],
        name='Cas par aire',
        marker_color='#E4032E'
    ))
    
    fig_dist.update_layout(
        yaxis_title='Nombre de cas',
        height=250
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

# Analyse par âge (si disponible)
if 'Age_Mois' in df.columns:
    st.subheader("👶 Distribution par Âge")
    
    df_age = df[df['Age_Mois'] > 0].copy()
    df_age['Age_Annees'] = df_age['Age_Mois'] / 12
    df_age['Groupe_Age'] = pd.cut(
        df_age['Age_Annees'],
        bins=[0, 1, 5, 10, 15, 100],
        labels=['<1 an', '1-4 ans', '5-9 ans', '10-14 ans', '≥15 ans']
    )
    
    age_stats = df_age.groupby('Groupe_Age').size().reset_index(name='cas')
    
    fig_age = px.bar(
        age_stats,
        x='Groupe_Age',
        y='cas',
        title='Distribution des Cas par Groupe d\'Âge',
        labels={'Groupe_Age': 'Groupe d\'âge', 'cas': 'Nombre de cas'},
        color='cas',
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Indicateurs enfants <5 ans
    enfants_5 = len(df_age[df_age['Age_Annees'] < 5])
    pct_enfants_5 = (enfants_5 / len(df_age)) * 100
    
    st.info(f"👶 **{enfants_5} cas ({pct_enfants_5:.1f}%)** chez les enfants < 5 ans")

# Analyse par statut vaccinal (si disponible)
if 'Statut_Vaccinal' in df.columns:
    st.subheader("💉 Analyse par Statut Vaccinal")
    
    vacc_stats = df[df['Statut_Vaccinal'].isin(['Oui', 'Non'])].groupby('Statut_Vaccinal').size().reset_index(name='cas')
    
    fig_vacc = px.pie(
        vacc_stats,
        values='cas',
        names='Statut_Vaccinal',
        title='Répartition par Statut Vaccinal',
        color='Statut_Vaccinal',
        color_discrete_map={'Oui': '#4caf50', 'Non': '#f44336'}
    )
    
    st.plotly_chart(fig_vacc, use_container_width=True)
    
    # Indicateur clé
    non_vaccines = vacc_stats[vacc_stats['Statut_Vaccinal'] == 'Non']['cas'].sum() if 'Non' in vacc_stats['Statut_Vaccinal'].values else 0
    total_connus = vacc_stats['cas'].sum()
    pct_non_vaccines = (non_vaccines / total_connus) * 100 if total_connus > 0 else 0
    
    if pct_non_vaccines > 50:
        st.error(f"⚠️ **{pct_non_vaccines:.1f}%** des cas chez les non-vaccinés → Échec vaccinal faible, problème de couverture")
    else:
        st.warning(f"💡 **{pct_non_vaccines:.1f}%** des cas chez les non-vaccinés → Possible échec vaccinal à investiguer")

# ============================================================
# SECTION MODÉLISATION PRÉDICTIVE (COMPLÈTE)
# ============================================================

st.markdown("---")
st.header("🤖 Modélisation Prédictive")

st.markdown("""
<div style="background:#f0f2f6; padding:1rem; border-radius:8px; border-left:4px solid #E4032E; margin:1rem 0;">
<b>Note:</b> La modélisation nécessite au moins 8 semaines de données. Les variables de vaccination 
et démographiques sont cruciales pour la rougeole (maladie évitable par vaccination).
</div>
""", unsafe_allow_html=True)

# Vérifier conditions minimales
if n_weeks < 8:
    st.warning(f"⚠️ Nombre de semaines insuffisant ({n_weeks}/8 minimum). Ajoutez plus de données historiques.")
    st.stop()

# Bouton pour lancer la modélisation
if st.button("🚀 LANCER LA MODÉLISATION", type="primary", use_container_width=True):
    
    with st.spinner("⏳ Préparation des données et entraînement du modèle..."):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Étape 1: Préparation des données (20%)
        status_text.text("📊 Préparation des données hebdomadaires...")
        progress_bar.progress(0.2)
        
        # Créer dataset hebdomadaire par aire
        weekly_data = []
        
        for area in sa_gdf['health_area'].unique():
            df_area = df[df['health_area'] == area]
            
            if len(df_area) == 0:
                continue
            
            # Agréger par semaine
            area_weekly = df_area.groupby(['Annee', 'Semaine_Epi']).size().reset_index(name='Cas_Observes')
            
            for _, row in area_weekly.iterrows():
                week_label = f"{int(row['Annee'])}-S{int(row['Semaine_Epi']):02d}"
                
                weekly_data.append({
                    'health_area': area,
                    'Annee': int(row['Annee']),
                    'Semaine_Epi': int(row['Semaine_Epi']),
                    'SemLabel': week_label,
                    'Cas_Observes': int(row['Cas_Observes'])
                })
        
        weekly_features = pd.DataFrame(weekly_data)
        
        # Ajouter numéro de semaine global
        weekly_features['weeknum'] = weekly_features['Semaine_Epi']
        
        # Étape 2: Création des features (30%)
        status_text.text("🔧 Création des features temporelles...")
        progress_bar.progress(0.3)
        
        # Trier par aire et semaine
        weekly_features = weekly_features.sort_values(['health_area', 'weeknum'])
        
        # Lags
        for lag in [1, 2, 4]:
            weekly_features[f'CasLag{lag}'] = weekly_features.groupby('health_area')['Cas_Observes'].shift(lag)
        
        # Moyennes mobiles
        for window in [2, 4]:
            weekly_features[f'CasMA{window}'] = weekly_features.groupby('health_area')['Cas_Observes'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Taux de croissance
        weekly_features['GrowthRate'] = weekly_features.groupby('health_area')['Cas_Observes'].pct_change().fillna(0)
        
        # Features cycliques
        weekly_features['sin_week'] = np.sin(2 * np.pi * weekly_features['weeknum'] / 52)
        weekly_features['cos_week'] = np.cos(2 * np.pi * weekly_features['weeknum'] / 52)
        
        # Étape 3: Ajouter vaccination (35%)
        if vaccination_df is not None:
            status_text.text("💉 Ajout des données de vaccination...")
            progress_bar.progress(0.35)
            
            weekly_features = weekly_features.merge(
                vaccination_df[['health_area', 'Taux_Vaccination']],
                on='health_area',
                how='left'
            )
            
            # Features dérivées vaccination
            weekly_features['Non_Vaccines_Pct'] = 100 - weekly_features['Taux_Vaccination'].fillna(80)
            weekly_features['Susceptibles'] = weekly_features['Non_Vaccines_Pct'] / 100
        
        # Étape 4: Ajouter démographie si disponible (40%)
        if MODULES_AVAILABLE and dm and dm.has_worldpop_data():
            status_text.text("👥 Ajout des données démographiques...")
            progress_bar.progress(0.4)
            
            df_worldpop = dm.get_worldpop_data()
            
            weekly_features = weekly_features.merge(
                df_worldpop[['health_area', 'Pop_Totale', 'Pop_Enfants_0_14', 'Densite_Pop']],
                on='health_area',
                how='left'
            )
        
        # Étape 5: Sélection des features (45%)
        status_text.text("🔍 Sélection des features...")
        progress_bar.progress(0.45)
        
        feature_cols = ['weeknum', 'sin_week', 'cos_week']
        
        # Lags
        for lag in [1, 2, 4]:
            if f'CasLag{lag}' in weekly_features.columns:
                feature_cols.append(f'CasLag{lag}')
        
        # Moyennes mobiles
        for window in [2, 4]:
            if f'CasMA{window}' in weekly_features.columns:
                feature_cols.append(f'CasMA{window}')
        
        # Growth rate
        if 'GrowthRate' in weekly_features.columns:
            feature_cols.append('GrowthRate')
        
        # Vaccination
        for col in ['Taux_Vaccination', 'Non_Vaccines_Pct', 'Susceptibles']:
            if col in weekly_features.columns:
                feature_cols.append(col)
        
        # Démographie
        for col in ['Pop_Totale', 'Pop_Enfants_0_14', 'Densite_Pop']:
            if col in weekly_features.columns:
                feature_cols.append(col)
        
        # Supprimer NaN
        df_model = weekly_features.dropna(subset=feature_cols + ['Cas_Observes'])
        
        st.info(f"📊 {len(df_model)} observations • {len(feature_cols)} features • {df_model['health_area'].nunique()} aires")
        
        # Afficher features
        with st.expander("📋 Features utilisées"):
            st.write(feature_cols)
        
        # Étape 6: Mapping des poids manuels (si activé) (50%)
        column_weights = {}
        
        if mode_importance == "Manuel (Expert)" and poids_normalises:
            status_text.text("⚖️ Application des poids manuels...")
            progress_bar.progress(0.5)
            
            # Mapper les features aux catégories
            for col in feature_cols:
                if 'Lag' in col or 'MA' in col or 'Growth' in col:
                    column_weights[col] = poids_normalises.get('Historique_Cas', 1.0)
                elif 'Vacc' in col or 'Suscept' in col:
                    column_weights[col] = poids_normalises.get('Vaccination', 1.0)
                elif 'Pop' in col or 'Densite' in col:
                    column_weights[col] = poids_normalises.get('Demographie', 1.0)
                elif 'sin' in col or 'cos' in col:
                    column_weights[col] = poids_normalises.get('Climat', 1.0)
                else:
                    column_weights[col] = 1.0
            
            st.markdown("**Poids appliqués:**")
            with st.expander("Voir les poids par feature"):
                for col, weight in column_weights.items():
                    st.markdown(f"- {col}: {weight:.3f}")
        
        # Étape 7: Train/Test split (55%)
        status_text.text("🔀 Split temporel des données...")
        progress_bar.progress(0.55)
        
        X = df_model[feature_cols]
        y = df_model['Cas_Observes']
        
        # Split 80/20
        split_idx = int(len(df_model) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Étape 8: Normalisation (60%)
        status_text.text("📐 Normalisation...")
        progress_bar.progress(0.6)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Appliquer poids manuels si activés
        if mode_importance == "Manuel (Expert)" and column_weights:
            for idx, col in enumerate(feature_cols):
                if col in column_weights:
                    X_train_scaled[:, idx] *= column_weights[col]
                    X_test_scaled[:, idx] *= column_weights[col]
        
        # Étape 9: Entraînement (70%)
        status_text.text("🧠 Entraînement du modèle...")
        progress_bar.progress(0.7)
        
        if modele_choisi == "GradientBoosting (Recommandé)":
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
        elif modele_choisi == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif modele_choisi == "Ridge Regression":
            model = Ridge(alpha=1.0)
        elif modele_choisi == "Lasso Regression":
            model = Lasso(alpha=0.1)
        else:
            model = DecisionTreeRegressor(max_depth=10, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        
        # Étape 10: Validation croisée (80%)
        status_text.text("✅ Validation croisée...")
        progress_bar.progress(0.8)
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Étape 11: Évaluation (85%)
        status_text.text("📊 Calcul des métriques...")
        progress_bar.progress(0.85)
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Étape 12: Prédictions futures (90%)
        status_text.text("🔮 Génération des prédictions futures...")
        progress_bar.progress(0.9)
        
        last_week = df_model['weeknum'].max()
        future_predictions = []
        
        for area in df_model['health_area'].unique():
            df_area = df_model[df_model['health_area'] == area].tail(10)
            
            if len(df_area) == 0:
                continue
            
            # Features constantes
            constant_features = {}
            for col in ['Taux_Vaccination', 'Non_Vaccines_Pct', 'Susceptibles', 'Pop_Totale', 'Pop_Enfants_0_14', 'Densite_Pop']:
                if col in df_area.columns and col in feature_cols:
                    constant_features[col] = df_area[col].mean()
            
            # Prédictions itératives
            prev_predictions = list(df_area['Cas_Observes'].tail(4))
            
            for i in range(1, n_weeks_pred + 1):
                future_week = last_week + i
                
                future_row = {
                    'weeknum': future_week,
                    'sin_week': np.sin(2 * np.pi * (future_week / 52)),
                    'cos_week': np.cos(2 * np.pi * (future_week / 52))
                }
                
                # Lags
                future_row['CasLag1'] = prev_predictions[-1] if len(prev_predictions) > 0 else df_area['Cas_Observes'].mean()
                future_row['CasLag2'] = prev_predictions[-2] if len(prev_predictions) > 1 else df_area['Cas_Observes'].mean()
                future_row['CasLag4'] = prev_predictions[-4] if len(prev_predictions) > 3 else df_area['Cas_Observes'].mean()
                
                # MA
                future_row['CasMA2'] = np.mean(prev_predictions[-2:]) if len(prev_predictions) > 1 else df_area['Cas_Observes'].mean()
                future_row['CasMA4'] = np.mean(prev_predictions[-4:]) if len(prev_predictions) > 3 else df_area['Cas_Observes'].mean()
                
                # Growth
                if len(prev_predictions) > 1:
                    future_row['GrowthRate'] = (prev_predictions[-1] - prev_predictions[-2]) / (prev_predictions[-2] + 1)
                else:
                    future_row['GrowthRate'] = 0
                
                # Constantes
                for col, val in constant_features.items():
                    future_row[col] = val
                
                # Prédire
                X_future = pd.DataFrame([future_row])[feature_cols]
                X_future_scaled = scaler.transform(X_future)
                
                # Appliquer poids manuels
                if mode_importance == "Manuel (Expert)" and column_weights:
                    for idx, col in enumerate(feature_cols):
                        if col in column_weights:
                            X_future_scaled[:, idx] *= column_weights[col]
                
                pred_cases = max(0, model.predict(X_future_scaled)[0])
                
                prev_predictions.append(pred_cases)
                
                future_predictions.append({
                    'health_area': area,
                    'SemainePic': f"S{int(future_week)}",
                    'PredictedCases': pred_cases
                })
        
        future_df = pd.DataFrame(future_predictions)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Modélisation terminée !")
        
        # AFFICHAGE DES RÉSULTATS
        st.success("✅ Modélisation terminée !")
        
        # Métriques
        st.subheader("📊 Performance du Modèle")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("R² Train", f"{r2_train:.3f}")
        
        with col2:
            st.metric("R² Test", f"{r2_test:.3f}")
        
        with col3:
            st.metric("R² CV", f"{cv_mean:.3f}")
        
        with col4:
            st.metric("MAE", f"{mae_test:.1f}")
        
        with col5:
            st.metric("RMSE", f"{rmse_test:.1f}")
        
        # Interprétation
        if r2_test > 0.75:
            st.success("🎯 Excellent modèle ! R² > 0.75")
        elif r2_test > 0.55:
            st.info("👍 Bon modèle. R² > 0.55")
        else:
            st.warning("⚠️ Modèle à améliorer. Ajoutez plus de données ou features.")
        
        # Graphique prédictions vs observations
        st.subheader("📈 Prédictions vs Observations")
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=y_test.values,
            y=y_pred_test,
            mode='markers',
            name='Prédictions',
            marker=dict(color='#E4032E', size=8, opacity=0.6)
        ))
        
        max_val = max(y_test.max(), y_pred_test.max())
        fig_pred.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Prédiction parfaite',
            line=dict(color='gray', dash='dash')
        ))
        
        fig_pred.update_layout(
            xaxis_title='Cas observés',
            yaxis_title='Cas prédits',
            height=400
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Importance des variables (si disponible)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Importance des Variables")
            
            feature_importance = pd.DataFrame({
                'Variable': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Variable',
                orientation='h',
                title='Top 10 Variables'
            )
            
            fig_imp.update_traces(marker_color='#E4032E')
            
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # Prédictions futures
        st.subheader(f"🔮 Prédictions ({n_weeks_pred} semaines)")
        
        # Analyse des risques
        moyenne_historique = weekly_features.groupby('health_area')['Cas_Observes'].mean().reset_index()
        moyenne_historique.columns = ['health_area', 'MoyenneHistorique']
        
        risk_df = future_df.groupby('health_area').agg({
            'PredictedCases': ['sum', 'max', 'mean'],
            'SemainePic': lambda x: future_df.loc[x.idxmax(), 'SemainePic'] if len(x) > 0 else 'NA'
        }).reset_index()
        
        risk_df.columns = ['health_area', 'CasPreditsTotal', 'CasPreditsMax', 'CasPreditsMoyen', 'SemainePic']
        
        risk_df = risk_df.merge(moyenne_historique, on='health_area', how='left')
        risk_df['VariationPct'] = ((risk_df['CasPreditsMoyen'] - risk_df['MoyenneHistorique']) / risk_df['MoyenneHistorique'].replace(0, 1)) * 100
        
        risk_df['CategorieVariation'] = pd.cut(
            risk_df['VariationPct'],
            bins=[-np.inf, -seuil_baisse, -10, 10, seuil_hausse, np.inf],
            labels=['Forte baisse', 'Baisse modérée', 'Stable', 'Hausse modérée', 'Forte hausse']
        )
        
        # Tableau de synthèse avec alertes
        tab1, tab2, tab3 = st.tabs(["🚨 Alertes Hausse", "📉 Baisses", "📋 Tableau Complet"])
        
        with tab1:
            st.subheader(f"Aires avec Hausse Significative (≥{seuil_hausse}%)")
            
            hausse_df = risk_df[risk_df['VariationPct'] >= seuil_hausse].copy()
            
            if len(hausse_df) > 0:
                def highlight_hausse(row):
                    return ['background-color: #ffebee'] * len(row)
                
                st.dataframe(
                    hausse_df[['health_area', 'MoyenneHistorique', 'CasPreditsMoyen', 'VariationPct', 'SemainePic', 'CasPreditsMax']]
                    .style.apply(highlight_hausse, axis=1)
                    .format({
                        'MoyenneHistorique': '{:.1f}',
                        'CasPreditsMoyen': '{:.1f}',
                        'VariationPct': '{:.1f}%',
                        'CasPreditsMax': '{:.0f}'
                    }),
                    use_container_width=True
                )
                
                st.warning(f"⚠️ **{len(hausse_df)} aires** nécessitent une vigilance accrue")
            else:
                st.success("✅ Aucune aire avec hausse significative")
        
        with tab2:
            st.subheader(f"Aires avec Baisse Significative (≥{seuil_baisse}%)")
            
            baisse_df = risk_df[risk_df['VariationPct'] <= -seuil_baisse].copy()
            
            if len(baisse_df) > 0:
                def highlight_baisse(row):
                    return ['background-color: #e8f5e9'] * len(row)
                
                st.dataframe(
                    baisse_df[['health_area', 'MoyenneHistorique', 'CasPreditsMoyen', 'VariationPct', 'SemainePic', 'CasPreditsMax']]
                    .style.apply(highlight_baisse, axis=1)
                    .format({
                        'MoyenneHistorique': '{:.1f}',
                        'CasPreditsMoyen': '{:.1f}',
                        'VariationPct': '{:.1f}%',
                        'CasPreditsMax': '{:.0f}'
                    }),
                    use_container_width=True
                )
                
                st.success(f"✅ {len(baisse_df)} aires montrent une amélioration")
            else:
                st.info("ℹ️ Aucune aire avec baisse significative")
        
        with tab3:
            st.subheader("Tableau Complet des Prédictions")
            st.dataframe(risk_df.sort_values('CasPreditsTotal', ascending=False), use_container_width=True)
        
        # Top 10 aires à risque
        st.subheader("🎯 Top 10 Aires à Risque (Prédictions)")
        
        top_pred = risk_df.nlargest(10, 'CasPreditsTotal')
        
        fig_top = px.bar(
            top_pred,
            x='CasPreditsTotal',
            y='health_area',
            orientation='h',
            title=f'Cas Prédits Totaux ({n_weeks_pred} semaines)',
            labels={'CasPreditsTotal': 'Cas prédits', 'health_area': 'Aire de santé'}
        )
        
        fig_top.update_traces(marker_color='#E4032E')
        
        st.plotly_chart(fig_top, use_container_width=True)
        
        # EXPORTS
        st.subheader("💾 Téléchargements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = future_df.to_csv(index=False)
            st.download_button(
                label="📥 Prédictions (CSV)",
                data=csv,
                file_name=f"predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            gdf_pred = sa_gdf.merge(
                risk_df[['health_area', 'CasPreditsTotal']],
                on='health_area',
                how='left'
            )
            
            geojson_str = gdf_pred.to_json()
            st.download_button(
                label="🗺️ Carte (GeoJSON)",
                data=geojson_str,
                file_name=f"carte_predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Résumé
                summary_df = pd.DataFrame({
                    'Métrique': ['R² Train', 'R² Test', 'R² CV', 'MAE', 'RMSE', 'Nb Features', 'Nb Observations'],
                    'Valeur': [r2_train, r2_test, cv_mean, mae_test, rmse_test, len(feature_cols), len(df_model)]
                })
                summary_df.to_excel(writer, sheet_name='Résumé', index=False)
                
                # Prédictions
                risk_df.to_excel(writer, sheet_name='Synthèse', index=False)
                future_df.to_excel(writer, sheet_name='Détail Semaines', index=False)
                
                # Historique
                cases_by_area.to_excel(writer, sheet_name='Cas Observés', index=False)
                
                # Stats âge si disponible
                if 'Age_Mois' in df.columns:
                    df_age.groupby('Groupe_Age').size().reset_index(name='cas').to_excel(writer, sheet_name='Analyse Âge', index=False)
                
                # Historique hebdo
                weekly_cases.to_excel(writer, sheet_name='Historique Hebdo', index=False)
            
            st.download_button(
                label="📊 Rapport Complet (Excel)",
                data=output.getvalue(),
                file_name=f"rapport_rougeole_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Recommandations
        st.subheader("💡 Recommandations Opérationnelles")
        
        aires_critiques_hausse = risk_df[risk_df['VariationPct'] >= seuil_hausse]['health_area'].tolist()
        aires_amelioration = risk_df[risk_df['VariationPct'] <= -seuil_baisse]['health_area'].tolist()
        
        if aires_critiques_hausse:
            st.error(f"🚨 **{len(aires_critiques_hausse)} aires à risque CRITIQUE** (hausse ≥{seuil_hausse}%)")
            
            for i, aire in enumerate(aires_critiques_hausse[:5], 1):
                st.markdown(f"{i}. **{aire}** → Intensifier surveillance + Envisager CVR")
        
        if aires_amelioration:
            st.success(f"✅ **{len(aires_amelioration)} aires** en amélioration (baisse ≥{seuil_baisse}%)")
            st.info("💡 Analyser les facteurs de succès pour répliquer dans autres zones")
        
        # Alertes vaccination
        if vaccination_df is not None:
            aires_faible_couv = vaccination_df[vaccination_df['Taux_Vaccination'] < 80]['health_area'].tolist()
            
            if aires_faible_couv:
                st.warning(f"⚠️ **{len(aires_faible_couv)} aires** avec couverture vaccinale <80%")
                st.markdown("**Action recommandée:** Campagne de rattrapage vaccinal (AVS)")

# ============================================================
# FOOTER MSF
# ============================================================

st.markdown("---")

if MODULES_AVAILABLE:
    msf_footer()
else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; background-color:#f8f9fa; border-radius:8px;">
        <p style="color:#58595B; margin:0;">Développé par <b>Youssoupha MBODJI</b></p>
        <p style="color:#58595B; margin:0;">© 2026 - Médecins Sans Frontières</p>
    </div>
    """, unsafe_allow_html=True)

