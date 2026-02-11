"""
============================================================
APPLICATION ROUGEOLE - VERSION 4.0 MODULAIRE
Utilise les modules partagés pour le chargement des données
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
from modules.ui_components import apply_msf_branding, msf_header, msf_footer
from modules.data_loader import DataManager
from modules.geo_loader import GeoLoader
from modules.climate_loader import ClimateLoader
from modules.worldpop_loader import WorldPopLoader
from modules.utils import safe_int, safe_float, format_large_number

# ============================================================
# APPLIQUER LE BRANDING MSF
# ============================================================
apply_msf_branding()

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
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
msf_header(
    "🦠 Dashboard de Surveillance et Prédiction - Rougeole",
    "Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques"
)

# ============================================================
# MAPPING PAYS
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
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

dm = st.session_state.data_manager

# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

st.sidebar.header("📂 Configuration de l'Analyse")

# Mode démo
st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives pour tester l'application"
)

# Aires de santé
st.sidebar.subheader("🗺️ Aires de Santé")

if dm.has_geodata():
    gdf_info = GeoLoader.get_geodata_info(dm.get_geodata())
    st.sidebar.success(f"✅ {gdf_info['n_features']} aires chargées (réutilisées)")
    
    if st.sidebar.button("🔄 Recharger de nouvelles aires"):
        dm.clear_by_type('geodata')
        st.rerun()
    
    sa_gdf = dm.get_geodata()

else:
    option_aire = st.sidebar.radio(
        "Source des données géographiques",
        ["Fichier local (ao_hlthArea.zip)", "Upload personnalisé"],
        key='option_aire'
    )
    
    pays_selectionne = None
    iso3_pays = None
    
    if option_aire == "Fichier local (ao_hlthArea.zip)":
        pays_selectionne = st.sidebar.selectbox(
            "🌍 Sélectionner le pays",
            list(PAYS_ISO3_MAP.keys()),
            key='pays_select'
        )
        
        iso3_pays = PAYS_ISO3_MAP[pays_selectionne]
        
        with st.spinner(f"⏳ Chargement des aires de {pays_selectionne}..."):
            sa_gdf = GeoLoader.load_local_ao_hltharea(iso3_pays)
            
            if sa_gdf is not None and not sa_gdf.empty:
                dm.set_geodata(sa_gdf, source=f"local_{iso3_pays}")
                st.sidebar.success(f"✅ {len(sa_gdf)} aires de santé chargées")
            else:
                st.sidebar.error(f"❌ Impossible de charger les données pour {pays_selectionne}")
                sa_gdf = None
    
    else:
        upload_file = st.sidebar.file_uploader(
            "Charger un fichier géographique",
            type=["shp", "geojson", "zip"],
            help="Format : Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'"
        )
        
        if upload_file is not None:
            sa_gdf = GeoLoader.load_from_file(upload_file)
            
            if sa_gdf is not None:
                dm.set_geodata(sa_gdf, source="upload")
                st.sidebar.success(f"✅ {len(sa_gdf)} aires de santé chargées")
            else:
                sa_gdf = None
        else:
            sa_gdf = None
            st.sidebar.info("👆 Uploadez un fichier pour commencer")

# Données épidémiologiques
st.sidebar.subheader("📊 Données Épidémiologiques")

if mode_demo == "🧪 Mode démo (données simulées)":
    linelist_file = None
    vaccination_file = None
    st.sidebar.info("📊 Mode démo activé - Données simulées")
else:
    if dm.has_epidemio_data('rougeole'):
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
            help="Format : health_area, Semaine_Epi, Cas_Total OU Date_Debut_Eruption, Aire_Sante..."
        )
    
    if dm.has_vaccination_data():
        st.sidebar.success("✅ Couverture vaccinale (réutilisée)")
        vaccination_df = dm.get_vaccination_data()
    else:
        vaccination_file = st.sidebar.file_uploader(
            "💉 Couverture vaccinale (CSV - optionnel)",
            type=["csv"],
            help="Format : health_area, Taux_Vaccination (en %)"
        )

# Période d'analyse
st.sidebar.subheader("📅 Période d'Analyse")

col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Date début",
        value=datetime(2024, 1, 1),
        key='start_date'
    )

with col2:
    end_date = st.date_input(
        "Date fin",
        value=datetime.today(),
        key='end_date'
    )

# Paramètres de prédiction
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

# Choix du modèle
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
# CHARGEMENT DES DONNÉES
# ============================================================

# Fonction de génération de données fictives
def generate_dummy_linelists(sa_gdf, n=500, start=None, end=None):
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
    np.random.seed(42)
    return pd.DataFrame({
        'health_area': sa_gdf['health_area'],
        'Taux_Vaccination': np.random.beta(a=8, b=2, size=len(sa_gdf)) * 100
    })

# Chargement selon le mode
if sa_gdf is None or sa_gdf.empty:
    st.error("❌ Aucune aire de santé chargée. Configurez dans la sidebar.")
    st.stop()

with st.spinner("⏳ Chargement des données de cas..."):
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
        st.sidebar.info(f"✅ {len(df)} cas simulés générés")
    else:
        # Chargement des données réelles
        if not dm.has_epidemio_data('rougeole') and linelist_file is not None:
            try:
                df_raw = pd.read_csv(linelist_file)
                
                # Déterminer le format
                if 'Semaine_Epi' in df_raw.columns and 'Cas_Total' in df_raw.columns:
                    # Format agrégé -> désagréger
                    expanded_rows = []
                    
                    for _, row in df_raw.iterrows():
                        aire = row.get('health_area') or row.get('Aire_Sante') or row.get('name_fr')
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
                
                elif 'Date_Debut_Eruption' in df_raw.columns:
                    df = df_raw.copy()
                    
                    for col in ['Date_Debut_Eruption', 'Date_Notification']:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                
                else:
                    st.error("❌ Format CSV non reconnu")
                    st.stop()
                
                dm.set_epidemio_data(df, disease='rougeole')
                st.sidebar.success(f"✅ {len(df)} cas chargés")
                
            except Exception as e:
                st.error(f"❌ Erreur CSV : {e}")
                st.stop()
        
        elif dm.has_epidemio_data('rougeole'):
            df = dm.get_epidemio_data('rougeole')
        
        else:
            st.error("❌ Veuillez uploader un fichier CSV de lineliste")
            st.stop()
        
        # Vaccination
        if not dm.has_vaccination_data() and vaccination_file is not None:
            try:
                vaccination_df = pd.read_csv(vaccination_file)
                dm.set_vaccination_data(vaccination_df)
                st.sidebar.success(f"✅ Couverture vaccinale chargée ({len(vaccination_df)} aires)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erreur vaccination CSV : {e}")
                vaccination_df = None
        
        elif dm.has_vaccination_data():
            vaccination_df = dm.get_vaccination_data()
        
        else:
            # Extraire de la linelist si disponible
            if 'Statut_Vaccinal' in df.columns:
                vacc_by_area = df.groupby('Aire_Sante').agg({
                    'Statut_Vaccinal': lambda x: (x == 'Oui').sum() / len(x) * 100 if len(x) > 0 else 0
                }).reset_index()
                
                vacc_by_area.columns = ['health_area', 'Taux_Vaccination']
                vaccination_df = vacc_by_area
                st.sidebar.info("💉 Taux vaccination extrait de la linelist")
            else:
                vaccination_df = None
                st.sidebar.info("ℹ️ Pas de données de vaccination")

# Normalisation des colonnes
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

# Ajouter la semaine épidémiologique
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

# ============================================================
# SUITE: CARTOGRAPHIE, ANALYSE ET MODÉLISATION
# (Continuez avec la logique de votre app_rougeole.py existante)
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
    gdf_cases = gdf_cases.merge(vaccination_df, on='health_area', how='left')

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

# Ajouter des popups
for idx, row in gdf_cases.iterrows():
    popup_html = f"""
    <div style="width:280px; font-family:Arial; font-size:12px;">
        <h4 style="color:#E4032E; margin:0;">{row['health_area']}</h4>
        <hr style="margin:5px 0;">
        <table style="width:100%;">
            <tr><td><b>🦠 Cas:</b></td><td>{int(row['cas_total'])}</td></tr>
    """
    
    if 'Taux_Vaccination' in row and not pd.isna(row['Taux_Vaccination']):
        popup_html += f"<tr><td><b>💉 Vaccination:</b></td><td>{row['Taux_Vaccination']:.1f}%</td></tr>"
    
    popup_html += """
        </table>
    </div>
    """
    
    folium.CircleMarker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        radius=5,
        popup=folium.Popup(popup_html, max_width=280),
        color='#E4032E',
        fill=True,
        fillColor='#E4032E'
    ).add_to(m)

folium.LayerControl().add_to(m)

st_folium(m, width=1200, height=600, key='rougeole_map')

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
msf_footer()
