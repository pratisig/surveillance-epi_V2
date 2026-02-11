"""
============================================================
APPLICATION PALUDISME - VERSION 4.0 MODULAIRE
Utilise les modules partagés pour le chargement des données
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import Popup, Tooltip, CircleMarker, GeoJson, LayerControl, DivIcon
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
import requests
import json
from shapely.geometry import Point
import sys
import os

# Ajouter le dossier modules au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Imports des modules partagés
from modules.ui_components import apply_msf_branding, msf_header, msf_footer
from modules.data_loader import DataManager
from modules.geo_loader import GeoLoader
from modules.climate_loader import ClimateLoader
from modules.worldpop_loader import WorldPopLoader
from modules.utils import safe_int, safe_float, format_large_number

warnings.filterwarnings('ignore')

# ============================================================
# APPLIQUER LE BRANDING MSF
# ============================================================
apply_msf_branding()

# CSS spécifique à l'app Paludisme
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #E4032E;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
msf_header(
    "🦟 Surveillance et Modélisation du Paludisme",
    "Plateforme d'analyse avancée intégrant données épidémiologiques, environnementales et climatiques"
)

st.markdown("""
<div class="info-box">
    <b>Plateforme d'analyse avancée</b> intégrant données épidémiologiques, environnementales et climatiques<br>
    Modélisation prédictive multi-factorielle avec Machine Learning et validation croisée temporelle
</div>
""", unsafe_allow_html=True)

# ============================================================
# INITIALISATION DU GESTIONNAIRE DE DONNÉES
# ============================================================
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

dm = st.session_state.data_manager

# ============================================================
# INITIALISATION DE LA SESSION
# ============================================================
for key in ['gdf_health', 'df_cases', 'temp_raster', 'flood_raster', 'rivers_gdf', 
            'precipitation_raster', 'humidity_raster', 'elevation_raster', 
            'model_results', 'df_climate_aggregated']:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def week_to_daterange(week_num, year=2024):
    """Convertit un numéro de semaine en plage de dates"""
    week_num = int(week_num)
    year = int(year)
    jan_first = datetime(year, 1, 1)
    week_start = jan_first + timedelta(weeks=week_num - 1)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end

def ensure_wgs84(gdf):
    """Assure que le GeoDataFrame est en WGS84"""
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')
    return gdf

def extract_raster_statistics(gdf, raster, stat='mean'):
    """Extrait des statistiques depuis un raster pour chaque géométrie"""
    stats = []
    for geom in gdf.geometry:
        try:
            out_img, _ = mask(raster, [geom], crop=True)
            data = out_img[0].astype(float)
            
            nodata = raster.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            
            if stat == 'mean':
                value = np.nanmean(data)
            elif stat == 'max':
                value = np.nanmax(data)
            elif stat == 'min':
                value = np.nanmin(data)
            elif stat == 'std':
                value = np.nanstd(data)
            else:
                value = np.nanmean(data)
            
            if np.isinf(value) or np.isnan(value):
                value = np.nan
            
            stats.append(value)
        except Exception:
            stats.append(np.nan)
    
    return stats

def distance_to_nearest_line(point, lines_gdf):
    """Calcule la distance au cours d'eau le plus proche"""
    if lines_gdf.empty:
        return np.nan
    return lines_gdf.geometry.apply(lambda x: point.distance(x)).min() * 111  # km

def create_advanced_features(df):
    """Crée des features temporelles avancées"""
    df = df.sort_values(['health_area', 'weeknum'])
    
    # Lags
    for lag in [1, 2, 4]:
        df[f'cases_lag{lag}'] = df.groupby('health_area')['cases'].shift(lag)
    
    # Moyennes mobiles
    for window in [2, 4]:
        df[f'cases_ma{window}'] = df.groupby('health_area')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Taux de croissance
    df['growth_rate'] = df.groupby('health_area')['cases'].pct_change().fillna(0)
    
    # Features cycliques
    df['week_of_year'] = df['weeknum'] / 52
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'])
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'])
    
    return df

def create_population_features(df):
    """Crée des features dérivées de population"""
    df = df.copy()
    
    # Taux d'incidence
    if 'Pop_Totale' in df.columns:
        df['incidence_rate'] = (df['cases'] / df['Pop_Totale']) * 10000
        df['incidence_rate'] = df['incidence_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Risque enfants
    if 'Pop_Enfants_0_14' in df.columns:
        df['child_risk'] = (df['cases'] / df['Pop_Enfants_0_14']) * 1000
        df['child_risk'] = df['child_risk'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Pression démographique
    if 'Densite_Pop' in df.columns and 'incidence_rate' in df.columns:
        df['demo_pressure'] = df['Densite_Pop'] * df['incidence_rate']
        df['demo_pressure'] = df['demo_pressure'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

def create_environmental_features(gdf_map):
    """Crée des features environnementales combinées"""
    if 'flood_mean' in gdf_map.columns and 'dist_river' in gdf_map.columns:
        gdf_map['flood_risk'] = gdf_map['flood_mean'] / (gdf_map['dist_river'] + 0.1)
    
    if 'temp_mean' in gdf_map.columns and 'humidity_mean' in gdf_map.columns:
        gdf_map['climate_index'] = np.exp(-((gdf_map['temp_mean'] - 27.5)**2 / 50)) * (gdf_map['humidity_mean'] / 100)
    
    if 'temp_mean' in gdf_map.columns and 'precipitation_mean' in gdf_map.columns:
        gdf_map['temp_precip_interaction'] = gdf_map['temp_mean'] * gdf_map['precipitation_mean']
    
    return gdf_map

# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

st.sidebar.header("📂 Configuration de l'Analyse")

# Section 1 : Aires de santé
st.sidebar.subheader("🗺️ Aires de Santé")

# Vérifier si des données géographiques sont déjà chargées
if dm.has_geodata():
    gdf_info = GeoLoader.get_geodata_info(dm.get_geodata())
    st.sidebar.success(f"✅ {gdf_info['n_features']} aires chargées (réutilisées)")
    
    if st.sidebar.button("🔄 Recharger de nouvelles aires"):
        dm.clear_by_type('geodata')
        st.rerun()
    
    gdf_health = dm.get_geodata()

else:
    # Charger de nouvelles données
    option_geo = st.sidebar.radio(
        "Source des données géographiques",
        ["Upload fichier (GeoJSON/Shapefile)", "Fichier de démo"],
        key='option_geo'
    )
    
    if option_geo == "Upload fichier (GeoJSON/Shapefile)":
        uploaded_geo = st.sidebar.file_uploader(
            "📁 Charger un fichier géographique",
            type=['geojson', 'zip', 'shp'],
            help="Format : GeoJSON ou ZIP (shapefile) avec colonne 'health_area'"
        )
        
        if uploaded_geo is not None:
            with st.spinner("⏳ Chargement des données géographiques..."):
                gdf_health = GeoLoader.load_from_file(uploaded_geo)
                
                if gdf_health is not None:
                    valid, msg = GeoLoader.validate_geodata(gdf_health)
                    
                    if valid:
                        gdf_health = ensure_wgs84(gdf_health)
                        dm.set_geodata(gdf_health, source="upload")
                        st.sidebar.success(f"✅ {len(gdf_health)} aires de santé chargées")
                    else:
                        st.sidebar.error(f"❌ {msg}")
                        gdf_health = None
        else:
            gdf_health = None
            st.sidebar.info("👆 Uploadez un fichier pour commencer")
    
    else:
        # Données de démo
        if st.sidebar.button("🎯 Générer des données de démo"):
            with st.spinner("⏳ Génération des données de démo..."):
                # Créer un GeoDataFrame de démo
                np.random.seed(42)
                n_areas = 20
                
                lons = np.random.uniform(-5, 5, n_areas)
                lats = np.random.uniform(10, 15, n_areas)
                
                from shapely.geometry import Point
                geometries = [Point(lon, lat).buffer(0.1) for lon, lat in zip(lons, lats)]
                
                gdf_health = gpd.GeoDataFrame({
                    'health_area': [f'Aire_{i+1}' for i in range(n_areas)],
                    'geometry': geometries
                }, crs='EPSG:4326')
                
                dm.set_geodata(gdf_health, source="demo")
                st.sidebar.success(f"✅ {n_areas} aires de démo générées")
        else:
            gdf_health = None
            st.sidebar.info("👆 Cliquez pour générer des données de démo")

# Section 2 : Données épidémiologiques
st.sidebar.subheader("📊 Données Épidémiologiques")

if dm.has_epidemio_data('paludisme'):
    epidemio_info = dm.get_summary()['source_info'].get('epidemio_paludisme', {})
    st.sidebar.success(f"✅ {epidemio_info.get('n_records', 0)} enregistrements (réutilisés)")
    
    if st.sidebar.button("🔄 Recharger les données épidémio"):
        dm.clear_by_type('epidemio')
        st.rerun()
    
    df_cases = dm.get_epidemio_data('paludisme')

else:
    uploaded_cases = st.sidebar.file_uploader(
        "📋 Linelists paludisme (CSV)",
        type=['csv'],
        help="Format: health_area, week_, cases, deaths (optionnel)"
    )
    
    if uploaded_cases is not None:
        try:
            df_cases = pd.read_csv(uploaded_cases)
            
            # Vérifier les colonnes obligatoires
            required_cols = ['health_area', 'week_', 'cases']
            missing_cols = [col for col in required_cols if col not in df_cases.columns]
            
            if missing_cols:
                st.sidebar.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
                df_cases = None
            else:
                # Ajouter colonne deaths si absente
                if 'deaths' not in df_cases.columns:
                    df_cases['deaths'] = 0
                
                dm.set_epidemio_data(df_cases, disease='paludisme')
                st.sidebar.success(f"✅ {len(df_cases)} enregistrements chargés")
        
        except Exception as e:
            st.sidebar.error(f"❌ Erreur de lecture : {str(e)}")
            df_cases = None
    else:
        df_cases = None
        st.sidebar.info("👆 Uploadez un fichier CSV")

# Section 3 : Données climatiques (NASA POWER)
st.sidebar.subheader("🌡️ Données Climatiques")

use_climate = st.sidebar.checkbox(
    "☑️ Activer NASA POWER API",
    value=False,
    help="Récupère température, précipitations et humidité"
)

if use_climate and gdf_health is not None and df_cases is not None:
    if dm.has_climate_data():
        st.sidebar.success("✅ Données climatiques (réutilisées)")
        df_climate = dm.get_climate_data()
    
    else:
        if st.sidebar.button("📡 Récupérer données climatiques"):
            with st.spinner("⏳ Récupération depuis NASA POWER..."):
                # Déterminer la période
                start_date = datetime(2024, 1, 1)
                end_date = datetime.today()
                
                df_climate = ClimateLoader.fetch_climate_for_geodata(
                    gdf_health, start_date, end_date, progress_bar=True
                )
                
                if df_climate is not None:
                    # Agréger par semaine
                    df_climate_weekly = ClimateLoader.aggregate_to_weekly(df_climate)
                    dm.set_climate_data(df_climate_weekly)
                    st.sidebar.success("✅ Données climatiques récupérées")
                else:
                    st.sidebar.error("❌ Échec de récupération")
else:
    df_climate = None

# Section 4 : WorldPop (GEE)
st.sidebar.subheader("👥 Données Démographiques")

use_worldpop = st.sidebar.checkbox(
    "☑️ Activer WorldPop (GEE)",
    value=False,
    help="Récupère données de population via Google Earth Engine"
)

if use_worldpop and gdf_health is not None:
    if dm.has_worldpop_data():
        st.sidebar.success("✅ Données WorldPop (réutilisées)")
        df_worldpop = dm.get_worldpop_data()
    
    else:
        if WorldPopLoader.init_gee():
            if st.sidebar.button("👥 Récupérer WorldPop"):
                with st.spinner("⏳ Récupération depuis GEE..."):
                    df_worldpop = WorldPopLoader.fetch_worldpop_for_geodata(
                        gdf_health, year=2020, progress_bar=True
                    )
                    
                    if df_worldpop is not None:
                        dm.set_worldpop_data(df_worldpop)
                        st.sidebar.success("✅ Données WorldPop récupérées")
                    else:
                        st.sidebar.error("❌ Échec de récupération")
        else:
            st.sidebar.error("❌ GEE non initialisé")
            df_worldpop = None
else:
    df_worldpop = None

# ============================================================
# VÉRIFICATION DES DONNÉES MINIMALES
# ============================================================

if gdf_health is None or df_cases is None:
    st.info("👆 **Chargez les aires de santé et les données épidémiologiques dans la sidebar pour commencer l'analyse**")
    st.stop()

# ============================================================
# FUSION DES DONNÉES
# ============================================================

st.markdown("---")
st.header("📊 Données Consolidées")

# Fusion de base
df_merged = df_cases.copy()

# Ajouter les géométries
gdf_merged = gdf_health[['health_area', 'geometry']].merge(
    df_merged, on='health_area', how='right'
)

# Ajouter données climatiques si disponibles
if df_climate is not None:
    gdf_merged = gdf_merged.merge(
        df_climate, on=['health_area', 'week_'], how='left'
    )
    st.success("✅ Données climatiques fusionnées")

# Ajouter données WorldPop si disponibles
if df_worldpop is not None:
    gdf_merged = gdf_merged.merge(
        df_worldpop, on='health_area', how='left'
    )
    st.success("✅ Données WorldPop fusionnées")

# Convertir en GeoDataFrame
if not isinstance(gdf_merged, gpd.GeoDataFrame):
    gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry', crs='EPSG:4326')

# Afficher un aperçu
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🗺️ Aires de santé", len(gdf_health))

with col2:
    st.metric("📋 Enregistrements", len(df_merged))

with col3:
    total_cases = df_merged['cases'].sum()
    st.metric("🦟 Cas totaux", f"{int(total_cases):,}")

with col4:
    total_deaths = df_merged['deaths'].sum()
    st.metric("💀 Décès totaux", int(total_deaths))

# Afficher les premières lignes
with st.expander("👀 Aperçu des données fusionnées"):
    st.dataframe(gdf_merged.head(20))

