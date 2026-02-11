"""
============================================================
VERSION 4.0 - APPLICATION PALUDISME COMPLÈTE
Garde toutes les fonctionnalités + modules partagés
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
import base64
from io import BytesIO

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

warnings.filterwarnings('ignore')

# ============================================================
# APPLIQUER LE BRANDING MSF (si modules disponibles)
# ============================================================
if MODULES_AVAILABLE:
    apply_msf_branding()
else:
    # CSS de base si modules non disponibles
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

# ============================================================
# HEADER
# ============================================================
if MODULES_AVAILABLE:
    msf_header(
        "🦟 Surveillance et Modélisation du Paludisme",
        "Plateforme d'analyse avancée intégrant données épidémiologiques, environnementales et climatiques"
    )
else:
    st.markdown('<h1 class="main-header">🦟 Surveillance et Modélisation Épidémiologique du Paludisme</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#f0f2f6; padding:1rem; border-radius:8px; border-left:4px solid #E4032E; margin:1rem 0;">
    <b>Plateforme d'analyse avancée</b> intégrant données épidémiologiques, environnementales et climatiques<br>
    Modélisation prédictive multi-factorielle avec Machine Learning et validation croisée temporelle
</div>
""", unsafe_allow_html=True)

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
# INITIALISATION DE LA SESSION (TOUTES LES VARIABLES ORIGINALES)
# ============================================================
for key in ['gdf_health', 'df_cases', 'temp_raster', 'flood_raster', 'rivers_gdf', 
            'precipitation_raster', 'humidity_raster', 'elevation_raster', 
            'model_results', 'df_climate_aggregated']:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# FONCTIONS UTILITAIRES (TOUTES CONSERVÉES)
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
    for window in [2, 4, 8]:
        df[f'cases_ma{window}'] = df.groupby('health_area')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Taux de croissance
    df['growth_rate'] = df.groupby('health_area')['cases'].pct_change().fillna(0)
    
    # Features cycliques
    df['week_of_year'] = df['weeknum'] / 52
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'])
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'])
    
    # Features cycliques harmoniques
    df['sin_week_2'] = np.sin(4 * np.pi * df['week_of_year'])
    df['cos_week_2'] = np.cos(4 * np.pi * df['week_of_year'])
    
    # Min/Max roulants
    for window in [4, 8]:
        df[f'cases_min_{window}'] = df.groupby('health_area')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f'cases_max_{window}'] = df.groupby('health_area')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    
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

def normalize_week_format(week_series):
    """Normalise les différents formats de semaine"""
    unique_weeks = week_series.unique()
    week_mapping = {}
    
    for i, week in enumerate(sorted(unique_weeks), start=1):
        week_str = str(week)
        
        if 'W' in week_str or 'w' in week_str:
            # Format 2024-W01 ou W01
            num = ''.join(filter(str.isdigit, week_str.split('-')[-1]))
        elif 'S' in week_str or 's' in week_str:
            # Format S01
            num = ''.join(filter(str.isdigit, week_str))
        else:
            # Format numérique simple
            num = ''.join(filter(str.isdigit, week_str.split('-')[-1]))
        
        week_mapping[week] = int(num) if num else i
    
    return week_series.map(week_mapping)

# Fonction pour ajouter raster à la carte
def add_raster_to_map(m, raster, name):
    """Ajoute un raster à la carte Folium"""
    try:
        import branca.colormap as cm
        
        bounds = raster.bounds
        data = raster.read(1).astype(float)
        
        if raster.nodata is not None:
            data[data == raster.nodata] = np.nan
        
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        
        h, w = data.shape
        
        # Créer image RGBA
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Choisir colormap selon le type
        if 'Inondation' in name or 'flood' in name.lower():
            cmap = cm.linear.Blues_09.scale(vmin, vmax)
        elif 'Température' in name or 'temp' in name.lower():
            cmap = cm.linear.YlOrRd_09.scale(vmin, vmax)
        elif 'Précipitation' in name or 'precip' in name.lower():
            cmap = cm.linear.BuPu_09.scale(vmin, vmax)
        elif 'Humidité' in name or 'humid' in name.lower():
            cmap = cm.linear.GnBu_09.scale(vmin, vmax)
        else:
            cmap = cm.linear.Viridis_09.scale(vmin, vmax)
        
        for i in range(h):
            for j in range(w):
                if np.isnan(data[i, j]):
                    rgba[i, j] = [0, 0, 0, 0]
                else:
                    hex_color = cmap(data[i, j]).lstrip('#')
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    rgba[i, j] = [r, g, b, 180]
        
        # Convertir en image
        img = Image.fromarray(rgba, mode='RGBA')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        img_url = f'data:image/png;base64,{encoded}'
        
        folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            name=name,
            opacity=0.7,
            interactive=True,
            zindex=1
        ).add_to(m)
        
        # Ajouter colormap
        colormap = cmap
        colormap.caption = name
        colormap.add_to(m)
        
    except Exception as e:
        st.warning(f"⚠️ Impossible d'ajouter le raster '{name}': {str(e)}")

# ============================================================
# SIDEBAR - CONFIGURATION (TOUTES LES OPTIONS CONSERVÉES)
# ============================================================

st.sidebar.header("📂 Configuration de l'Analyse")

# ============================================================
# Section 1 : Aires de santé (UNIFIÉ avec option fichier local)
# ============================================================
st.sidebar.subheader("🗺️ Aires de Santé")

# Vérifier si des données géographiques sont déjà chargées via DataManager
if MODULES_AVAILABLE and dm and dm.has_geodata():
    gdf_info = GeoLoader.get_geodata_info(dm.get_geodata())
    st.sidebar.success(f"✅ {gdf_info['n_features']} aires chargées (réutilisées)")
    
    if st.sidebar.button("🔄 Recharger de nouvelles aires"):
        dm.clear_by_type('geodata')
        st.rerun()
    
    gdf_health = dm.get_geodata()
    gdf_health = ensure_wgs84(gdf_health)
    st.session_state.gdf_health = gdf_health

else:
    # NOUVELLE OPTION : Choix de la source
    option_geo = st.sidebar.radio(
        "Source des données géographiques",
        ["Upload fichier (GeoJSON/Shapefile)", "Fichier local (ao_hlthArea.zip)", "Fichier de démo"],
        key='option_geo_palu'
    )
    
    gdf_health = None
    
    # OPTION 1 : Upload
    if option_geo == "Upload fichier (GeoJSON/Shapefile)":
        uploaded_geo = st.sidebar.file_uploader(
            "📁 Charger un fichier géographique",
            type=['geojson', 'zip', 'shp'],
            help="Format : GeoJSON ou ZIP (shapefile) avec colonne 'health_area'",
            key='upload_geo_palu'
        )
        
        if uploaded_geo is not None:
            with st.spinner("⏳ Chargement des données géographiques..."):
                if MODULES_AVAILABLE:
                    gdf_health = GeoLoader.load_from_file(uploaded_geo)
                else:
                    # Fallback sans modules
                    try:
                        if uploaded_geo.name.endswith('.geojson'):
                            gdf_health = gpd.read_file(uploaded_geo)
                        elif uploaded_geo.name.endswith('.zip'):
                            import tempfile
                            import zipfile
                            with tempfile.TemporaryDirectory() as tmpdir:
                                with zipfile.ZipFile(uploaded_geo, 'r') as z:
                                    z.extractall(tmpdir)
                                shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                                if shp_files:
                                    gdf_health = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                    except Exception as e:
                        st.sidebar.error(f"❌ Erreur : {e}")
                
                if gdf_health is not None:
                    # Normaliser les noms de colonnes
                    if 'health_area' not in gdf_health.columns:
                        for col in ['HEALTHAREA', 'name_fr', 'NAME', 'nom', 'NOM', 'aire_sante']:
                            if col in gdf_health.columns:
                                gdf_health['health_area'] = gdf_health[col]
                                break
                    
                    gdf_health = ensure_wgs84(gdf_health)
                    st.session_state.gdf_health = gdf_health
                    
                    if MODULES_AVAILABLE and dm:
                        dm.set_geodata(gdf_health, source="upload")
                    
                    st.sidebar.success(f"✅ {len(gdf_health)} aires de santé chargées")
        else:
            st.sidebar.info("👆 Uploadez un fichier pour commencer")
    
    # OPTION 2 : Fichier local ao_hlthArea.zip (COMME ROUGEOLE)
    elif option_geo == "Fichier local (ao_hlthArea.zip)":
        # Mapping pays (réutilisable)
        PAYS_ISO3_MAP = {
            "Niger": "ner",
            "Burkina Faso": "bfa",
            "Mali": "mli",
            "Mauritanie": "mrt"
        }
        
        pays_selectionne = st.sidebar.selectbox(
            "🌍 Sélectionner le pays",
            list(PAYS_ISO3_MAP.keys()),
            key='pays_select_palu'
        )
        
        iso3_pays = PAYS_ISO3_MAP[pays_selectionne]
        
        if st.sidebar.button("📥 Charger les aires"):
            with st.spinner(f"⏳ Chargement des aires de {pays_selectionne}..."):
                if MODULES_AVAILABLE:
                    gdf_health = GeoLoader.load_local_ao_hltharea(iso3_pays)
                else:
                    # Fallback sans modules
                    try:
                        zip_path = os.path.join("data", "ao_hlthArea.zip")
                        if not os.path.exists(zip_path):
                            zip_path = "ao_hlthArea.zip"
                        
                        gdf_full = gpd.read_file(f"zip://{zip_path}")
                        
                        if 'iso3' in gdf_full.columns:
                            gdf_health = gdf_full[gdf_full['iso3'].str.lower() == iso3_pays.lower()].copy()
                        else:
                            gdf_health = gdf_full
                    except Exception as e:
                        st.sidebar.error(f"❌ Erreur : {e}")
                        gdf_health = None
                
                if gdf_health is not None and not gdf_health.empty:
                    gdf_health = ensure_wgs84(gdf_health)
                    st.session_state.gdf_health = gdf_health
                    
                    if MODULES_AVAILABLE and dm:
                        dm.set_geodata(gdf_health, source=f"local_{iso3_pays}")
                    
                    st.sidebar.success(f"✅ {len(gdf_health)} aires de santé chargées")
                else:
                    st.sidebar.error(f"❌ Impossible de charger les données pour {pays_selectionne}")
    
    # OPTION 3 : Données de démo
    else:
        if st.sidebar.button("🎯 Générer des données de démo"):
            with st.spinner("⏳ Génération des données de démo..."):
                np.random.seed(42)
                n_areas = 20
                
                lons = np.random.uniform(-5, 5, n_areas)
                lats = np.random.uniform(10, 15, n_areas)
                
                geometries = [Point(lon, lat).buffer(0.1) for lon, lat in zip(lons, lats)]
                
                gdf_health = gpd.GeoDataFrame({
                    'health_area': [f'Aire_{i+1}' for i in range(n_areas)],
                    'geometry': geometries
                }, crs='EPSG:4326')
                
                st.session_state.gdf_health = gdf_health
                
                if MODULES_AVAILABLE and dm:
                    dm.set_geodata(gdf_health, source="demo")
                
                st.sidebar.success(f"✅ {n_areas} aires de démo générées")
        else:
            st.sidebar.info("👆 Cliquez pour générer des données de démo")

# Récupérer depuis session_state
if st.session_state.gdf_health is not None:
    gdf_health = st.session_state.gdf_health
else:
    gdf_health = None

# ============================================================
# Section 2 : Données épidémiologiques (AVEC MEILLEURE DÉTECTION)
# ============================================================
st.sidebar.subheader("📊 Données Épidémiologiques")

# Vérifier si déjà chargées via DataManager
if MODULES_AVAILABLE and dm and dm.has_epidemio_data('paludisme'):
    epidemio_info = dm.get_summary()['source_info'].get('epidemio_paludisme', {})
    st.sidebar.success(f"✅ {epidemio_info.get('n_records', 0)} enregistrements (réutilisés)")
    
    if st.sidebar.button("🔄 Recharger les données épidémio"):
        dm.clear_by_type('epidemio')
        st.rerun()
    
    df_cases = dm.get_epidemio_data('paludisme')
    st.session_state.df_cases = df_cases

else:
    uploaded_cases = st.sidebar.file_uploader(
        "📋 Linelists paludisme (CSV)",
        type=['csv'],
        help="Format flexible : health_area/aire_sante, week_/semaine, cases/cas, deaths/deces",
        key='upload_cases_palu'
    )
    
    if uploaded_cases is not None:
        try:
            # Lire le CSV avec différents encodages possibles
            try:
                df_cases = pd.read_csv(uploaded_cases, encoding='utf-8')
            except UnicodeDecodeError:
                df_cases = pd.read_csv(uploaded_cases, encoding='latin1')
            
            st.sidebar.info(f"📋 Colonnes détectées : {', '.join(df_cases.columns)}")
            
            # AMÉLIORATION : Détection intelligente des colonnes
            COLUMN_MAPPING = {
                'health_area': [
                    'health_area', 'HEALTH_AREA', 'HealthArea',
                    'healtharea', 'HEALTHAREA',
                    'aire_sante', 'Aire_Sante', 'AIRE_SANTE', 'airesante',
                    'name_fr', 'NAME_FR', 'name', 'NAME',
                    'nom', 'NOM', 'Nom',
                    'district', 'DISTRICT', 'District',
                    'location', 'LOCATION', 'Location',
                    'zone', 'ZONE', 'Zone',
                    'area', 'AREA', 'Area'
                ],
                'week_': [
                    'week_', 'WEEK_', 'Week_',
                    'week', 'WEEK', 'Week',
                    'semaine', 'SEMAINE', 'Semaine',
                    'Semaine_Epi', 'semaine_epi', 'SEMAINE_EPI',
                    'epi_week', 'EPI_WEEK', 'EpiWeek',
                    'epiweek', 'EPIWEEK',
                    'SE', 'se', 'Se',
                    'S', 's',
                    'wk', 'WK', 'Wk',
                    'num_semaine', 'NUM_SEMAINE', 'NumSemaine'
                ],
                'cases': [
                    'cases', 'CASES', 'Cases',
                    'cas', 'CAS', 'Cas',
                    'Cas_Total', 'cas_total', 'CAS_TOTAL',
                    'nb_cas', 'NB_CAS', 'NbCas',
                    'nombre_cas', 'NOMBRE_CAS', 'NombreCas',
                    'total_cases', 'TOTAL_CASES', 'TotalCases',
                    'confirmed_cases', 'CONFIRMED_CASES',
                    'cas_confirmes', 'CAS_CONFIRMES',
                    'nbr_cas', 'NBR_CAS'
                ],
                'deaths': [
                    'deaths', 'DEATHS', 'Deaths',
                    'deces', 'DECES', 'Deces', 'Décès', 'DÉCÈS',
                    'Deces_Total', 'deces_total', 'DECES_TOTAL',
                    'nb_deces', 'NB_DECES', 'NbDeces',
                    'nombre_deces', 'NOMBRE_DECES', 'NombreDeces',
                    'total_deaths', 'TOTAL_DEATHS',
                    'morts', 'MORTS', 'Morts',
                    'nbr_deces', 'NBR_DECES'
                ]
            }
            
            # Renommer automatiquement
            rename_dict = {}
            found_cols = {'health_area': False, 'week_': False, 'cases': False, 'deaths': False}
            
            for standard_col, possible_cols in COLUMN_MAPPING.items():
                for col in possible_cols:
                    if col in df_cases.columns:
                        if col != standard_col:
                            rename_dict[col] = standard_col
                        found_cols[standard_col] = True
                        break
            
            if rename_dict:
                df_cases = df_cases.rename(columns=rename_dict)
                st.sidebar.success(f"🔄 Colonnes renommées : {', '.join(rename_dict.keys())}")
            
            # Afficher le statut de détection
            st.sidebar.markdown("**Statut de détection :**")
            for col, found in found_cols.items():
                if col != 'deaths':  # deaths est optionnel
                    icon = "✅" if found else "❌"
                    st.sidebar.markdown(f"{icon} `{col}`")
            
            # Vérifier les colonnes obligatoires APRÈS renommage
            required_cols = ['health_area', 'week_', 'cases']
            missing_cols = [col for col in required_cols if col not in df_cases.columns]
            
            if missing_cols:
                st.sidebar.error(f"❌ Colonnes manquantes après détection : {', '.join(missing_cols)}")
                
                # Afficher un tableau d'aide
                st.sidebar.markdown("---")
                st.sidebar.markdown("**📋 Colonnes détectées dans votre fichier :**")
                for col in df_cases.columns:
                    st.sidebar.code(col)
                
                st.sidebar.markdown("---")
                st.sidebar.markdown("**💡 Solutions :**")
                st.sidebar.markdown("""
                1. Vérifiez que votre CSV contient bien des colonnes pour :
                   - **Aire de santé** (ex: health_area, aire_sante, nom)
                   - **Semaine** (ex: week_, semaine, SE)
                   - **Cas** (ex: cases, cas, nb_cas)
                
                2. Exemple de format attendu :
                ```csv
                health_area,week_,cases,deaths
                Dakar Centre,1,45,2
                Dakar Centre,2,52,1
                ```
                
                3. Ou renommez manuellement vos colonnes avant l'upload
                """)
                
                df_cases = None
            else:
                # Ajouter colonne deaths si absente
                if 'deaths' not in df_cases.columns:
                    df_cases['deaths'] = 0
                    st.sidebar.info("ℹ️ Colonne 'deaths' créée (valeurs = 0)")
                
                # Normaliser le format de semaine
                df_cases['weeknum'] = normalize_week_format(df_cases['week_'])
                
                # Validation des données
                n_rows_before = len(df_cases)
                
                # Supprimer lignes avec valeurs manquantes dans colonnes critiques
                df_cases = df_cases.dropna(subset=['health_area', 'weeknum', 'cases'])
                
                # Convertir cases et deaths en numériques
                df_cases['cases'] = pd.to_numeric(df_cases['cases'], errors='coerce').fillna(0).astype(int)
                df_cases['deaths'] = pd.to_numeric(df_cases['deaths'], errors='coerce').fillna(0).astype(int)
                
                # Supprimer les valeurs négatives
                df_cases = df_cases[df_cases['cases'] >= 0]
                df_cases = df_cases[df_cases['deaths'] >= 0]
                
                n_rows_after = len(df_cases)
                
                if n_rows_before != n_rows_after:
                    st.sidebar.warning(f"⚠️ {n_rows_before - n_rows_after} lignes supprimées (valeurs manquantes ou invalides)")
                
                # Stocker dans session_state
                st.session_state.df_cases = df_cases
                
                if MODULES_AVAILABLE and dm:
                    dm.set_epidemio_data(df_cases, disease='paludisme')
                
                st.sidebar.success(f"✅ {len(df_cases)} enregistrements chargés")
                
                # Afficher un aperçu
                with st.sidebar.expander("👀 Aperçu des données"):
                    st.dataframe(df_cases[['health_area', 'week_', 'weeknum', 'cases', 'deaths']].head(10))
        
        except Exception as e:
            st.sidebar.error(f"❌ Erreur de lecture CSV : {str(e)}")
            st.sidebar.code(f"Détails : {type(e).__name__}")
            df_cases = None
    else:
        df_cases = None
        st.sidebar.info("👆 Uploadez un fichier CSV")

# Récupérer depuis session_state
if st.session_state.df_cases is not None:
    df_cases = st.session_state.df_cases
else:
    df_cases = None


# ============================================================
# Section 3 : Données Environnementales (RASTERS - CONSERVÉES)
# ============================================================
st.sidebar.subheader("🌍 Données Environnementales (Optionnel)")

with st.sidebar.expander("📁 Charger les rasters environnementaux", expanded=False):
    st.markdown("*Ces données améliorent significativement les prédictions*")
    
    # Température
    temp_raster_file = st.file_uploader(
        "🌡️ Raster Température",
        type=['tif', 'tiff'],
        help="Température moyenne annuelle ou saisonnière",
        key='temp_raster_palu'
    )
    
    if temp_raster_file is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(temp_raster_file.read())
                tmp_path = tmp.name
            
            st.session_state.temp_raster = rasterio.open(tmp_path)
            st.success("✅ Raster température chargé")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    
    # Inondation
    flood_raster_file = st.file_uploader(
        "🌊 Raster Zones Inondables",
        type=['tif', 'tiff'],
        help="Probabilité ou hauteur d'inondation",
        key='flood_raster_palu'
    )
    
    if flood_raster_file is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(flood_raster_file.read())
                tmp_path = tmp.name
            
            st.session_state.flood_raster = rasterio.open(tmp_path)
            st.success("✅ Raster inondation chargé")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    
    # Précipitations
    precip_raster_file = st.file_uploader(
        "🌧️ Raster Précipitations",
        type=['tif', 'tiff'],
        help="Cumul annuel ou saisonnier de pluies",
        key='precip_raster_palu'
    )
    
    if precip_raster_file is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(precip_raster_file.read())
                tmp_path = tmp.name
            
            st.session_state.precipitation_raster = rasterio.open(tmp_path)
            st.success("✅ Raster précipitations chargé")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    
    # Humidité
    humid_raster_file = st.file_uploader(
        "💧 Raster Humidité",
        type=['tif', 'tiff'],
        help="Humidité relative moyenne",
        key='humid_raster_palu'
    )
    
    if humid_raster_file is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(humid_raster_file.read())
                tmp_path = tmp.name
            
            st.session_state.humidity_raster = rasterio.open(tmp_path)
            st.success("✅ Raster humidité chargé")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    
    # Altitude
    elev_raster_file = st.file_uploader(
        "⛰️ Raster Altitude (DEM)",
        type=['tif', 'tiff'],
        help="Modèle numérique de terrain",
        key='elev_raster_palu'
    )
    
    if elev_raster_file is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(elev_raster_file.read())
                tmp_path = tmp.name
            
            st.session_state.elevation_raster = rasterio.open(tmp_path)
            st.success("✅ Raster altitude chargé")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    
    # Rivières (vecteur)
    rivers_file = st.file_uploader(
        "🏞️ Réseau Hydrographique (GeoJSON/Shapefile)",
        type=['geojson', 'zip', 'shp'],
        help="Cours d'eau (lignes)",
        key='rivers_palu'
    )
    
    if rivers_file is not None:
        try:
            if rivers_file.name.endswith('.geojson'):
                st.session_state.rivers_gdf = gpd.read_file(rivers_file)
            elif rivers_file.name.endswith('.zip'):
                import tempfile
                import zipfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(rivers_file, 'r') as z:
                        z.extractall(tmpdir)
                    shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                    if shp_files:
                        st.session_state.rivers_gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
            
            st.session_state.rivers_gdf = ensure_wgs84(st.session_state.rivers_gdf)
            st.success("✅ Réseau hydrographique chargé")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# ============================================================
# Section 4 : Données Climatiques NASA POWER (AVEC MODULES)
# ============================================================
st.sidebar.subheader("🌡️ Données Climatiques")

use_climate = st.sidebar.checkbox(
    "☑️ Activer NASA POWER API",
    value=False,
    help="Récupère température, précipitations et humidité depuis NASA POWER"
)

if use_climate and gdf_health is not None and df_cases is not None:
    # Vérifier si déjà chargées
    if MODULES_AVAILABLE and dm and dm.has_climate_data():
        st.sidebar.success("✅ Données climatiques (réutilisées)")
        df_climate = dm.get_climate_data()
        st.session_state.df_climate_aggregated = df_climate
    
    else:
        if st.sidebar.button("📡 Récupérer données climatiques"):
            with st.spinner("⏳ Récupération depuis NASA POWER API..."):
                # Déterminer la période depuis les données de cas
                if 'weeknum' in df_cases.columns:
                    min_week = df_cases['weeknum'].min()
                    max_week = df_cases['weeknum'].max()
                    start_date = datetime(2024, 1, 1) + timedelta(weeks=int(min_week)-1)
                    end_date = datetime(2024, 1, 1) + timedelta(weeks=int(max_week))
                else:
                    start_date = datetime(2024, 1, 1)
                    end_date = datetime.today()
                
                if MODULES_AVAILABLE:
                    df_climate = ClimateLoader.fetch_climate_for_geodata(
                        gdf_health, start_date, end_date, progress_bar=True
                    )
                    
                    if df_climate is not None:
                        # Agréger par semaine
                        df_climate_weekly = ClimateLoader.aggregate_to_weekly(df_climate)
                        st.session_state.df_climate_aggregated = df_climate_weekly
                        dm.set_climate_data(df_climate_weekly)
                        st.sidebar.success("✅ Données climatiques récupérées")
                    else:
                        st.sidebar.error("❌ Échec de récupération")
                else:
                    st.sidebar.warning("⚠️ Modules non disponibles. Fonction désactivée.")
else:
    df_climate = None

# ============================================================
# Section 5 : WorldPop (GEE) - AVEC MODULES
# ============================================================
st.sidebar.subheader("👥 Données Démographiques")

use_worldpop = st.sidebar.checkbox(
    "☑️ Activer WorldPop (GEE)",
    value=False,
    help="Récupère données de population via Google Earth Engine"
)

if use_worldpop and gdf_health is not None:
    # Vérifier si déjà chargées
    if MODULES_AVAILABLE and dm and dm.has_worldpop_data():
        st.sidebar.success("✅ Données WorldPop (réutilisées)")
        df_worldpop = dm.get_worldpop_data()
    
    else:
        if MODULES_AVAILABLE and WorldPopLoader.init_gee():
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
            st.sidebar.error("❌ GEE non initialisé ou modules indisponibles")
            df_worldpop = None
else:
    df_worldpop = None

# ============================================================
# Section 6 : FILTRES (CONSERVÉS)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtres d'Analyse")

# Filtre par semaine
if df_cases is not None and 'weeknum' in df_cases.columns:
    min_week = int(df_cases['weeknum'].min())
    max_week = int(df_cases['weeknum'].max())
    
    selected_weeks = st.sidebar.slider(
        "Filtrer par semaine",
        min_value=min_week,
        max_value=max_week,
        value=(min_week, max_week),
        help="Sélectionnez la plage de semaines à analyser"
    )
    
    # Appliquer le filtre
    df_cases_filtered = df_cases[
        (df_cases['weeknum'] >= selected_weeks[0]) & 
        (df_cases['weeknum'] <= selected_weeks[1])
    ].copy()
else:
    df_cases_filtered = df_cases

# Filtre par aire de santé
if df_cases_filtered is not None and gdf_health is not None:
    all_areas = sorted(gdf_health['health_area'].unique())
    
    selected_areas = st.sidebar.multiselect(
        "Filtrer par aire de santé",
        options=all_areas,
        default=all_areas,
        help="Sélectionnez les aires à inclure dans l'analyse"
    )
    
    if selected_areas:
        df_cases_filtered = df_cases_filtered[df_cases_filtered['health_area'].isin(selected_areas)].copy()
        gdf_health_filtered = gdf_health[gdf_health['health_area'].isin(selected_areas)].copy()
    else:
        gdf_health_filtered = gdf_health
else:
    gdf_health_filtered = gdf_health

# ============================================================
# VÉRIFICATION DES DONNÉES MINIMALES
# ============================================================

if gdf_health is None or df_cases is None:
    st.info("👆 **Chargez les aires de santé et les données épidémiologiques dans la sidebar pour commencer l'analyse**")
    st.stop()

# Utiliser les données filtrées
if df_cases_filtered is not None:
    df_cases = df_cases_filtered
if gdf_health_filtered is not None:
    gdf_health = gdf_health_filtered

# ============================================================
# EXTRACTION DES FEATURES ENVIRONNEMENTALES (SI DISPONIBLES)
# ============================================================

st.markdown("---")
st.header("🔄 Préparation des Données")

with st.spinner("⏳ Extraction des features environnementales..."):
    
    # Créer une copie de gdf_health pour enrichir
    gdf_enriched = gdf_health.copy()
    
    # Extraire depuis les rasters si disponibles
    features_extracted = []
    
    if st.session_state.temp_raster is not None:
        st.info("📊 Extraction température...")
        gdf_enriched['temp_mean'] = extract_raster_statistics(gdf_enriched, st.session_state.temp_raster, 'mean')
        features_extracted.append("Température")
    
    if st.session_state.flood_raster is not None:
        st.info("📊 Extraction zones inondables...")
        gdf_enriched['flood_mean'] = extract_raster_statistics(gdf_enriched, st.session_state.flood_raster, 'mean')
        features_extracted.append("Inondation")
    
    if st.session_state.precipitation_raster is not None:
        st.info("📊 Extraction précipitations...")
        gdf_enriched['precipitation_mean'] = extract_raster_statistics(gdf_enriched, st.session_state.precipitation_raster, 'mean')
        features_extracted.append("Précipitations")
    
    if st.session_state.humidity_raster is not None:
        st.info("📊 Extraction humidité...")
        gdf_enriched['humidity_mean'] = extract_raster_statistics(gdf_enriched, st.session_state.humidity_raster, 'mean')
        features_extracted.append("Humidité")
    
    if st.session_state.elevation_raster is not None:
        st.info("📊 Extraction altitude...")
        gdf_enriched['elevation_mean'] = extract_raster_statistics(gdf_enriched, st.session_state.elevation_raster, 'mean')
        features_extracted.append("Altitude")
    
    # Distance aux rivières
    if st.session_state.rivers_gdf is not None:
        st.info("📊 Calcul distance aux rivières...")
        gdf_enriched['dist_river'] = gdf_enriched.geometry.centroid.apply(
            lambda x: distance_to_nearest_line(x, st.session_state.rivers_gdf)
        )
        features_extracted.append("Distance rivières")
    
    if features_extracted:
        st.success(f"✅ Features extraites : {', '.join(features_extracted)}")
    else:
        st.info("ℹ️ Aucune donnée environnementale chargée")

# ============================================================
# FUSION DES DONNÉES
# ============================================================

st.markdown("---")
st.header("📊 Données Consolidées")

# Fusion de base
df_merged = df_cases.copy()

# Ajouter les géométries et features environnementales
gdf_merged = gdf_enriched[['health_area', 'geometry'] + 
                           [col for col in gdf_enriched.columns if col not in ['health_area', 'geometry']]].merge(
    df_merged, on='health_area', how='right'
)

# Ajouter données climatiques si disponibles
if st.session_state.df_climate_aggregated is not None:
    df_climate = st.session_state.df_climate_aggregated
    
    # Normaliser les semaines pour fusion
    if 'week_' in df_climate.columns and 'weeknum' in gdf_merged.columns:
        # Créer une correspondance semaine
        df_climate['weeknum'] = normalize_week_format(df_climate['week_'])
        
        gdf_merged = gdf_merged.merge(
            df_climate[['health_area', 'weeknum', 'temp_api', 'precip_api', 'humidity_api']], 
            on=['health_area', 'weeknum'], 
            how='left'
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
    # Sélectionner les colonnes intéressantes
    display_cols = ['health_area', 'weeknum', 'cases', 'deaths']
    
    # Ajouter colonnes climatiques si disponibles
    for col in ['temp_api', 'precip_api', 'humidity_api']:
        if col in gdf_merged.columns:
            display_cols.append(col)
    
    # Ajouter colonnes environnementales
    for col in ['temp_mean', 'flood_mean', 'elevation_mean', 'dist_river']:
        if col in gdf_merged.columns:
            display_cols.append(col)
    
    # Ajouter colonnes démographiques
    for col in ['Pop_Totale', 'Pop_Enfants_0_14', 'Densite_Pop']:
        if col in gdf_merged.columns:
            display_cols.append(col)
    
    available_cols = [col for col in display_cols if col in gdf_merged.columns]
    
    st.dataframe(gdf_merged[available_cols].head(20))

# ============================================================
# SECTION CARTOGRAPHIE INTERACTIVE (COMPLÈTE)
# ============================================================

st.markdown("---")
st.header("🗺️ Cartographie Interactive")

# Agréger les données par aire de santé
gdf_map = gdf_merged.groupby('health_area').agg({
    'cases': 'sum',
    'deaths': 'sum',
    'geometry': 'first'
}).reset_index()

# Ajouter toutes les features disponibles (moyennes)
for col in gdf_enriched.columns:
    if col not in ['health_area', 'geometry'] and col not in gdf_map.columns:
        if col in gdf_merged.columns:
            gdf_map = gdf_map.merge(
                gdf_merged.groupby('health_area')[col].mean().reset_index(),
                on='health_area',
                how='left'
            )

# Ajouter les données démographiques si disponibles
if df_worldpop is not None:
    gdf_map = gdf_map.merge(df_worldpop, on='health_area', how='left')

# Ajouter les moyennes climatiques si disponibles
if st.session_state.df_climate_aggregated is not None:
    climate_means = df_climate.groupby('health_area').agg({
        'temp_api': 'mean',
        'precip_api': 'mean',
        'humidity_api': 'mean'
    }).reset_index()
    
    gdf_map = gdf_map.merge(climate_means, on='health_area', how='left')

# Calculer le taux d'incidence si population disponible
if 'Pop_Totale' in gdf_map.columns:
    gdf_map['incidence_rate'] = (gdf_map['cases'] / gdf_map['Pop_Totale']) * 10000
    gdf_map['incidence_rate'] = gdf_map['incidence_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Convertir en GeoDataFrame
gdf_map = gpd.GeoDataFrame(gdf_map, geometry='geometry', crs='EPSG:4326')

# Créer la carte
center_lat = gdf_map.geometry.centroid.y.mean()
center_lon = gdf_map.geometry.centroid.x.mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=7,
    tiles='CartoDB positron'
)

# Choroplèthe - Nombre de cas
max_cases = gdf_map['cases'].max()

folium.Choropleth(
    geo_data=gdf_map,
    data=gdf_map,
    columns=['health_area', 'cases'],
    key_on='feature.properties.health_area',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Nombre de cas de paludisme',
    name='Cas de paludisme'
).add_to(m)

# Ajouter les rasters à la carte si disponibles
if st.session_state.temp_raster is not None:
    add_raster_to_map(m, st.session_state.temp_raster, "Température")

if st.session_state.flood_raster is not None:
    add_raster_to_map(m, st.session_state.flood_raster, "Zones Inondables")

if st.session_state.precipitation_raster is not None:
    add_raster_to_map(m, st.session_state.precipitation_raster, "Précipitations")

if st.session_state.humidity_raster is not None:
    add_raster_to_map(m, st.session_state.humidity_raster, "Humidité")

if st.session_state.elevation_raster is not None:
    add_raster_to_map(m, st.session_state.elevation_raster, "Altitude")

# Ajouter le réseau hydrographique si disponible
if st.session_state.rivers_gdf is not None:
    folium.GeoJson(
        st.session_state.rivers_gdf,
        name='Réseau hydrographique',
        style_function=lambda x: {
            'color': 'blue',
            'weight': 2,
            'opacity': 0.7
        }
    ).add_to(m)

# Ajouter des popups enrichis
for idx, row in gdf_map.iterrows():
    # Construction du HTML du popup
    popup_html = f"""
    <div style="width:340px; font-family:Arial; font-size:12px;">
        <h4 style="color:#E4032E; margin:0;">{row['health_area']}</h4>
        <hr style="margin:5px 0;">
        <table style="width:100%;">
            <tr><td><b>📊 Cas:</b></td><td>{safe_int(row['cases'])}</td></tr>
            <tr><td><b>💀 Décès:</b></td><td>{safe_int(row['deaths'])}</td></tr>
    """
    
    if 'Pop_Totale' in row and not pd.isna(row['Pop_Totale']):
        popup_html += f"<tr><td><b>👥 Population:</b></td><td>{int(row['Pop_Totale']):,}</td></tr>"
    
    if 'Pop_Enfants_0_14' in row and not pd.isna(row['Pop_Enfants_0_14']):
        popup_html += f"<tr><td><b>👶 Enfants 0–14:</b></td><td>{int(row['Pop_Enfants_0_14']):,}</td></tr>"
    
    if 'Densite_Pop' in row and not pd.isna(row['Densite_Pop']):
        popup_html += f"<tr><td><b>📏 Densité:</b></td><td>{safe_float(row['Densite_Pop']):.2f} hab/km²</td></tr>"
    
    if 'temp_api' in row and not pd.isna(row['temp_api']):
        popup_html += f"<tr style='background:#FFF3E0;'><td><b>🌡️ Température:</b></td><td>{safe_float(row['temp_api']):.1f}°C</td></tr>"
    
    if 'precip_api' in row and not pd.isna(row['precip_api']):
        popup_html += f"<tr style='background:#E1F5FE;'><td><b>🌧️ Précipitations:</b></td><td>{safe_float(row['precip_api']):.1f}mm</td></tr>"
    
    if 'humidity_api' in row and not pd.isna(row['humidity_api']):
        popup_html += f"<tr style='background:#E8F5E9;'><td><b>💧 Humidité:</b></td><td>{safe_float(row['humidity_api']):.1f}%</td></tr>"
    
    if 'flood_mean' in row and not pd.isna(row['flood_mean']):
        popup_html += f"<tr><td><b>🌊 Inondation:</b></td><td>{safe_float(row['flood_mean']):.2f}</td></tr>"
    
    if 'elevation_mean' in row and not pd.isna(row['elevation_mean']):
        popup_html += f"<tr><td><b>⛰️ Élévation:</b></td><td>{safe_float(row['elevation_mean']):.0f}m</td></tr>"
    
    if 'dist_river' in row and not pd.isna(row['dist_river']):
        popup_html += f"<tr><td><b>🏞️ Dist. rivière:</b></td><td>{safe_float(row['dist_river']):.2f}km</td></tr>"
    
    if 'incidence_rate' in row and not pd.isna(row['incidence_rate']):
        popup_html += f"<tr><td><b>📈 Taux incidence:</b></td><td>{safe_float(row['incidence_rate']):.1f}/10K</td></tr>"
    
    popup_html += """
        </table>
    </div>
    """
    
    # Ajouter un marker au centroid
    folium.CircleMarker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        radius=5,
        popup=folium.Popup(popup_html, max_width=340),
        color='#E4032E',
        fill=True,
        fillColor='#E4032E',
        fillOpacity=0.7
    ).add_to(m)

# Contrôle des couches
folium.LayerControl(collapsed=False).add_to(m)

# Afficher la carte
st_folium(m, width=1200, height=700, key='main_map')


# ============================================================
# SECTION ANALYSE TEMPORELLE
# ============================================================

st.markdown("---")
st.header("📈 Analyse Temporelle")

# Tendance hebdomadaire globale
weekly_cases = gdf_merged.groupby('weeknum')['cases'].sum().reset_index()
weekly_cases = weekly_cases.sort_values('weeknum')

fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=weekly_cases['weeknum'],
    y=weekly_cases['cases'],
    mode='lines+markers',
    name='Cas hebdomadaires',
    line=dict(color='#E4032E', width=2),
    marker=dict(size=6)
))

fig_trend.update_layout(
    title='Évolution Hebdomadaire des Cas de Paludisme',
    xaxis_title='Semaine',
    yaxis_title='Nombre de cas',
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_trend, use_container_width=True)

# Top 10 aires les plus touchées
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔝 Top 10 Aires Touchées")
    top10 = gdf_map.nlargest(10, 'cases')[['health_area', 'cases', 'deaths']]
    st.dataframe(top10, use_container_width=True)

with col2:
    st.subheader("📊 Distribution des Cas")
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Box(
        y=gdf_map['cases'],
        name='Cas par aire',
        marker_color='#E4032E'
    ))
    
    fig_dist.update_layout(
        yaxis_title='Nombre de cas',
        height=300
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

# Analyse par aire de santé
st.subheader("📊 Analyse Détaillée par Aire de Santé")

# Tableau récapitulatif
summary_cols = ['health_area', 'cases', 'deaths']

if 'Pop_Totale' in gdf_map.columns:
    summary_cols.append('Pop_Totale')

if 'incidence_rate' in gdf_map.columns:
    summary_cols.append('incidence_rate')

if 'Pop_Enfants_0_14' in gdf_map.columns:
    summary_cols.append('Pop_Enfants_0_14')

available_summary_cols = [col for col in summary_cols if col in gdf_map.columns]

st.dataframe(
    gdf_map[available_summary_cols].sort_values('cases', ascending=False),
    use_container_width=True
)

# ============================================================
# SECTION MODÉLISATION PRÉDICTIVE (COMPLÈTE)
# ============================================================

st.markdown("---")
st.header("🤖 Modélisation Prédictive")

st.markdown("""
<div style="background-color:#f0f2f6; padding:1rem; border-radius:8px; border-left:4px solid #E4032E; margin:1rem 0;">
<b>Note:</b> La modélisation nécessite au moins 8 semaines de données historiques et 
des features temporelles (lags, moyennes mobiles). Les données climatiques, démographiques et 
environnementales améliorent significativement la précision.
</div>
""", unsafe_allow_html=True)

# Vérifier les conditions minimales
n_weeks = gdf_merged['weeknum'].nunique()
n_areas = gdf_merged['health_area'].nunique()

if n_weeks < 8:
    st.warning(f"⚠️ Nombre de semaines insuffisant ({n_weeks}/8 minimum). Ajoutez plus de données historiques.")
    st.stop()

# Configuration du modèle
st.subheader("⚙️ Configuration du Modèle")

col1, col2, col3 = st.columns(3)

with col1:
    pred_weeks = st.slider(
        "Horizon de prédiction (semaines)",
        min_value=2,
        max_value=12,
        value=4,
        help="Nombre de semaines à prédire dans le futur"
    )

with col2:
    model_type = st.selectbox(
        "Algorithme de prédiction",
        ["Gradient Boosting (Recommandé)", "Random Forest", "Extra Trees", "Régression Linéaire"],
        help="Choisissez l'algorithme de machine learning"
    )

with col3:
    use_pca = st.checkbox(
        "Utiliser ACP",
        value=True,
        help="Réduction dimensionnelle (recommandé si >20 features)"
    )

# Options avancées
with st.expander("⚙️ Options Avancées"):
    col1, col2 = st.columns(2)
    
    with col1:
        use_clustering = st.checkbox(
            "Clustering spatial (K-Means)",
            value=True,
            help="Grouper les zones géographiquement similaires"
        )
        
        if use_clustering:
            n_clusters = st.slider("Nombre de clusters", 3, 10, 5)
    
    with col2:
        use_spatial_lag = st.checkbox(
            "Lag spatial",
            value=True,
            help="Influence des zones voisines"
        )
        
        if use_spatial_lag:
            n_neighbors = st.slider("Nombre de voisins", 3, 10, 5)

# Bouton pour lancer la modélisation
if st.button("🚀 LANCER LA MODÉLISATION", type="primary", use_container_width=True):
    
    with st.spinner("⏳ Préparation des données et entraînement du modèle..."):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Étape 1: Création des features (20%)
        status_text.text("📊 Création des features temporelles...")
        progress_bar.progress(0.2)
        
        # Créer les features avancées
        df_features = create_advanced_features(gdf_merged)
        df_features = create_population_features(df_features)
        
        # Créer features environnementales combinées
        if 'flood_mean' in df_features.columns or 'temp_mean' in df_features.columns:
            # Créer un GeoDataFrame temporaire pour create_environmental_features
            gdf_temp = gpd.GeoDataFrame(df_features, geometry='geometry')
            gdf_temp = create_environmental_features(gdf_temp)
            df_features = pd.DataFrame(gdf_temp.drop(columns='geometry'))
        
        # Étape 2: Clustering spatial (30%)
        if use_clustering:
            status_text.text("🗺️ Clustering spatial...")
            progress_bar.progress(0.3)
            
            # Extraire coordonnées des centroïdes
            coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in df_features['geometry']])
            
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_features['cluster'] = kmeans.fit_predict(coords)
            
            # One-hot encoding des clusters
            for i in range(n_clusters):
                df_features[f'cluster_{i}'] = (df_features['cluster'] == i).astype(int)
        
        # Étape 3: Lag spatial (35%)
        if use_spatial_lag:
            status_text.text("📍 Calcul lag spatial...")
            progress_bar.progress(0.35)
            
            # Calculer matrice de distances
            coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in df_features['geometry']])
            
            # Pour chaque semaine, calculer lag spatial
            spatial_lags = []
            
            for week in df_features['weeknum'].unique():
                df_week = df_features[df_features['weeknum'] == week].copy()
                
                if len(df_week) > n_neighbors:
                    # NearestNeighbors
                    week_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in df_week['geometry']])
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(week_coords)
                    distances, indices = nbrs.kneighbors(week_coords)
                    
                    # Calculer lag spatial (moyenne pondérée des voisins)
                    cases_array = df_week['cases'].values
                    
                    for i in range(len(df_week)):
                        neighbor_indices = indices[i][1:]  # Exclure le point lui-même
                        neighbor_distances = distances[i][1:]
                        
                        # Poids inversement proportionnels à la distance
                        weights = 1 / (neighbor_distances + 0.001)
                        weights = weights / weights.sum()
                        
                        spatial_lag = np.sum(cases_array[neighbor_indices] * weights)
                        spatial_lags.append(spatial_lag)
                else:
                    # Pas assez de points, utiliser moyenne simple
                    spatial_lags.extend([df_week['cases'].mean()] * len(df_week))
            
            df_features['spatial_lag'] = spatial_lags
        
        # Étape 4: Sélection des features (40%)
        status_text.text("🔍 Sélection des features...")
        progress_bar.progress(0.4)
        
        feature_cols = ['weeknum', 'sin_week', 'cos_week', 'sin_week_2', 'cos_week_2']
        
        # Ajouter les lags
        for lag in [1, 2, 4]:
            if f'cases_lag{lag}' in df_features.columns:
                feature_cols.append(f'cases_lag{lag}')
        
        # Ajouter les moyennes mobiles
        for window in [2, 4, 8]:
            if f'cases_ma{window}' in df_features.columns:
                feature_cols.append(f'cases_ma{window}')
        
        # Ajouter growth_rate
        if 'growth_rate' in df_features.columns:
            feature_cols.append('growth_rate')
        
        # Ajouter min/max roulants
        for window in [4, 8]:
            for stat in ['min', 'max']:
                col = f'cases_{stat}_{window}'
                if col in df_features.columns:
                    feature_cols.append(col)
        
        # Ajouter les features climatiques API
        for col in ['temp_api', 'precip_api', 'humidity_api']:
            if col in df_features.columns:
                feature_cols.append(col)
        
        # Ajouter les features environnementales (rasters)
        for col in ['temp_mean', 'flood_mean', 'precipitation_mean', 'humidity_mean', 'elevation_mean', 'dist_river']:
            if col in df_features.columns:
                feature_cols.append(col)
        
        # Ajouter les features environnementales combinées
        for col in ['flood_risk', 'climate_index', 'temp_precip_interaction']:
            if col in df_features.columns:
                feature_cols.append(col)
        
        # Ajouter les features démographiques
        for col in ['Pop_Totale', 'Pop_Enfants_0_14', 'Densite_Pop', 'incidence_rate', 'child_risk', 'demo_pressure']:
            if col in df_features.columns:
                feature_cols.append(col)
        
        # Ajouter les clusters
        if use_clustering:
            for i in range(n_clusters):
                feature_cols.append(f'cluster_{i}')
        
        # Ajouter spatial lag
        if use_spatial_lag:
            feature_cols.append('spatial_lag')
        
        # Supprimer les lignes avec NaN dans les features
        df_model = df_features.dropna(subset=feature_cols + ['cases'])
        
        st.info(f"📊 {len(df_model)} observations • {len(feature_cols)} features • {df_model['health_area'].nunique()} aires")
        
        # Afficher les features utilisées
        with st.expander("📋 Features utilisées"):
            st.write(feature_cols)
        
        # Étape 5: Préparation train/test (50%)
        status_text.text("🔀 Split temporel des données...")
        progress_bar.progress(0.5)
        
        X = df_model[feature_cols]
        y = df_model['cases']
        
        # Split temporel (80/20)
        split_idx = int(len(df_model) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Étape 6: Normalisation (55%)
        status_text.text("📐 Normalisation...")
        progress_bar.progress(0.55)
        
        scaler = RobustScaler()  # Plus robuste aux outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Étape 7: ACP si demandé (60%)
        if use_pca and len(feature_cols) > 10:
            status_text.text("🔬 Analyse en Composantes Principales...")
            progress_bar.progress(0.6)
            
            pca = PCA(n_components=0.95, random_state=42)  # Conserver 95% de variance
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
            
            st.info(f"🔬 ACP: {len(feature_cols)} features → {X_train_scaled.shape[1]} composantes ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
        
        # Étape 8: Entraînement du modèle (70%)
        status_text.text("🧠 Entraînement du modèle...")
        progress_bar.progress(0.7)
        
        if model_type == "Gradient Boosting (Recommandé)":
            model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
        elif model_type == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == "Extra Trees":
            model = ExtraTreesRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42
            )
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        # Étape 9: Validation croisée temporelle (80%)
        status_text.text("✅ Validation croisée temporelle...")
        progress_bar.progress(0.8)
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Étape 10: Évaluation (85%)
        status_text.text("📊 Calcul des métriques...")
        progress_bar.progress(0.85)
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Métriques
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Étape 11: Prédictions futures (90%)
        status_text.text("🔮 Génération des prédictions futures...")
        progress_bar.progress(0.9)
        
        last_week = df_model['weeknum'].max()
        future_predictions = []
        
        for area in df_model['health_area'].unique():
            df_area = df_model[df_model['health_area'] == area].tail(20)
            
            if len(df_area) == 0:
                continue
            
            # Récupérer les features constantes (environnementales, démo)
            constant_features = {}
            for col in feature_cols:
                if col not in ['weeknum', 'sin_week', 'cos_week', 'sin_week_2', 'cos_week_2', 
                               'cases_lag1', 'cases_lag2', 'cases_lag4',
                               'cases_ma2', 'cases_ma4', 'cases_ma8',
                               'growth_rate', 'cases_min_4', 'cases_max_4', 
                               'cases_min_8', 'cases_max_8', 'spatial_lag']:
                    if col in df_area.columns and not df_area[col].isna().all():
                        constant_features[col] = df_area[col].mean()
            
            # Prédictions itératives
            prev_predictions = list(df_area['cases'].tail(8))
            
            for i in range(1, pred_weeks + 1):
                future_week = last_week + i
                
                # Créer les features pour la semaine future
                future_row = {
                    'weeknum': future_week,
                    'sin_week': np.sin(2 * np.pi * (future_week / 52)),
                    'cos_week': np.cos(2 * np.pi * (future_week / 52)),
                    'sin_week_2': np.sin(4 * np.pi * (future_week / 52)),
                    'cos_week_2': np.cos(4 * np.pi * (future_week / 52))
                }
                
                # Lags
                future_row['cases_lag1'] = prev_predictions[-1] if len(prev_predictions) > 0 else df_area['cases'].mean()
                future_row['cases_lag2'] = prev_predictions[-2] if len(prev_predictions) > 1 else df_area['cases'].mean()
                future_row['cases_lag4'] = prev_predictions[-4] if len(prev_predictions) > 3 else df_area['cases'].mean()
                
                # Moyennes mobiles
                future_row['cases_ma2'] = np.mean(prev_predictions[-2:]) if len(prev_predictions) > 1 else df_area['cases'].mean()
                future_row['cases_ma4'] = np.mean(prev_predictions[-4:]) if len(prev_predictions) > 3 else df_area['cases'].mean()
                future_row['cases_ma8'] = np.mean(prev_predictions[-8:]) if len(prev_predictions) > 7 else df_area['cases'].mean()
                
                # Growth rate
                if len(prev_predictions) > 1:
                    future_row['growth_rate'] = (prev_predictions[-1] - prev_predictions[-2]) / (prev_predictions[-2] + 1)
                else:
                    future_row['growth_rate'] = 0
                
                # Min/Max
                if len(prev_predictions) > 3:
                    future_row['cases_min_4'] = min(prev_predictions[-4:])
                    future_row['cases_max_4'] = max(prev_predictions[-4:])
                else:
                    future_row['cases_min_4'] = df_area['cases'].min()
                    future_row['cases_max_4'] = df_area['cases'].max()
                
                if len(prev_predictions) > 7:
                    future_row['cases_min_8'] = min(prev_predictions[-8:])
                    future_row['cases_max_8'] = max(prev_predictions[-8:])
                else:
                    future_row['cases_min_8'] = df_area['cases'].min()
                    future_row['cases_max_8'] = df_area['cases'].max()
                
                # Spatial lag (utiliser la moyenne des prédictions précédentes)
                if use_spatial_lag:
                    future_row['spatial_lag'] = np.mean(prev_predictions[-5:]) if len(prev_predictions) > 4 else df_area['cases'].mean()
                
                # Ajouter features constantes
                for col, val in constant_features.items():
                    future_row[col] = val
                
                # Faire la prédiction
                X_future = pd.DataFrame([future_row])[feature_cols]
                X_future_scaled = scaler.transform(X_future)
                
                if use_pca and len(feature_cols) > 10:
                    X_future_scaled = pca.transform(X_future_scaled)
                
                pred_cases = max(0, model.predict(X_future_scaled)[0])
                
                # Ajouter à l'historique
                prev_predictions.append(pred_cases)
                
                future_predictions.append({
                    'health_area': area,
                    'week_': f"S{int(future_week)}",
                    'weeknum': future_week,
                    'predicted_cases': pred_cases
                })
        
        df_predictions = pd.DataFrame(future_predictions)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Modélisation terminée !")
        
        # Afficher les résultats
        st.success("✅ Modélisation terminée !")
        
        # Métriques de performance
        st.subheader("📊 Performance du Modèle")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("R² Train", f"{r2_train:.3f}")
        
        with col2:
            st.metric("R² Test", f"{r2_test:.3f}")
        
        with col3:
            st.metric("R² CV (Moy)", f"{cv_mean:.3f}")
        
        with col4:
            st.metric("MAE", f"{mae_test:.1f}")
        
        with col5:
            st.metric("RMSE", f"{rmse_test:.1f}")
        
        # Stabilité du modèle
        st.markdown(f"**Stabilité (CV):** R² = {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Interprétation
        if r2_test > 0.8:
            st.success("🎯 Excellent modèle ! Le R² > 0.80 indique une très bonne capacité prédictive.")
        elif r2_test > 0.6:
            st.info("👍 Bon modèle. Le R² > 0.60 indique une capacité prédictive correcte.")
        else:
            st.warning("⚠️ Modèle à améliorer. Essayez d'ajouter plus de données historiques ou de features.")
        
        if cv_std < 0.05:
            st.success("✅ Modèle très stable (écart-type CV < 0.05)")
        elif cv_std < 0.10:
            st.info("ℹ️ Modèle stable (écart-type CV < 0.10)")
        else:
            st.warning("⚠️ Modèle instable. Les performances varient selon les périodes.")
        
        # Graphique des prédictions vs réel
        st.subheader("📈 Prédictions vs Observations (Ensemble de Test)")
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=y_test.values,
            y=y_pred_test,
            mode='markers',
            name='Prédictions',
            marker=dict(color='#E4032E', size=8, opacity=0.6)
        ))
        
        # Ligne de référence parfaite
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
            height=400,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Importance des variables
        if hasattr(model, 'feature_importances_') and not use_pca:
            st.subheader("🔍 Importance des Variables")
            
            feature_importance = pd.DataFrame({
                'Variable': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(
                feature_importance.head(15),
                x='Importance',
                y='Variable',
                orientation='h',
                title='Top 15 Variables les Plus Importantes'
            )
            
            fig_imp.update_traces(marker_color='#E4032E')
            
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Interprétation
            top_var = feature_importance.iloc[0]['Variable']
            top_imp = feature_importance.iloc[0]['Importance']
            
            st.info(f"💡 La variable la plus importante est **{top_var}** (importance: {top_imp:.3f})")
        
        # Prédictions futures par aire
        st.subheader(f"🔮 Prédictions pour les {pred_weeks} Prochaines Semaines")
        
        # Top 10 aires avec les prédictions les plus élevées
        top_pred_areas = df_predictions.groupby('health_area')['predicted_cases'].sum().nlargest(10).reset_index()
        
        fig_top_pred = px.bar(
            top_pred_areas,
            x='predicted_cases',
            y='health_area',
            orientation='h',
            title=f'Top 10 Aires - Cas Prédits (Total {pred_weeks} semaines)',
            labels={'predicted_cases': 'Cas prédits', 'health_area': 'Aire de santé'}
        )
        
        fig_top_pred.update_traces(marker_color='#E4032E')
        
        st.plotly_chart(fig_top_pred, use_container_width=True)
        
        # Graphique temporel des prédictions
        st.subheader("📊 Évolution Temporelle des Prédictions")
        
        # Agréger par semaine
        weekly_pred = df_predictions.groupby('weeknum')['predicted_cases'].sum().reset_index()
        
        # Ajouter l'historique
        weekly_hist = gdf_merged.groupby('weeknum')['cases'].sum().reset_index()
        weekly_hist.columns = ['weeknum', 'cases']
        
        fig_timeline = go.Figure()
        
        # Historique
        fig_timeline.add_trace(go.Scatter(
            x=weekly_hist['weeknum'],
            y=weekly_hist['cases'],
            mode='lines+markers',
            name='Historique (observé)',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=6)
        ))
        
        # Prédictions
        fig_timeline.add_trace(go.Scatter(
            x=weekly_pred['weeknum'],
            y=weekly_pred['predicted_cases'],
            mode='lines+markers',
            name='Prédictions',
            line=dict(color='#E4032E', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig_timeline.update_layout(
            title='Historique vs Prédictions',
            xaxis_title='Semaine',
            yaxis_title='Nombre de cas',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Tableau détaillé des prédictions
        with st.expander("📋 Tableau Complet des Prédictions"):
            pivot_table = df_predictions.pivot(
                index='health_area',
                columns='week_',
                values='predicted_cases'
            ).fillna(0).round(1)
            
            # Ajouter colonne total
            pivot_table['TOTAL'] = pivot_table.sum(axis=1)
            pivot_table = pivot_table.sort_values('TOTAL', ascending=False)
            
            st.dataframe(pivot_table, use_container_width=True)
        
        # Téléchargements
        st.subheader("💾 Téléchargements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_predictions.to_csv(index=False)
            st.download_button(
                label="📥 Prédictions (CSV)",
                data=csv,
                file_name=f"predictions_paludisme_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Créer un GeoJSON avec les prédictions
            gdf_pred = gdf_health.merge(
                df_predictions.groupby('health_area')['predicted_cases'].sum().reset_index(),
                on='health_area',
                how='left'
            )
            
            geojson_str = gdf_pred.to_json()
            st.download_button(
                label="🗺️ Carte (GeoJSON)",
                data=geojson_str,
                file_name=f"carte_predictions_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Rapport Excel complet
            from io import BytesIO
            import openpyxl
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Feuille 1: Résumé
                summary_df = pd.DataFrame({
                    'Métrique': ['R² Train', 'R² Test', 'R² CV Moyen', 'R² CV Std', 'MAE', 'RMSE', 'Nb Features', 'Nb Observations'],
                    'Valeur': [r2_train, r2_test, cv_mean, cv_std, mae_test, rmse_test, len(feature_cols), len(df_model)]
                })
                summary_df.to_excel(writer, sheet_name='Résumé', index=False)
                
                # Feuille 2: Prédictions
                df_predictions.to_excel(writer, sheet_name='Prédictions', index=False)
                
                # Feuille 3: Features importance (si disponible)
                if hasattr(model, 'feature_importances_') and not use_pca:
                    feature_importance.to_excel(writer, sheet_name='Importance Features', index=False)
                
                # Feuille 4: Données historiques
                gdf_merged[['health_area', 'weeknum', 'cases', 'deaths']].to_excel(writer, sheet_name='Historique', index=False)
            
            st.download_button(
                label="📊 Rapport Complet (Excel)",
                data=output.getvalue(),
                file_name=f"rapport_paludisme_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

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
