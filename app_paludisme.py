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

# ============================================================
# SUITE DE app_paludisme.py - CARTOGRAPHIE ET MODÉLISATION
# ============================================================

# ============================================================
# SECTION CARTOGRAPHIE INTERACTIVE
# ============================================================

st.markdown("---")
st.header("🗺️ Cartographie Interactive")

# Agréger les données par aire de santé
gdf_map = gdf_merged.groupby('health_area').agg({
    'cases': 'sum',
    'deaths': 'sum',
    'geometry': 'first'
}).reset_index()

# Ajouter les données démographiques si disponibles
if df_worldpop is not None:
    gdf_map = gdf_map.merge(df_worldpop, on='health_area', how='left')

# Ajouter les moyennes climatiques si disponibles
if df_climate is not None:
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
    legend_name='Nombre de cas',
    name='Cas de paludisme'
).add_to(m)

# Ajouter des popups enrichis
for idx, row in gdf_map.iterrows():
    # Construction du HTML du popup
    popup_html = f"""
    <div style="width:300px; font-family:Arial; font-size:12px;">
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
        popup=folium.Popup(popup_html, max_width=300),
        color='#E4032E',
        fill=True,
        fillColor='#E4032E',
        fillOpacity=0.7
    ).add_to(m)

# Contrôle des couches
folium.LayerControl(collapsed=False).add_to(m)

# Afficher la carte
st_folium(m, width=1200, height=600, key='main_map')

# Légende
st.markdown(f"""
<div class="info-box">
📊 <b>Légende</b><br>
🟡 Faible: 0-{max_cases//3:.0f} cas<br>
🟠 Moyen: {max_cases//3:.0f}-{2*max_cases//3:.0f} cas<br>
🔴 Élevé: >{2*max_cases//3:.0f} cas
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION ANALYSE TEMPORELLE
# ============================================================

st.markdown("---")
st.header("📈 Analyse Temporelle")

# Tendance hebdomadaire globale
weekly_cases = gdf_merged.groupby('week_')['cases'].sum().reset_index()
weekly_cases = weekly_cases.sort_values('week_')

fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=weekly_cases['week_'],
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

# ============================================================
# SECTION MODÉLISATION PRÉDICTIVE
# ============================================================

st.markdown("---")
st.header("🤖 Modélisation Prédictive")

st.markdown("""
<div class="info-box">
<b>Note:</b> La modélisation nécessite au moins 8 semaines de données historiques et 
des features temporelles (lags, moyennes mobiles). Les données climatiques et démographiques 
améliorent significativement la précision.
</div>
""", unsafe_allow_html=True)

# Vérifier les conditions minimales
n_weeks = gdf_merged['week_'].nunique()
n_areas = gdf_merged['health_area'].nunique()

if n_weeks < 8:
    st.warning(f"⚠️ Nombre de semaines insuffisant ({n_weeks}/8 minimum). Ajoutez plus de données historiques.")
    st.stop()

# Configuration du modèle
st.subheader("⚙️ Configuration du Modèle")

col1, col2 = st.columns(2)

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
        ["Gradient Boosting (Recommandé)", "Random Forest", "Régression Linéaire"],
        help="Choisissez l'algorithme de machine learning"
    )

# Bouton pour lancer la modélisation
if st.button("🚀 LANCER LA MODÉLISATION", type="primary", use_container_width=True):
    
    with st.spinner("⏳ Préparation des données et entraînement du modèle..."):
        
        progress_bar = st.progress(0)
        
        # Étape 1: Création des features (20%)
        progress_bar.progress(0.2)
        
        # Normaliser le numéro de semaine
        gdf_merged['weeknum'] = gdf_merged['week_'].astype(str).str.extract('(\d+)').astype(int)
        
        # Créer les features avancées
        df_features = create_advanced_features(gdf_merged)
        df_features = create_population_features(df_features)
        
        # Étape 2: Sélection des features (40%)
        progress_bar.progress(0.4)
        
        feature_cols = ['weeknum', 'sin_week', 'cos_week']
        
        # Ajouter les lags
        for lag in [1, 2, 4]:
            if f'cases_lag{lag}' in df_features.columns:
                feature_cols.append(f'cases_lag{lag}')
        
        # Ajouter les moyennes mobiles
        for window in [2, 4]:
            if f'cases_ma{window}' in df_features.columns:
                feature_cols.append(f'cases_ma{window}')
        
        # Ajouter growth_rate
        if 'growth_rate' in df_features.columns:
            feature_cols.append('growth_rate')
        
        # Ajouter les features climatiques si disponibles
        for col in ['temp_api', 'precip_api', 'humidity_api']:
            if col in df_features.columns:
                feature_cols.append(col)
        
        # Ajouter les features démographiques si disponibles
        for col in ['Pop_Totale', 'Pop_Enfants_0_14', 'Densite_Pop']:
            if col in df_features.columns:
                feature_cols.append(col)
        
        # Supprimer les lignes avec NaN dans les features
        df_model = df_features.dropna(subset=feature_cols + ['cases'])
        
        st.info(f"📊 {len(df_model)} observations • {len(feature_cols)} features • {df_model['health_area'].nunique()} aires")
        
        # Étape 3: Préparation train/test (60%)
        progress_bar.progress(0.6)
        
        X = df_model[feature_cols]
        y = df_model['cases']
        
        # Split temporel (80/20)
        split_idx = int(len(df_model) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Étape 4: Entraînement du modèle (80%)
        progress_bar.progress(0.8)
        
        if model_type == "Gradient Boosting (Recommandé)":
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        # Étape 5: Évaluation (90%)
        progress_bar.progress(0.9)
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Métriques
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Étape 6: Prédictions futures (100%)
        progress_bar.progress(1.0)
        
        # Générer les prédictions futures
        last_week = df_model['weeknum'].max()
        future_predictions = []
        
        for area in df_model['health_area'].unique():
            df_area = df_model[df_model['health_area'] == area].tail(10)
            
            for i in range(1, pred_weeks + 1):
                future_week = last_week + i
                
                # Créer les features pour la semaine future
                future_row = {
                    'weeknum': future_week,
                    'sin_week': np.sin(2 * np.pi * (future_week / 52)),
                    'cos_week': np.cos(2 * np.pi * (future_week / 52))
                }
                
                # Utiliser les dernières valeurs pour les lags
                if len(df_area) > 0:
                    future_row['cases_lag1'] = df_area['cases'].iloc[-1]
                    future_row['cases_lag2'] = df_area['cases'].iloc[-2] if len(df_area) > 1 else df_area['cases'].iloc[-1]
                    future_row['cases_lag4'] = df_area['cases'].iloc[-4] if len(df_area) > 3 else df_area['cases'].mean()
                    future_row['cases_ma2'] = df_area['cases'].tail(2).mean()
                    future_row['cases_ma4'] = df_area['cases'].tail(4).mean()
                    future_row['growth_rate'] = df_area['growth_rate'].iloc[-1] if 'growth_rate' in df_area.columns else 0
                
                # Ajouter les features climatiques/démographiques moyennes
                for col in feature_cols:
                    if col not in future_row and col in df_area.columns:
                        future_row[col] = df_area[col].mean()
                
                # Faire la prédiction
                X_future = pd.DataFrame([future_row])[feature_cols]
                X_future_scaled = scaler.transform(X_future)
                pred_cases = max(0, model.predict(X_future_scaled)[0])
                
                future_predictions.append({
                    'health_area': area,
                    'week_': f"S{future_week}",
                    'predicted_cases': pred_cases
                })
        
        df_predictions = pd.DataFrame(future_predictions)
        
        progress_bar.empty()
        
        # Afficher les résultats
        st.success("✅ Modélisation terminée !")
        
        # Métriques de performance
        st.subheader("📊 Performance du Modèle")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Train", f"{r2_train:.3f}")
        
        with col2:
            st.metric("R² Test", f"{r2_test:.3f}")
        
        with col3:
            st.metric("MAE", f"{mae_test:.1f}")
        
        with col4:
            st.metric("RMSE", f"{rmse_test:.1f}")
        
        # Interprétation
        if r2_test > 0.8:
            st.success("🎯 Excellent modèle ! Le R² > 0.80 indique une très bonne capacité prédictive.")
        elif r2_test > 0.6:
            st.info("👍 Bon modèle. Le R² > 0.60 indique une capacité prédictive correcte.")
        else:
            st.warning("⚠️ Modèle à améliorer. Essayez d'ajouter plus de données historiques ou de features.")
        
        # Graphique des prédictions vs réel
        st.subheader("📈 Prédictions vs Observations")
        
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
                title='Top 10 Variables les Plus Importantes'
            )
            
            fig_imp.update_traces(marker_color='#E4032E')
            
            st.plotly_chart(fig_imp, use_container_width=True)
        
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
        
        # Tableau détaillé des prédictions
        with st.expander("📋 Tableau Complet des Prédictions"):
            st.dataframe(
                df_predictions.pivot(
                    index='health_area',
                    columns='week_',
                    values='predicted_cases'
                ).fillna(0).round(1),
                use_container_width=True
            )
        
        # Téléchargements
        st.subheader("💾 Téléchargements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_predictions.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les prédictions (CSV)",
                data=csv,
                file_name=f"predictions_paludisme_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
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
                label="🗺️ Télécharger la carte (GeoJSON)",
                data=geojson_str,
                file_name=f"carte_predictions_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json"
            )

# ============================================================
# FOOTER MSF
# ============================================================

st.markdown("---")
msf_footer()
