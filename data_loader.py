"""
============================================================
DATA LOADER - MODULE CENTRALISÉ DE CHARGEMENT DES DONNÉES
Gestion des données partagées entre Paludisme et Rougeole
============================================================
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import ee
import json
import requests
from datetime import datetime, timedelta
import zipfile
import tempfile
import os
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. INITIALISATION GOOGLE EARTH ENGINE
# ============================================================

@st.cache_resource
def init_gee():
    """
    Initialise Google Earth Engine pour accéder à WorldPop
    Retourne True si succès, False sinon
    """
    try:
        # Tentative avec service account (secrets Streamlit)
        key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"],
            key_data=json.dumps(key_dict)
        )
        ee.Initialize(credentials)
        return True
    except:
        try:
            # Tentative avec authentification par défaut
            ee.Initialize()
            return True
        except:
            return False

# ============================================================
# 2. CHARGEMENT DES AIRES DE SANTÉ (SHAPEFILES/GEOJSON)
# ============================================================

@st.cache_data
def load_shapefile_from_zip(zip_path, iso3_filter=None):
    """
    Charge un shapefile depuis un fichier ZIP

    Args:
        zip_path (str): Chemin vers le fichier ZIP
        iso3_filter (str): Code ISO3 pour filtrer (ex: 'ner', 'bfa')

    Returns:
        GeoDataFrame: Aires de santé chargées
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]

            if not shp_files:
                st.error("❌ Aucun fichier .shp trouvé dans le ZIP")
                return None

            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf = gpd.read_file(shp_path)

            # Normalisation des colonnes
            gdf.columns = gdf.columns.str.lower()

            # Filtrage par pays si demandé
            if iso3_filter and 'iso3' in gdf.columns:
                gdf = gdf[gdf['iso3'] == iso3_filter].copy()

            # Vérification colonne health_area
            if 'health_area' not in gdf.columns:
                st.error("❌ Colonne 'health_area' manquante")
                return None

            # Reprojection en WGS84 si nécessaire
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            return gdf

    except Exception as e:
        st.error(f"❌ Erreur chargement shapefile : {str(e)}")
        return None

def load_geojson_or_shapefile(uploaded_file):
    """
    Charge un fichier GeoJSON ou Shapefile uploadé

    Args:
        uploaded_file: Fichier uploadé via st.file_uploader

    Returns:
        GeoDataFrame: Aires de santé chargées
    """
    try:
        if uploaded_file.name.endswith('.geojson') or uploaded_file.name.endswith('.json'):
            gdf = gpd.read_file(uploaded_file)

        elif uploaded_file.name.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, 'upload.zip')
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)

                shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                if shp_files:
                    gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                else:
                    st.error("❌ Aucun .shp trouvé dans le ZIP")
                    return None

        elif uploaded_file.name.endswith('.shp'):
            gdf = gpd.read_file(uploaded_file)

        else:
            st.error("❌ Format non supporté")
            return None

        # Normalisation
        gdf.columns = gdf.columns.str.lower()

        if 'health_area' not in gdf.columns:
            st.error("❌ Colonne 'health_area' manquante")
            return None

        # Reprojection
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        return gdf

    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        return None

# ============================================================
# 3. DONNÉES CLIMATIQUES - NASA POWER API
# ============================================================

@st.cache_data(ttl=3600)
def get_nasa_power_data(lat, lon, start_date, end_date):
    """
    Récupère données climatiques depuis NASA POWER API

    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Date début 'YYYYMMDD'
        end_date (str): Date fin 'YYYYMMDD'

    Returns:
        DataFrame: Données climatiques quotidiennes
    """
    try:
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'T2M,PRECTOTCORR,RH2M',
            'community': 'AG',
            'longitude': lon,
            'latitude': lat,
            'start': start_date,
            'end': end_date,
            'format': 'JSON'
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            properties = data.get('properties', {}).get('parameter', {})

            if properties:
                df = pd.DataFrame({
                    'date': pd.to_datetime(list(properties['T2M'].keys()), format='%Y%m%d'),
                    'temperature': list(properties['T2M'].values()),
                    'precipitation': list(properties['PRECTOTCORR'].values()),
                    'humidity': list(properties['RH2M'].values())
                })

                # Remplacement valeurs manquantes (-999)
                df.replace(-999, np.nan, inplace=True)

                return df

        return None

    except Exception as e:
        st.warning(f"⚠️ NASA POWER API : {str(e)}")
        return None

def aggregate_climate_to_weekly(climate_df, week_column='week_'):
    """
    Agrège données climatiques quotidiennes en hebdomadaires

    Args:
        climate_df (DataFrame): Données quotidiennes
        week_column (str): Nom de la colonne semaine

    Returns:
        DataFrame: Données hebdomadaires
    """
    if climate_df is None or climate_df.empty:
        return None

    try:
        # Calcul semaine épidémiologique
        climate_df['year'] = climate_df['date'].dt.isocalendar().year
        climate_df['week'] = climate_df['date'].dt.isocalendar().week
        climate_df[week_column] = climate_df['year'].astype(str) + '-W' + climate_df['week'].astype(str).str.zfill(2)

        # Agrégation hebdomadaire
        weekly = climate_df.groupby(week_column).agg({
            'temperature': 'mean',
            'precipitation': 'sum',
            'humidity': 'mean'
        }).reset_index()

        weekly.columns = [week_column, 'temp_api', 'precip_api', 'humidity_api']

        return weekly

    except Exception as e:
        st.warning(f"⚠️ Agrégation hebdomadaire : {str(e)}")
        return None

# ============================================================
# 4. DONNÉES DÉMOGRAPHIQUES - WORLDPOP (GOOGLE EARTH ENGINE)
# ============================================================

@st.cache_data
def get_worldpop_data(gdf, year=2020):
    """
    Récupère données WorldPop pour chaque aire de santé

    Args:
        gdf (GeoDataFrame): Aires de santé
        year (int): Année des données (2020 par défaut)

    Returns:
        DataFrame: Données démographiques par aire
    """
    try:
        # Collections WorldPop
        collections = {
            'Pop_Totale': f'WorldPop/GP/100m/pop/{year}',
            'Pop_M_0_4': f'WorldPop/GP/100m/pop_age_sex_cons_unadj/WPGP_M_{year}_0',
            'Pop_F_0_4': f'WorldPop/GP/100m/pop_age_sex_cons_unadj/WPGP_F_{year}_0',
            'Pop_M_5_9': f'WorldPop/GP/100m/pop_age_sex_cons_unadj/WPGP_M_{year}_5',
            'Pop_F_5_9': f'WorldPop/GP/100m/pop_age_sex_cons_unadj/WPGP_F_{year}_5',
            'Pop_M_10_14': f'WorldPop/GP/100m/pop_age_sex_cons_unadj/WPGP_M_{year}_10',
            'Pop_F_10_14': f'WorldPop/GP/100m/pop_age_sex_cons_unadj/WPGP_F_{year}_10'
        }

        results = []

        for idx, row in gdf.iterrows():
            area_data = {'health_area': row['health_area']}

            # Conversion géométrie en feature EE
            geom_json = row['geometry'].__geo_interface__
            ee_geom = ee.Geometry(geom_json)
            area_km2 = ee_geom.area().divide(1e6).getInfo()

            # Extraction données pour chaque collection
            for var_name, collection_path in collections.items():
                try:
                    image = ee.Image(collection_path)
                    stats = image.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=ee_geom,
                        scale=100,
                        maxPixels=1e9
                    ).getInfo()

                    area_data[var_name] = stats.get('population', 0)
                except:
                    area_data[var_name] = 0

            # Calculs dérivés
            area_data['Pop_Enfants_0_14'] = sum([
                area_data.get('Pop_M_0_4', 0),
                area_data.get('Pop_F_0_4', 0),
                area_data.get('Pop_M_5_9', 0),
                area_data.get('Pop_F_5_9', 0),
                area_data.get('Pop_M_10_14', 0),
                area_data.get('Pop_F_10_14', 0)
            ])

            area_data['Densite_Pop'] = area_data['Pop_Totale'] / area_km2 if area_km2 > 0 else 0

            results.append(area_data)

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"❌ Erreur WorldPop : {str(e)}")
        return None

# ============================================================
# 5. FONCTION PRINCIPALE DE CHARGEMENT COORDONNÉ
# ============================================================

def load_shared_data(option_source, uploaded_file=None, iso3_filter=None, 
                     use_climate=False, use_worldpop=False, 
                     start_date=None, end_date=None):
    """
    Fonction principale pour charger toutes les données partagées

    Args:
        option_source (str): "local" ou "upload"
        uploaded_file: Fichier uploadé (si option_source="upload")
        iso3_filter (str): Code pays pour filtrage
        use_climate (bool): Activer données climatiques
        use_worldpop (bool): Activer données WorldPop
        start_date (datetime): Date début pour climat
        end_date (datetime): Date fin pour climat

    Returns:
        dict: {
            'gdf': GeoDataFrame aires de santé,
            'climate': DataFrame données climatiques (ou None),
            'population': DataFrame données démographiques (ou None),
            'gee_ok': bool statut GEE
        }
    """
    result = {
        'gdf': None,
        'climate': None,
        'population': None,
        'gee_ok': False
    }

    # 1. Chargement aires de santé
    if option_source == "local":
        zip_path = "data/ao_hlthArea.zip"
        if os.path.exists(zip_path):
            result['gdf'] = load_shapefile_from_zip(zip_path, iso3_filter)
        else:
            st.error(f"❌ Fichier {zip_path} introuvable")
            return result

    elif option_source == "upload" and uploaded_file:
        result['gdf'] = load_geojson_or_shapefile(uploaded_file)

    if result['gdf'] is None:
        return result

    st.success(f"✓ {len(result['gdf'])} aires de santé chargées")

    # 2. Initialisation GEE
    if use_worldpop:
        result['gee_ok'] = init_gee()
        if result['gee_ok']:
            st.sidebar.success("✓ GEE connecté")
        else:
            st.sidebar.warning("⚠️ GEE non disponible")

    # 3. Chargement données climatiques
    if use_climate and start_date and end_date:
        with st.spinner("🌦️ Chargement données climatiques NASA POWER..."):
            # Calcul centroïde moyen
            centroid = result['gdf'].geometry.centroid.unary_union.centroid

            climate_df = get_nasa_power_data(
                lat=centroid.y,
                lon=centroid.x,
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d')
            )

            if climate_df is not None:
                result['climate'] = aggregate_climate_to_weekly(climate_df)
                if result['climate'] is not None:
                    st.success(f"✓ Données climatiques : {len(result['climate'])} semaines")

    # 4. Chargement données WorldPop
    if use_worldpop and result['gee_ok']:
        with st.spinner("👥 Extraction données WorldPop (peut prendre 1-2 min)..."):
            result['population'] = get_worldpop_data(result['gdf'])
            if result['population'] is not None:
                st.success(f"✓ Données démographiques pour {len(result['population'])} aires")

    return result
