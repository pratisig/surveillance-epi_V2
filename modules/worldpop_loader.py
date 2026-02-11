"""
============================================================
WORLDPOP LOADER - GOOGLE EARTH ENGINE
Récupération données démographiques WorldPop
============================================================
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import ee
import json

class WorldPopLoader:
    """Gestionnaire de récupération des données WorldPop via GEE"""
    
    @staticmethod
    def init_gee():
        """Initialise Google Earth Engine"""
        try:
            # Méthode 1 : Service Account (Streamlit Cloud)
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"],
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            return True
        except:
            try:
                # Méthode 2 : Authentification locale
                ee.Initialize()
                return True
            except:
                return False
    
    @staticmethod
    @st.cache_data(ttl=3600*24*7)  # Cache 7 jours
    def fetch_worldpop_for_area(_geometry, year=2020):
        """
        Récupère les données WorldPop pour une géométrie
        
        Args:
            _geometry: Géométrie Shapely
            year (int): Année des données
        
        Returns:
            dict avec Pop_Totale, Pop_Enfants_0_14, Pop_M/F par tranche d'âge
        """
        try:
            # Convertir la géométrie en format GEE
            geom_json = json.loads(gpd.GeoSeries([_geometry]).to_json())
            ee_geom = ee.Geometry(geom_json['features'][0]['geometry'])
            
            # Charger les datasets WorldPop
            pop_total = ee.ImageCollection("WorldPop/GP/100m/pop").filter(
                ee.Filter.eq('year', year)
            ).mosaic()
            
            pop_age_sex = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex_cons_unadj").filter(
                ee.Filter.eq('year', year)
            ).mosaic()
            
            # Statistiques pour population totale
            stats_total = pop_total.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=ee_geom,
                scale=100,
                maxPixels=1e9
            ).getInfo()
            
            # Population par tranches d'âge
            bands_enfants = ['M_0', 'M_1', 'M_5', 'M_10', 'F_0', 'F_1', 'F_5', 'F_10']
            stats_age = pop_age_sex.select(bands_enfants).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=ee_geom,
                scale=100,
                maxPixels=1e9
            ).getInfo()
            
            # Calculer superficie pour densité
            area_km2 = ee_geom.area().divide(1e6).getInfo()
            
            pop_totale = stats_total.get('population', 0)
            pop_enfants_0_14 = sum([stats_age.get(band, 0) for band in bands_enfants])
            
            result = {
                'Pop_Totale': pop_totale,
                'Pop_Enfants_0_14': pop_enfants_0_14,
                'Densite_Pop': pop_totale / area_km2 if area_km2 > 0 else 0,
                'Superficie_km2': area_km2
            }
            
            # Ajouter détail par tranches d'âge
            for band in bands_enfants:
                result[f'Pop_{band}'] = stats_age.get(band, 0)
            
            return result
            
        except Exception as e:
            st.warning(f"⚠️ Erreur WorldPop pour cette aire : {str(e)}")
            return None
    
    @staticmethod
    def fetch_worldpop_for_geodata(gdf, year=2020, progress_bar=True):
        """
        Récupère WorldPop pour toutes les aires de santé
        
        Args:
            gdf (GeoDataFrame): Données géographiques
            year (int): Année des données
            progress_bar (bool): Afficher progression
        
        Returns:
            DataFrame avec données démographiques
        """
        if not WorldPopLoader.init_gee():
            st.error("❌ Impossible d'initialiser Google Earth Engine")
            return None
        
        all_data = []
        
        if progress_bar:
            progress = st.progress(0)
            status = st.empty()
        
        for idx, row in gdf.iterrows():
            if progress_bar:
                pct = (idx + 1) / len(gdf)
                progress.progress(pct)
                status.text(f"👥 Récupération WorldPop : {idx+1}/{len(gdf)} aires...")
            
            pop_data = WorldPopLoader.fetch_worldpop_for_area(row.geometry, year)
            
            if pop_data:
                pop_data['health_area'] = row['health_area']
                all_data.append(pop_data)
        
        if progress_bar:
            progress.empty()
            status.empty()
        
        if not all_data:
            st.error("❌ Aucune donnée WorldPop récupérée")
            return None
        
        result = pd.DataFrame(all_data)
        st.success(f"✅ Données WorldPop récupérées : {len(result)} aires de santé")
        
        return result
