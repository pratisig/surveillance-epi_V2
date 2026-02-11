"""
============================================================
GEO LOADER - CHARGEMENT DONNÉES GÉOGRAPHIQUES
Gère GeoJSON, Shapefile, ZIP pour les aires de santé
============================================================
"""

import streamlit as st
import geopandas as gpd
import zipfile
import tempfile
import os
from io import BytesIO

class GeoLoader:
    """Gestionnaire de chargement des données géographiques"""
    
    @staticmethod
    def load_from_file(uploaded_file):
        """
        Charge un fichier géographique (GeoJSON, Shapefile, ZIP)
        
        Args:
            uploaded_file: Fichier uploadé via st.file_uploader
            
        Returns:
            GeoDataFrame ou None si erreur
        """
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == 'geojson':
                return gpd.read_file(uploaded_file)
            
            elif file_ext == 'zip':
                return GeoLoader._load_from_zip(uploaded_file)
            
            elif file_ext == 'shp':
                st.error("⚠️ Pour un Shapefile, uploadez un fichier ZIP contenant tous les composants (.shp, .shx, .dbf, .prj)")
                return None
            
            else:
                st.error(f"❌ Format '{file_ext}' non supporté. Utilisez GeoJSON ou ZIP (shapefile).")
                return None
                
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier géographique : {str(e)}")
            return None
    
    @staticmethod
    def _load_from_zip(zip_file):
        """Charge un shapefile depuis un fichier ZIP"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extraire le ZIP
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(tmpdir)
            
            # Trouver le fichier .shp
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            
            if not shp_files:
                st.error("❌ Aucun fichier .shp trouvé dans le ZIP")
                return None
            
            if len(shp_files) > 1:
                st.warning(f"⚠️ Plusieurs fichiers .shp trouvés. Utilisation de '{shp_files[0]}'")
            
            shp_path = os.path.join(tmpdir, shp_files[0])
            return gpd.read_file(shp_path)
    
    @staticmethod
    def load_local_ao_hltharea(iso3_country):
        """
        Charge le fichier local ao_hlthArea.zip pour un pays spécifique
        
        Args:
            iso3_country (str): Code ISO3 du pays (ex: "ner", "bfa", "mli", "mrt")
            
        Returns:
            GeoDataFrame filtré sur le pays
        """
        try:
            zip_path = "ao_hlthArea.zip"
            
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier '{zip_path}' introuvable dans le dossier de l'application")
                return None
            
            # Charger tout le fichier
            gdf = gpd.read_file(f"zip://{zip_path}")
            
            # Filtrer sur le pays
            if 'iso3' in gdf.columns:
                gdf = gdf[gdf['iso3'].str.lower() == iso3_country.lower()].copy()
            
            if len(gdf) == 0:
                st.error(f"❌ Aucune donnée trouvée pour le pays '{iso3_country}'")
                return None
            
            return gdf
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de ao_hlthArea.zip : {str(e)}")
            return None
    
    @staticmethod
    def validate_geodata(gdf, required_cols=['health_area', 'geometry']):
        """
        Valide qu'un GeoDataFrame contient les colonnes nécessaires
        
        Args:
            gdf (GeoDataFrame): Données à valider
            required_cols (list): Colonnes obligatoires
            
        Returns:
            (bool, str): (valide, message d'erreur)
        """
        if gdf is None:
            return False, "GeoDataFrame vide"
        
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        
        if missing_cols:
            return False, f"Colonnes manquantes : {', '.join(missing_cols)}"
        
        if not isinstance(gdf, gpd.GeoDataFrame):
            return False, "Le fichier ne contient pas de géométries valides"
        
        return True, "OK"
    
    @staticmethod
    def get_geodata_info(gdf):
        """Retourne des informations sur les données géographiques"""
        if gdf is None:
            return None
        
        info = {
            'n_features': len(gdf),
            'crs': str(gdf.crs),
            'bounds': gdf.total_bounds,
            'columns': list(gdf.columns),
            'geom_type': gdf.geometry.geom_type.unique().tolist(),
            'has_iso3': 'iso3' in gdf.columns,
            'has_health_area': 'health_area' in gdf.columns
        }
        
        if 'iso3' in gdf.columns:
            info['countries'] = gdf['iso3'].unique().tolist()
        
        return info
