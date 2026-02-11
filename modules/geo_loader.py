"""
============================================================
GEO LOADER - CHARGEMENT DES DONNÉES GÉOGRAPHIQUES
Gère le chargement depuis fichiers locaux, uploads, et formats multiples
============================================================
"""

import streamlit as st
import geopandas as gpd
import os
import tempfile
import zipfile
from pathlib import Path

class GeoLoader:
    """Classe pour charger des données géographiques depuis différentes sources"""
    
    @staticmethod
    def load_from_file(uploaded_file):
        """
        Charge un GeoDataFrame depuis un fichier uploadé
        
        Args:
            uploaded_file: Fichier uploadé via st.file_uploader
            
        Returns:
            GeoDataFrame ou None si erreur
        """
        try:
            if uploaded_file.name.endswith('.geojson'):
                gdf = gpd.read_file(uploaded_file)
            
            elif uploaded_file.name.endswith('.zip'):
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, 'upload.zip')
                    
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(tmpdir)
                    
                    # Chercher fichier .shp
                    shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                    
                    if shp_files:
                        gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                    else:
                        st.error("❌ Aucun fichier .shp trouvé dans le ZIP")
                        return None
            
            elif uploaded_file.name.endswith('.shp'):
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Sauvegarder tous les fichiers du shapefile
                    shp_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(shp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    gdf = gpd.read_file(shp_path)
            
            else:
                st.error(f"❌ Format non supporté : {uploaded_file.name}")
                return None
            
            # Normaliser health_area
            if 'health_area' not in gdf.columns:
                for col in ['healtharea', 'HEALTHAREA', 'name_fr', 'NAME', 'nom', 'NOM', 'aire_sante', 'airesante']:
                    if col in gdf.columns:
                        gdf['health_area'] = gdf[col]
                        break
            
            # Valider géométries
            gdf = gdf[gdf.geometry.is_valid].copy()
            
            # WGS84
            if gdf.crs is None:
                gdf.set_crs('EPSG:4326', inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs('EPSG:4326')
            
            return gdf
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {str(e)}")
            return None
    
    @staticmethod
    def load_local_ao_hltharea(iso3_code):
        """
        Charge les aires de santé depuis le fichier local ao_hlthArea.zip
        
        Args:
            iso3_code (str): Code ISO3 du pays (ex: 'ner', 'bfa')
            
        Returns:
            GeoDataFrame ou None si erreur
        """
        try:
            # MULTIPLES CHEMINS POSSIBLES
            possible_paths = [
                "data/ao_hlthArea.zip",              # Dossier data/
                "ao_hlthArea.zip",                   # Racine du projet
                "../data/ao_hlthArea.zip",           # Un niveau au-dessus
                "../../data/ao_hlthArea.zip",        # Deux niveaux au-dessus
                str(Path.home() / "data" / "ao_hlthArea.zip"),  # Home directory
            ]
            
            # Chercher le fichier
            zip_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    zip_path = path
                    break
            
            if zip_path is None:
                st.error("❌ Fichier 'ao_hlthArea.zip' introuvable dans le dossier de l'application")
                st.info(f"""
                **📁 Chemins recherchés :**
                {chr(10).join(f'- {p}' for p in possible_paths)}
                
                **💡 Solutions :**
                1. Créer un dossier `data/` à la racine du projet
                2. Placer `ao_hlthArea.zip` dans ce dossier
                3. OU utiliser l'option "Upload personnalisé"
                """)
                return None
            
            st.info(f"✅ Fichier trouvé : {zip_path}")
            
            # Charger avec geopandas (supporte zip://)
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tmpdir)
                
                # Chercher fichier .shp
                shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                
                if not shp_files:
                    st.error("❌ Aucun fichier .shp trouvé dans ao_hlthArea.zip")
                    return None
                
                shp_path = os.path.join(tmpdir, shp_files[0])
                gdf_full = gpd.read_file(shp_path)
            
            # Filtrer par ISO3
            iso3_col = None
            for col in ['iso3', 'ISO3', 'iso_code', 'ISOCODE', 'countryiso', 'COUNTRYISO']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break
            
            if iso3_col is None:
                st.warning(f"⚠️ Colonne ISO3 non trouvée. Colonnes disponibles : {', '.join(gdf_full.columns)}")
                return gdf_full  # Retourner tout si pas de filtre possible
            
            # Filtrer
            gdf = gdf_full[gdf_full[iso3_col].str.lower() == iso3_code.lower()].copy()
            
            if gdf.empty:
                st.warning(f"⚠️ Aucune aire trouvée pour le code ISO3 '{iso3_code}'")
                st.info(f"Codes ISO3 disponibles : {', '.join(gdf_full[iso3_col].unique())}")
                return None
            
            # Normaliser health_area
            if 'health_area' not in gdf.columns:
                for col in ['healtharea', 'HEALTHAREA', 'name_fr', 'NAME', 'nom', 'NOM', 'aire_sante']:
                    if col in gdf.columns:
                        gdf['health_area'] = gdf[col]
                        break
                else:
                    # Si aucune colonne trouvée, créer des noms génériques
                    gdf['health_area'] = [f'Aire_{i+1}' for i in range(len(gdf))]
            
            # Valider géométries
            gdf = gdf[gdf.geometry.is_valid].copy()
            
            # WGS84
            if gdf.crs is None:
                gdf.set_crs('EPSG:4326', inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs('EPSG:4326')
            
            return gdf
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de ao_hlthArea.zip : {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    @staticmethod
    def validate_geodata(gdf):
        """
        Valide un GeoDataFrame
        
        Args:
            gdf (GeoDataFrame): GeoDataFrame à valider
            
        Returns:
            tuple: (bool, str) - (is_valid, message)
        """
        if gdf is None or gdf.empty:
            return False, "GeoDataFrame vide"
        
        if 'geometry' not in gdf.columns:
            return False, "Colonne 'geometry' manquante"
        
        if 'health_area' not in gdf.columns:
            return False, "Colonne 'health_area' manquante"
        
        # Vérifier géométries valides
        invalid_geoms = (~gdf.geometry.is_valid).sum()
        if invalid_geoms > 0:
            return False, f"{invalid_geoms} géométries invalides détectées"
        
        # Vérifier CRS
        if gdf.crs is None:
            return False, "CRS non défini"
        
        return True, "GeoDataFrame valide"
    
    @staticmethod
    def get_geodata_info(gdf):
        """
        Extrait des informations sur un GeoDataFrame
        
        Args:
            gdf (GeoDataFrame): GeoDataFrame
            
        Returns:
            dict: Informations (n_features, columns, crs, bounds)
        """
        if gdf is None or gdf.empty:
            return {
                'n_features': 0,
                'columns': [],
                'crs': None,
                'bounds': None
            }
        
        return {
            'n_features': len(gdf),
            'columns': list(gdf.columns),
            'crs': str(gdf.crs) if gdf.crs else None,
            'bounds': gdf.total_bounds.tolist() if hasattr(gdf, 'total_bounds') else None
        }
