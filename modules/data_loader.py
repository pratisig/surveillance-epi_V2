"""
============================================================
DATA MANAGER - GESTIONNAIRE CENTRALISÉ DES DONNÉES
Gère toutes les sources de données pour Paludisme et Rougeole
============================================================
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    """
    Gestionnaire centralisé de toutes les données de la plateforme.
    Utilise st.session_state pour partager les données entre applications.
    """
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialise les variables de session si elles n'existent pas"""
        if 'dm_initialized' not in st.session_state:
            st.session_state.dm_initialized = True
            st.session_state.dm_geodata = None
            st.session_state.dm_climate_data = None
            st.session_state.dm_worldpop_data = None
            st.session_state.dm_epidemio_data = None
            st.session_state.dm_vaccination_data = None
            st.session_state.dm_last_update = None
            st.session_state.dm_source_info = {}
    
    # ============================================================
    # DONNÉES GÉOGRAPHIQUES
    # ============================================================
    
    def set_geodata(self, gdf, source="upload"):
        """
        Enregistre les données géographiques dans session_state
        
        Args:
            gdf (GeoDataFrame): Données géographiques
            source (str): Source des données ("upload", "local", "demo")
        """
        st.session_state.dm_geodata = gdf
        st.session_state.dm_source_info['geodata'] = {
            'source': source,
            'n_features': len(gdf),
            'timestamp': datetime.now(),
            'columns': list(gdf.columns)
        }
        st.session_state.dm_last_update = datetime.now()
    
    def get_geodata(self):
        """Récupère les données géographiques"""
        return st.session_state.dm_geodata
    
    def has_geodata(self):
        """Vérifie si des données géographiques sont chargées"""
        return st.session_state.dm_geodata is not None
    
    # ============================================================
    # DONNÉES CLIMATIQUES (NASA POWER)
    # ============================================================
    
    def set_climate_data(self, df):
        """
        Enregistre les données climatiques
        
        Args:
            df (DataFrame): Données climatiques (temp, precip, humidity)
        """
        st.session_state.dm_climate_data = df
        st.session_state.dm_source_info['climate'] = {
            'source': 'NASA POWER API',
            'n_records': len(df),
            'timestamp': datetime.now(),
            'variables': [col for col in df.columns if any(x in col for x in ['temp', 'precip', 'humid'])]
        }
        st.session_state.dm_last_update = datetime.now()
    
    def get_climate_data(self):
        """Récupère les données climatiques"""
        return st.session_state.dm_climate_data
    
    def has_climate_data(self):
        """Vérifie si des données climatiques sont chargées"""
        return st.session_state.dm_climate_data is not None
    
    # ============================================================
    # DONNÉES WORLDPOP (GEE)
    # ============================================================
    
    def set_worldpop_data(self, df):
        """
        Enregistre les données démographiques WorldPop
        
        Args:
            df (DataFrame): Données population (Pop_Totale, Pop_Enfants, etc.)
        """
        st.session_state.dm_worldpop_data = df
        st.session_state.dm_source_info['worldpop'] = {
            'source': 'Google Earth Engine (WorldPop)',
            'n_records': len(df),
            'timestamp': datetime.now(),
            'variables': [col for col in df.columns if 'Pop' in col]
        }
        st.session_state.dm_last_update = datetime.now()
    
    def get_worldpop_data(self):
        """Récupère les données WorldPop"""
        return st.session_state.dm_worldpop_data
    
    def has_worldpop_data(self):
        """Vérifie si des données WorldPop sont chargées"""
        return st.session_state.dm_worldpop_data is not None
    
    # ============================================================
    # DONNÉES ÉPIDÉMIOLOGIQUES
    # ============================================================
    
    def set_epidemio_data(self, df, disease="paludisme"):
        """
        Enregistre les données épidémiologiques
        
        Args:
            df (DataFrame): Linelists (cases, deaths, health_area, week)
            disease (str): "paludisme" ou "rougeole"
        """
        if 'dm_epidemio_data' not in st.session_state:
            st.session_state.dm_epidemio_data = {}
        
        st.session_state.dm_epidemio_data[disease] = df
        st.session_state.dm_source_info[f'epidemio_{disease}'] = {
            'disease': disease,
            'n_records': len(df),
            'timestamp': datetime.now(),
            'date_range': (df['week_'].min() if 'week_' in df.columns else None,
                          df['week_'].max() if 'week_' in df.columns else None),
            'total_cases': df['cases'].sum() if 'cases' in df.columns else None
        }
        st.session_state.dm_last_update = datetime.now()
    
    def get_epidemio_data(self, disease="paludisme"):
        """Récupère les données épidémiologiques pour une maladie"""
        if 'dm_epidemio_data' in st.session_state:
            return st.session_state.dm_epidemio_data.get(disease)
        return None
    
    def has_epidemio_data(self, disease="paludisme"):
        """Vérifie si des données épidémio sont chargées pour une maladie"""
        return (hasattr(st.session_state, 'dm_epidemio_data') and 
                disease in st.session_state.dm_epidemio_data)
    
    # ============================================================
    # DONNÉES VACCINATION
    # ============================================================
    
    def set_vaccination_data(self, df):
        """
        Enregistre les données de couverture vaccinale
        
        Args:
            df (DataFrame): Couverture vaccinale par aire de santé
        """
        st.session_state.dm_vaccination_data = df
        st.session_state.dm_source_info['vaccination'] = {
            'n_records': len(df),
            'timestamp': datetime.now(),
            'coverage_mean': df['Taux_Vaccination'].mean() if 'Taux_Vaccination' in df.columns else None
        }
        st.session_state.dm_last_update = datetime.now()
    
    def get_vaccination_data(self):
        """Récupère les données de vaccination"""
        return st.session_state.dm_vaccination_data
    
    def has_vaccination_data(self):
        """Vérifie si des données de vaccination sont chargées"""
        return st.session_state.dm_vaccination_data is not None
    
    # ============================================================
    # UTILITAIRES
    # ============================================================
    
    def get_summary(self):
        """Retourne un résumé de toutes les données chargées"""
        summary = {
            'last_update': st.session_state.dm_last_update,
            'geodata': self.has_geodata(),
            'climate': self.has_climate_data(),
            'worldpop': self.has_worldpop_data(),
            'epidemio_paludisme': self.has_epidemio_data('paludisme'),
            'epidemio_rougeole': self.has_epidemio_data('rougeole'),
            'vaccination': self.has_vaccination_data(),
            'source_info': st.session_state.dm_source_info
        }
        return summary
    
    def clear_all(self):
        """Réinitialise toutes les données"""
        st.session_state.dm_geodata = None
        st.session_state.dm_climate_data = None
        st.session_state.dm_worldpop_data = None
        st.session_state.dm_epidemio_data = None
        st.session_state.dm_vaccination_data = None
        st.session_state.dm_last_update = None
        st.session_state.dm_source_info = {}
    
    def clear_by_type(self, data_type):
        """
        Réinitialise un type de données spécifique
        
        Args:
            data_type (str): "geodata", "climate", "worldpop", "epidemio", "vaccination"
        """
        if data_type == "geodata":
            st.session_state.dm_geodata = None
        elif data_type == "climate":
            st.session_state.dm_climate_data = None
        elif data_type == "worldpop":
            st.session_state.dm_worldpop_data = None
        elif data_type == "epidemio":
            st.session_state.dm_epidemio_data = None
        elif data_type == "vaccination":
            st.session_state.dm_vaccination_data = None
        
        # Supprimer les infos sources
        keys_to_remove = [k for k in st.session_state.dm_source_info.keys() if data_type in k]
        for key in keys_to_remove:
            del st.session_state.dm_source_info[key]
