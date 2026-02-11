"""
============================================================
CLIMATE LOADER - NASA POWER API
Récupération données climatiques (température, précipitations, humidité)
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

class ClimateLoader:
    """Gestionnaire de récupération des données climatiques NASA POWER"""
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    @staticmethod
    @st.cache_data(ttl=3600*24)  # Cache 24h
    def fetch_climate_data(lat, lon, start_date, end_date, params=['T2M', 'PRECTOTCORR', 'RH2M']):
        """
        Récupère les données climatiques pour un point géographique
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            start_date (str): Date début (YYYYMMDD)
            end_date (str): Date fin (YYYYMMDD)
            params (list): Paramètres à récupérer
                - T2M : Température à 2m (°C)
                - PRECTOTCORR : Précipitations (mm/day)
                - RH2M : Humidité relative à 2m (%)
        
        Returns:
            DataFrame avec les données climatiques
        """
        try:
            params_str = ",".join(params)
            
            url = (
                f"{ClimateLoader.BASE_URL}?"
                f"parameters={params_str}&"
                f"community=AG&"
                f"longitude={lon}&"
                f"latitude={lat}&"
                f"start={start_date}&"
                f"end={end_date}&"
                f"format=JSON"
            )
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extraire les données
            records = data['properties']['parameter']
            
            df = pd.DataFrame(records)
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            # Renommer les colonnes
            rename_map = {
                'T2M': 'temp_api',
                'PRECTOTCORR': 'precip_api',
                'RH2M': 'humidity_api'
            }
            df.rename(columns=rename_map, inplace=True)
            
            # Remplacer -999 par NaN
            for col in df.columns:
                if col != 'date':
                    df[col] = df[col].replace(-999, np.nan)
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Erreur NASA POWER pour lat={lat}, lon={lon}: {str(e)}")
            return None
    
    @staticmethod
    def fetch_climate_for_geodata(gdf, start_date, end_date, progress_bar=True):
        """
        Récupère les données climatiques pour chaque aire de santé
        
        Args:
            gdf (GeoDataFrame): Données géographiques avec geometries
            start_date (datetime): Date de début
            end_date (datetime): Date de fin
            progress_bar (bool): Afficher une barre de progression
        
        Returns:
            DataFrame avec données climatiques par aire et date
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        all_data = []
        
        if progress_bar:
            progress = st.progress(0)
            status = st.empty()
        
        for idx, row in gdf.iterrows():
            if progress_bar:
                pct = (idx + 1) / len(gdf)
                progress.progress(pct)
                status.text(f"📡 Récupération données climatiques : {idx+1}/{len(gdf)} aires...")
            
            centroid = row.geometry.centroid
            lat, lon = centroid.y, centroid.x
            
            climate_df = ClimateLoader.fetch_climate_data(lat, lon, start_str, end_str)
            
            if climate_df is not None:
                climate_df['health_area'] = row['health_area']
                all_data.append(climate_df)
        
        if progress_bar:
            progress.empty()
            status.empty()
        
        if not all_data:
            st.error("❌ Aucune donnée climatique récupérée")
            return None
        
        result = pd.concat(all_data, ignore_index=True)
        st.success(f"✅ Données climatiques récupérées : {len(result)} enregistrements")
        
        return result
    
    @staticmethod
    def aggregate_to_weekly(climate_df, date_col='date'):
        """
        Agrège les données quotidiennes en semaines épidémiologiques
        
        Args:
            climate_df (DataFrame): Données climatiques quotidiennes
            date_col (str): Nom de la colonne date
        
        Returns:
            DataFrame agrégé par semaine
        """
        if climate_df is None or climate_df.empty:
            return None
        
        climate_df = climate_df.copy()
        climate_df['week_'] = climate_df[date_col].dt.isocalendar().week
        climate_df['year'] = climate_df[date_col].dt.year
        
        # Agrégation
        agg_dict = {}
        if 'temp_api' in climate_df.columns:
            agg_dict['temp_api'] = 'mean'
        if 'precip_api' in climate_df.columns:
            agg_dict['precip_api'] = 'sum'  # Cumul hebdomadaire
        if 'humidity_api' in climate_df.columns:
            agg_dict['humidity_api'] = 'mean'
        
        weekly = climate_df.groupby(['health_area', 'year', 'week_']).agg(agg_dict).reset_index()
        
        return weekly
