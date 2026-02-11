"""
============================================================
UTILS - FONCTIONS UTILITAIRES COMMUNES
Fonctions partagées entre les applications
============================================================
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ============================================================
# FONCTIONS DE SÉCURITÉ POUR AFFICHAGE
# ============================================================

def safe_int(value):
    """Convertit en entier de manière sécurisée"""
    try:
        return int(value) if pd.notna(value) and not np.isinf(value) else 0
    except:
        return 0

def safe_float(value):
    """Convertit en float de manière sécurisée"""
    try:
        return float(value) if pd.notna(value) and not np.isinf(value) else 0.0
    except:
        return 0.0

# ============================================================
# CALCULS DE SEMAINES ÉPIDÉMIOLOGIQUES
# ============================================================

def get_epi_week(date):
    """
    Calcule la semaine épidémiologique au format ISO

    Args:
        date (datetime): Date

    Returns:
        str: Semaine au format 'YYYY-WXX'
    """
    if pd.isna(date):
        return None

    iso_year, iso_week, _ = date.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"

def parse_epi_week(week_str):
    """
    Parse une chaîne de semaine épidémiologique

    Args:
        week_str (str): Semaine au format 'YYYY-WXX' ou 'WXX' ou 'SXX'

    Returns:
        tuple: (année, numéro_semaine)
    """
    if pd.isna(week_str):
        return None, None

    week_str = str(week_str).strip()

    # Format YYYY-WXX
    if '-W' in week_str or '-w' in week_str:
        parts = week_str.upper().split('-W')
        return int(parts[0]), int(parts[1])

    # Format WXX ou SXX
    elif week_str.upper().startswith('W') or week_str.upper().startswith('S'):
        return datetime.now().year, int(week_str[1:])

    # Numéro seul
    else:
        try:
            return datetime.now().year, int(week_str)
        except:
            return None, None

def generate_future_weeks(last_week_str, n_weeks):
    """
    Génère n semaines futures à partir d'une semaine donnée

    Args:
        last_week_str (str): Dernière semaine connue 'YYYY-WXX'
        n_weeks (int): Nombre de semaines à générer

    Returns:
        list: Liste de semaines au format 'YYYY-WXX'
    """
    year, week = parse_epi_week(last_week_str)

    if year is None:
        return []

    future_weeks = []
    current_date = datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w")

    for i in range(1, n_weeks + 1):
        next_date = current_date + timedelta(weeks=i)
        future_weeks.append(get_epi_week(next_date))

    return future_weeks

# ============================================================
# CALCULS ÉPIDÉMIOLOGIQUES
# ============================================================

def calculate_incidence_rate(cases, population, per=10000):
    """
    Calcule le taux d'incidence

    Args:
        cases (int): Nombre de cas
        population (int): Population
        per (int): Pour combien d'habitants (10000 par défaut)

    Returns:
        float: Taux d'incidence
    """
    if population == 0 or pd.isna(population):
        return 0.0

    return (cases / population) * per

def calculate_attack_rate(cases, susceptible_population, per=10000):
    """
    Calcule le taux d'attaque

    Args:
        cases (int): Nombre de cas
        susceptible_population (int): Population susceptible
        per (int): Pour combien d'habitants

    Returns:
        float: Taux d'attaque
    """
    return calculate_incidence_rate(cases, susceptible_population, per)

def calculate_case_fatality_rate(deaths, cases):
    """
    Calcule le taux de létalité (CFR)

    Args:
        deaths (int): Nombre de décès
        cases (int): Nombre de cas

    Returns:
        float: CFR en pourcentage
    """
    if cases == 0 or pd.isna(cases):
        return 0.0

    return (deaths / cases) * 100

# ============================================================
# ENRICHISSEMENT DES DONNÉES
# ============================================================

def merge_climate_data(df, climate_df, on='week_'):
    """
    Fusionne les données climatiques avec un DataFrame principal

    Args:
        df (DataFrame): Données principales
        climate_df (DataFrame): Données climatiques
        on (str): Colonne de jointure

    Returns:
        DataFrame: Données fusionnées
    """
    if climate_df is None or climate_df.empty:
        return df

    return df.merge(climate_df, on=on, how='left')

def merge_population_data(df, population_df, on='health_area'):
    """
    Fusionne les données démographiques avec un DataFrame principal

    Args:
        df (DataFrame): Données principales
        population_df (DataFrame): Données démographiques
        on (str): Colonne de jointure

    Returns:
        DataFrame: Données fusionnées
    """
    if population_df is None or population_df.empty:
        return df

    return df.merge(population_df, on=on, how='left')

# ============================================================
# VALIDATION DES DONNÉES
# ============================================================

def validate_required_columns(df, required_cols, df_name="DataFrame"):
    """
    Valide la présence de colonnes requises

    Args:
        df (DataFrame): DataFrame à valider
        required_cols (list): Liste des colonnes requises
        df_name (str): Nom du DataFrame pour messages

    Returns:
        bool: True si toutes les colonnes présentes, False sinon
    """
    if df is None:
        st.error(f"❌ {df_name} est vide")
        return False

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"❌ Colonnes manquantes dans {df_name}: {', '.join(missing)}")
        return False

    return True

def clean_numeric_column(series, fill_value=0):
    """
    Nettoie une colonne numérique (NaN, inf, etc.)

    Args:
        series (Series): Série à nettoyer
        fill_value: Valeur de remplacement

    Returns:
        Series: Série nettoyée
    """
    series = pd.to_numeric(series, errors='coerce')
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.fillna(fill_value)
    return series

# ============================================================
# EXPORT DES DONNÉES
# ============================================================

def create_download_link(df, filename, file_label="Télécharger"):
    """
    Crée un bouton de téléchargement pour un DataFrame

    Args:
        df (DataFrame): Données à exporter
        filename (str): Nom du fichier
        file_label (str): Texte du bouton
    """
    csv = df.to_csv(index=False, encoding='utf-8-sig')

    st.download_button(
        label=f"📥 {file_label}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# ============================================================
# FORMATAGE POUR AFFICHAGE
# ============================================================

def format_large_number(num):
    """
    Formate un grand nombre avec séparateurs

    Args:
        num: Nombre à formater

    Returns:
        str: Nombre formaté
    """
    try:
        return f"{int(num):,}".replace(',', ' ')
    except:
        return "N/A"

def format_percentage(value, decimals=1):
    """
    Formate un pourcentage

    Args:
        value (float): Valeur à formater
        decimals (int): Nombre de décimales

    Returns:
        str: Pourcentage formaté
    """
    try:
        return f"{value:.{decimals}f}%"
    except:
        return "N/A"

def format_week_label(week_str):
    """
    Formate une semaine épidémiologique pour affichage

    Args:
        week_str (str): Semaine 'YYYY-WXX'

    Returns:
        str: Libellé formaté 'SXX (YYYY)'
    """
    year, week = parse_epi_week(week_str)

    if year and week:
        return f"S{week:02d} ({year})"

    return str(week_str)

# ============================================================
# GESTION DES ALERTES
# ============================================================

def display_alert(message, alert_type="info"):
    """
    Affiche une alerte formatée

    Args:
        message (str): Message à afficher
        alert_type (str): Type d'alerte ('info', 'success', 'warning', 'error')
    """
    alert_classes = {
        'success': 'alert-success',
        'warning': 'alert-warning',
        'error': 'alert-danger',
        'info': 'alert-info'
    }

    css_class = alert_classes.get(alert_type, 'alert-info')

    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

# ============================================================
# DÉTECTION ÉPIDÉMIES
# ============================================================

def detect_epidemic_threshold(cases_history, multiplier=2, window=4):
    """
    Détecte le seuil épidémique basé sur l'historique

    Args:
        cases_history (list/array): Historique des cas
        multiplier (float): Multiplicateur pour le seuil (2 par défaut)
        window (int): Fenêtre pour moyenne mobile

    Returns:
        float: Seuil épidémique
    """
    if len(cases_history) < window:
        return np.mean(cases_history) * multiplier if len(cases_history) > 0 else 10

    # Moyenne mobile des dernières semaines
    baseline = np.mean(cases_history[-window:])

    return baseline * multiplier

def classify_epidemic_level(current_cases, threshold):
    """
    Classifie le niveau épidémique

    Args:
        current_cases (int): Cas actuels
        threshold (float): Seuil épidémique

    Returns:
        dict: {
            'level': str ('normal', 'alert', 'epidemic'),
            'emoji': str,
            'color': str
        }
    """
    if current_cases < threshold * 0.75:
        return {
            'level': 'normal',
            'emoji': '🟢',
            'color': 'green'
        }
    elif current_cases < threshold:
        return {
            'level': 'alert',
            'emoji': '🟡',
            'color': 'orange'
        }
    else:
        return {
            'level': 'epidemic',
            'emoji': '🔴',
            'color': 'red'
        }
