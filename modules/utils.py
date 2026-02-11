"""
============================================================
UTILS - FONCTIONS UTILITAIRES
Fonctions helpers réutilisables
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime

def safe_int(value):
    """Convertit en int en gérant les NaN"""
    try:
        return int(value) if not pd.isna(value) else 0
    except:
        return 0

def safe_float(value):
    """Convertit en float en gérant les NaN"""
    try:
        return float(value) if not pd.isna(value) else 0.0
    except:
        return 0.0

def format_large_number(n):
    """Formate un grand nombre avec séparateurs"""
    try:
        return f"{int(n):,}".replace(',', ' ')
    except:
        return "N/A"

def calculate_epi_week(date):
    """Calcule la semaine épidémiologique depuis une date"""
    return date.isocalendar()[1]

def week_to_date(year, week):
    """Convertit année + semaine épidémio en date"""
    return datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
