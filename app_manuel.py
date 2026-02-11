"""
============================================================
MANUEL D'UTILISATION - VERSION 4.0 MODULAIRE
Documentation complète Paludisme + Rougeole avec branding MSF
============================================================
"""

import streamlit as st
import sys
import os

# Ajouter le dossier modules au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Imports des modules partagés
from modules.ui_components import apply_msf_branding, msf_header, msf_footer

# ============================================================
# APPLIQUER LE BRANDING MSF
# ============================================================
apply_msf_branding()

# CSS spécifique au manuel
st.markdown("""
<style>
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .step-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #E4032E;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .code-example {
        background: #263238;
        color: #aed581;
        padding: 0.8rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        margin: 0.5rem 0;
    }
    
    .feature-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .feature-table th {
        background: #E4032E;
        color: white;
        padding: 0.8rem;
        text-align: left;
        font-weight: bold;
    }
    
    .feature-table td {
        padding: 0.6rem;
        border-bottom: 1px solid #ddd;
    }
    
    .feature-table tr:hover {
        background: #f5f5f5;
    }
    
    .glossary-term {
        background: #e3f2fd;
        padding: 0.8rem;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
        margin: 0.8rem 0;
    }
    
    .glossary-term strong {
        color: #1565c0;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
msf_header(
    "📚 Manuel d'Utilisation - Plateforme de Surveillance Épidémiologique",
    "Guide complet pour l'utilisation des modules Paludisme et Rougeole"
)

st.markdown("---")

# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================
tab_palu, tab_rougeole, tab_glossaire, tab_architecture = st.tabs([
    "🦟 Paludisme",
    "🦠 Rougeole",
    "📖 Glossaire & Méthodologie",
    "🏗️ Architecture v4.0"
])

# ============================================================
# TAB 1 : PALUDISME
# ============================================================
with tab_palu:
    st.markdown('<div class="section-header">🦟 Application de Surveillance du Paludisme</div>', unsafe_allow_html=True)
    
    # Section 1 : Introduction
    st.markdown("## 📋 Vue d'Ensemble")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4 style="color:#E4032E; margin-top:0;">C'est quoi EpiPalu Predict ?</h4>
        
        **EpiPalu Predict** est un outil intelligent qui vous aide à :
        
        - 🗺️ **Cartographier** les zones à risque paludisme
        - 📊 **Analyser** les facteurs de transmission
        - 🔮 **Prédire** l'évolution future des cas
        - 🎯 **Cibler** les interventions prioritaires
        
        **Nouveauté v4.0** : Architecture modulaire avec données partagées entre Paludisme et Rougeole !
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
        <h4 style="color:#2e7d32; margin-top:0;">✨ Points Forts</h4>
        
        - ✅ **Gratuit** et **open source**
        - ✅ **Sans programmation** requise
        - ✅ **Données réutilisables** entre applications
        - ✅ **API externes** : NASA POWER, WorldPop
        - ✅ **Machine Learning** intégré
        - ✅ **Validation temporelle** rigoureuse
        </div>
        """, unsafe_allow_html=True)
    
    # Section 2 : Démarrage rapide
    st.markdown('<div class="section-header">🚀 Démarrage Rapide (5 minutes)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 1️⃣ : Charger les Aires de Santé</h4>
    
    **📁 Dans la sidebar → 🗺️ Aires de Santé**
    
    - **Option A** : Upload votre fichier GeoJSON ou Shapefile (ZIP)
    - **Option B** : Utiliser les données de démo
    
    **Format requis :**
    - `health_area` : Nom de l'aire de santé
    - `geometry` : Géométrie (polygones)
    
    **Vous devez voir :**
    ```
    ✓ X aires de santé chargées
    ```
    
    💡 **Astuce v4.0** : Les aires chargées restent disponibles pour l'app Rougeole !
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 2️⃣ : Charger les Données Épidémiologiques</h4>
    
    **📁 Dans la sidebar → 📊 Données Épidémiologiques**
    
    Uploadez un fichier CSV avec **colonnes obligatoires** :
    
    <table class="feature-table">
        <tr>
            <th>Colonne</th>
            <th>Description</th>
            <th>Exemple</th>
        </tr>
        <tr>
            <td><code>health_area</code></td>
            <td>Nom de l'aire (doit correspondre au fichier géographique)</td>
            <td>Dakar Centre</td>
        </tr>
        <tr>
            <td><code>week_</code></td>
            <td>Numéro ou nom de semaine</td>
            <td>S01, 2024-W01</td>
        </tr>
        <tr>
            <td><code>cases</code></td>
            <td>Nombre de cas confirmés</td>
            <td>45</td>
        </tr>
        <tr>
            <td><code>deaths</code></td>
            <td>Nombre de décès (optionnel)</td>
            <td>2</td>
        </tr>
    </table>
    
    💡 **Astuce v4.0** : Les données épidémio sont stockées séparément pour Paludisme et Rougeole.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 3️⃣ : Activer les Données Climatiques (Optionnel mais Recommandé)</h4>
    
    **📁 Dans la sidebar → 🌡️ Données Climatiques**
    
    1. ☑️ Cocher **"Activer NASA POWER API"**
    2. Cliquer sur **"📡 Récupérer données climatiques"**
    3. Attendre la récupération (30-60 secondes)
    
    **Source : NASA POWER API** : Données météo depuis 1981, sans inscription
    
    **Pourquoi c'est important ?**
    
    Le paludisme est une **maladie climatosensible** :
    - 🌡️ **Température** : Optimal 25-30°C pour reproduction du parasite
    - 🌧️ **Pluies** : Créent des gîtes larvaires pour les moustiques
    - 💧 **Humidité** : Favorise survie des vecteurs
    
    **Impact sur prédictions :** +20-30% de précision avec climat vs sans climat
    
    💡 **Astuce v4.0** : Les données climatiques sont mises en cache et réutilisables !
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 4️⃣ : Activer WorldPop (Optionnel mais Recommandé)</h4>
    
    **📁 Dans la sidebar → 👥 Données Démographiques**
    
    1. ☑️ Cocher **"Activer WorldPop (GEE)"**
    2. Cliquer sur **"👥 Récupérer WorldPop"**
    3. Attendre l'extraction (1-2 minutes selon le nombre d'aires)
    
    **Source : Google Earth Engine** : Données populationnelles mondiales, résolution 100m
    
    **Prérequis :** Compte Google Earth Engine configuré (voir `✓ GEE connecté` en haut)
    
    **Variables extraites :**
    
    <table class="feature-table">
        <tr>
            <th>Variable</th>
            <th>Description</th>
            <th>Utilisation</th>
        </tr>
        <tr>
            <td>Pop_Totale</td>
            <td>Nombre total d'habitants</td>
            <td>Dénominateur taux d'incidence</td>
        </tr>
        <tr>
            <td>Pop_Enfants_0_14</td>
            <td>Population pédiatrique</td>
            <td>Estimation besoins en MII/médicaments</td>
        </tr>
        <tr>
            <td>Densite_Pop</td>
            <td>Habitants par km²</td>
            <td>Risque de transmission (densité élevée = plus de contacts)</td>
        </tr>
        <tr>
            <td>Pop_M/F_0_4, 5_9...</td>
            <td>Tranches d'âge par sexe</td>
            <td>Pyramide des âges (visualisation)</td>
        </tr>
    </table>
    
    💡 **Astuce v4.0** : Les données WorldPop sont partagées entre Paludisme et Rougeole !
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h4 style="color:#2e7d32; margin-top:0;">✅ Félicitations !</h4>
    
    Vous avez maintenant toutes les données nécessaires. L'interface affiche automatiquement :
    
    - 🗺️ **Carte interactive** avec popups détaillés
    - 📈 **Graphiques temporels** de l'évolution des cas
    - 📊 **Statistiques descriptives** par aire de santé
    - 🤖 **Bouton de modélisation** pour les prédictions
    </div>
    """, unsafe_allow_html=True)
    
    # Section 3 : Fonctionnalités avancées
    st.markdown('<div class="section-header">⚙️ Fonctionnalités Avancées</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🗺️ Cartographie Interactive
    
    **Ce que vous voyez :**
    - Choroplèthe coloré selon le nombre de cas
    - Popups enrichis avec toutes les données disponibles
    - Centroïdes des aires avec markers cliquables
    
    **Actions possibles :**
    - Zoomer/Dézoomer sur la carte
    - Cliquer sur une aire pour voir les détails
    - Exporter la carte (GeoJSON)
    
    ---
    
    ### 📊 Analyse des Données
    
    **Taux d'incidence** :
    ```
    Taux d'incidence = (Cas / Pop_Totale) × 10 000
    ```
    Permet de comparer le risque entre zones de tailles différentes
    
    **Risque pédiatrique** :
    ```
    Risque enfants = (Cas / Pop_Enfants_0_14) × 1 000
    ```
    Cible les zones où les enfants sont plus vulnérables
    
    ---
    
    ### 🤖 Modélisation Prédictive
    
    **Configuration :**
    1. **Horizon de prédiction** : 2 à 12 semaines
    2. **Algorithme** : Gradient Boosting (recommandé), Random Forest ou Régression Linéaire
    
    **Features utilisées :**
    - ⏰ **Temporelles** : Numéro de semaine, cycles saisonniers (sin/cos)
    - 📈 **Lags** : Cas des 1, 2, 4 semaines précédentes
    - 📊 **Moyennes mobiles** : MA2, MA4, MA8
    - 🌡️ **Climatiques** : Température, précipitations, humidité
    - 👥 **Démographiques** : Population, densité
    - 🌍 **Environnementales** : Altitude, distance rivières, zones inondables
    
    **Sorties du modèle :**
    - ✅ **Métriques** : R², MAE, RMSE
    - 📈 **Graphiques** : Prédictions vs Observations
    - 🔍 **Importance des variables** : Quelles features comptent le plus ?
    - 🗺️ **Carte des prédictions** : Visualisation spatiale
    - 💾 **Exports** : CSV et GeoJSON
    
    **Validation :**
    
    Le modèle utilise un **split temporel 80/20** pour éviter le sur-apprentissage :
    - 80% des données → Entraînement
    - 20% des données les plus récentes → Test
    
    **Interprétation du R² :**
    - R² > 0.80 → 🎯 Excellent modèle
    - R² > 0.60 → 👍 Bon modèle
    - R² < 0.60 → ⚠️ À améliorer (plus de données ou de features)
    """)
    
    # Section 4 : Cas d'usage
    st.markdown('<div class="section-header">💼 Cas d'Usage Terrain</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Cas 1 : Planification d'une Distribution de MII
    
    **Contexte :** Vous disposez de 50 000 moustiquaires à distribuer dans 100 aires de santé.
    
    **Étapes avec EpiPalu Predict :**
    
    1. Charger les aires et les cas des 3 derniers mois
    2. Activer WorldPop pour avoir la population enfants 0-14 ans
    3. Lancer une prédiction sur 2 mois
    4. Identifier les **10 aires** avec :
       - Plus haut risque prédit
       - Plus forte population pédiatrique
    5. Calculer les besoins :
       ```
       Besoins = Pop_Enfants_0_14 × 1.2 (1 MII pour 2 enfants + marge)
       ```
    
    **Résultat :** Distribution ciblée maximisant l'impact épidémiologique
    
    ---
    
    ### 🎯 Cas 2 : Riposte à une Flambée Épidémique
    
    **Contexte :** Pic de cas dans plusieurs districts, ressources limitées pour investigation.
    
    **Étapes :**
    
    1. Charger les données de la semaine en cours
    2. Activer les données climatiques
    3. Observer la carte des taux d'incidence (pas seulement les cas absolus !)
    4. Identifier les aires avec :
       - Taux d'incidence élevé (> seuil alerte)
       - Précipitations récentes importantes
       - Proximité de cours d'eau
    5. Prioriser les investigations entomologiques
    
    **Résultat :** Identification rapide des foyers actifs de transmission
    
    ---
    
    ### 🎯 Cas 3 : Évaluation d'Impact Post-Intervention
    
    **Contexte :** Campagne de pulvérisation intradomiciliaire (PID) réalisée en semaine 20.
    
    **Étapes :**
    
    1. Charger données des semaines 1-30 (incluant avant et après PID)
    2. Entraîner le modèle sur données S1-S19 (avant intervention)
    3. Prédire S21-S30
    4. Comparer prédictions vs observations réelles
    5. Calculer l'impact :
       ```
       Cas évités = Σ(Cas prédits - Cas observés)
       ```
    
    **Résultat :** Quantification objective de l'impact de l'intervention
    """)

# ============================================================
# TAB 2 : ROUGEOLE
# ============================================================
with tab_rougeole:
    st.markdown('<div class="section-header">🦠 Application de Surveillance de la Rougeole</div>', unsafe_allow_html=True)
    
    st.markdown("## 📋 Vue d'Ensemble")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4 style="color:#E4032E; margin-top:0;">C'est quoi l'App Rougeole ?</h4>
        
        Outil spécialisé de **surveillance et prédiction** des épidémies de rougeole :
        
        - 📅 **Suivi par semaines épidémiologiques**
        - 💉 **Analyse de couverture vaccinale**
        - 🎯 **Détection des poches de susceptibilité**
        - 🔮 **Prédiction des flambées**
        - 🗺️ **Multi-pays** : Niger, Burkina, Mali, Mauritanie
        
        **Nouveauté v4.0** : Partage intelligent des données avec l'app Paludisme !
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4 style="color:#ef6c00; margin-top:0;">⚠️ Différences avec Paludisme</h4>
        
        - 📁 **Fichier local intégré** : `ao_hlthArea.zip` intégré
        - 📊 **Format linelist** : Cas individuels avec dates
        - 💉 **Vaccination** : Variable critique pour prédiction
        - 👶 **Âge** : Focus sur enfants < 5 ans
        - 🚨 **Seuils OMS** : Alertes épidémiques automatiques
        </div>
        """, unsafe_allow_html=True)
    
    # Démarrage rapide
    st.markdown('<div class="section-header">🚀 Démarrage Rapide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 1️⃣ : Charger les Aires de Santé</h4>
    
    **Option A : Fichier local intégré**
    
    1. Sélectionner **"Fichier local (ao_hlthArea.zip)"**
    2. Choisir le **pays** : Niger, Burkina Faso, Mali ou Mauritanie
    3. Les aires se chargent automatiquement ✅
    
    **Format attendu :**
    - Colonnes : `iso3`, `health_area`, `geometry`
    
    **Option B : Upload personnalisé**
    
    Uploadez votre propre fichier (GeoJSON ou ZIP)
    
    💡 **Astuce v4.0** : Si vous avez déjà chargé des aires dans l'app Paludisme, elles sont réutilisées !
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 2️⃣ : Charger les Données de Cas</h4>
    
    **Deux formats acceptés :**
    
    **Format 1 : Agrégé par semaine**
    
    <table class="feature-table">
        <tr>
            <th>Colonne</th>
            <th>Description</th>
        </tr>
        <tr>
            <td><code>health_area</code></td>
            <td>Nom aire santé</td>
        </tr>
        <tr>
            <td><code>Semaine_Epi</code></td>
            <td>Semaine épidémiologique (ex: 2024-W05)</td>
        </tr>
        <tr>
            <td><code>Cas_Total</code></td>
            <td>Nombre de cas</td>
        </tr>
    </table>
    
    **Format 2 : Linelist individuelle (Recommandé)**
    
    <table class="feature-table">
        <tr>
            <th>Colonne</th>
            <th>Description</th>
        </tr>
        <tr>
            <td><code>Aire_Sante</code></td>
            <td>Lieu du cas</td>
        </tr>
        <tr>
            <td><code>Date_Debut_Eruption</code></td>
            <td>Date début éruption cutanée</td>
        </tr>
    </table>
    
    *→ Agrégation automatique par semaine épidémiologique*
    
    💡 **Avantage linelist** : Permet analyse par âge, sexe, statut vaccinal
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 3️⃣ : Couverture Vaccinale (Optionnel mais Crucial)</h4>
    
    **Format CSV :**
    - `health_area` : Aire de santé
    - `Taux_Vaccination` : % population vaccinée (0-100)
    
    **Si absent :** L'application extrait depuis la linelist si colonne `Statut_Vaccinal` présente
    
    **Interprétation :**
    
    <table class="feature-table">
        <tr>
            <th>Taux</th>
            <th>Statut</th>
        </tr>
        <tr>
            <td>&lt; 80%</td>
            <td>🔴 Très insuffisant - Risque épidémie majeure</td>
        </tr>
        <tr>
            <td>80-94%</td>
            <td>🟡 Insuffisant - Risque flambées localisées</td>
        </tr>
        <tr>
            <td>≥ 95%</td>
            <td>🟢 Objectif atteint - Immunité collective</td>
        </tr>
    </table>
    
    **Seuil OMS rougeole :** 95% pour immunité de groupe
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
    <h4>Étape 4️⃣ : Mode Démo (Pour Tester)</h4>
    
    **📁 Dans la sidebar → Mode d'utilisation**
    
    1. Sélectionner **"🧪 Mode démo (données simulées)"**
    2. Choisir un pays
    3. Génération automatique de 52 semaines avec :
       - Saisonnalité réaliste (pic mars-mai)
       - Distribution géographique hétérogène
       - Couverture vaccinale variable
    
    *💡 Conseil : Commencez par le mode démo pour comprendre le fonctionnement*
    </div>
    """, unsafe_allow_html=True)
    
    # Fonctionnalités
    st.markdown('<div class="section-header">⚙️ Fonctionnalités Clés</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Analyse Épidémiologique
    
    **Courbe épidémique :**
    - Évolution hebdomadaire des cas
    - Identification des pics saisonniers
    - Comparaison avec seuils d'alerte
    
    **Seuil d'alerte épidémique :**
    ```
    Seuil = Moyenne + 2 × Écart-type
    ```
    
    *→ L'app calcule automatiquement le seuil adapté à chaque zone*
    
    **Détection de clusters :**
    
    **Critères OMS :**
    - ≥ 3 cas dans une même aire en 4 semaines = **Cluster confirmé**
    - ≥ 5 cas liés épidémiologiquement = **Flambée**
    
    **Action :** Campagne vaccination réactive (CVR) dans rayon 30 km autour du cluster
    
    ---
    
    ### 💉 Analyse Vaccinale
    
    **Carte de couverture :**
    - Choroplèthe par niveau de couverture
    - Identification des "poches" < 80%
    - Calcul des besoins en doses
    
    **Estimation enfants non-vaccinés :**
    ```
    Non-vaccinés = Pop_Enfants × (1 - Taux_Vaccination/100)
    ```
    
    ---
    
    ### 🤖 Modélisation Prédictive
    
    **Algorithmes disponibles :**
    1. **Gradient Boosting** (Recommandé) - Capture relations complexes
    2. **Random Forest** - Robuste aux données manquantes
    3. **Ridge/Lasso** - Simple, rapide, interprétable
    
    **Features utilisées :**
    - 📈 **Historique des cas** : Lags 1, 2, 4 semaines
    - 💉 **Couverture vaccinale** : % vaccinés, nb susceptibles
    - 👥 **Démographie** : Population totale, enfants < 5 ans
    - 🏙️ **Urbanisation** : Type habitat (urbain/rural)
    - 🌡️ **Climat** : Température, humidité (optionnel)
    
    **Horizon de prédiction :**
    - 1 à 12 mois (4 à 48 semaines épidémio)
    - Prédictions hebdomadaires
    
    **Validation :**
    - Split temporel 80/20
    - Validation croisée 5-fold
    - Métriques : R², MAE, RMSE
    
    ---
    
    ### 🗺️ Cartographie Avancée
    
    **Couches disponibles :**
    - Nombre de cas
    - Taux d'incidence
    - Couverture vaccinale
    - Clusters détectés
    - Prédictions futures
    
    **Exports :**
    - 📥 CSV : Données tabulaires
    - 🗺️ GeoJSON : Carte avec attributs
    - 📊 Excel : Rapport complet multi-onglets
    """)
    
    # Cas d'usage
    st.markdown('<div class="section-header">💼 Cas d'Usage Terrain</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Cas 1 : Planification Campagne Vaccination Préventive (AVS)
    
    **Contexte :** Période pré-épidémique (janvier), planifier campagne mars-avril
    
    **Workflow :**
    1. Charger données N-1 (année précédente complète)
    2. Charger couverture vaccinale actuelle
    3. Lancer prédiction sur 16 semaines (4 mois)
    4. Identifier les **aires prioritaires** :
       - Couverture < 90%
       - Prédiction > 5 cas/semaine
       - Population enfants > 1000
    5. Calculer besoins :
       ```
       Doses = Pop_Enfants × (95% - Taux_actuel)/100 × 1.1 (pertes)
       ```
    
    **Résultat :** Liste priorisée d'aires + quantification besoins
    
    ---
    
    ### 🎯 Cas 2 : Riposte à une Flambée (CVR)
    
    **Contexte :** 8 cas confirmés rougeole en semaine 15 dans district X
    
    **Actions immédiates :**
    1. Charger linelist des 8 cas
    2. Identifier sur carte les aires touchées
    3. Tracer cercle 30 km autour du cluster
    4. Lister toutes les aires dans le cercle
    5. Extraire population cible (9 mois - 15 ans)
    6. Organiser CVR sous 72h
    
    **Suivi post-CVR (S+4) :**
    1. Recharger données S16-S19
    2. Comparer avec prédiction pré-CVR
    3. Évaluer impact :
       ```
       Efficacité = (Cas_prédits - Cas_observés) / Cas_prédits × 100
       ```
    
    ---
    
    ### 🎯 Cas 3 : Surveillance Sentinelle Multi-Pays
    
    **Contexte :** Coordinateur régional MSF (Niger + Burkina + Mali)
    
    **Workflow hebdomadaire :**
    1. **Lundi matin** : Charger linelists des 3 pays (semaine précédente)
    2. **Analyse rapide** :
       - Nombre de cas par pays
       - Identification clusters émergents
       - Comparaison avec seuils alerte
    3. **Prédiction courte (4 semaines)** pour chaque pays
    4. **Rapport synthétique** :
       - 🔴 Pays en alerte (>= seuil)
       - 🟡 Pays en vigilance (proche seuil)
       - 🟢 Pays stables
    5. **Briefing équipes** mardi
    
    **Fréquence :** Hebdomadaire (automatisable avec scripts Python)
    """)

# ============================================================
# TAB 3 : GLOSSAIRE & MÉTHODOLOGIE
# ============================================================
with tab_glossaire:
    st.markdown('<div class="section-header">📖 Glossaire & Méthodologie</div>', unsafe_allow_html=True)
    
    # Sous-onglets
    subtab1, subtab2, subtab3 = st.tabs([
        "📚 Variables Temporelles",
        "🌍 Variables Environnementales",
        "🤖 Méthodologie ML"
    ])
    
    with subtab1:
        st.markdown("## 📚 Variables Temporelles et Épidémiologiques")
        
        st.markdown("""
        <div class="glossary-term">
        <strong>📅 Numéro de Semaine (weeknum)</strong>
        
        **Signification :** Numéro séquentiel de la semaine (1, 2, 3...)
        
        **Utilité :** Capture la tendance générale dans le temps
        
        **Exemple :** Semaine 20 → printemps (hausse attendue paludisme)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>🌀 Features Cycliques (sin_week, cos_week)</strong>
        
        **Signification :** Représentation mathématique des cycles annuels
        
        **Utilité :** Capture les variations saisonnières (pic saison pluies)
        
        **Calcul :** sin(2π × semaine / 52) et cos(2π × semaine / 52)
        
        **Pourquoi ?** Permet au modèle de savoir que la semaine 1 et 52 sont proches
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>📉 Lags (cases_lag1, cases_lag2, cases_lag4)</strong>
        
        **Signification :** Nombre de cas 1, 2 ou 4 semaines avant
        
        **Utilité :** **Variable la plus importante** - Tendance récente
        
        **Exemple :** 50 cas en S24 → Prédiction S25 ≈ 48-52 cas
        
        **Pourquoi ça marche ?** Inertie épidémique (cas d'aujourd'hui → cas de demain)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>📊 Moyennes Mobiles (cases_ma2, cases_ma4, cases_ma8)</strong>
        
        **Signification :** Moyenne des 2, 4 ou 8 dernières semaines
        
        **Utilité :** Lisse les fluctuations, montre tendance globale
        
        **Calcul :** MA2 = (S-1 + S-2) / 2
        
        **Avantage :** Moins sensitive aux pics isolés
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>📈 Taux de Croissance (growth_rate)</strong>
        
        **Signification :** Variation % entre 2 semaines consécutives
        
        **Formule :** (Cas S - Cas S-1) / Cas S-1
        
        **Exemple :** 40→50 cas → +25% (croissance rapide)
        
        **Utilité :** Détecte accélérations/décélérations épidémiques
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>📐 Min/Max Roulants (cases_min_4, cases_max_8)</strong>
        
        **Signification :** Valeurs extrêmes sur fenêtres 4 et 8 semaines
        
        **Utilité :** Capture amplitude variations récentes
        
        **Exemple :** Max_4 très élevé → Pic récent = zone à risque
        </div>
        """, unsafe_allow_html=True)
    
    with subtab2:
        st.markdown("## 🌍 Variables Environnementales et Climatiques")
        
        st.markdown("""
        <div class="glossary-term">
        <strong>🌡️ Température (temp_api)</strong>
        
        **Signification :** Température moyenne hebdomadaire en degrés Celsius
        
        **Source :** NASA POWER API
        
        **Impact paludisme :**
        - **Optimal** : 25-30°C (développement optimal du parasite)
        - **< 18°C** : Transmission ralentie
        - **> 35°C** : Mortalité accrue des moustiques
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>🌧️ Précipitations (precip_api)</strong>
        
        **Signification :** Cumul pluies hebdomadaires en millimètres
        
        **Impact paludisme :**
        - **Lag 1-2 semaines** : Création gîtes larvaires
        - **Pic transmission** : 2-4 semaines après pic pluies
        - **Excès** : Emporte les larves (effet négatif)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>💧 Humidité (humidity_api)</strong>
        
        **Signification :** Humidité moyenne hebdomadaire en %
        
        **Impact paludisme :**
        - **Optimal** : 60-80% (survie moustiques)
        - **< 50%** : Dessiccation des vecteurs
        - **Rougeole** : Climat sec favorise transmission (aérosols)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>🌊 Zones Inondables (flood_mean)</strong>
        
        **Signification :** Hauteur d'eau moyenne zone inondable (raster)
        
        **Utilité :** Zones inondables = gîtes larvaires permanents
        
        **Impact :** Transmission stable toute l'année (pas seulement saison pluies)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>⛰️ Altitude (elevation_mean)</strong>
        
        **Signification :** Altitude moyenne de l'aire en mètres
        
        **Impact paludisme :**
        - **< 1000m** : Transmission forte
        - **1000-1500m** : Transmission modérée
        - **> 2000m** : Transmission rare (température trop basse)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>🏞️ Distance aux Rivières (dist_river)</strong>
        
        **Signification :** Distance centroïde aire → cours d'eau le plus proche (km)
        
        **Utilité :** Proximité rivière = risque accru (reproduction *Anopheles*)
        
        **Seuil :** < 2 km = zone à très haut risque
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>👥 Population Totale (Pop_Totale)</strong>
        
        **Source :** WorldPop (Google Earth Engine)
        
        **Utilité :** Dénominateur pour taux d'incidence
        
        **Formule :** Taux incidence = (Cas / Pop_Totale) × 10 000
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>👶 Population Enfants (Pop_Enfants_0_14)</strong>
        
        **Signification :** Somme des tranches 0-4, 5-9, 10-14 ans
        
        **Utilité Paludisme :**
        - Groupe le plus vulnérable (immunité faible)
        - Calcul besoins en MII pédiatriques
        
        **Utilité Rougeole :**
        - Population cible vaccination
        - Calcul enfants susceptibles
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glossary-term">
        <strong>📏 Densité de Population (Densite_Pop)</strong>
        
        **Signification :** Habitants par km²
        
        **Utilité :**
        - **Paludisme** : Densité élevée = plus de transmission (réservoir humain)
        - **Rougeole** : Densité élevée = transmission rapide (contacts fréquents)
        </div>
        """, unsafe_allow_html=True)
    
    with subtab3:
        st.markdown("## 🤖 Méthodologie Machine Learning")
        
        st.markdown("""
        ### 🧮 Analyse en Composantes Principales (ACP)
        
        **Objectif :** Réduire la dimensionnalité tout en conservant l'information
        
        L'ACP transforme un ensemble de variables **corrélées** en un ensemble réduit de variables **non-corrélées** (composantes principales) qui capturent la majorité de la variance.
        
        **Situation initiale :**
        - 50 variables (lags, climat, démo, environnement)
        - Beaucoup de corrélations entre elles
        - Risque de sur-apprentissage
        
        **Après ACP :**
        - 10-15 composantes principales
        - Capturent 90-95% de la variance
        - Variables décorrélées
        - Modèle plus robuste
        
        **Comment ça marche ?**
        
        Chaque composante est une **combinaison linéaire** des variables originales. Les coefficients indiquent la contribution de chaque variable.
        
        **Exemple :**
        ```
        CP1 = 0.4×temp + 0.5×precip + 0.3×humidity + ...
        CP2 = 0.6×elevation - 0.4×dist_river + ...
        ```
        
        **Interprétation :**
        - **CP1** = "Facteur climatique général"
        - **CP2** = "Facteur topographique"
        
        ---
        
        ### 🗺️ Clustering Spatial (K-Means)
        
        **Objectif :** Identifier des **groupes de zones géographiques homogènes** ayant des profils épidémiologiques similaires.
        
        **Pourquoi ?**
        - Zones similaires → Risques similaires
        - Permet de "transférer" l'apprentissage d'une zone à une autre
        - Améliore les prédictions pour zones avec peu de données
        
        **Exemple de clusters :**
        
        <table class="feature-table">
            <tr>
                <th>Cluster</th>
                <th>Caractéristiques</th>
                <th>Cas Moy.</th>
            </tr>
            <tr>
                <td>Cluster 0 (Côtier)</td>
                <td>Basse altitude, près rivières, forte humidité</td>
                <td>120/sem</td>
            </tr>
            <tr>
                <td>Cluster 1 (Urbain)</td>
                <td>Dense, assainissement variable</td>
                <td>85/sem</td>
            </tr>
            <tr>
                <td>Cluster 2 (Rural plaine)</td>
                <td>Rizières, marais, forte transmission</td>
                <td>150/sem</td>
            </tr>
            <tr>
                <td>Cluster 3 (Montagne)</td>
                <td>Altitude > 800m, faible transmission</td>
                <td>20/sem</td>
            </tr>
            <tr>
                <td>Cluster 4 (Semi-aride)</td>
                <td>Faibles précipitations, transmission saisonnière</td>
                <td>45/sem</td>
            </tr>
        </table>
        
        **Encodage pour le modèle :**
        
        Chaque cluster devient une **variable binaire** (one-hot encoding) :
        ```
        cluster_0 =   # Zone côtière
        cluster_1 =   # Zone urbaine
        cluster_2 =   # Rural plaine
        ```
        
        *Le modèle apprend poids spécifiques pour chaque cluster*
        
        ---
        
        ### 📍 Lag Spatial (Spatial Lag)
        
        **Objectif :** Capturer l'influence des **zones voisines**
        
        Le **lag spatial** mesure l'influence des zones **voisines** sur le nombre de cas d'une zone.
        
        *Hypothèse :* Si mes voisins ont beaucoup de cas, j'ai probablement plus de risques (migration moustiques, mouvements population).
        
        **Formule :**
        ```
        Lag_spatial(zone_i) = Σ w_ij × Cas_j pour j = k voisins les plus proches
        
        où w_ij = 1 / distance(i, j) (poids inversement proportionnel à la distance)
        ```
        
        **Exemple :**
        - Zone A : 50 cas
        - Voisins (< 10 km) : B=80 cas, C=60 cas, D=40 cas
        - Lag_spatial(A) ≈ 60 (moyenne pondérée)
        
        *💡 En pratique, k=5 fonctionne bien pour la plupart des contextes épidémiologiques.*
        
        ---
        
        ### ⏰ Validation Croisée Temporelle (Time Series Split)
        
        **Pourquoi pas une CV classique ?**
        
        En épidémiologie, **l'ordre temporel est crucial**. On ne peut pas tester le modèle sur des données *antérieures* à celles d'entraînement (ça n'a pas de sens de "prédire le passé" !)
        
        **Principe :**
        
        On divise les données en **folds successifs** en respectant l'ordre temporel :
        
        ```
        Fold 1: Entraînement [S1-S30] → Test [S31-S40] → r² = 0.82
        Fold 2: Entraînement [S1-S35] → Test [S36-S44] → r² = 0.78
        Fold 3: Entraînement [S1-S40] → Test [S41-S48] → r² = 0.85
        Fold 4: Entraînement [S1-S44] → Test [S45-S50] → r² = 0.80
        Fold 5: Entraînement [S1-S48] → Test [S49-S52] → r² = 0.83
        
        **Performance finale : r² = 0.82 ± 0.03 → Robuste !**
        ```
        
        **Avantage :**
        - Évalue la **stabilité** du modèle
        - Détecte le sur-apprentissage
        - Estime la performance réelle en production
        
        **Interprétation de l'écart-type :**
        
        <table class="feature-table">
            <tr>
                <th>Écart-type r²</th>
                <th>Signification</th>
            </tr>
            <tr>
                <td>&lt; 0.05</td>
                <td>🟢 Très stable</td>
            </tr>
            <tr>
                <td>0.05-0.10</td>
                <td>🟡 Acceptable</td>
            </tr>
            <tr>
                <td>&gt; 0.10</td>
                <td>🔴 Instable</td>
            </tr>
        </table>
        
        ---
        
        ### 🔄 Pipeline Complet de Modélisation
        
        **1. Feature Engineering**
        
        **Variables créées :**
        - Lags (1, 2, 4 semaines)
        - Moyennes mobiles (2, 4, 8 semaines)
        - Taux de croissance
        - Cycles saisonniers (sin/cos)
        - Min/Max roulants
        
        **2. Réduction dimensionnelle :**
        - ACP (50 → 15 composantes, 90% variance)
        
        **3. Analyse géographique :**
        - K-Means clustering (5 clusters)
        - Lag spatial (k=5 voisins)
        - One-hot encoding clusters
        
        **4. Normalisation :**
        - StandardScaler (μ=0, σ=1)
        - Ou RobustScaler si valeurs aberrantes
        
        **5. Entraînement :**
        - Split temporel 80/20
        - Gradient Boosting ou Random Forest
        - Optimisation hyperparamètres (optionnel)
        
        **6. Test rigoureux :**
        - Validation croisée temporelle 5-fold
        - Calcul R², MAE, RMSE
        - Analyse importance des variables
        
        **7. Prédiction :**
        - Génération features futures
        - Prédictions par aire et semaine
        - Intervalles de confiance (optionnel)
        
        ---
        
        ### 📊 Métriques de Performance
        
        **R² (Coefficient de Détermination)**
        ```
        R² = 1 - (Somme carrés résidus / Somme carrés totale)
        ```
        - **Interprétation :** % de variance expliquée
        - **Valeurs :** 0 (mauvais) à 1 (parfait)
        - **Seuils :** >0.8 excellent, >0.6 bon, <0.4 insuffisant
        
        **MAE (Mean Absolute Error)**
        ```
        MAE = (1/n) × Σ |y_réel - y_prédit|
        ```
        - **Interprétation :** Erreur moyenne en nombre de cas
        - **Exemple :** MAE=5 → erreur moyenne de 5 cas
        
        **RMSE (Root Mean Squared Error)**
        ```
        RMSE = √[(1/n) × Σ (y_réel - y_prédit)²]
        ```
        - **Interprétation :** Erreur avec pénalité pour grandes erreurs
        - **Utilisation :** Comparer plusieurs modèles
        """)

# ============================================================
# TAB 4 : ARCHITECTURE v4.0
# ============================================================
with tab_architecture:
    st.markdown('<div class="section-header">🏗️ Architecture Modulaire v4.0</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## ✨ Nouveauté Version 4.0
    
    La plateforme MSF a été **complètement refactorisée** avec une **architecture modulaire** qui permet :
    
    - 🔄 **Données partagées** entre applications Paludisme et Rougeole
    - 💾 **Cache intelligent** pour éviter les rechargements
    - 🎨 **Branding MSF uniforme** sur toutes les pages
    - 🧩 **Modules réutilisables** faciles à maintenir
    - 📈 **Scalabilité** : ajout facile de nouvelles maladies
    
    ---
    
    ## 📁 Structure des Fichiers
    
    ```
    📁 Plateforme_MSF/
    ├── 📄 main_app.py                    # Navigation principale
    ├── 📄 app_paludisme.py               # App Paludisme (modulaire)
    ├── 📄 app_rougeole.py                # App Rougeole (modulaire)
    ├── 📄 app_manuel.py                  # Manuel (vous êtes ici !)
    ├── 📄 requirements.txt               # Dépendances Python
    │
    ├── 📁 modules/                       # MODULES PARTAGÉS
    │   ├── __init__.py
    │   ├── data_loader.py               # Gestionnaire centralisé des données
    │   ├── geo_loader.py                # Chargement géographique
    │   ├── climate_loader.py            # NASA POWER API
    │   ├── worldpop_loader.py           # Google Earth Engine
    │   ├── ui_components.py             # Composants UI (branding MSF)
    │   └── utils.py                     # Fonctions utilitaires
    │
    └── 📁 data/                         # Données locales (optionnel)
        └── ao_hlthArea.zip
    ```
    
    ---
    
    ## 🧩 Modules Partagés
    
    ### 1. `data_loader.py` - Gestionnaire Centralisé
    
    **Rôle :** Stocke toutes les données dans `st.session_state` pour qu'elles soient accessibles par toutes les applications.
    
    **Fonctionnalités :**
    - `set_geodata()` / `get_geodata()` : Aires de santé
    - `set_climate_data()` / `get_climate_data()` : Données NASA POWER
    - `set_worldpop_data()` / `get_worldpop_data()` : Données démographiques
    - `set_epidemio_data(disease)` : Linelists par maladie
    - `set_vaccination_data()` : Couverture vaccinale
    - `get_summary()` : Résumé de toutes les données chargées
    - `clear_all()` : Réinitialisation complète
    
    **Exemple d'utilisation :**
    ```python
    from modules.data_loader import DataManager
    
    dm = DataManager()
    
    # Charger des données géographiques
    dm.set_geodata(gdf, source="upload")
    
    # Les récupérer plus tard (même dans une autre app)
    gdf = dm.get_geodata()
    
    # Vérifier si disponibles
    if dm.has_geodata():
        print("Aires déjà chargées !")
    ```
    
    ---
    
    ### 2. `geo_loader.py` - Chargement Géographique
    
    **Rôle :** Gère tous les formats géographiques (GeoJSON, Shapefile, ZIP)
    
    **Fonctionnalités :**
    - `load_from_file(uploaded_file)` : Charge depuis upload Streamlit
    - `load_local_ao_hltharea(iso3)` : Charge fichier local par pays
    - `validate_geodata(gdf)` : Vérifie colonnes obligatoires
    - `get_geodata_info(gdf)` : Extrait métadonnées
    
    **Exemple :**
    ```python
    from modules.geo_loader import GeoLoader
    
    # Charger depuis upload
    gdf = GeoLoader.load_from_file(uploaded_file)
    
    # Valider
    valid, msg = GeoLoader.validate_geodata(gdf)
    if valid:
        dm.set_geodata(gdf)
    ```
    
    ---
    
    ### 3. `climate_loader.py` - NASA POWER API
    
    **Rôle :** Récupération automatisée des données climatiques
    
    **Fonctionnalités :**
    - `fetch_climate_data(lat, lon, start, end)` : Point unique (avec cache)
    - `fetch_climate_for_geodata(gdf, start, end)` : Toutes les aires
    - `aggregate_to_weekly(df)` : Agrégation quotidien → hebdo
    
    **Cache :** 24h pour éviter requêtes répétées
    
    **Exemple :**
    ```python
    from modules.climate_loader import ClimateLoader
    
    # Récupérer pour toutes les aires
    df_climate = ClimateLoader.fetch_climate_for_geodata(
        gdf, 
        start_date=datetime(2024, 1, 1),
        end_date=datetime.today()
    )
    
    # Agréger par semaine
    df_weekly = ClimateLoader.aggregate_to_weekly(df_climate)
    
    # Sauvegarder
    dm.set_climate_data(df_weekly)
    ```
    
    ---
    
    ### 4. `worldpop_loader.py` - Google Earth Engine
    
    **Rôle :** Extraction données démographiques WorldPop
    
    **Fonctionnalités :**
    - `init_gee()` : Initialisation GEE (service account ou local)
    - `fetch_worldpop_for_area(geometry, year)` : Zone unique (avec cache)
    - `fetch_worldpop_for_geodata(gdf, year)` : Toutes les aires
    
    **Cache :** 7 jours (données annuelles)
    
    **Exemple :**
    ```python
    from modules.worldpop_loader import WorldPopLoader
    
    # Initialiser GEE
    if WorldPopLoader.init_gee():
        # Récupérer pour toutes les aires
        df_pop = WorldPopLoader.fetch_worldpop_for_geodata(gdf, year=2020)
        dm.set_worldpop_data(df_pop)
    ```
    
    ---
    
    ### 5. `ui_components.py` - Branding MSF
    
    **Rôle :** Composants UI standardisés avec identité visuelle MSF
    
    **Fonctionnalités :**
    - `apply_msf_branding()` : Applique le CSS MSF
    - `msf_header(title, subtitle)` : En-tête standardisé
    - `msf_footer()` : Footer standardisé
    - `data_status_panel(dm)` : Panneau de statut des données
    
    **Exemple :**
    ```python
    from modules.ui_components import apply_msf_branding, msf_header, msf_footer
    
    # Appliquer le branding
    apply_msf_branding()
    
    # En-tête
    msf_header(
        "Mon Application",
        "Sous-titre explicatif"
    )
    
    # ... Contenu de l'app ...
    
    # Footer
    msf_footer()
    ```
    
    ---
    
    ### 6. `utils.py` - Fonctions Utilitaires
    
    **Rôle :** Fonctions helpers réutilisables
    
    **Contenu :**
    - `safe_int(value)` : Conversion int avec gestion NaN
    - `safe_float(value)` : Conversion float avec gestion NaN
    - `format_large_number(n)` : Formatage avec séparateurs
    - `calculate_epi_week(date)` : Calcul semaine épidémio
    - `week_to_date(year, week)` : Conversion inverse
    
    ---
    
    ## 🔄 Workflow de Partage des Données
    
    **Scénario : Utiliser les mêmes aires pour Paludisme et Rougeole**
    
    ### Étape 1 : Charger dans Paludisme
    ```
    1. Ouvrir app Paludisme
    2. Uploader fichier GeoJSON → 100 aires chargées
    3. DataManager stocke dans st.session_state
    ```
    
    ### Étape 2 : Naviguer vers Rougeole
    ```
    1. Menu sidebar → Sélectionner "Rougeole"
    2. App Rougeole se charge
    3. DataManager détecte : "Aires déjà chargées !"
    4. Affiche : ✅ 100 aires chargées (réutilisées)
    ```
    
    ### Étape 3 : Aucun rechargement nécessaire ! 🎉
    
    **Avantages :**
    - ⚡ **Gain de temps** : Pas de re-upload
    - 💾 **Économie de ressources** : Pas de re-traitement
    - 🎯 **Cohérence** : Mêmes données = analyses comparables
    
    ---
    
    ## 📊 Panneau de Statut des Données
    
    **Dans la sidebar de main_app.py :**
    
    ```
    ### 📊 Données Chargées
    
    ✅ Aires de santé
    ✅ Données climatiques
    ✅ WorldPop (GEE)
    ✅ Linelists Paludisme
    ⬜ Linelists Rougeole
    ⬜ Couverture vaccinale
    
    ⏱️ MAJ : 14:23:45
    ```
    
    **Interprétation :**
    - ✅ = Données disponibles et réutilisables
    - ⬜ = Données non chargées
    - 🔄 Bouton "Réinitialiser" pour tout effacer
    
    ---
    
    ## 🚀 Ajouter une Nouvelle Maladie
    
    **Exemple : Ajouter une app Méningite**
    
    ### 1. Créer `app_meningite.py`
    ```python
    import streamlit as st
    from modules.ui_components import apply_msf_branding, msf_header
    from modules.data_loader import DataManager
    from modules.geo_loader import GeoLoader
    from modules.worldpop_loader import WorldPopLoader
    
    apply_msf_branding()
    msf_header("🧠 Surveillance Méningite", "Analyse et prédiction")
    
    dm = DataManager()
    
    # Réutiliser les aires si déjà chargées
    if dm.has_geodata():
        gdf = dm.get_geodata()
        st.success(f"✅ {len(gdf)} aires réutilisées")
    else:
        # Charger nouvelles aires
        uploaded = st.file_uploader("Charger aires")
        if uploaded:
            gdf = GeoLoader.load_from_file(uploaded)
            dm.set_geodata(gdf)
    
    # Charger linelists méningite
    uploaded_cases = st.file_uploader("Linelists méningite")
    if uploaded_cases:
        df_cases = pd.read_csv(uploaded_cases)
        dm.set_epidemio_data(df_cases, disease='meningite')
    
    # ... Reste de l'app ...
    ```
    
    ### 2. Ajouter au menu de `main_app.py`
    ```python
    page = st.selectbox(
        "Choisir une application",
        ["Accueil", "Paludisme", "Rougeole", "Méningite", "Manuel"]
    )
    
    # ...
    
    elif st.session_state.page_choice == "Méningite":
        load_app("app_meningite.py")
    ```
    
    **C'est tout ! 🎉** L'app Méningite peut maintenant réutiliser toutes les données déjà chargées.
    
    ---
    
    ## 🎨 Personnalisation du Branding
    
    **Pour modifier les couleurs MSF :**
    
    Éditer `modules/ui_components.py` :
    
    ```python
    # Couleur principale MSF
    MSF_RED = "#E4032E"
    MSF_DARK_RED = "#C4032A"
    MSF_GRAY = "#58595B"
    
    def apply_msf_branding():
        st.markdown(f'''
        <style>
            .header-banner {{
                background: linear-gradient(135deg, {MSF_RED}, {MSF_DARK_RED});
                ...
            }}
            
            .stButton > button {{
                background: {MSF_RED};
                ...
            }}
        </style>
        ''', unsafe_allow_html=True)
    ```
    
    **Modification automatiquement appliquée partout ! 🎨**
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
msf_footer()

# Section contact
st.markdown("""
<div class="info-box">
<h3 style="color:#E4032E; margin-top:0;">📧 Support Technique</h3>

**Email :** youssoupha.mbodji@example.com

**Questions fréquentes :** FAQ (à venir)

**Documentation complète :** Ce manuel utilisateur

**Code source :** Disponible sur demande (Licence Open Source MIT)

---

**Version 4.0** | Développé par **Youssoupha MBODJI** | © 2026 - Médecins Sans Frontières
</div>
""", unsafe_allow_html=True)
