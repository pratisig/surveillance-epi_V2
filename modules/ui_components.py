"""
============================================================
UI COMPONENTS - BRANDING MSF
Composants UI réutilisables pour toutes les applications
============================================================
"""

import streamlit as st

def apply_msf_branding():
    """Applique le CSS MSF à toute page de l'application"""
    st.markdown("""
    <style>
        /* Bannière en-tête MSF */
        .header-banner {
            background: linear-gradient(135deg, #E4032E 0%, #C4032A 100%);
            border-radius: 12px;
            padding: 1.5rem 1rem;
            text-align: center;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(228, 3, 46, 0.2);
        }
        
        .msf-logo {
            font-size: 1.1rem;
            font-weight: bold;
            letter-spacing: 2px;
            margin-bottom: 0.3rem;
        }
        
        .header-banner h1 {
            font-size: 1.8rem;
            margin: 0.3rem 0;
            font-weight: 600;
        }
        
        .header-banner p {
            font-size: 0.95rem;
            margin: 0.2rem 0;
            opacity: 0.95;
        }
        
        /* Boutons MSF */
        .stButton > button {
            background: #E4032E;
            color: white;
            border: none;
            padding: 0.7rem 1.5rem;
            font-size: 1rem;
            font-weight: bold;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 3px 10px rgba(228, 3, 46, 0.3);
        }
        
        .stButton > button:hover {
            background: #C4032A;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(228, 3, 46, 0.4);
        }
        
        /* Cartes info */
        .info-card {
            background: white;
            border-radius: 10px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 4px solid #E4032E;
            margin: 1rem 0;
        }
        
        .info-card h3 {
            color: #E4032E;
            margin-top: 0;
        }
        
        /* Footer MSF */
        .msf-footer {
            text-align: center;
            color: #58595B;
            padding: 1.5rem;
            background: #F8F9FA;
            border-radius: 10px;
            border-top: 3px solid #E4032E;
            margin-top: 2rem;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

def msf_header(title, subtitle=None):
    """
    Affiche un en-tête MSF standardisé
    
    Args:
        title (str): Titre principal
        subtitle (str): Sous-titre optionnel
    """
    header_html = f"""
    <div class="header-banner">
        <div class="msf-logo">⚕️ MÉDECINS SANS FRONTIÈRES</div>
        <h1>{title}</h1>
    """
    
    if subtitle:
        header_html += f"<p>{subtitle}</p>"
    
    header_html += "</div>"
    
    st.markdown(header_html, unsafe_allow_html=True)

def msf_footer():
    """Affiche le footer MSF standardisé"""
    st.markdown("""
    <div class="msf-footer">
        <p style="font-weight:bold; color:#E4032E; margin-bottom:0.5rem;">
            ⚕️ MÉDECINS SANS FRONTIÈRES
        </p>
        <p style="margin:0.3rem 0;"><strong>Développé par Youssoupha MBODJI</strong></p>
        <p style="margin:0.3rem 0;">📧 youssoupha.mbodji@example.com</p>
        <p style="margin-top:0.8rem;">Version 3.0 | © 2026 MSF | Afrique de l'Ouest</p>
        <p style="font-size:0.85rem; margin-top:0.8rem; font-style:italic; color:#7f8c8d;">
            "Bringing medical assistance to people affected by conflict, epidemics, disasters, or exclusion from healthcare"
        </p>
    </div>
    """, unsafe_allow_html=True)

def data_status_panel(data_manager):
    """
    Affiche un panneau de statut des données chargées
    
    Args:
        data_manager (DataManager): Instance du gestionnaire de données
    """
    summary = data_manager.get_summary()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Données Chargées")
    
    status_icons = {
        True: "✅",
        False: "⬜"
    }
    
    st.sidebar.markdown(f"{status_icons[summary['geodata']]} Aires de santé")
    st.sidebar.markdown(f"{status_icons[summary['climate']]} Données climatiques")
    st.sidebar.markdown(f"{status_icons[summary['worldpop']]} WorldPop (GEE)")
    st.sidebar.markdown(f"{status_icons[summary['epidemio_paludisme']]} Linelists Paludisme")
    st.sidebar.markdown(f"{status_icons[summary['epidemio_rougeole']]} Linelists Rougeole")
    st.sidebar.markdown(f"{status_icons[summary['vaccination']]} Couverture vaccinale")
    
    if summary['last_update']:
        st.sidebar.caption(f"Dernière MAJ : {summary['last_update'].strftime('%H:%M:%S')}")
