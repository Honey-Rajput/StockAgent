# utils.py
import streamlit as st

def inject_premium_css():
    """
    Injects global premium dark-mode, glassmorphism, and neon-themed CSS
    into the Streamlit web application.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Global CSS Overrides */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: 'Outfit', sans-serif;
            background-color: #090d16 !important;
            color: #e2e8f0 !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0d1321 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* ======= FORCE TAB WRAPPING — Streamlit 1.56 ======= */
        /* Target every possible wrapper that clips/scrolls the tab bar */
        [data-testid="stTabs"],
        [data-testid="stTabs"] > div,
        [data-testid="stTabs"] > div > div,
        [data-testid="stTabs"] > div > div > div {
            overflow: visible !important;
            overflow-x: visible !important;
        }
        /* The actual tab-list flexbox row */
        [data-testid="stTabs"] [data-baseweb="tab-list"],
        [role="tablist"] {
            flex-wrap: wrap !important;
            gap: 8px 6px !important;
            overflow: visible !important;
            overflow-x: visible !important;
            max-width: 100% !important;
            padding-bottom: 6px !important;
        }
        /* Individual tabs – spaced out with padding */
        [data-testid="stTabs"] [data-baseweb="tab-list"] button[role="tab"],
        [role="tablist"] button[role="tab"] {
            white-space: nowrap !important;
            flex-shrink: 0 !important;
            padding: 6px 14px !important;
            margin: 2px 3px !important;
            border-radius: 6px !important;
            font-size: 0.82rem !important;
        }
        /* Hide the left/right scroll arrow buttons */
        [data-testid="stTabs"] [data-baseweb="tab-list"] ~ button,
        [data-testid="stTabs"] button[aria-label*="croll"],
        [data-testid="stTabs"] button[data-baseweb="button"],
        [data-testid="stTabs"] > div > div > button,
        [data-testid="stTabs"] svg[data-testid="stIconMaterial"],
        [data-testid="stTabs"] > div > div > div > button:not([role="tab"]) {
            display: none !important;
        }
        /* Hide the bottom tab border/highlight bar if it overflows */
        [data-testid="stTabs"] [data-baseweb="tab-border"],
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        
        /* Premium Card Component */
        .glass-card {
            background: rgba(17, 24, 39, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            border-color: rgba(41, 182, 246, 0.3);
            box-shadow: 0 12px 35px 0 rgba(41, 182, 246, 0.08);
            transform: translateY(-2px);
        }
        
        /* Metric Cards */
        .metric-glow-green {
            border-left: 4px solid #00e676;
            box-shadow: inset 0 0 15px rgba(0, 230, 118, 0.02);
        }
        
        .metric-glow-blue {
            border-left: 4px solid #29b6f6;
            box-shadow: inset 0 0 15px rgba(41, 182, 246, 0.02);
        }
        
        .metric-glow-amber {
            border-left: 4px solid #ffa000;
            box-shadow: inset 0 0 15px rgba(255, 160, 0, 0.02);
        }
        
        /* Highlight Glow Card for Scores >= 70 */
        .high-signal-card {
            background: rgba(255, 160, 0, 0.06) !important;
            border: 1px solid rgba(255, 160, 0, 0.35) !important;
            box-shadow: 0 0 20px rgba(255, 160, 0, 0.1) !important;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: transform 0.2s ease;
        }
        .high-signal-card:hover {
            transform: scale(1.008);
            box-shadow: 0 0 25px rgba(255, 160, 0, 0.15) !important;
        }
        
        /* Custom Header Gradients */
        .gradient-title {
            background: linear-gradient(135deg, #00e676 0%, #00e5ff 50%, #29b6f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        
        .gradient-subtitle {
            background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 25px;
        }
        
        /* Styling Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap !important;
            gap: 8px 10px !important;
            background-color: transparent !important;
            height: auto !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            min-height: 40px !important;
            height: auto !important;
            white-space: nowrap !important;
            background-color: rgba(255, 255, 255, 0.02) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.03) !important;
            color: #94a3b8 !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(41, 182, 246, 0.05) !important;
            color: #29b6f6 !important;
            border-color: rgba(41, 182, 246, 0.15) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(41, 182, 246, 0.12) !important;
            border-color: rgba(41, 182, 246, 0.35) !important;
            color: #29b6f6 !important;
            font-weight: 600 !important;
        }
        
        /* Premium Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            padding: 6px 16px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #29b6f6 0%, #0288d1 100%) !important;
            border-color: #29b6f6 !important;
            box-shadow: 0 4px 15px rgba(41, 182, 246, 0.2) !important;
            color: white !important;
            transform: scale(1.03);
        }
        
        /* Add to Watchlist Row Buttons */
        div.stButton > button.add-btn {
            background: rgba(0, 230, 118, 0.08) !important;
            color: #00e676 !important;
            border: 1px solid rgba(0, 230, 118, 0.3) !important;
        }
        div.stButton > button.add-btn:hover {
            background: #00e676 !important;
            color: #090d16 !important;
            box-shadow: 0 4px 12px rgba(0, 230, 118, 0.3) !important;
            border-color: #00e676 !important;
        }
        
        /* Remove Watchlist Row Buttons */
        div.stButton > button.remove-btn {
            background: rgba(239, 68, 68, 0.08) !important;
            color: #ef4444 !important;
            border: 1px solid rgba(239, 68, 68, 0.3) !important;
        }
        div.stButton > button.remove-btn:hover {
            background: #ef4444 !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
            border-color: #ef4444 !important;
        }
        
        /* Badges */
        .custom-badge {
            display: inline-block;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 20px;
            text-align: center;
        }
        
        .badge-green {
            background: rgba(0, 230, 118, 0.12);
            color: #00e676;
            border: 1px solid rgba(0, 230, 118, 0.25);
        }
        
        .badge-amber {
            background: rgba(255, 160, 0, 0.12);
            color: #ffa000;
            border: 1px solid rgba(255, 160, 0, 0.25);
            box-shadow: 0 0 8px rgba(255, 160, 0, 0.1);
        }
        
        .badge-blue {
            background: rgba(41, 182, 246, 0.12);
            color: #29b6f6;
            border: 1px solid rgba(41, 182, 246, 0.25);
        }
        
        .badge-red {
            background: rgba(239, 68, 68, 0.12);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }
        
        .badge-grey {
            background: rgba(148, 163, 184, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

def get_signal_badge_html(score: float) -> str:
    """
    Generates a premium HTML badge based on the signal strength score.
    """
    if score >= 70.0:
        return f'<span class="custom-badge badge-amber">🔥 Strong ({score} pts)</span>'
    elif score >= 50.0:
        return f'<span class="custom-badge badge-blue">📈 Moderate ({score} pts)</span>'
    else:
        return f'<span class="custom-badge badge-grey">⏳ Weak ({score} pts)</span>'

def get_day_change_badge_html(pct: float) -> str:
    """
    Generates an HTML badge for price percentage change.
    """
    if pct > 0:
        return f'<span class="custom-badge badge-green">▲ +{pct:.2f}%</span>'
    elif pct < 0:
        return f'<span class="custom-badge badge-red">▼ {pct:.2f}%</span>'
    else:
        return f'<span class="custom-badge badge-grey">0.00%</span>'
