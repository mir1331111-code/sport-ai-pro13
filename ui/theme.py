"""ui/theme.py — Neon Glass 2.0 с анимациями и жёстким sidebar."""
from __future__ import annotations
import streamlit as st


CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&display=swap');

* { font-family: 'Inter', -apple-system, sans-serif; }

html, body, .stApp, .stApp > div,
[data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div,
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stBottom"], [data-testid="stBottom"] > div,
section.main, section.main > div, .main, .main > div, .block-container {
    background: #05070f !important;
    background-image:
        radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,.10), transparent 60%),
        radial-gradient(1000px 500px at 90% 0%, rgba(168,85,247,.10), transparent 55%),
        radial-gradient(900px 600px at 50% 110%, rgba(236,72,153,.07), transparent 60%),
        linear-gradient(180deg, #05070f 0%, #070b15 50%, #05070f 100%) !important;
    background-attachment: fixed !important;
    color: #e6eaf2 !important;
}
header, #MainMenu, footer { visibility: hidden; height: 0; }

/* ============ ЖЁСТКО ОТКРЫТЫЙ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: rgba(8,11,20,.85) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255,255,255,.06);
    transform: none !important;
    margin-left: 0 !important;
    min-width: 290px !important;
    max-width: 290px !important;
    visibility: visible !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    transform: none !important;
    visibility: visible !important;
}
/* Убираем кнопку сворачивания и стрелки */
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"] {
    display: none !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #e6eaf2 !important; }

/* ============ КНОПКИ ============ */
section.stButton > button,
section.stDownloadButton > button {
    position: relative;
    background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #ec4899 100%);
    background-size: 200% 200%;
    color: #fff !important;
    border: none !important;
    border-radius: 14px;
    font-weight: 700;
    letter-spacing: .3px;
    padding: 12px 22px;
    overflow: hidden;
    box-shadow:
        0 8px 24px rgba(139,92,246,.35),
        inset 0 0 0 1px rgba(255,255,255,.10);
    transition: all .4s cubic-bezier(.4,0,.2,1);
}
section.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 0; height: 0;
    border-radius: 50%;
    background: rgba(255,255,255,.35);
    transform: translate(-50%, -50%);
    transition: width .5s, height .5s;
}
section.stButton > button:hover::before {
    width: 400px; height: 400px;
}
section.stButton > button:hover,
section.stDownloadButton > button:hover {
    background-position: 100% 100%;
    transform: translateY(-2px) scale(1.02);
    box-shadow:
        0 16px 40px rgba(139,92,246,.55),
        0 0 40px rgba(34,211,238,.35),
        inset 0 0 0 1px rgba(255,255,255,.20);
}
section.stButton > button:active { transform: translateY(0) scale(.98); }

/* ============ HERO ============ */
.hero {
    position: relative;
    padding: 32px 36px;
    border-radius: 26px;
    margin-bottom: 22px;
    overflow: hidden;
    background: linear-gradient(130deg, rgba(14,165,233,.18), rgba(139,92,246,.16), rgba(236,72,153,.14));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 24px 60px -20px rgba(139,92,246,.35);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    animation: hero-fade-in .8s ease-out;
}
@keyframes hero-fade-in {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(600px 300px at 15% 0%, rgba(34,211,238,.20), transparent 60%),
                radial-gradient(500px 300px at 85% 100%, rgba(236,72,153,.18), transparent 60%);
    pointer-events: none;
    animation: hero-glow 6s ease-in-out infinite alternate;
}
@keyframes hero-glow {
    0% { opacity: .7; }
    100% { opacity: 1; }
}
.hero h1 {
    position: relative;
    margin: 0;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -.5px;
    background: linear-gradient(92deg, #22d3ee 0%, #a78bfa 45%, #f472b6 100%);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: nbr-hero-shift 6s linear infinite;
}
@keyframes nbr-hero-shift {
    0% { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}
.hero p {
    position: relative;
    margin: 8px 0 0 0;
    color: #b8c2d6;
    font-size: .92rem;
    font-weight: 500;
}

/* ============ KPI ============ */
.kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-top: 18px;
    position: relative;
}
.kpi {
    background: rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 18px;
    padding: 16px 18px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: all .35s cubic-bezier(.4,0,.2,1);
    position: relative;
    overflow: hidden;
    animation: kpi-slide-up .6s ease-out backwards;
}
.kpi:nth-child(1) { animation-delay: .05s; }
.kpi:nth-child(2) { animation-delay: .1s; }
.kpi:nth-child(3) { animation-delay: .15s; }
.kpi:nth-child(4) { animation-delay: .2s; }
@keyframes kpi-slide-up {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(34,211,238,.8), transparent);
    animation: kpi-line 3s ease-in-out infinite;
}
@keyframes kpi-line {
    0%, 100% { left: -100%; }
    50% { left: 100%; }
}
.kpi:hover {
    border-color: rgba(34,211,238,.4);
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(34,211,238,.20);
}
.kpi .t {
    color: #7dd3fc;
    font-size: .68rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1.4px;
    margin-bottom: 6px;
}
.kpi .v {
    font-size: 1.7rem;
    font-weight: 900;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -.5px;
    line-height: 1.1;
}
.kpi .v.g { color: #34d399; text-shadow: 0 0 24px rgba(52,211,153,.45); }
.kpi .v.y { color: #fbbf24; text-shadow: 0 0 24px rgba(251,191,36,.45); }
.kpi .v.r { color: #f87171; text-shadow: 0 0 24px rgba(248,113,113,.45); }

/* ============ VERDICT CARD ============ */
.vcard {
    border-radius: 22px;
    margin-bottom: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.10);
    box-shadow:
        0 20px 50px -20px rgba(0,0,0,.6),
        inset 0 0 0 1px rgba(255,255,255,.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: transform .35s cubic-bezier(.4,0,.2,1), box-shadow .35s;
    animation: vcard-slide-in .7s ease-out backwards;
}
.vcard:nth-child(1) { animation-delay: .05s; }
.vcard:nth-child(2) { animation-delay: .1s; }
.vcard:nth-child(3) { animation-delay: .15s; }
.vcard:nth-child(4) { animation-delay: .2s; }
.vcard:nth-child(5) { animation-delay: .25s; }
@keyframes vcard-slide-in {
    from { opacity: 0; transform: translateY(24px) scale(.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}
.vcard:hover {
    transform: translateY(-4px);
    box-shadow:
        0 32px 72px -20px rgba(34,211,238,.30),
        inset 0 0 0 1px rgba(255,255,255,.10);
}

.team-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}
.team-badge {
    width: 40px; height: 40px;
    object-fit: contain;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,.6));
    flex-shrink: 0;
    transition: transform .4s cubic-bezier(.4,0,.2,1);
}
.vcard:hover .team-badge { transform: scale(1.1) rotate(-3deg); }
.team-badge-fallback {
    width: 40px; height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(34,211,238,.3), rgba(139,92,246,.3));
    display: inline-flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 900; font-size: 1.1rem;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,.15);
}
.team-name {
    font-size: 1.35rem;
    font-weight: 800;
    color: #fff;
    text-shadow: 0 2px 12px rgba(0,0,0,.8);
    letter-spacing: -.3px;
}

.league-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(34,211,238,.12);
    border: 1px solid rgba(34,211,238,.25);
    color: #a5f3fc;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .5px;
}
.league-badge img {
    width: 16px; height: 16px;
    object-fit: contain;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,.5));
}
.date-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(251,191,36,.12);
    border: 1px solid rgba(251,191,36,.25);
    color: #fde68a;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .5px;
}

/* ============ BET CARD (pulse на pending) ============ */
.betcard {
    background: rgba(10,14,24,.72);
    border: 1px solid rgba(255,255,255,.08);
    border-left: 4px solid rgba(148,163,184,.4);
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: .88rem;
    color: #e6eaf2;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: all .35s ease;
    animation: betcard-fade .5s ease-out backwards;
}
.betcard:hover { transform: translateX(4px); border-left-width: 6px; }
.betcard.pending {
    border-left-color: #fbbf24;
    box-shadow: -6px 0 24px -8px rgba(251,191,36,.6);
    animation: betcard-fade .5s ease-out backwards, pending-pulse 2.5s ease-in-out infinite;
}
@keyframes pending-pulse {
    0%, 100% { box-shadow: -6px 0 24px -8px rgba(251,191,36,.6); }
    50% { box-shadow: -6px 0 32px -4px rgba(251,191,36,.9); }
}
@keyframes betcard-fade {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
.betcard.won { border-left-color: #34d399; box-shadow: -6px 0 24px -8px rgba(52,211,153,.6); }
.betcard.lost { border-left-color: #f87171; box-shadow: -6px 0 24px -8px rgba(248,113,113,.6); }
.betcard.push { border-left-color: #94a3b8; }
.betcard.void { border-left-color: #64748b; opacity: .6; }

/* ============ LOADER ============ */
.nbr-loader {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 24px;
    background: rgba(10,14,24,.72);
    border: 1px solid rgba(34,211,238,.35);
    border-radius: 18px;
    margin-bottom: 14px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    animation: loader-appear .4s ease-out;
}
@keyframes loader-appear {
    from { opacity: 0; transform: scale(.95); }
    to { opacity: 1; transform: scale(1); }
}
.nbr-ring {
    width: 40px; height: 40px; border-radius: 50%;
    border: 3px solid rgba(34,211,238,.15);
    border-top-color: #22d3ee;
    border-right-color: #a78bfa;
    animation: nbr-spin 1s linear infinite;
    flex-shrink: 0;
    box-shadow: 0 0 24px rgba(34,211,238,.5);
}
@keyframes nbr-spin { to { transform: rotate(360deg); } }
.nbr-text { flex: 1; color: #a5f3fc; font-size: .98rem; }
.nbr-text b { color: #fff; font-weight: 700; }
.nbr-bar {
    height: 6px; background: rgba(255,255,255,.08);
    border-radius: 3px; overflow: hidden; margin-top: 10px;
}
.nbr-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
    background-size: 200% 100%;
    animation: nbr-bar-move 2s linear infinite;
    border-radius: 3px;
    transition: width .4s ease;
    box-shadow: 0 0 12px rgba(168,85,247,.6);
}
@keyframes nbr-bar-move {
    0% { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

/* ============ METRICS ============ */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
    color: #fff !important;
    animation: metric-pop .5s ease-out;
}
@keyframes metric-pop {
    from { opacity: 0; transform: scale(.9); }
    to { opacity: 1; transform: scale(1); }
}
[data-testid="stMetricLabel"] {
    color: #7dd3fc !important;
    text-transform: uppercase !important;
    font-size: .72rem !important;
    letter-spacing: 1px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ============ TABS ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,.03);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 18px;
    color: #b8c2d6;
    font-weight: 600;
    font-size: .9rem;
    transition: all .3s cubic-bezier(.4,0,.2,1);
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(34,211,238,.08);
    color: #e6eaf2;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(34,211,238,.22), rgba(139,92,246,.22)) !important;
    color: #fff !important;
    box-shadow:
        inset 0 0 0 1px rgba(34,211,238,.4),
        0 4px 16px rgba(34,211,238,.25);
    transform: translateY(-1px);
}

/* ============ INPUTS ============ */
input[type="text"], input[type="password"], textarea {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    border-radius: 12px !important;
    color: #e6eaf2 !important;
    transition: all .3s ease;
}
input[type="text"]:focus, input[type="password"]:focus {
    border-color: rgba(34,211,238,.5) !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,.12), 0 0 20px rgba(34,211,238,.15) !important;
}

/* ============ SLIDER ============ */
[data-testid="stSlider"] [role="slider"] {
    background: linear-gradient(135deg, #22d3ee, #a78bfa) !important;
    box-shadow: 0 0 12px rgba(34,211,238,.6) !important;
    transition: transform .2s ease;
}
[data-testid="stSlider"] [role="slider"]:hover {
    transform: scale(1.15);
    box-shadow: 0 0 20px rgba(34,211,238,.9) !important;
}

/* ============ SELECTBOX ============ */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,.04) !important;
    border-color: rgba(255,255,255,.10) !important;
    border-radius: 12px !important;
    transition: all .3s ease;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(34,211,238,.4) !important;
}

/* ============ EXPANDER ============ */
.streamlit-expanderHeader {
    background: rgba(255,255,255,.03) !important;
    border-radius: 12px !important;
    transition: all .3s ease;
}
.streamlit-expanderHeader:hover {
    background: rgba(34,211,238,.06) !important;
}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,.02); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(34,211,238,.5), rgba(139,92,246,.5));
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, rgba(34,211,238,.8), rgba(139,92,246,.8));
}

/* ============ GLOW ANIMATION для активных кнопок ============ */
section.stButton > button[kind="primary"] {
    animation: btn-glow 2.5s ease-in-out infinite alternate;
}
@keyframes btn-glow {
    0% {
        box-shadow: 0 8px 24px rgba(139,92,246,.35), inset 0 0 0 1px rgba(255,255,255,.10);
    }
    100% {
        box-shadow: 0 12px 36px rgba(139,92,246,.60), 0 0 30px rgba(34,211,238,.35), inset 0 0 0 1px rgba(255,255,255,.20);
    }
}
</style>"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
