"""ui/theme.py — CSS и глобальные стили."""
from __future__ import annotations
import streamlit as st


CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html,body,#root,.stApp,.stApp>div,
[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>div,
[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stBottom"],[data-testid="stBottom"]>div,
[data-testid="stAppViewBlockContainer"],[data-testid="stVerticalBlock"],
section.main,section.main>div,.main,.main>div,.block-container{
background-color:#05070f!important;
background-image:linear-gradient(180deg,#05070f 0%,#0b0f1a 50%,#05070f 100%)!important;}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent!important;}
.stMarkdown,.stMarkdown p,.stMarkdown li{color:#e6eaf2;}
header,#MainMenu{visibility:hidden}
section[data-testid="stSidebar"]{background:rgba(8,11,20,.85)!important;
 border-right:1px solid rgba(255,255,255,.07);}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span{color:#e6eaf2!important;}
section.stButton>button{background:linear-gradient(135deg,#0ea5e9,#8b5cf6,#ec4899);
 color:#fff;border:none;border-radius:14px;font-weight:800;}
.hero{padding:24px 28px;border-radius:22px;margin-bottom:16px;
 border:1px solid rgba(255,255,255,.10);
 background:linear-gradient(130deg,rgba(14,165,233,.20),rgba(139,92,246,.16),rgba(236,72,153,.14));}
.hero h1{margin:0;font-size:2.2rem;font-weight:800;
 background:linear-gradient(92deg,#22d3ee,#a78bfa,#f472b6);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px;}
.kpi{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);
 border-radius:16px;padding:12px 14px;}
.kpi .t{color:#7dd3fc;font-size:.65rem;text-transform:uppercase;font-weight:700;}
.kpi .v{font-size:1.4rem;font-weight:800;color:#fff;}
.kpi .v.g{color:#34d399;}.kpi .v.y{color:#fbbf24;}.kpi .v.r{color:#f87171;}
.betcard{background:rgba(10,14,24,.72);border:1px solid rgba(255,255,255,.09);
 border-left:4px solid rgba(148,163,184,.4);border-radius:14px;
 padding:12px 16px;margin-bottom:8px;font-size:.87rem;color:#e6eaf2;}
.betcard.pending{border-left-color:#fbbf24;}
.betcard.won{border-left-color:#34d399;}
.betcard.lost{border-left-color:#f87171;}
.betcard.push{border-left-color:#94a3b8;}
.betcard.void{border-left-color:#64748b;opacity:.7;}
.betcard .score{font-weight:900;padding:2px 10px;border-radius:9px;
 margin-left:6px;background:rgba(52,211,153,.25);color:#6ee7b7;}
.nbr-loader{display:flex;align-items:center;gap:14px;padding:18px;
 background:rgba(10,14,24,.72);border:1px solid rgba(34,211,238,.35);
 border-radius:16px;margin-bottom:12px;}
.nbr-ring{width:36px;height:36px;border-radius:50%;
 border:3px solid rgba(34,211,238,.2);border-top-color:#22d3ee;
 animation:nbr-spin 1s linear infinite;flex-shrink:0;}
@keyframes nbr-spin{to{transform:rotate(360deg);}}
.nbr-text{flex:1;color:#a5f3fc;font-size:.95rem;}
.nbr-text b{color:#fff;}
.nbr-bar{height:6px;background:rgba(255,255,255,.08);border-radius:3px;
 overflow:hidden;margin-top:8px;}
.nbr-bar-fill{height:100%;background:linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);
 background-size:200% 100%;animation:nbr-bar-move 2s linear infinite;
 border-radius:3px;transition:width .4s ease;}
@keyframes nbr-bar-move{0%{background-position:0% 0%;}
 100%{background-position:200% 0%;}}
</style>"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
