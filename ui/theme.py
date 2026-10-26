"""Unified product design system for NEURO BET PRO."""
from __future__ import annotations

import streamlit as st


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root {
  --bg: #080d12;
  --panel: #101820;
  --panel-raised: #141f29;
  --line: #263642;
  --text: #edf4f5;
  --muted: #91a4ad;
  --mint: #72e0bd;
  --mint-strong: #36c99c;
  --blue: #83baff;
  --amber: #f3c879;
  --red: #ff8d91;
  --radius: 16px;
}
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}
[data-testid="stHeader"] { background: rgba(8,13,18,.92) !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1500px !important; padding: 1.8rem 2.3rem 3.5rem !important; }
p, label, .stMarkdown { color: var(--text); }
small, .muted { color: var(--muted) !important; }
h1,h2,h3,h4 { color: var(--text) !important; letter-spacing: -.035em !important; }
h1 { font-size: clamp(1.8rem, 3vw, 2.65rem) !important; font-weight: 800 !important; }
h2 { font-size: 1.55rem !important; font-weight: 750 !important; }
h3 { font-size: 1.18rem !important; font-weight: 700 !important; }
hr { border-color: var(--line) !important; }

/* Sidebar: calm navigation, not a permanently forced drawer. */
section[data-testid="stSidebar"] {
  background: #0c131a !important;
  border-right: 1px solid #1d2a33 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #c4d2d7 !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: var(--mint) !important; }

/* Shared surface system */
[data-testid="stMetric"], .kpi, .signal-metric {
  background: linear-gradient(145deg, #121d26, #0f171e) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
  color: var(--muted) !important; font-size: .75rem !important;
  text-transform: uppercase; letter-spacing: .09em; font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
  color: var(--text) !important; font-family: 'IBM Plex Mono', monospace !important;
  font-size: 1.7rem !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="stAlert"] { border-radius: 12px !important; }

/* Buttons: clear hierarchy, restrained interaction. */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  min-height: 42px; border-radius: 11px !important; padding: .55rem 1rem !important;
  border: 1px solid #344650 !important; background: #17232c !important;
  color: var(--text) !important; font-weight: 700 !important; box-shadow: none !important;
  transition: background .16s ease, border-color .16s ease, transform .16s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #20333d !important; border-color: #55716f !important; transform: translateY(-1px);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: var(--mint) !important; border-color: var(--mint) !important;
  color: #07231c !important; box-shadow: 0 5px 18px rgba(54,201,156,.12) !important;
}
.stButton > button[kind="primary"]:hover {
  background: #91ebcf !important; border-color: #91ebcf !important;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 2px solid var(--blue) !important; outline-offset: 3px;
}

/* Inputs */
input, textarea, [data-baseweb="select"] > div, [data-testid="stDateInput"] input {
  background: #0b141b !important; border-color: #2a3a45 !important;
  color: var(--text) !important; border-radius: 10px !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--mint) !important; border-color: var(--mint) !important; }
[data-testid="stCheckbox"] label { color: #c4d2d7 !important; }

/* Tabs and expanders */
.stTabs [data-baseweb="tab-list"] {
  gap: 5px; background: #0d151c; border: 1px solid #202f39;
  padding: 5px; border-radius: 13px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 9px; padding: 9px 15px; color: var(--muted) !important;
  font-weight: 700; transition: background .15s ease;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important; background: #20342f !important;
  box-shadow: inset 0 0 0 1px #355a4f !important;
}
div[data-testid="stExpander"] {
  border: 1px solid #263640 !important; border-radius: 13px !important;
  background: #0e171e !important; overflow: hidden;
}
div[data-testid="stExpander"] details summary { padding: 12px 15px !important; }

/* Scanner product header */
.scanner-hero {
  position: relative; overflow: hidden; padding: 30px 32px 25px; margin: 0 0 20px;
  border-radius: 19px; border: 1px solid #2b4944;
  background: linear-gradient(115deg, #12231f 0%, #101b22 52%, #121b27 100%);
  box-shadow: 0 16px 42px rgba(0,0,0,.17);
}
.scanner-hero:after {
  content: ""; position: absolute; width: 300px; height: 300px; right: -125px; top: -180px;
  border-radius: 50%; background: rgba(114,224,189,.11); filter: blur(18px); pointer-events: none;
}
.scanner-kicker { position: relative; font-size: .68rem; font-weight: 800; letter-spacing: .16em; color: var(--mint); margin-bottom: 9px; }
.scanner-title { position: relative; font-size: clamp(1.8rem,3vw,2.55rem); font-weight: 800; letter-spacing: -.055em; color: #f3faf8; line-height: 1.12; }
.scanner-subtitle { position: relative; color: #a9bbc1; font-size: .94rem; margin-top: 10px; max-width: 760px; line-height: 1.6; }
.scanner-steps { display: flex; flex-wrap: wrap; gap: 7px; align-items: center; margin-top: 20px; color: #cbd8d9; font-size: .75rem; font-weight: 650; }
.scanner-steps span { padding: 7px 10px; border-radius: 8px; background: rgba(255,255,255,.035); border: 1px solid #2b3e43; }
.scanner-steps i { color: #59766f; font-style: normal; }
.scanner-help-pill { padding: 11px 12px; margin-bottom: 4px; border: 1px solid #31564b; border-radius: 10px; background: #12271f; color: #a9e9d1; font-size: .76rem; font-weight: 700; text-align: center; }

/* Match and pick cards; retain existing component contracts. */
.vcard {
  border-radius: 16px !important; margin-bottom: 15px; overflow: hidden;
  border: 1px solid #293b45 !important; background: #101a22 !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.12) !important;
  transition: border-color .16s ease, transform .16s ease;
}
.vcard:hover { border-color: #47645e !important; transform: translateY(-2px); }
.team-name { color: #f0f7f5 !important; font-size: 1.12rem !important; font-weight: 750 !important; }
.team-badge-fallback { background: #1d3932 !important; border: 1px solid #376354 !important; color: var(--mint) !important; }
.league-badge { background: #162c29 !important; border: 1px solid #2b5148 !important; color: #a4e8d1 !important; }
.date-badge { background: #2c2619 !important; border: 1px solid #57472a !important; color: #f1d49b !important; }
.signal-grid { display: grid; grid-template-columns: repeat(6,minmax(0,1fr)); gap: 8px; margin-top: 8px; }
.signal-metric { padding: 10px 11px !important; min-width: 0; }
.signal-metric span { display:block; color: var(--muted); font-size: .62rem; font-weight: 800; letter-spacing: .06em; margin-bottom: 5px; }
.signal-metric b { color: var(--text); font-family: 'IBM Plex Mono', monospace; font-size: .98rem; overflow-wrap: anywhere; }
.signal-metric.model b { color: #a5d0ff; }
.signal-metric.market b { color: #c5b4ff; }
.signal-metric.value b { color: var(--mint); }
.signal-layers { display:flex; flex-wrap:wrap; gap:7px; margin-top:11px; }
.layer { padding:5px 9px; border-radius:7px; font-size:.65rem; font-weight:750; border:1px solid #2c3b43; }
.model-layer { color:#b2d6ff; background:#142333; }
.market-layer { color:#d0c2ff; background:#211d32; }
.value-layer { color:#a3e8ce; background:#142820; }
.betcard {
  background: #101a22 !important; border: 1px solid #293a44 !important;
  border-left: 3px solid #4b626b !important; border-radius: 12px !important;
  padding: 14px 16px; margin-bottom: 9px; color: var(--text) !important;
}
.betcard.pending { border-left-color: var(--amber) !important; }
.betcard.won { border-left-color: var(--mint-strong) !important; }
.betcard.lost { border-left-color: var(--red) !important; }
.betcard.push, .betcard.void { border-left-color: #82949b !important; }
.nbr-loader { display:flex; align-items:center; gap:14px; padding:18px 20px; background:#101b22; border:1px solid #2b5148; border-radius:14px; margin-bottom:14px; }
.nbr-ring { width:30px; height:30px; border-radius:50%; border:3px solid #29423c; border-top-color:var(--mint); animation:nbr-spin .9s linear infinite; flex-shrink:0; }
@keyframes nbr-spin { to { transform: rotate(360deg); } }
.nbr-text { color:#a9e9d1; font-size:.92rem; }
.nbr-text b { color:#f1faf6; }
.nbr-bar { height:5px; background:#26343a; border-radius:99px; overflow:hidden; margin-top:9px; }
.nbr-bar-fill { height:100%; background:var(--mint-strong); border-radius:99px; transition:width .25s ease; }

/* Legacy overview components use the same unified surfaces. */
.hero {
  padding: 27px 30px; border-radius: 18px; margin-bottom: 20px;
  border: 1px solid #2b4944; background: linear-gradient(115deg,#12231f,#101b22 58%,#121b27);
  box-shadow: 0 12px 34px rgba(0,0,0,.16);
}
.hero h1 { color:#f3faf8 !important; -webkit-text-fill-color: initial !important; background:none !important; }
.hero p { color:#a9bbc1 !important; }
.hero-eyebrow { color:var(--mint) !important; letter-spacing:.14em; font-size:.68rem; font-weight:800; }
.hero-status { color:#a9e9d1; background:#12271f; border:1px solid #31564b; border-radius:99px; padding:6px 10px; }
.status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--mint); margin-right:6px; }
.kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:16px 0; }
.kpi .t { color:var(--muted); font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; font-weight:800; margin-bottom:6px; }
.kpi .v { color:var(--text); font-family:'IBM Plex Mono',monospace; font-size:1.65rem; font-weight:700; }
.kpi .v.g { color:var(--mint); } .kpi .v.y { color:var(--amber); } .kpi .v.r { color:var(--red); }

/* Motion and editorial pick cards: restrained, purposeful, reduced-motion safe. */
@keyframes nbr-enter {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes nbr-scan-glow {
  0%,100% { box-shadow: 0 0 0 rgba(114,224,189,0); }
  50% { box-shadow: 0 0 28px rgba(114,224,189,.08); }
}
.scanner-hero { animation: nbr-enter .42s cubic-bezier(.2,.7,.2,1) both; }
.scanner-steps span { transition: transform .18s ease, border-color .18s ease, background .18s ease; }
.scanner-steps span:hover { transform: translateY(-2px); border-color:#426a5e; background:rgba(114,224,189,.07); }
.pick-feed { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:14px 0 24px; }
.pick-story {
  position:relative; overflow:hidden; min-width:0; padding:20px 21px 18px;
  background:linear-gradient(135deg,#121f27 0%,#101820 70%,#13231f 100%);
  border:1px solid #2b3e46; border-radius:17px;
  animation:nbr-enter .36s cubic-bezier(.2,.7,.2,1) both;
  transition:transform .2s ease,border-color .2s ease,background .2s ease;
}
.pick-story:hover { transform:translateY(-3px); border-color:#4c7869; }
.pick-story:before { content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--mint); opacity:.85; }
.pick-story-top { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:17px; }
.pick-league { color:#9aafb6; font-size:.72rem; font-weight:700; letter-spacing:.04em; }
.pick-status { border:1px solid #31594b; background:#142a22; color:#a9e9d1; border-radius:999px; padding:4px 8px; font-size:.67rem; font-weight:800; white-space:nowrap; }
.pick-status.watch { color:#f1d49b; background:#2b261b; border-color:#5a4b2e; }
.pick-match { color:#f2f7f6; font-size:1.18rem; font-weight:800; line-height:1.32; letter-spacing:-.025em; margin:0 0 7px; }
.pick-time { color:#81969f; font-size:.76rem; margin-bottom:17px; }
.pick-divider { border-top:1px solid #293a42; margin:0 0 15px; }
.pick-choice-label { color:#81969f; font-size:.67rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:6px; }
.pick-choice { color:#8de6c6; font-size:1.27rem; font-weight:850; line-height:1.3; overflow-wrap:anywhere; }
.pick-meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.pick-chip { color:#c4d2d7; border:1px solid #30414a; background:#111d24; border-radius:8px; padding:6px 9px; font-size:.75rem; }
.pick-ai { position:relative; margin-top:15px; padding:12px 13px; background:rgba(8,14,18,.48); border:1px solid #293c43; border-radius:11px; color:#b8c8cc; font-size:.83rem; line-height:1.58; }
.pick-ai b { color:#a9e9d1; display:block; font-size:.67rem; letter-spacing:.1em; margin-bottom:5px; text-transform:uppercase; }
.pick-ai-empty { color:#8a9ca3; font-style:italic; }
@media (max-width: 760px) {
  .block-container { padding:1rem .85rem 2.2rem !important; }
  .scanner-hero { padding:22px 19px 19px; }
  .pick-feed { grid-template-columns:1fr; gap:11px; }
  .pick-story { padding:17px; }
  .pick-match { font-size:1.1rem; }
  .signal-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
}

/* Tables and code should remain readable. */
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #263640; border-radius: 12px; overflow: hidden; }
code, pre { font-family: 'IBM Plex Mono', monospace !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0b1116; }
::-webkit-scrollbar-thumb { background: #31434b; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #4b6668; }

@media (max-width: 900px) {
  .block-container { padding: 1.2rem 1rem 2.5rem !important; }
  .signal-grid { grid-template-columns: repeat(3,minmax(0,1fr)); }
  .scanner-hero { padding: 23px 19px 20px; }
  .scanner-steps i { display: none; }
}
@media (max-width: 560px) {
  .signal-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .team-name { font-size: 1rem !important; }
  .scanner-steps span { font-size: .68rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
}

/* Product-wide finishing pass: consistent cards, spacing, and mobile layouts. */
.scanner-section-label {
  color: var(--muted); font-size: .68rem; font-weight: 800;
  letter-spacing: .14em; margin: 8px 0 10px;
}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { min-width: 0; }
.stMarkdown p { line-height: 1.6; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }
.team-row {
  display:flex; align-items:center; flex-wrap:wrap; gap:10px;
  padding: 4px 0 2px; min-width:0;
}
.team-row .team-badge, .team-badge {
  width:34px; height:34px; object-fit:contain; border-radius:50%;
  background:#f7faf9; padding:3px; border:1px solid #344a51; flex-shrink:0;
}
.team-badge-fallback {
  display:inline-flex; width:34px; height:34px; align-items:center; justify-content:center;
  border-radius:50%; font-weight:800; flex-shrink:0;
}
.vcard > div { min-width:0; }
.vcard .league-badge, .vcard .date-badge {
  display:inline-flex; align-items:center; gap:6px; padding:5px 9px;
  border-radius:7px; font-size:.68rem; font-weight:750; line-height:1.3;
}
.league-badge img { width:15px; height:15px; object-fit:contain; }
.vcard ul { color:#b9c9ce !important; }
.vcard li { padding:2px 0; }
.vcard [style*="JetBrains Mono"] { font-family:'IBM Plex Mono',monospace !important; }
.betcard { line-height:1.55; overflow-wrap:anywhere; }
.stDataFrame, [data-testid="stDataFrame"] { background:#0e171e; }
.stProgress > div > div { background:var(--mint) !important; }
[data-testid="stStatusWidget"] { border-radius:10px; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.65rem; }
section[data-testid="stSidebar"] [data-testid="stExpander"] { background:#0e171e !important; }
button[aria-label="View fullscreen"] { color:var(--muted) !important; }
@media (max-width: 760px) {
  .block-container { padding: .85rem .7rem 2rem !important; }
  [data-testid="stMetricValue"] { font-size:1.35rem !important; }
  .vcard > div { padding-left:14px !important; padding-right:14px !important; }
  .team-row { gap:7px; }
  .team-badge, .team-badge-fallback { width:29px; height:29px; }
  .team-name { font-size:.93rem !important; }
  .kpis { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
</style>
"""


def inject() -> None:
    """Inject the app-wide design system once per Streamlit render."""
    st.markdown(CSS, unsafe_allow_html=True)
