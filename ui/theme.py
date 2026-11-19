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

/* v14 analytics-terminal navigation and section shell */
.terminal-section-head {
  position:relative; display:flex; align-items:flex-end; justify-content:space-between;
  gap:18px; padding:20px 23px; margin:10px 0 18px;
  border:1px solid #2b4944; border-radius:16px;
  background:linear-gradient(110deg,#12251f 0%,#101b22 58%,#131d27 100%);
  box-shadow:0 10px 30px rgba(0,0,0,.14); overflow:hidden;
}
.terminal-section-head:after {
  content:""; position:absolute; width:210px; height:210px; right:-80px; top:-135px;
  border-radius:50%; background:rgba(114,224,189,.10); filter:blur(13px);
}
.terminal-section-kicker {
  position:relative; z-index:1; color:#72e0bd; font-size:.66rem;
  font-weight:850; letter-spacing:.17em; text-transform:uppercase; margin-bottom:6px;
}
.terminal-section-title {
  position:relative; z-index:1; color:#f2f8f6; font-size:1.55rem;
  letter-spacing:-.045em; line-height:1.15; font-weight:850;
}
.terminal-section-subtitle {
  position:relative; z-index:1; color:#9db0b7; font-size:.84rem;
  line-height:1.5; margin-top:7px; max-width:760px;
}
.terminal-section-status {
  position:relative; z-index:1; flex-shrink:0; color:#a9e9d1;
  background:#142a22; border:1px solid #31594b; border-radius:999px;
  padding:7px 11px; font-size:.66rem; font-weight:850; letter-spacing:.08em;
}
div[data-testid="stRadio"] > label { display:none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] {
  display:flex !important; flex-wrap:wrap !important; gap:7px !important;
  align-items:center !important; padding:7px !important; margin:0 0 17px !important;
  background:#0c151c !important; border:1px solid #273943 !important;
  border-radius:14px !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label {
  position:relative !important; display:inline-flex !important; align-items:center !important;
  gap:0 !important; padding:9px 13px !important; margin:0 !important;
  border:1px solid transparent !important; border-radius:10px !important;
  background:transparent !important; cursor:pointer !important;
  transition:background .16s ease,border-color .16s ease,transform .16s ease !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
  background:#14241f !important; border-color:#2c4a40 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
  background:linear-gradient(135deg,#1b3a30,#18302c) !important;
  border-color:#3c6e5d !important; box-shadow:inset 0 0 0 1px rgba(114,224,189,.07) !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
  display:none !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label p {
  color:#9eb0b6 !important; font-size:.76rem !important; font-weight:800 !important;
  letter-spacing:.035em !important; white-space:nowrap !important; margin:0 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
  color:#b6f1dc !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] input { position:absolute !important; opacity:0 !important; pointer-events:none !important; }
.analytics-grid-gap { margin-top:10px; }
div[data-testid="stMetric"] { min-height:91px; }
[data-testid="stDataFrame"] { box-shadow:0 8px 24px rgba(0,0,0,.10); }
@media (max-width: 900px) {
  div[data-testid="stRadio"] div[role="radiogroup"] { gap:5px !important; }
  div[data-testid="stRadio"] div[role="radiogroup"] > label { padding:8px 10px !important; }
  .terminal-section-head { padding:17px 18px; }
}
@media (max-width: 600px) {
  div[data-testid="stRadio"] div[role="radiogroup"] { overflow-x:auto !important; flex-wrap:nowrap !important; }
  div[data-testid="stRadio"] div[role="radiogroup"] > label { flex:0 0 auto !important; }
  .terminal-section-title { font-size:1.28rem; }
  .terminal-section-status { display:none; }
}

/* NEURO BET PRO · premium football intelligence workspace */
.sp-appbar {
  display:flex; align-items:center; gap:12px; min-height:56px; margin:0 0 12px;
  padding:10px 15px; border:1px solid #243a42; border-radius:14px;
  background:linear-gradient(100deg,#0e191f 0%,#111c23 64%,#14251f 100%);
  box-shadow:0 8px 25px rgba(0,0,0,.16);
}
.sp-brandmark {
  width:34px;height:34px;display:grid;place-items:center;border-radius:10px;
  color:#071e17;font-size:1.04rem;font-weight:950;
  background:linear-gradient(145deg,#a5f2d7,#42c99c);
  box-shadow:0 5px 16px rgba(66,201,156,.18);
}
.sp-brandmark span { color:#167956;font-size:.55rem; margin-left:1px; }
.sp-brandname { color:#f2f8f6;font-size:.95rem;font-weight:900;letter-spacing:.04em;line-height:1.1; }
.sp-brandname em { color:#72e0bd;font-style:normal; }
.sp-brand-sub { color:#789099;font-size:.57rem;font-weight:800;letter-spacing:.14em;margin-top:4px; }
.sp-appbar-spacer { flex:1; }
.sp-appbar-chip {
  display:inline-flex;align-items:center;gap:7px;padding:7px 10px;
  border:1px solid #315246;border-radius:999px;background:#13271f;
  color:#a9e9d1;font-size:.65rem;font-weight:850;letter-spacing:.07em;white-space:nowrap;
}
.sp-appbar-chip i,.sp-system-pill i {
  width:7px;height:7px;border-radius:50%;background:#72e0bd;
  box-shadow:0 0 10px rgba(114,224,189,.55);display:inline-block;
}
.sp-appbar-bank { color:#d4e0e1;font:700 .74rem 'IBM Plex Mono',monospace;white-space:nowrap; }
.sp-dashboard-hero {
  position:relative;display:flex;align-items:center;justify-content:space-between;gap:24px;
  overflow:hidden;min-height:190px;padding:28px 32px;margin:0 0 17px;
  border:1px solid #31554b;border-radius:21px;
  background:radial-gradient(circle at 82% 32%,rgba(54,201,156,.15),transparent 28%),
             linear-gradient(112deg,#142b25 0%,#101d24 54%,#101a25 100%);
  box-shadow:0 17px 42px rgba(0,0,0,.18);
}
.sp-dashboard-hero:after {
  content:"";position:absolute;inset:auto -5% -110px 42%;height:190px;
  background:radial-gradient(ellipse,rgba(73,190,157,.10),transparent 65%);pointer-events:none;
}
.sp-hero-copy { position:relative;z-index:2;max-width:830px; }
.sp-eyebrow { color:#73dfbd;font-size:.65rem;font-weight:900;letter-spacing:.19em;margin-bottom:9px; }
.sp-dashboard-hero h1 {
  color:#f3faf7!important;font-size:clamp(1.8rem,3vw,2.65rem)!important;
  font-weight:900!important;letter-spacing:-.055em!important;margin:0!important;
}
.sp-dashboard-hero p { color:#a9bbc0!important;font-size:.9rem;line-height:1.55;margin:9px 0 0;max-width:710px; }
.sp-hero-tags { display:flex;gap:7px;flex-wrap:wrap;margin-top:18px; }
.sp-hero-tags span {
  padding:6px 9px;border:1px solid rgba(125,174,159,.22);border-radius:7px;
  background:rgba(255,255,255,.035);color:#c4d8d1;font-size:.61rem;font-weight:850;letter-spacing:.08em;
}
.sp-hero-mark { position:relative;z-index:1;display:flex;flex-direction:column;align-items:center;gap:8px;flex:0 0 180px; }
.sp-pitch {
  position:relative;width:154px;height:100px;border:1px solid rgba(151,219,194,.46);
  border-radius:9px;background:linear-gradient(145deg,rgba(50,125,97,.24),rgba(12,34,30,.25));
  box-shadow:inset 0 0 28px rgba(95,222,181,.07),0 0 30px rgba(65,191,147,.08);
}
.sp-pitch:before { content:"";position:absolute;left:50%;top:0;bottom:0;border-left:1px solid rgba(151,219,194,.35); }
.sp-pitch:after { content:"";position:absolute;left:50%;top:50%;width:36px;height:36px;border:1px solid rgba(151,219,194,.35);border-radius:50%;transform:translate(-50%,-50%); }
.sp-pitch i,.sp-pitch b {
  position:absolute;top:28px;width:22px;height:42px;border:1px solid rgba(151,219,194,.35);
}
.sp-pitch i { left:0;border-left:0; }.sp-pitch b { right:0;border-right:0; }
.sp-pitch em { position:absolute;right:20%;top:48%;width:6px;height:6px;background:#8af0ca;border-radius:50%;box-shadow:0 0 12px #72e0bd; }
.sp-hero-mark-caption { color:#70928a;font-size:.57rem;font-weight:900;letter-spacing:.2em;text-align:center;line-height:1.55; }
.sp-panel {
  height:100%;min-width:0;padding:17px 17px 16px;margin:0 0 15px;
  border:1px solid #283c45;border-radius:16px;
  background:linear-gradient(145deg,#121e26 0%,#0f181f 78%,#12231e 100%);
  box-shadow:0 9px 26px rgba(0,0,0,.13);
}
.sp-panel-kicker { color:#72dabb;font-size:.59rem;font-weight:900;letter-spacing:.17em;margin-bottom:6px; }
.sp-panel-title { color:#f0f7f5;font-size:1.05rem;font-weight:850;letter-spacing:-.025em;line-height:1.25; }
.sp-panel-detail { color:#80969f;font-size:.72rem;line-height:1.45;margin:5px 0 13px; }
.sp-match-tile { padding:11px 0 10px;border-top:1px solid #26363d; }
.sp-match-meta { display:flex;justify-content:space-between;gap:8px;color:#8298a0;font-size:.59rem;font-weight:800;letter-spacing:.035em; }
.sp-league-meta { display:inline-flex;align-items:center;gap:6px;min-width:0; }
.sp-league-emblem { display:inline-grid;place-items:center;flex:0 0 17px;width:17px;height:17px;border:1px solid #354a48;border-radius:50%;overflow:hidden;background:#142720; }
.sp-league-emblem img { width:100%;height:100%;object-fit:contain;background:#edf4f1;padding:2px; }
.sp-league-emblem.fallback { color:#87d9b7;font-size:.58rem; }
.sp-team-line { display:flex;align-items:center;justify-content:space-between;gap:9px;margin:10px 0 9px; }
.sp-team-line b { color:#edf5f2;font-size:.8rem;font-weight:800;line-height:1.3;overflow-wrap:anywhere; }
.sp-team-side { display:flex;align-items:center;gap:7px;min-width:0;flex:1; }
.sp-team-side.away { justify-content:flex-end; }
.sp-team-side b { color:#edf5f2;font-size:.78rem;font-weight:800;line-height:1.3;overflow-wrap:anywhere; }
.sp-team-emblem {
  position:relative;display:grid;place-items:center;flex:0 0 27px;width:27px;height:27px;
  border:1px solid #3a514d;border-radius:50%;background:#172d27;overflow:hidden;
}
.sp-team-fallback {
  position:absolute;inset:0;display:grid;place-items:center;color:#a4e8d0;
  font-size:.65rem;font-weight:900;
}
.sp-team-emblem img {
  position:absolute;inset:0;width:100%;height:100%;object-fit:contain;
  background:#f4f7f5;border-radius:50%;padding:2px;
}
.sp-vs { display:grid;place-items:center;flex:0 0 26px;height:26px;border:1px solid #35484e;border-radius:8px;color:#9bb1b5;font-size:.55rem;font-weight:900;background:#17242b; }
.sp-match-bottom { display:flex;align-items:center;justify-content:space-between;gap:8px; }
.sp-pick { color:#8be6c4;font:800 .72rem 'IBM Plex Mono',monospace; }
.sp-ai-tag { color:#a5c4ff;background:#17283a;border:1px solid #304a65;border-radius:6px;padding:4px 6px;font-size:.55rem;font-weight:900;white-space:nowrap; }
.sp-ai-tag.muted,.sp-muted { color:#83969d; }
.sp-ai-score { display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0 14px; }
.sp-ai-score>div { min-width:0;padding:11px;border:1px solid #2c4148;border-radius:11px;background:#0e171e; }
.sp-ai-score small { display:block;color:#82989f;font-size:.55rem;font-weight:900;letter-spacing:.1em; }
.sp-ai-score strong { display:block;color:#8ce7c7;font:800 1.65rem 'IBM Plex Mono',monospace;margin:5px 0 1px; }
.sp-ai-score span { display:block;color:#b7c8c9;font-size:.65rem;overflow-wrap:anywhere; }
.sp-prob-row { display:grid;grid-template-columns:24px minmax(0,1fr) 40px;align-items:center;gap:8px;margin:9px 0; }
.sp-prob-row>span { color:#8da1a6;font-size:.67rem;font-weight:850; }
.sp-prob-row>b { color:#e8f2ee;font:700 .72rem 'IBM Plex Mono',monospace;text-align:right; }
.sp-track,.sp-context-track,.sp-league-track { height:6px;overflow:hidden;border-radius:99px;background:#26363c; }
.sp-track i,.sp-context-track i,.sp-league-track i { display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#3fbb91,#91efcc); }
.sp-track i.draw { background:linear-gradient(90deg,#6487ba,#9dc5f2); }
.sp-track i.away { background:linear-gradient(90deg,#937cc6,#c0a7f2); }
.sp-ai-summary { margin-top:14px;padding:11px 12px;border:1px solid #2a4940;border-radius:10px;background:rgba(13,36,29,.58); }
.sp-ai-summary b { color:#82dfbd;font-size:.58rem;font-weight:900;letter-spacing:.12em; }
.sp-ai-summary p { color:#b9cbc8;font-size:.73rem;line-height:1.55;margin:5px 0 0; }
.sp-ai-summary.muted { border-color:#303b40;background:#111a20; }
.sp-ai-summary.muted b { color:#9aabb0; }
.sp-context-score { display:flex;align-items:baseline;gap:9px;margin:13px 0 15px; }
.sp-context-score strong { color:#8ce7c7;font:800 1.8rem 'IBM Plex Mono',monospace; }
.sp-context-score span { color:#8ea2a6;font-size:.68rem; }
.sp-context-row { display:flex;justify-content:space-between;gap:8px;color:#b7c8c8;font-size:.68rem;margin:12px 0 6px; }
.sp-context-row b { color:#edf5f2;font:700 .68rem 'IBM Plex Mono',monospace; }
.sp-context-track { height:5px;margin-bottom:9px; }
.sp-context-note { color:#80949b;font-size:.65rem;line-height:1.5;margin-top:14px; }
.sp-league-row { display:grid;grid-template-columns:minmax(0,1.2fr) minmax(70px,1fr) 20px;gap:10px;align-items:center;margin:14px 0; }
.sp-league-row span { color:#bdcccd;font-size:.7rem;overflow-wrap:anywhere; }
.sp-league-row b { color:#e5f0eb;font:700 .7rem 'IBM Plex Mono',monospace;text-align:right; }
.sp-league-track { height:6px; }.sp-league-track i { background:linear-gradient(90deg,#2eae82,#9ce9c9); }
.sp-divider { border-top:1px solid #2b3d43;margin:15px 0 12px; }
.sp-analysis-kpis { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px; }
.sp-analysis-kpis>div { min-width:0; }
.sp-analysis-kpis small,.sp-xg-row small { display:block;color:#81959b;font-size:.54rem;font-weight:900;letter-spacing:.08em; }
.sp-analysis-kpis b { display:block;color:#e5f0ec;font:800 1.08rem 'IBM Plex Mono',monospace;margin:5px 0; }
.sp-analysis-kpis b.positive { color:#7fe0b9; }.sp-analysis-kpis b.negative { color:#ff9699; }
.sp-analysis-kpis span { display:block;color:#8da1a6;font-size:.59rem;line-height:1.35; }
.sp-market-match { margin:13px 0 9px; }
.sp-market-match b { display:block;color:#dfece8;font-size:.78rem;line-height:1.35; }
.sp-market-match span { display:block;color:#84989e;font-size:.64rem;margin-top:4px; }
.sp-odds { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:10px; }
.sp-odds span { display:block;padding:10px;border:1px solid #2d4149;border-radius:9px;background:#0d171e;text-align:center; }
.sp-odds small { display:block;color:#82989e;font-size:.6rem;font-weight:850; }
.sp-odds b { display:block;color:#a8edcf;font:800 1.05rem 'IBM Plex Mono',monospace;margin-top:4px; }
.sp-source { color:#7b9692;font-size:.55rem;font-weight:900;letter-spacing:.12em;margin-top:8px; }
.sp-xg-row { display:grid;grid-template-columns:1fr 36px 1fr;align-items:center;gap:8px;text-align:center; }
.sp-xg-row b { display:block;color:#94e9c8;font:800 1.45rem 'IBM Plex Mono',monospace;margin-top:5px; }
.sp-xg-sep { color:#6c8587;font-size:.6rem;font-weight:900; }
.sp-risk-strip { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:14px; }
.sp-risk-strip>div:not(.sp-risk-state) { min-width:0;padding:11px 12px;border:1px solid #2c4148;border-radius:10px;background:#0d171e; }
.sp-risk-strip small { display:block;color:#82989e;font-size:.54rem;font-weight:900;letter-spacing:.08em; }
.sp-risk-strip b { display:block;color:#a8edcf;font:800 1.08rem 'IBM Plex Mono',monospace;margin:5px 0; }
.sp-risk-strip span { display:block;color:#81969c;font-size:.59rem;line-height:1.35; }
.sp-risk-state { grid-column:1/-1;padding:8px 10px;border-radius:8px;font-size:.63rem;font-weight:800; }
.sp-risk-state.ok { color:#9fe9cb;background:#142820;border:1px solid #315747; }
.sp-risk-state.blocked { color:#ffc1ad;background:#321f1c;border:1px solid #684139; }
.sp-empty-inline { color:#92a3a8;font-size:.72rem;line-height:1.5;padding:12px;border:1px dashed #34474d;border-radius:10px;margin-top:12px; }
.sp-empty-state { display:flex;flex-direction:column;align-items:center;gap:8px;padding:30px 16px 25px;text-align:center; }
.sp-empty-icon { display:grid;place-items:center;width:54px;height:54px;border:1px solid #38594d;border-radius:16px;background:#142b23;font-size:1.45rem; }
.sp-empty-state b { color:#e6f2ed;font-size:.9rem; }
.sp-empty-state span { color:#889ca1;font-size:.75rem;max-width:450px;line-height:1.5; }
.sp-footer-note { color:#71868d;font-size:.62rem;line-height:1.5;margin:5px 0 18px; }
.sp-footer-note span { color:#a0e5ca;font-weight:900;letter-spacing:.08em; }
@media (max-width: 1100px) {
  .sp-hero-mark { flex-basis:125px;transform:scale(.85); }
  .sp-dashboard-hero { padding:23px 22px; }
}
@media (max-width: 760px) {
  .sp-appbar { flex-wrap:wrap; }.sp-appbar-spacer { display:none; }
  .sp-appbar-chip,.sp-appbar-bank { display:none; }
  .sp-dashboard-hero { padding:21px 18px;min-height:unset; }
  .sp-hero-mark { display:none; }
  .sp-analysis-kpis { grid-template-columns:1fr 1fr; }
  .sp-risk-strip { grid-template-columns:1fr 1fr; }
  .sp-panel { padding:14px; }
  .sp-brand-sub { font-size:.52rem; }
}



/* SPORT-AI-PRO13 · Neural Terminal dashboard v2 */
.block-container {
  max-width: 1780px !important;
  padding: .9rem 1.35rem 2.2rem !important;
}
section[data-testid="stSidebar"] {
  min-width: 258px !important;
  max-width: 258px !important;
  background: linear-gradient(180deg,#071a1b 0%,#09151a 65%,#0b191d 100%) !important;
  border-right: 1px solid #183b36 !important;
}
section[data-testid="stSidebar"] > div {
  padding: 1rem .78rem 1.4rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #aec4c4 !important;
}
.neo-side-brand {
  display:flex; align-items:center; gap:10px; padding:5px 6px 18px;
  margin-bottom:12px; border-bottom:1px solid #1c3b38;
}
.neo-side-mark {
  width:39px;height:39px;display:grid;place-items:center;flex:0 0 39px;
  border:1px solid #3a826c;border-radius:12px;color:#08231b;
  background:linear-gradient(145deg,#8ff0c9,#26bd8b);
  font-size:1.25rem;font-weight:950;box-shadow:0 0 22px rgba(43,204,151,.14);
}
.neo-side-brand b {
  display:block;color:#f1faf6;font-size:.88rem;font-weight:900;line-height:1.15;
}
.neo-side-brand span {
  display:block;color:#7fa39a;font-size:.57rem;font-weight:850;
  letter-spacing:.13em;margin-top:4px;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
  display:none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
  display:flex !important;flex-direction:column !important;gap:5px !important;
  padding:0 !important;margin:0 !important;border:0 !important;background:transparent !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
  width:100% !important;min-height:41px !important;display:flex !important;
  align-items:center !important;margin:0 !important;padding:9px 11px !important;
  border:1px solid transparent !important;border-radius:10px !important;
  background:transparent !important;cursor:pointer !important;
  transition:background .15s ease,border-color .15s ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
  background:#102a27 !important;border-color:#234b42 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
  background:linear-gradient(100deg,#164b3b,#10362f) !important;
  border-color:#2d725b !important;box-shadow:inset 3px 0 #42d49e,0 5px 15px rgba(0,0,0,.12) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
  display:none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label p {
  color:#afc3c2 !important;font-size:.79rem !important;font-weight:750 !important;
  margin:0 !important;letter-spacing:.005em !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
  color:#e4fff3 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] input {
  position:absolute !important;opacity:0 !important;pointer-events:none !important;
}
.neo-side-status {
  margin:17px 0 13px;padding:13px 12px;border:1px solid #245344;
  border-radius:13px;background:linear-gradient(145deg,#102b23,#0c211f);
}
.neo-side-status-head {display:flex;align-items:center;gap:8px;color:#8ce7c1;font-size:.78rem;font-weight:850;}
.neo-side-status-dot {width:9px;height:9px;border-radius:50%;background:#3ed59a;box-shadow:0 0 10px rgba(62,213,154,.65);}
.neo-side-status small {display:block;color:#8ba8a0;font-size:.65rem;line-height:1.55;margin-top:7px;}
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
  margin-top:10px;border-color:#24433d !important;background:#0d201f !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
  color:#c8ddd5 !important;font-size:.77rem;font-weight:800 !important;
}
.neo-dashboard-header {
  display:flex;align-items:center;justify-content:space-between;gap:18px;
  padding:4px 2px 17px;margin:0 0 4px;border-bottom:1px solid #203a3b;
}
.neo-dashboard-eyebrow {
  color:#54d9a9;font-size:.59rem;font-weight:900;letter-spacing:.18em;margin-bottom:5px;
}
.neo-dashboard-header h1 {
  color:#f3fbf8 !important;font-size:1.63rem !important;font-weight:900 !important;
  letter-spacing:-.045em !important;line-height:1.12 !important;margin:0 !important;
}
.neo-dashboard-header p {
  color:#8ea7a9 !important;font-size:.74rem;margin:5px 0 0 !important;
}
.neo-header-meta {
  display:flex;align-items:center;justify-content:flex-end;gap:12px;flex-wrap:wrap;
  color:#809b99;font-size:.62rem;font-weight:750;
}
.neo-live-pill {
  display:inline-flex;align-items:center;gap:7px;padding:7px 10px;
  color:#99f0ca;background:#112d24;border:1px solid #2b7258;border-radius:999px;
  white-space:nowrap;font-size:.59rem;letter-spacing:.08em;
}
.neo-live-pill i {width:7px;height:7px;border-radius:50%;background:#40d99b;box-shadow:0 0 9px #40d99b;}
.neo-kpi-grid {
  display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin:13px 0 8px;
}
.neo-kpi {
  min-width:0;min-height:105px;padding:13px 13px 11px;
  border:1px solid #213e3e;border-radius:12px;
  background:linear-gradient(145deg,#11252a 0%,#0d1c23 74%,#102821 100%);
  box-shadow:0 8px 22px rgba(0,0,0,.12);
}
.neo-kpi-label {
  color:#93aaa9;font-size:.59rem;font-weight:850;letter-spacing:.075em;
  line-height:1.35;text-transform:uppercase;
}
.neo-kpi-value {
  color:#f0faf6;font:800 1.58rem 'IBM Plex Mono',monospace;
  letter-spacing:-.045em;margin:8px 0 3px;overflow-wrap:anywhere;
}
.neo-kpi-note {color:#59d5a7;font-size:.61rem;font-weight:750;line-height:1.35;}
.neo-market-delta {
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:9px 12px;
  margin:0 0 13px;border:1px solid #22443a;border-radius:10px;
  background:linear-gradient(90deg,#10271f,#102027);
}
.neo-market-delta span {color:#81a79a;font-size:.59rem;font-weight:900;letter-spacing:.1em;}
.neo-market-delta b {color:#8ce7bd;font:800 .92rem 'IBM Plex Mono',monospace;}
.neo-market-delta small {color:#91a7a5;font-size:.62rem;}
.neo-panel {
  min-width:0;height:100%;padding:15px 15px 13px;margin:0 0 13px;
  border:1px solid #24403f;border-radius:14px;
  background:linear-gradient(145deg,#102027 0%,#0d1b22 72%,#10251f 100%);
  box-shadow:0 9px 25px rgba(0,0,0,.13);
}
.neo-panel-heading {
  display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:0 0 10px;margin:0 0 8px;border-bottom:1px solid #24403d;
}
.neo-panel-kicker {display:block;color:#5bd7ab;font-size:.55rem;font-weight:900;letter-spacing:.16em;margin-bottom:4px;}
.neo-panel-heading h3 {color:#eef8f3 !important;font-size:.98rem !important;font-weight:850 !important;margin:0 !important;letter-spacing:-.025em !important;}
.neo-panel-icon {display:grid;place-items:center;width:31px;height:31px;border:1px solid #285b4c;border-radius:9px;color:#5bd7ab;background:#112e27;font-size:1.05rem;}
.neo-count-pill {
  flex-shrink:0;padding:5px 8px;border:1px solid #295d4c;border-radius:999px;
  background:#102a22;color:#9be8c7;font:750 .59rem 'IBM Plex Mono',monospace;
}
.neo-table-scroll {width:100%;overflow-x:auto;}
.neo-match-table {width:100%;min-width:790px;border-collapse:separate;border-spacing:0;font-size:.68rem;}
.neo-match-table th {
  padding:9px 8px;text-align:left;white-space:nowrap;
  color:#7f9d9b;background:#0d1a20;border-bottom:1px solid #28403d;
  font-size:.56rem;font-weight:900;letter-spacing:.07em;text-transform:uppercase;
}
.neo-match-table td {
  padding:11px 8px;border-bottom:1px solid rgba(46,75,71,.58);
  color:#d6e5df;vertical-align:middle;white-space:nowrap;
}
.neo-match-table tbody tr:hover td {background:rgba(39,102,78,.12);}
.neo-match-table tr:last-child td {border-bottom:0;}
.neo-time {color:#8da7a5 !important;font:650 .61rem 'IBM Plex Mono',monospace;}
.neo-league {
  display:inline-flex;padding:5px 7px;border:1px solid #2a4b45;border-radius:6px;
  color:#9bdcc2;background:#122b25;font-size:.58rem;font-weight:800;
}
.neo-fixture {max-width:235px;white-space:normal !important;line-height:1.35;}
.neo-fixture b {color:#f0f8f4;font-weight:800;}
.neo-fixture span {color:#63817b;padding:0 2px;}
.neo-prob {font:750 .67rem 'IBM Plex Mono',monospace;color:#c8e8dc !important;}
.neo-odds {font:650 .61rem 'IBM Plex Mono',monospace;color:#92b8a8 !important;}
.neo-pick {
  display:inline-block;padding:5px 7px;border:1px solid #2f5148;border-radius:7px;
  color:#b5d9cb;background:#142820;font-size:.59rem;font-weight:900;
}
.neo-pick.value {color:#a6f3ce;background:#12412e;border-color:#2f9b6b;}
.neo-pick-cell {white-space:normal !important;min-width:105px;}
.neo-pick-note {display:block;margin-top:4px;font-size:.47rem;line-height:1.25;font-weight:800;white-space:normal;}
.neo-pick-note.proven {color:#70d7ad;}
.neo-pick-note.provisional {color:#d9b77b;}
.neo-pick-note.rank-score {color:#8bb8d9;}
.neo-bet-status {
  display:block;margin-top:5px;padding:4px 6px;border-radius:5px;
  font-size:.49rem;line-height:1.3;font-weight:900;letter-spacing:.025em;
  white-space:normal;
}
.neo-bet-status.approved {
  color:#a8f3cd;background:rgba(22,105,70,.26);border:1px solid #2e9566;
}
.neo-bet-status.blocked {
  color:#efb0a0;background:rgba(118,48,38,.20);border:1px solid #8b5146;
}
.neo-bet-status.forecast-only {
  color:#d7c18c;background:rgba(112,89,35,.18);border:1px solid #756037;
}
.neo-ranking-disclaimer {
  margin:0 0 8px;padding:8px 10px;border-radius:7px;
  background:rgba(104,76,31,.16);border:1px solid #725c32;
  color:#d7c18c;font-size:.58rem;line-height:1.45;
}
.neo-ranking-disclaimer b {color:#f0d79b;}
.neo-ranking-disclaimer strong {color:#a8f3cd;font-weight:900;}
.neo-pick-cell {min-width:160px;}
.neo-empty-cell {padding:34px 12px !important;text-align:center;color:#91a8a4 !important;white-space:normal !important;}
.neo-table-footer {padding-top:9px;color:#78928e;font-size:.6rem;line-height:1.4;}
.neo-chart-wrap {padding:2px 0 0;}
.neo-chart-wrap svg {display:block;width:100%;height:auto;overflow:visible;}
.neo-gridline {stroke:#28403e;stroke-width:1;opacity:.68;}
.neo-ideal {stroke:#a2b8b0;stroke-width:1.8;stroke-dasharray:6 5;opacity:.8;}
.neo-axis {fill:#7e9b97;font:10px 'IBM Plex Mono',monospace;}
.neo-axis-title {fill:#9ab2ad;font:10px 'DM Sans',sans-serif;}
.neo-chart-legend {display:flex;flex-wrap:wrap;gap:9px 13px;margin:5px 0 0 39px;}
.neo-chart-legend span {display:inline-flex;align-items:center;gap:5px;color:#9ab2ad;font-size:.58rem;}
.neo-chart-legend i {width:7px;height:7px;border-radius:50%;display:inline-block;}
.neo-ideal-key {background:transparent;border:1px dashed #a2b8b0;border-radius:0 !important;}
.neo-chart-empty {
  min-height:188px;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:9px;padding:20px;text-align:center;border:1px dashed #2c5148;border-radius:11px;
  background:rgba(11,33,29,.5);
}
.neo-chart-empty strong {color:#c5dfd4;font-size:.77rem;}
.neo-chart-empty span {max-width:300px;color:#829e98;font-size:.66rem;line-height:1.55;}
.neo-panel-footnote {color:#79928d;font-size:.59rem;line-height:1.45;margin:10px 0 0;}
.neo-bottom-panel {min-height:265px;}
.neo-quality-summary {display:flex;align-items:baseline;gap:9px;margin:6px 0 13px;}
.neo-quality-summary strong {color:#61e0ae;font:800 1.8rem 'IBM Plex Mono',monospace;}
.neo-quality-summary span {color:#8ca6a0;font-size:.61rem;}
.neo-quality-row {margin:11px 0 12px;}
.neo-quality-row > div:first-child {display:flex;justify-content:space-between;gap:8px;margin-bottom:6px;}
.neo-quality-row span {color:#a9bfba;font-size:.64rem;}
.neo-quality-row b {color:#d9eee4;font:750 .64rem 'IBM Plex Mono',monospace;}
.neo-progress {height:6px;overflow:hidden;border-radius:99px;background:#263b39;}
.neo-progress i {display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#19ac78,#6be1b1);}
.neo-status-line {display:flex;align-items:center;gap:8px;padding:8px 0 11px;color:#a9c1ba;font-size:.68rem;}
.neo-status-line b {color:#9aebc6;font-size:.65rem;letter-spacing:.04em;}
.neo-status-line > span:last-child {margin-left:auto;color:#6f9487;font-size:.6rem;}
.neo-status-dot {width:9px;height:9px;flex:0 0 9px;border-radius:50%;background:#6b879e;}
.neo-status-dot.ok {background:#42d69b;box-shadow:0 0 9px rgba(66,214,155,.55);}
.neo-status-dot.error {background:#ff777d;box-shadow:0 0 9px rgba(255,119,125,.4);}
.neo-status-dot.pending {background:#d6ab58;}
.neo-backup-time {padding:10px 0;border-top:1px solid #25413d;border-bottom:1px solid #25413d;color:#88a69b;font-size:.62rem;}
.neo-backup-time b {float:right;color:#d3e9df;font:650 .62rem 'IBM Plex Mono',monospace;}
.neo-backup-detail {color:#b4c9c1 !important;font-size:.66rem;line-height:1.5;margin:11px 0 0 !important;}
.neo-history-row {display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 0;border-bottom:1px solid rgba(41,68,62,.7);}
.neo-history-row span {color:#9bb4ac;font-size:.63rem;}
.neo-history-row b {color:#e3f2e9;font:750 .72rem 'IBM Plex Mono',monospace;}
.neo-history-divider {border-top:1px solid #2b4941;margin:9px 0 8px;}
.neo-history-foot {display:flex;flex-direction:column;gap:5px;color:#8da69e;font-size:.58rem;line-height:1.4;}
.neo-baseline-note {color:#e1c17e;font-size:.59rem;font-weight:850;margin-top:4px;}
.neo-dashboard-footer {padding:4px 2px 14px;color:#68857d;font-size:.58rem;line-height:1.5;}
.neo-dashboard-footer span {color:#5bd7ab;font-weight:900;letter-spacing:.08em;}
.terminal-section-head {
  padding:15px 18px !important;margin:0 0 14px !important;border-radius:13px !important;
}
.terminal-section-title {font-size:1.28rem !important;}
@media (max-width: 1500px) {
  .neo-kpi-grid {grid-template-columns:repeat(3,minmax(0,1fr));}
  .neo-bottom-panel {min-height:unset;}
}
@media (max-width: 1050px) {
  .neo-dashboard-header {align-items:flex-start;flex-direction:column;}
  .neo-header-meta {justify-content:flex-start;}
}
@media (max-width: 760px) {
  .block-container {padding:.75rem .6rem 1.5rem !important;}
  section[data-testid="stSidebar"] {min-width:245px !important;max-width:245px !important;}
  .neo-kpi-grid {grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;}
  .neo-kpi {min-height:92px;padding:11px;}
  .neo-kpi-value {font-size:1.25rem;}
  .neo-market-delta {gap:7px;}
  .neo-panel {padding:12px;}
  .neo-dashboard-header h1 {font-size:1.35rem !important;}
  .neo-header-meta {gap:7px;}
}


/* Neural Terminal · reference dashboard refresh v3 */
.block-container { max-width: 1760px !important; padding: .85rem 1.35rem 2rem !important; }
.neo-dashboard-header { padding: 3px 2px 13px; margin-bottom: 8px; }
.neo-dashboard-header h1 { font-size: 1.55rem !important; }
.neo-dashboard-header p { font-size: .70rem; }
.neo-header-meta { gap: 9px; }
.neo-kpi-grid { grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; margin: 12px 0 8px; }
.neo-kpi {
  min-height: 124px; padding: 13px 14px 10px;
  border-color: #21483c;
  background: linear-gradient(145deg,#102720 0%,#10202a 58%,#102922 100%);
  box-shadow: 0 8px 22px rgba(0,0,0,.14);
}
.neo-kpi-label-row { display:flex;align-items:center;justify-content:space-between;gap:8px; }
.neo-kpi-label { color:#a3bcb3; font-size:.62rem; letter-spacing:.075em; }
.neo-kpi-mark { width:7px;height:7px;border-radius:50%;background:#36d69c;box-shadow:0 0 10px rgba(54,214,156,.42); }
.neo-kpi-main { display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0;margin:6px 0 3px; }
.neo-kpi-value { font-size:1.7rem; margin:0; white-space:nowrap; }
.neo-kpi-spark { flex:0 0 40%;max-width:128px;min-width:54px;height:36px;display:flex;align-items:center; }
.neo-sparkline { display:block;width:100%;height:34px;overflow:visible; }
.neo-kpi-note { color:#70d9b0;font-size:.59rem;line-height:1.35; }
.neo-market-delta { padding:8px 11px; margin:0 0 12px; gap:10px; }
.neo-market-delta span { font-size:.56rem; }
.neo-market-delta b { font-size:.85rem; }
.neo-market-delta small { font-size:.59rem; }
.neo-meta-spacer { margin-left:auto; }
.neo-panel { padding:13px 13px 11px; border-radius:13px; }
.neo-panel-heading { padding-bottom:8px; margin-bottom:7px; }
.neo-panel-heading h3 { font-size:.92rem !important; }
.neo-panel-kicker { font-size:.53rem; }
.neo-count-pill { padding:4px 7px; font-size:.56rem; }
.neo-match-table { min-width:930px; font-size:.64rem; }
.neo-match-table th { padding:8px 7px; font-size:.53rem; }
.neo-match-table td { padding:9px 7px; }
.neo-fixture { max-width:none; }
.neo-fixture-line { display:flex;align-items:center;gap:5px;min-width:240px; }
.neo-team { display:inline-flex;align-items:center;gap:5px;min-width:0; }
.neo-team b { color:#edf8f2;font-size:.66rem;font-weight:800;white-space:normal;line-height:1.2; }
.neo-team-badge {
  position:relative;display:inline-grid;place-items:center;flex:0 0 var(--badge-size);
  width:var(--badge-size);height:var(--badge-size);overflow:hidden;border-radius:50%;
  border:1px solid #2f5949;background:#17382d;color:#8ce8c1;
  font-size:.61rem;font-weight:900;
}
.neo-team-badge img { width:100%;height:100%;object-fit:contain;padding:2px; }
.neo-vs { color:#64867b;font-size:.58rem;flex:0 0 auto; }
.neo-league { gap:5px;max-width:150px;overflow:hidden;text-overflow:ellipsis;font-size:.55rem;padding:4px 6px; }
.neo-league img { width:14px;height:14px;object-fit:contain; }
.neo-prob-pill {
  display:inline-flex;min-width:33px;justify-content:center;padding:4px 5px;
  border:1px solid #275344;border-radius:6px;background:#102c22;
  color:#a5efd0;font:750 .61rem 'IBM Plex Mono',monospace;
}
.neo-odds { font-size:.60rem !important; }
.neo-pick { padding:4px 6px;font-size:.56rem; }
.neo-table-footer { font-size:.57rem; }
.neo-reliability-panel { min-height:100%; }
.neo-chart-wrap { padding:1px 0 0; }
.neo-calibration-bar { fill:#39d49b;opacity:.84;transition:opacity .15s ease; }
.neo-calibration-bar:hover { opacity:1; }
.neo-chart-legend { margin-left:35px; }
.neo-chart-legend span { font-size:.56rem; }
.neo-chart-empty { min-height:182px; }
.neo-quality-layout { display:flex;align-items:center;gap:12px;min-height:156px; }
.neo-gauge { position:relative;flex:0 0 118px;width:118px;height:118px; }
.neo-gauge-svg { width:100%;height:100%;display:block; }
.neo-gauge-center { position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none; }
.neo-gauge-center strong { color:#e9fff4;font:800 1.25rem 'IBM Plex Mono',monospace; }
.neo-gauge-center span { color:#8eaaa0;font-size:.49rem;font-weight:900;letter-spacing:.11em;margin-top:2px; }
.neo-quality-list { flex:1;min-width:0; }
.neo-quality-row { margin:8px 0 9px; }
.neo-quality-row > div:first-child { margin-bottom:4px; }
.neo-quality-row span { font-size:.58rem; }
.neo-quality-row b { font-size:.59rem; }
.neo-progress { height:5px; }
.neo-bottom-panel { min-height:240px; }
.neo-status-line { padding:7px 0 9px; }
.neo-backup-time { padding:9px 0; }
.neo-history-row { padding:6px 0; }
.neo-history-row span { font-size:.59rem; }
.neo-history-row b { font-size:.67rem; }
.neo-history-foot { font-size:.55rem; }
.neo-panel-footnote { font-size:.56rem; }
.neo-dashboard-footer { font-size:.55rem; }
div[data-testid="stSelectbox"] > div > div {
  min-height:38px !important;background:#102820 !important;border:1px solid #285342 !important;
  border-radius:9px !important;
}
div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p { color:#b8d6c9 !important; }
div[data-testid="stButton"] > button {
  min-height:38px !important;width:100%;padding:0 .5rem !important;
  border-radius:9px !important;background:#123427 !important;border-color:#2b7454 !important;
  color:#8eebbe !important;font-size:1rem !important;
}
@media (max-width: 1500px) {
  .neo-kpi-grid { grid-template-columns:repeat(4,minmax(0,1fr)); }
}
@media (max-width: 1150px) {
  .neo-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .neo-quality-layout { align-items:flex-start; }
}
@media (max-width: 760px) {
  .neo-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .neo-kpi { min-height:104px;padding:10px; }
  .neo-kpi-value { font-size:1.32rem; }
  .neo-kpi-spark { max-width:82px;min-width:34px; }
  .neo-meta-spacer { margin-left:0; }
  .neo-quality-layout { flex-direction:column;align-items:center; }
  .neo-quality-list { width:100%; }
}


/* Reference pass v4: keep the dashboard identity visible and tighten the match desk. */
.neo-dashboard-header {
  position:sticky;top:0;z-index:35;
  padding:9px 12px 12px;margin:0 -2px 8px;
  background:linear-gradient(100deg,rgba(7,19,20,.98),rgba(8,18,23,.96) 64%,rgba(10,26,21,.96));
  border-bottom:1px solid #24443b;border-radius:0 0 10px 10px;
  box-shadow:0 8px 18px rgba(0,0,0,.18);backdrop-filter:blur(12px);
}
.neo-dashboard-header h1 {font-size:1.48rem !important;}
.neo-dashboard-header p {margin-top:4px !important;}
.neo-kpi {min-height:116px;padding:12px 13px 10px;}
.neo-kpi-label-row {min-height:17px;}
.neo-kpi-status {
  display:inline-flex;align-items:center;justify-content:center;
  padding:3px 6px;border:1px solid #355146;border-radius:999px;
  background:#10251f;color:#8fb8a6;font-size:.46rem;font-weight:900;
  letter-spacing:.07em;white-space:nowrap;
}
.neo-kpi-spark {align-items:center;justify-content:flex-end;}
.neo-spark-empty {
  display:flex;align-items:center;justify-content:center;gap:5px;
  width:100%;height:29px;padding:0 7px;
  border:1px dashed #345145;border-radius:7px;
  background:repeating-linear-gradient(135deg,rgba(56,112,86,.08) 0 5px,transparent 5px 10px);
  color:#88ab9a;font:700 .49rem 'IBM Plex Mono',monospace;white-space:nowrap;
}
.neo-spark-empty b {color:#b7d7c8;font-size:.72rem;}
.neo-panel {padding:12px 12px 10px;}
.neo-panel-heading {padding-bottom:7px;margin-bottom:7px;}
.neo-match-table {min-width:820px;}
.neo-match-table th {padding:7px 6px;}
.neo-match-table td {padding:8px 6px;}
.neo-fixture-line {min-width:220px;gap:4px;}
.neo-team {gap:5px;}
.neo-team b {font-size:.65rem;}
.neo-team-badge {
  box-shadow:0 2px 7px rgba(0,0,0,.22);
  overflow:hidden;
}
.neo-team-badge img {
  position:absolute;inset:1px;width:calc(100% - 2px);height:calc(100% - 2px);
  object-fit:contain;padding:2px;background:#10251f;
}
.neo-vs {font-size:.55rem;}
.neo-league {max-width:160px;gap:5px;}
.neo-league img {width:17px;height:17px;object-fit:contain;flex:0 0 17px;}
.neo-league-flag {
  display:inline-grid;place-items:center;flex:0 0 17px;
  width:17px;height:17px;font-size:13px;line-height:1;
}
.neo-reliability-panel {min-height:unset;}
.neo-chart-empty {min-height:175px;}
.neo-bottom-panel {min-height:205px;}
.neo-quality-layout {min-height:138px;}
.neo-gauge {flex-basis:108px;width:108px;height:108px;}
.neo-quality-row {margin:7px 0 8px;}
.neo-backup-detail {margin-top:8px !important;}
.neo-history-row {padding:5px 0;}
.neo-dashboard-footer {padding-top:2px;}
@media (max-width: 1500px) {
  .neo-kpi-grid {grid-template-columns:repeat(4,minmax(0,1fr));}
}
@media (max-width: 1000px) {
  .neo-kpi-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
  .neo-dashboard-header {position:relative;}
}


/* Separate upcoming forecast feed: visible predictions are never betting approval */
.neo-forecast-panel { margin-top:12px; border-color:#3a5145; }
.neo-forecast-panel .neo-panel-kicker { color:#8bd8b5; }
.neo-forecast-panel .neo-pick-cell { min-width:210px;max-width:285px;white-space:normal; }
.neo-forecast-panel .neo-pick-note.provisional { display:block;white-space:normal;line-height:1.4;margin-top:5px; }
.neo-forecast-panel .neo-bet-status.blocked { display:block; }
/* Grok independent-analysis panel for Neural Terminal */
.neo-grok-panel { margin-top:12px; border-color:#2b4d68; background:linear-gradient(145deg,#101d2a,#101a21 65%,#10231e); }
.neo-grok-panel .neo-panel-kicker { color:#83c7f5; }
.neo-grok-probs { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin:8px 0; }
.neo-grok-probs span { display:flex;align-items:center;justify-content:space-between;gap:5px;padding:6px 8px;border:1px solid #30485a;border-radius:7px;background:#142534;color:#aec8d8;font-size:.65rem; }
.neo-grok-probs b { color:#d5ecff;font:800 .72rem 'IBM Plex Mono',monospace; }
.neo-grok-row { padding:9px 0;border-top:1px solid #293b47; }
.neo-grok-match { color:#eef7fc;font-size:.74rem;font-weight:850;margin-bottom:5px; }
.neo-grok-meta { display:block;color:#8eabbc;font-size:.58rem;line-height:1.45;margin:4px 0; }
.neo-grok-meta b { color:#a9e0ff; }
.neo-grok-copy { color:#c4d4df;font-size:.67rem;line-height:1.5;margin:5px 0; }
.neo-grok-risk { color:#edc78c;font-size:.62rem;line-height:1.45;margin:5px 0; }
.neo-grok-empty { display:flex;flex-direction:column;gap:5px;padding:13px;border:1px dashed #385266;border-radius:9px;background:#111f2a;color:#a9c0cf;font-size:.64rem;line-height:1.5; }
.neo-grok-empty b { color:#d4eaff;font-size:.72rem; }

</style>
"""


def inject() -> None:
    """Inject the app-wide design system once per Streamlit render."""
    st.markdown(CSS, unsafe_allow_html=True)
