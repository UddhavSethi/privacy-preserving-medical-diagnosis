"""OPT-6 — visual theme for the Streamlit demo.

Second design pass (2026-08-30, owner-directed): the first pass rendered the
whole app as a literal parchment "page" (light background, SVG paper-grain,
blackletter headings) — reading as a research paper/notebook document, not a
finished product. This pass keeps the same underlying palette family as the
notebook portfolio (`projects/portfolio`) for continuity, but pulls from its
dark COVER tokens rather than its light PAGE tokens: near-black surfaces, warm
cream text, one restrained accent color. Blackletter (UnifrakturCook) and the
handwritten accent (Caveat) are dropped from the main experience — they read
as "academic manuscript," which works against "this is a finished
application." Geist (the same UI typeface the portfolio itself uses for its
own chrome) now carries headings/navigation/labels; EB Garamond is kept only
for longer-form prose (the "How it works" explanations), where a refined serif
reads as considered typography rather than decoration.

Pure presentation constants and a CSS string; no research/inference logic
lives here.
"""
from __future__ import annotations

# Palette — dark half of the notebook portfolio's token family
# (projects/portfolio/src/styles/notebook-theme.css: --cover-1/2, --blood*;
# projects/portfolio/src/app/globals.css: --background/--foreground).
BG = "#050403"
SURFACE = "#100d0a"
SURFACE_RAISED = "#171310"
BORDER = "rgba(184, 168, 119, 0.16)"
BORDER_STRONG = "rgba(184, 168, 119, 0.30)"
TEXT_PRIMARY = "#ece3cf"
TEXT_MUTED = "#a89a80"
TEXT_FAINT = "#6f6553"
ACCENT = "#c0394a"  # brightened from the notebook's --blood for dark-surface legibility
ACCENT_DIM = "rgba(192, 57, 74, 0.14)"
UNCERTAIN = "#c9a227"  # amber -- distinct from ACCENT (Pneumonia/attention, red) and
                       # TEXT_PRIMARY (Normal, neutral): a genuine third state (2026-09-01,
                       # abstention band), never conflated with either forced prediction
UNCERTAIN_DIM = "rgba(201, 162, 39, 0.14)"

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">

<style>
:root {{
    --bg: {BG};
    --surface: {SURFACE};
    --surface-raised: {SURFACE_RAISED};
    --border: {BORDER};
    --border-strong: {BORDER_STRONG};
    --text-primary: {TEXT_PRIMARY};
    --text-muted: {TEXT_MUTED};
    --text-faint: {TEXT_FAINT};
    --accent: {ACCENT};
    --accent-dim: {ACCENT_DIM};
    --uncertain: {UNCERTAIN};
    --uncertain-dim: {UNCERTAIN_DIM};
}}

html, body {{ background: var(--bg) !important; }}
.stApp {{
    background: radial-gradient(120% 100% at 50% -12%, #0c0a07 0%, var(--bg) 55%, #030201 100%);
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer {{ visibility: hidden; }}

[data-testid="stMainBlockContainer"] {{
    max-width: 760px;
    margin: 0 auto !important;
    padding: 3rem 1.4rem 4rem 1.4rem !important;
}}

html, body, [class*="css"], .stMarkdown, p, span, div, label {{
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}}
h1, h2, h3, h4 {{
    font-family: 'Geist', sans-serif;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    margin: 0;
}}
.prose {{
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.08rem;
    line-height: 1.7;
    color: var(--text-muted);
}}
code, .mono, [data-testid="stDataFrame"], pre {{
    font-family: 'Geist Mono', monospace !important;
}}

/* ---- brand / hero ---- */
.brand-row {{ display:flex; align-items:baseline; gap:0.6rem; margin-bottom:0.35rem; }}
.brand-mark {{ font-size:1.15rem; font-weight:800; letter-spacing:-0.02em; }}
.brand-tag {{ font-size:0.78rem; color:var(--text-faint); }}
.hero-title {{
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.9rem 0 0.5rem 0;
    line-height: 1.18;
}}
.hero-subtitle {{
    font-size: 1.02rem;
    color: var(--text-muted);
    max-width: 46ch;
    line-height: 1.55;
    margin-bottom: 1.6rem;
}}

/* ---- section headings (How it works / Research) ---- */
.section-title {{
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
}}
.section-lede {{
    color: var(--text-muted);
    font-size: 0.96rem;
    margin-bottom: 1.4rem;
}}
.subhead {{
    font-size: 1.02rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 1.6rem 0 0.35rem 0;
}}

hr {{ border: none !important; border-top: 1px solid var(--border) !important; margin: 2.2rem 0 !important; }}

/* ---- result focus ---- */
.result-eyebrow {{
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-bottom: 0.4rem;
}}
.result-label {{
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin-bottom: 0.3rem;
}}
.result-label.attention {{ color: var(--accent); }}
.result-label.uncertain {{ color: var(--uncertain); }}
.confidence-row {{ display:flex; align-items:baseline; gap:0.5rem; margin-bottom:1.1rem; }}
.confidence-value {{ font-size:1.3rem; font-weight:700; }}
.confidence-caption {{ font-size:0.85rem; color:var(--text-muted); }}

.meter-track {{
    width: 100%;
    height: 6px;
    background: var(--surface-raised);
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 1.6rem;
}}
.meter-fill {{
    height: 100%;
    border-radius: 999px;
    background: var(--text-primary);
}}
.meter-fill.attention {{ background: var(--accent); }}
.meter-fill.uncertain {{ background: var(--uncertain); }}

.confidence-note {{
    font-size: 0.94rem;
    line-height: 1.6;
    padding: 0.8rem 1rem;
    border-left: 2px solid var(--border-strong);
    color: var(--text-muted);
    margin: 0.9rem 0 1.6rem 0;
}}
.confidence-note.attention {{
    border-left-color: var(--accent);
    color: var(--text-primary);
    background: var(--accent-dim);
}}
.confidence-note.uncertain {{
    border-left-color: var(--uncertain);
    color: var(--text-primary);
    background: var(--uncertain-dim);
}}

/* ---- architecture diagram ---- */
.diagram {{
    font-family: 'Geist Mono', monospace;
    font-size: 0.84rem;
    line-height: 1.75;
    color: var(--text-muted);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    white-space: pre;
    overflow-x: auto;
}}

.disclaimer {{
    font-size: 0.8rem;
    color: var(--text-faint);
    margin-top: 2.4rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}}

/* ---- widgets ---- */
.stButton button {{
    background: var(--text-primary) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Geist', sans-serif !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: none !important;
    transition: opacity 0.15s ease;
}}
.stButton button:hover {{ opacity: 0.85; }}

div[data-baseweb="select"] > div, .stTextInput input {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}}
div[data-baseweb="select"] * {{ color: var(--text-primary) !important; }}
label, .stSelectbox label, .stTextInput label {{
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: var(--surface) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploaderDropzone"] * {{ color: var(--text-muted) !important; }}
[data-testid="stFileUploaderDropzone"] button {{
    background: var(--surface-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
}}

[data-testid="stExpander"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] summary p {{
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}}

[data-testid="stAlert"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-muted) !important;
}}
[data-testid="stAlert"] p {{ color: var(--text-muted) !important; }}

[data-testid="stCaptionContainer"], .stCaption {{ color: var(--text-faint) !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--border) !important; border-radius: 8px; }}
figcaption, [data-testid="stImageCaption"] {{ color: var(--text-faint) !important; font-size: 0.8rem !important; }}
[data-testid="stSpinner"] * {{ color: var(--text-muted) !important; }}

/* Tabs — thin underline, not filled boxes */
[data-testid="stTabs"] button[role="tab"] {{
    font-family: 'Geist', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: var(--text-primary) !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ background-color: var(--border) !important; }}

img {{ border-radius: 10px; }}
</style>
"""
