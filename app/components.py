"""OPT-6 — small HTML-rendering helpers for the Streamlit demo (second design
pass, 2026-08-30: consumer-product presentation — result-focused, plain
language, no research/dashboard vocabulary). Returns HTML strings for
`st.markdown(..., unsafe_allow_html=True)`. No research/inference logic."""
from __future__ import annotations

import html


def brand_row(mark: str, tag: str) -> str:
    return f'<div class="brand-row"><span class="brand-mark">{html.escape(mark)}</span><span class="brand-tag">{html.escape(tag)}</span></div>'


def hero_title(text: str) -> str:
    return f'<div class="hero-title">{html.escape(text)}</div>'


def hero_subtitle(text: str) -> str:
    return f'<div class="hero-subtitle">{html.escape(text)}</div>'


def section_title(text: str) -> str:
    return f'<div class="section-title">{html.escape(text)}</div>'


def section_lede(text: str) -> str:
    return f'<div class="section-lede">{html.escape(text)}</div>'


def subhead(text: str) -> str:
    return f'<div class="subhead">{html.escape(text)}</div>'


def prose(text: str) -> str:
    return f'<div class="prose">{text}</div>'


def result_label(label: str, attention: bool = False, uncertain: bool = False) -> str:
    if uncertain:
        cls = "result-label uncertain"
    elif attention:
        cls = "result-label attention"
    else:
        cls = "result-label"
    return f'<div class="result-eyebrow">Result</div><div class="{cls}">{html.escape(label)}</div>'


def confidence_meter(fraction: float, attention: bool = False, uncertain: bool = False) -> str:
    frac = max(0.0, min(1.0, fraction))
    pct = frac * 100
    # `.0f` rounding let any confidence >= 99.95% (a real, common MC Dropout
    # output for a confident prediction, not a bug) display as a literal
    # "100%" -- reading as absolute certainty the model never actually
    # claims. One decimal place reflects the real output; a sub-1.0 fraction
    # is additionally floored just under 100% so it never rounds up to a
    # false "100.0%" either.
    if frac >= 1.0:
        pct_label = "100.0%"
    elif pct >= 99.95:
        pct_label = "99.9%"
    else:
        pct_label = f"{pct:.1f}%"
    fill_cls = "meter-fill uncertain" if uncertain else ("meter-fill attention" if attention else "meter-fill")
    return (
        f'<div class="confidence-row"><span class="confidence-value">{pct_label}</span>'
        f'<span class="confidence-caption">confidence</span></div>'
        f'<div class="meter-track"><div class="{fill_cls}" style="width:{pct:.1f}%"></div></div>'
    )


def confidence_note(text: str, attention: bool = False, uncertain: bool = False) -> str:
    cls = "confidence-note uncertain" if uncertain else ("confidence-note attention" if attention else "confidence-note")
    return f'<div class="{cls}">{html.escape(text)}</div>'


def diagram(text: str) -> str:
    return f'<div class="diagram">{html.escape(text)}</div>'


def disclaimer(text: str) -> str:
    return f'<div class="disclaimer">{html.escape(text)}</div>'
