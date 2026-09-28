"""PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Risk #2) — catches a
stale or missing committed `dist/` bundle before it reaches Streamlit Cloud
(which never runs `npm run build` itself; the built bundle must already be
in git). Does not run `npm run build` itself (that's the separate CI
frontend job the plan describes) — this only checks the bundle actually
committed to the repo right now is internally consistent: it exists, and
every asset path it references resolves to a real file using a relative
(not absolute) path, so it works when served from the component's own
`path=dist` root regardless of Streamlit's own mount path.
"""
from __future__ import annotations

import re
from pathlib import Path

DIST_DIR = Path(__file__).resolve().parents[1] / "app" / "pneumoscan_component" / "frontend" / "dist"


def test_dist_index_html_exists():
    assert (DIST_DIR / "index.html").exists(), (
        f"{DIST_DIR}/index.html is missing — run `npm run build` in "
        "app/pneumoscan_component/frontend and commit the resulting dist/."
    )


def test_dist_referenced_assets_exist_and_are_relative():
    html = (DIST_DIR / "index.html").read_text()
    referenced = set(re.findall(r'(?:src|href)="([^"]+)"', html))
    assert referenced, "index.html references no src/href assets — build likely broken"
    for path in referenced:
        assert not path.startswith("/"), f"absolute asset path {path!r} breaks Streamlit's component mount"
        resolved = (DIST_DIR / path).resolve()
        assert resolved.exists(), f"index.html references missing asset: {path}"
