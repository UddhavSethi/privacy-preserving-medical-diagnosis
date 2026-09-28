"""PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Risk #2) — catches a
stale or missing committed `dist/` bundle before it reaches Streamlit Cloud
(which never runs `npm run build` itself; the built bundle must already be
in git). Does not run `npm run build` itself (that's the separate CI
frontend job the plan describes) — this only checks the bundle actually
committed to the repo right now is internally consistent: it exists, and
every asset path it references resolves to a real file using a relative
(not absolute) path, so it works when served from the component's own
`path=dist` root regardless of Streamlit's own mount path.

Also covers the Phase 2 case the plan's own testing section names
explicitly: `lung-model.html` and its vendored three.js files. This is the
regression test for a real bug found live (2026-09-28): the repo's blanket
`.gitignore` `build/` rule (meant for Python packaging) silently swallowed
`three.module.js`/`three.core.js` because they sit under a path segment
literally named `build/` (three.js's own release layout) -- even though
`dist/` itself has its own negation rule, that negation doesn't cascade to
a *different*, later pattern matching a nested path. The bug was invisible
locally (the files were still physically on disk) and would only have
surfaced as a broken 3D viewer on a fresh clone or real deployment. These
tests parse `lung-model.html`'s own `<script type="importmap">` and assert
every path it names is a real committed file, not ignored.
"""
from __future__ import annotations

import json
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


def _is_git_tracked(path: Path) -> bool:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=DIST_DIR.parents[3],
        capture_output=True,
    )
    return result.returncode == 0


def test_lung_model_html_exists():
    assert (DIST_DIR / "lung-model.html").exists(), (
        "dist/lung-model.html is missing — copy it from public/lung-model.html via `npm run build`."
    )


def test_lung_model_vendored_three_js_exists_and_is_git_tracked():
    """Regression test for the real `build/`-gitignore trap (see module
    docstring): existence on disk isn't enough -- these files must also
    actually be tracked by git, or a fresh clone/deployment silently loses
    them exactly as happened live."""
    html = (DIST_DIR / "lung-model.html").read_text()
    match = re.search(r'<script type="importmap">\s*(\{.*?\})\s*</script>', html, re.DOTALL)
    assert match, "lung-model.html has no <script type=\"importmap\"> block — three.js won't resolve"
    import_map = json.loads(match.group(1))

    all_paths = set(import_map.get("imports", {}).values()) | set(import_map.get("integrity", {}).keys())
    assert all_paths, "import map has no imports/integrity entries"

    for path in all_paths:
        assert not path.startswith("http"), f"import map still points at a live CDN, not vendored: {path}"
        resolved = (DIST_DIR / path).resolve()
        assert resolved.exists(), f"lung-model.html's import map references a missing vendored file: {path}"
        assert _is_git_tracked(resolved), (
            f"{resolved} exists on disk but isn't tracked by git — it will be missing from a fresh "
            "clone or deployment even though local tests/builds pass. Check .gitignore for a rule "
            "matching a path segment in this file (e.g. a blanket `build/` rule)."
        )
