"""Phase 0 -- the PneumoScan Streamlit custom component wrapper.

Two modes, switched by an env var (never a code change):
- Dev: `PNEUMOSCAN_COMPONENT_DEV_URL` set (typically
  "http://localhost:5173", the Vite dev server) -- live-reloading frontend,
  used while iterating on the React app locally.
- Prod (default): serves the committed, pre-built `frontend/dist/` bundle.
  Streamlit Community Cloud's build environment is Python/pip only and
  cannot run npm, so `dist/` must already exist in the repo -- see the
  root .gitignore's negation rule for `frontend/dist/`, and
  `tests/test_frontend_bundle.py` (Phase 1) which fails CI if it's missing
  or stale.

If `dist/index.html` is missing in prod mode, this raises loudly at import
time rather than silently serving a blank iframe -- a blank iframe with no
error is exactly the failure mode Risk #1 in the plan warns about, and it's
much easier to debug a clear ImportError than "the app loads but is empty."
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"
_DIST_DIR = _FRONTEND_DIR / "dist"
_DEV_URL_ENV_VAR = "PNEUMOSCAN_COMPONENT_DEV_URL"

_dev_url = os.environ.get(_DEV_URL_ENV_VAR)

if _dev_url:
    _component_func = components.declare_component("pneumoscan", url=_dev_url)
else:
    if not (_DIST_DIR / "index.html").exists():
        raise ImportError(
            f"PneumoScan component bundle not found at {_DIST_DIR}/index.html. "
            f"Run `npm run build` in {_FRONTEND_DIR} and commit the resulting "
            f"dist/ directory, or set {_DEV_URL_ENV_VAR} to a running Vite dev "
            "server for local development."
        )
    _component_func = components.declare_component("pneumoscan", path=str(_DIST_DIR))


def pneumoscan(props: dict, *, key: str | None = None, default=None):
    """Render the PneumoScan component. `props` becomes the JS side's
    `RenderData.args` dict (each key a separate named kwarg to
    `declare_component`'s returned callable -- not nested under an `args`
    key, despite the name; see `CustomComponent.__call__`). Returns the last
    event dict sent back from JS (the plan's "JS -> Python events"
    contract), or `default` if nothing has been sent yet."""
    return _component_func(**props, key=key, default=default)
