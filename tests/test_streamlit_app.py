"""PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Phase 1B/1C/1D) —
a smoke test that `app/streamlit_app.py` (the state machine + event
dispatcher wired around the real backend) runs cleanly from fresh state with
a real v1 custom component rendered, per the plan's own "Testing and
verification" section: verifying `AppTest` tolerates a v1 component was an
open question through Phase 0, resolved here.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_streamlit_app_runs_from_fresh_state_with_no_exception():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert len(at.exception) == 0
