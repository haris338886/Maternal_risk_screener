"""AppTest smoke tests — includes out-of-range vitals requirement."""

import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py")


def _run_app():
    at = AppTest.from_file(APP_PATH).run(timeout=60)
    assert not at.exception, f"App raised: {at.exception}"
    return at


def _all_text(at) -> str:
    """Concatenate text from all renderable widget groups."""
    chunks = []
    for group in ("markdown", "warning", "caption", "info", "error", "title", "subheader"):
        for el in getattr(at, group, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def test_app_loads():
    """Smoke: app renders, has 6 inputs, and shows disclaimer."""
    at = _run_app()
    assert len(at.number_input) >= 6
    text = _all_text(at).lower()
    assert "disclaimer" in text or "not a substitute" in text


def test_out_of_range_input_shows_error():
    """Requirement: out-of-range vitals must be rejected."""
    at = _run_app()

    # Age min is 10 → set to 5
    age = at.number_input(key="input_Age")
    age.set_value(5.0)

    submit = next(b for b in at.button if "Assess" in b.label)
    submit.click().run()

    assert not at.exception
    assert any("invalid input" in e.value.lower() for e in at.error)


def test_valid_input_produces_traffic_light():
    at = _run_app()

    submit = next(b for b in at.button if "Assess" in b.label)
    submit.click().run()

    assert not at.exception
    text = _all_text(at)
    assert any(emoji in text for emoji in ["🟢", "🟡", "🔴"])