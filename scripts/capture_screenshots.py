#!/usr/bin/env python3
"""
scripts/capture_screenshots.py — The Dashboard, Photographed

WHY THIS FILE EXISTS:
    The three dashboard screenshots in the README are the only claim in this
    repository a reader cannot check by running something. This script drives
    the real Streamlit UI against the real API in a real browser and saves
    them, so they can be regenerated rather than trusted — the same reason
    `plot_horizon.py` is committed next to the chart it draws.

    Nothing here is mocked. Every number in the resulting images came out of
    the API, and the AI report came out of a local model.

HOW IT WORKS:
    Rewinds the dashboard to 2024-10-31 hour 6 — the state the sidebar itself
    recommends, hours before machine 51's recorded failure — then captures the
    fleet, machine-detail, and AI-report views.

    Needs the stack running (`make docker-up-d`, or `make run-api` and
    `make run-dashboard`), Playwright's Chromium, and, for the report view, an
    Ollama server holding the model.

    pip install -r requirements-dev.txt
    playwright install chromium
    python scripts/capture_screenshots.py
    python scripts/capture_screenshots.py --model llama3.1:8b
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent


def settle(page, timeout=600_000):
    """Wait for Streamlit to stop re-running."""
    page.wait_for_timeout(800)
    try:
        page.wait_for_selector(
            '[data-testid="stStatusWidget"]', state="detached", timeout=timeout
        )
    except Exception:
        pass
    page.wait_for_timeout(1500)


def pick_view(page, name):
    page.get_by_test_id("stSidebar").get_by_text(name, exact=True).click()
    settle(page)


def choose(page, label, value):
    """Pick `value` in the combobox labelled `label`."""
    box = page.get_by_test_id("stMain").get_by_role("combobox", name=label)
    box.click()
    page.wait_for_timeout(600)
    box.fill(value)
    page.wait_for_timeout(900)
    # Enter would take the first fuzzy match ("51" -> "5"); click the exact one.
    page.get_by_role("option", name=value, exact=True).click()
    settle(page)


def rewind(page, date, hour):
    """Turn on Rewind and set the as-of timestamp."""
    sidebar = page.get_by_test_id("stSidebar")
    sidebar.get_by_text("Rewind", exact=True).click()
    settle(page)

    date_box = sidebar.locator('input[aria-label="Select a date."]')
    date_box.click()
    date_box.press("ControlOrMeta+a")
    date_box.type(date, delay=60)
    date_box.press("Enter")
    page.keyboard.press("Escape")
    settle(page)

    slider = sidebar.locator('input[type="range"][aria-label="Hour"]')
    slider.focus()
    for _ in range(25):  # the slider defaults to hour 23; floor it at 0
        slider.press("ArrowLeft")
    page.wait_for_timeout(300)
    for _ in range(hour):
        slider.press("ArrowRight")
    settle(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "images")
    ap.add_argument("--machine", default="51")
    ap.add_argument("--date", default="2024/10/31")
    ap.add_argument("--hour", type=int, default=6)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1500, "height": 1050}, device_scale_factor=2
        )
        page = ctx.new_page()
        page.goto(args.url, wait_until="load", timeout=120_000)
        settle(page)

        rewind(page, args.date, args.hour)

        pick_view(page, "Fleet overview")
        page.screenshot(path=str(args.out / "dashboard-fleet.png"))
        print("captured dashboard-fleet.png")

        pick_view(page, "Machine detail")
        choose(page, "Machine", args.machine)
        page.screenshot(path=str(args.out / "dashboard-machine.png"))
        print("captured dashboard-machine.png")

        pick_view(page, "AI report")
        choose(page, "Machine", args.machine)
        choose(page, "Provider", args.provider)
        page.get_by_test_id("stMain").get_by_placeholder("e.g. qwen2.5-coder:7b").fill(
            args.model
        )
        page.wait_for_timeout(400)
        page.get_by_test_id("stMain").get_by_role("button", name="Generate").click()
        settle(page)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(args.out / "dashboard-report.png"))
        print("captured dashboard-report.png")

        ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
