#!/usr/bin/env python3
"""Capture README dashboard screenshots after Plotly charts finish rendering."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
URL = "http://localhost:8765"

CAPTURES: list[tuple[str, str, int]] = [
    ("dashboard-trends.png", "Trends", 2),
    ("dashboard-sentiment.png", "Sentiment", 2),
    ("dashboard-vocabulary.png", "Vocabulary", 4),
    ("dashboard-voices.png", "Voices", 0),
    ("dashboard-semantic.png", "Semantic Map", 1),
    ("dashboard-analyst.png", "Analyst Take", 0),
]


def wait_for_plotly(page, min_count: int, timeout_ms: int = 20000) -> None:
    if min_count <= 0:
        page.wait_for_timeout(1200)
        return

    page.wait_for_function(
        """(minCount) => {
            const plots = Array.from(document.querySelectorAll('.js-plotly-plot'));
            const rendered = plots.filter((plot) => {
                const svg = plot.querySelector('svg.main-svg');
                if (!svg) return false;
                const paths = svg.querySelectorAll('path, rect, line, circle');
                return paths.length > 8;
            });
            return rendered.length >= minCount;
        }""",
        arg=min_count,
        timeout=timeout_ms,
    )
    page.wait_for_timeout(1800)


def capture_screenshots() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2500)

        for filename, tab_name, min_plots in CAPTURES:
            print(f"Capturing {filename} ({tab_name})...")
            page.get_by_role("tab", name=tab_name, exact=True).click()
            page.wait_for_timeout(900)

            try:
                wait_for_plotly(page, min_plots)
            except PlaywrightTimeoutError:
                print(f"  warning: timed out waiting for {min_plots} plot(s); saving anyway")

            if tab_name == "Sentiment":
                page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.35)")
                page.wait_for_timeout(700)
            elif tab_name == "Vocabulary":
                page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.42)")
                page.wait_for_timeout(700)

            page.screenshot(path=str(ASSETS / filename), full_page=True)

        browser.close()


if __name__ == "__main__":
    try:
        capture_screenshots()
    except Exception as exc:
        print(f"Screenshot capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Saved screenshots to {ASSETS}")
