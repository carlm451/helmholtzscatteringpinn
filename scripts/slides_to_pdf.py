#!/usr/bin/env python3
"""Convert slides.html to PDF by screenshotting each slide via Playwright.

Usage:
    .venv/bin/python scripts/slides_to_pdf.py [docs/slides.html] [output.pdf]
"""

import sys
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

SLIDES_HTML = sys.argv[1] if len(sys.argv) > 1 else "docs/slides.html"
OUTPUT_PDF = sys.argv[2] if len(sys.argv) > 2 else "docs/HelmholtzPINN_Slides.pdf"

WIDTH = 1920
HEIGHT = 1080

# Light-theme CSS overrides for print-friendly PDF
PRINT_CSS = """
/* ── Light theme overrides ── */
html, body { background: #ffffff !important; }
.slide {
  background: #ffffff !important;
  color: #1a1a2e !important;
  padding: 2.8rem 4rem 2.5rem !important;
}

/* Top accent line — keep but darken for print */
.slide::before {
  background: linear-gradient(90deg, #4f46e5 0%, #0d9488 50%, transparent 100%) !important;
  opacity: .7 !important;
}

/* ── Headings ── */
.slide h2 {
  color: #1a1a2e !important;
  font-size: 1.9rem !important;
  margin-bottom: 1.2rem !important;
}
.slide.physics h2 { border-left-color: #4f46e5 !important; }
.slide.results h2 { border-left-color: #0d9488 !important; }
.slide.method h2 { border-bottom-color: rgba(0,0,0,.15) !important; }

/* ── Title slide ── */
.slide.title h1 {
  color: #1a1a2e !important;
  font-size: clamp(3rem, 5.5vw, 4.5rem) !important;
}
.slide.title .sub { color: #4a4a5e !important; font-size: 1.45rem !important; }
.slide.title .author { color: #6b6b80 !important; font-size: 1.1rem !important; }
.slide.title .title-accent { opacity: .08 !important; }

/* ── Typography — scale up for PDF readability ── */
p { color: #2a2a3e !important; font-size: 1.15rem !important; }
ul { color: #2a2a3e !important; font-size: 1.15rem !important; }
li { color: #2a2a3e !important; }
li strong { color: #1a1a2e !important; }
h3 { color: #2a2a3e !important; font-size: 1.2rem !important; }
.small { color: #5a5a6e !important; font-size: 1rem !important; }
.tiny { color: #7a7a8e !important; font-size: .9rem !important; }
a { color: #4f46e5 !important; }
code { background: rgba(0,0,0,.05) !important; color: #2a2a3e !important; }

/* ── Tables ── */
table { font-size: 1.1rem !important; }
th {
  background: rgba(0,0,0,.04) !important;
  color: #5a5a6e !important;
  border-bottom-color: rgba(0,0,0,.15) !important;
  font-size: .9rem !important;
}
td {
  border-bottom-color: rgba(0,0,0,.08) !important;
  color: #2a2a3e !important;
}
td:first-child { color: #1a1a2e !important; }
.mono { color: #2a2a3e !important; font-size: 1.05rem !important; }
.best td { background: rgba(13,148,136,.08) !important; }

/* ── Equations ── */
.eq {
  background: rgba(79,70,229,.06) !important;
  border-color: rgba(79,70,229,.2) !important;
  font-size: 1.15em !important;
  padding: .9rem 1.4rem !important;
}
.eq-hero {
  background: rgba(79,70,229,.08) !important;
  border-color: rgba(79,70,229,.3) !important;
  font-size: 1.3em !important;
  padding: 1.2rem 2rem !important;
}

/* ── Callouts ── */
.highlight {
  background: rgba(13,148,136,.06) !important;
  border-left-color: #0d9488 !important;
  color: #1a3a35 !important;
  font-size: 1.1rem !important;
}
.highlight strong { color: #0d9488 !important; }
.callout-result {
  background: rgba(79,70,229,.06) !important;
  border-left-color: #4f46e5 !important;
  color: #1a1a3e !important;
  font-size: 1.1rem !important;
}

/* ── Architecture flow ── */
.arch-node {
  border-color: rgba(0,0,0,.2) !important;
  color: #1a1a2e !important;
  background: rgba(0,0,0,.02) !important;
  font-size: 1.05rem !important;
}
.arch-node .detail { color: #5a5a6e !important; }
.arch-arrow { color: #9a9aae !important; }
.node-ff { background: rgba(79,70,229,.1) !important; border-color: rgba(79,70,229,.3) !important; color: #4f46e5 !important; }
.node-res { background: rgba(13,148,136,.08) !important; border-color: rgba(13,148,136,.25) !important; color: #0d9488 !important; }
.node-out { background: rgba(217,119,6,.08) !important; border-color: rgba(217,119,6,.25) !important; color: #b45309 !important; }

/* ── Big result numbers ── */
.big-result { color: #0d9488 !important; font-size: 2.5rem !important; }
.big-result .unit { color: #5a5a6e !important; }

/* ── Takeaways ── */
.takeaway .num { color: #4f46e5 !important; font-size: 2rem !important; }
.takeaway p { color: #2a2a3e !important; font-size: 1.1rem !important; }
.takeaway strong { color: #1a1a2e !important; }

/* ── Field plot links ── */
.field-links a {
  border-color: rgba(0,0,0,.12) !important;
  color: #4f46e5 !important;
  font-size: 1rem !important;
}

/* ── Hide nav elements ── */
#progress, #counter, #nav-hint { display: none !important; }

/* ── SVG text colors for light background ── */
svg text { fill: #2a2a3e !important; }
svg text[fill*="818cf8"], svg text[fill*="a5b4fc"] { fill: #4f46e5 !important; }
svg text[fill*="5eead4"], svg text[fill*="99f6e4"] { fill: #0d9488 !important; }
svg text[fill*="6b6880"], svg text[fill*="4b4868"], svg text[fill*="3a3754"] { fill: #7a7a8e !important; }

/* ── MathJax color fix ── */
mjx-container, mjx-math, mjx-mi, mjx-mo, mjx-mn, mjx-mrow {
  color: #1a1a2e !important;
}
"""

# JS to recolor Plotly charts for light theme
PLOTLY_LIGHT_JS = """
document.querySelectorAll('.plotly-graph-div').forEach(function(gd) {
    if (gd.data) {
        Plotly.relayout(gd, {
            'paper_bgcolor': 'rgba(255,255,255,0)',
            'plot_bgcolor': 'rgba(255,255,255,0)',
            'font.color': '#2a2a3e',
            'xaxis.gridcolor': 'rgba(0,0,0,0.08)',
            'xaxis.zerolinecolor': 'rgba(0,0,0,0.12)',
            'xaxis.tickfont.color': '#5a5a6e',
            'xaxis.title.font.color': '#2a2a3e',
            'yaxis.gridcolor': 'rgba(0,0,0,0.08)',
            'yaxis.zerolinecolor': 'rgba(0,0,0,0.12)',
            'yaxis.tickfont.color': '#5a5a6e',
            'yaxis.title.font.color': '#2a2a3e',
            'legend.font.color': '#2a2a3e',
        });
    }
});
"""


def main():
    slides_path = Path(SLIDES_HTML).resolve()
    if not slides_path.exists():
        print(f"Error: {slides_path} not found")
        sys.exit(1)

    url = f"file://{slides_path}"
    tmp_dir = Path("docs/.slide_screenshots")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(url, wait_until="networkidle")

        # Wait for MathJax to finish rendering
        page.wait_for_timeout(3000)

        # Inject light theme CSS for print-friendly PDF
        page.add_style_tag(content=PRINT_CSS)

        # Recolor Plotly charts for light background
        page.evaluate(PLOTLY_LIGHT_JS)
        page.wait_for_timeout(1000)

        # Count slides
        n_slides = page.evaluate("document.querySelectorAll('.slide').length")
        print(f"Found {n_slides} slides")

        screenshots = []
        for i in range(n_slides):
            # Navigate to slide i
            page.evaluate(f"show({i})")
            # Wait for Plotly charts and MathJax to render
            page.wait_for_timeout(1500)

            # Re-apply Plotly light theme (new charts may have rendered)
            page.evaluate(PLOTLY_LIGHT_JS)
            page.wait_for_timeout(500)

            path = tmp_dir / f"slide_{i:02d}.png"
            page.screenshot(path=str(path), type="png")
            screenshots.append(str(path))
            print(f"  Slide {i+1}/{n_slides}: {path.name}")

        browser.close()

    # Combine PNGs into PDF
    print("Combining into PDF...")
    try:
        from PIL import Image
    except ImportError:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "Pillow", "--quiet"],
            check=True,
        )
        from PIL import Image

    images = [Image.open(s).convert("RGB") for s in screenshots]
    images[0].save(
        OUTPUT_PDF,
        save_all=True,
        append_images=images[1:],
        resolution=200,
    )

    # Cleanup screenshots
    for s in screenshots:
        os.remove(s)
    tmp_dir.rmdir()

    size_mb = os.path.getsize(OUTPUT_PDF) / 1024 / 1024
    print(f"\nDone! {OUTPUT_PDF} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
