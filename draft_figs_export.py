"""draft_figs_export.py SRC_DIR DEST_DIR — every SRC_DIR/<id>.html printed to DEST_DIR/<id>.pdf at the size of its #fig
element, light colours, with headless Chromium: playwright's own build (`playwright install chromium`), or the
executable in $CHROMIUM."""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

EXE = os.environ.get("CHROMIUM")
src, dest = Path(sys.argv[1]), Path(sys.argv[2])
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE) if EXE else p.chromium.launch()
    pg = b.new_page(viewport={"width": 1040, "height": 900}, color_scheme="light")
    for f in sorted(src.glob("*.html")):
        pg.goto(f.as_uri(), wait_until="networkidle")
        pg.emulate_media(media="screen", color_scheme="light")
        box = pg.locator("#fig").bounding_box()
        pg.pdf(path=str(dest / f"{f.stem}.pdf"), width=f"{box['width'] + 2:.0f}px", height=f"{box['height'] + 2:.0f}px",
               print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}, page_ranges="1")
        print("pdf", f.stem, round(box["width"]), round(box["height"]))
    b.close()
