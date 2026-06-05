"""
download_noto_fonts.py
──────────────────────────────────────────────────────────────────────────────
One-time script: downloads the Noto Sans fonts needed for rendering
non-Latin scripts (Hindi, Odia, Tamil, Bengali, Telugu) in the webcam feed.

Run once before first use:
    python download_noto_fonts.py
──────────────────────────────────────────────────────────────────────────────
"""

import os
import urllib.request

FONT_DIR = os.path.join("assets", "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

# Direct download URLs from Google Fonts / GitHub releases
FONTS = {
    "NotoSansDevanagari-Regular.ttf": (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
    ),
    "NotoSansOriya-Regular.ttf": (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansOriya/NotoSansOriya-Regular.ttf"
    ),
    "NotoSansTamil-Regular.ttf": (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansTamil/NotoSansTamil-Regular.ttf"
    ),
    "NotoSansBengali-Regular.ttf": (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansBengali/NotoSansBengali-Regular.ttf"
    ),
    "NotoSansTelugu-Regular.ttf": (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansTelugu/NotoSansTelugu-Regular.ttf"
    ),
}

for filename, url in FONTS.items():
    dest = os.path.join(FONT_DIR, filename)
    if os.path.exists(dest):
        print(f"  [SKIP]  {filename} already exists")
        continue
    print(f"  [GET]   {filename} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK]    saved → {dest}")
    except Exception as exc:
        print(f"  [FAIL]  {filename}: {exc}")
        print(f"          Manual download: {url}")

print("\nDone. Font directory:", os.path.abspath(FONT_DIR))