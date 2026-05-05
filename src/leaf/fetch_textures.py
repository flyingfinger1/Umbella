"""Download CC0 outdoor texture photos from Pexels for synth background augmentation.

Pexels is free (no attribution required) and has a generous API rate limit
(200 requests/hour). Sign up for a free API key at https://www.pexels.com/api/,
then run with the key in the PEXELS_API_KEY environment variable.

Output: data/outdoor_textures/<category>/*.jpg, ~8 images per category × 8
categories = ~64 subject-free outdoor backgrounds.

Usage:
    set PEXELS_API_KEY=your_key_here
    .venv/Scripts/python.exe -m src.leaf.fetch_textures
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import requests


API = "https://api.pexels.com/v1/search"
USER_AGENT = "Umbella plant-research project (educational)"

# Categories chosen to cover the realistic outdoor backgrounds a plant photo
# might have. Each category will get up to N images via Pexels search.
QUERIES = {
    "meadow":      "meadow grass field",
    "lawn":        "lawn grass close up",
    "forest_floor": "forest floor leaves",
    "soil":        "soil dirt ground texture",
    "hedge":       "hedge green leaves",
    "bark":        "tree bark texture",
    "wall":        "stone wall texture",
    "sky":         "blue sky clouds",
}

PER_CATEGORY = 8
SLEEP_BETWEEN_REQUESTS_S = 0.5


def fetch_category(query: str, api_key: str, count: int, out_dir: Path,
                   log=print) -> int:
    headers = {"Authorization": api_key, "User-Agent": USER_AGENT}
    params = {"query": query, "per_page": count, "orientation": "landscape"}
    r = requests.get(f"{API}?{urlencode(params)}", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    photos = data.get("photos", [])
    log(f"    API returned {len(photos)} photos. Response keys: {list(data.keys())}")
    if photos:
        first = photos[0]
        log(f"    first photo keys: {list(first.keys())}")
        if "src" in first:
            log(f"    first photo src keys: {list(first['src'].keys())}")
    saved = 0
    for photo in photos:
        photo_id = photo.get("id")
        url = photo.get("src", {}).get("large")  # ~1500-2000 px
        if not url or photo_id is None:
            log(f"    skipped photo: id={photo_id}, url={url}")
            continue
        fname = out_dir / f"{photo_id}.jpg"
        if fname.exists():
            continue
        try:
            img_r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            img_r.raise_for_status()
            fname.write_bytes(img_r.content)
            saved += 1
        except Exception as e:
            log(f"    failed {url}: {e}")
        time.sleep(SLEEP_BETWEEN_REQUESTS_S * 0.5)
    return saved


def main() -> None:
    api_key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not api_key:
        sys.exit("ERROR: set PEXELS_API_KEY environment variable\n"
                 "  (free signup at https://www.pexels.com/api/ — takes 2 minutes)")

    out_root = Path(__file__).resolve().parents[2] / "data" / "bg_textures"
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for category, query in QUERIES.items():
        out_dir = out_root / category
        out_dir.mkdir(exist_ok=True)
        print(f"\n=== {category} (query: {query!r}) ===")
        n = fetch_category(query, api_key, PER_CATEGORY, out_dir)
        print(f"  saved {n} new images")
        total += n
        time.sleep(SLEEP_BETWEEN_REQUESTS_S)

    print(f"\ntotal: {total} new outdoor texture images in {out_root}")


if __name__ == "__main__":
    main()
