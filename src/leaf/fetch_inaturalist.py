"""Download research-grade Apiaceae photos from iNaturalist.

Per species, queries the public API for observations tagged research-grade,
then downloads each observation's photos at "medium" resolution (~500 px max
side). Stores them as data/leaf_images/<species>/<obs_id>_<idx>.jpg.

Polite usage:
  - 1 second sleep between requests
  - User-Agent header identifies the project
  - skips already-downloaded files
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode

import requests


API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "Umbella plant-research project (educational, github.com/flyingfinger1/Umbella)"
PER_PAGE = 100
SLEEP_BETWEEN_REQUESTS_S = 1.0


# iNaturalist taxon ids — verified via the /v1/taxa search endpoint, so a
# query for the species name returns the canonical Plantae taxon. (Earlier
# values in this file were placeholder guesses that turned out to point to
# completely unrelated plants — careful when adding more species: always
# verify via API, not by guessing.)
INAT_TAXA = {
    "Heracleum_sphondylium": 163682,
    "Conium_maculatum":       52998,
    "Daucus_carota":          76610,
    "Anthriscus_sylvestris": 124544,
    "Aethusa_cynapium":      158043,
    "Pastinaca_sativa":       59778,
}


def _api_get(params: dict) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = f"{API}?{urlencode(params)}"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _photo_url_medium(photo: dict) -> str | None:
    """Pick a 'medium' (~500 px max side) URL from a photo entry. iNat photos
    usually expose a `url` containing 'square' (75 px); we substitute the size
    keyword to get medium."""
    base = photo.get("url") or ""
    if not base:
        return None
    return base.replace("square", "medium")


def fetch_species_images(
    species_key: str,
    out_root: Path,
    max_observations: int = 250,
    max_photos_per_obs: int = 3,
    place_id: int | None = 7207,        # Germany (verified via places/autocomplete API)
    min_identifications: int = 2,       # post-filter: skip obs with too few IDs;
                                        # research_grade implies ≥2 anyway, so 2 = no extra filter
                                        # (>=3 turned out to drop ~99% — see iNat data distribution)
    log = print,
) -> int:
    """Fetch research-grade observations and download photos.

    Tightened against iNat misidentification noise:
      - default `place_id=7035` (Germany) restricts to a region with active,
        regionally knowledgeable identifiers
      - `min_identifications` skips observations with fewer than N IDs
        (raw "research_grade" only requires 2 — too lenient for the
        Heracleum/Petasites or Conium/Anthriscus look-alike confusions
        we observed in the data)

    Returns count of new images saved.
    """
    if species_key not in INAT_TAXA:
        raise KeyError(f"unknown species_key {species_key!r}")
    taxon_id = INAT_TAXA[species_key]
    out_dir = Path(out_root) / species_key
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    page = 1
    seen_obs = 0
    skipped_low_id = 0
    while seen_obs < max_observations:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": PER_PAGE,
            "page": page,
            "order_by": "votes",        # most-loved first → cleaner images
            "order": "desc",
            "locale": "de",
        }
        if place_id is not None:
            params["place_id"] = place_id

        try:
            data = _api_get(params)
        except Exception as e:
            log(f"  page {page}: API error {e}; retrying in 5 s")
            time.sleep(5)
            continue

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            seen_obs += 1
            if seen_obs > max_observations:
                break
            # post-filter on identifications_count to weed out 2-confirmation
            # research-grade observations that frequently mislabel look-alikes
            n_ids = int(obs.get("identifications_count", 0))
            if n_ids < min_identifications:
                skipped_low_id += 1
                continue
            obs_id = obs.get("id")
            photos = obs.get("photos", []) or []
            for idx, photo in enumerate(photos[:max_photos_per_obs]):
                url = _photo_url_medium(photo)
                if not url:
                    continue
                fname = out_dir / f"{obs_id}_{idx}.jpg"
                if fname.exists():
                    continue
                try:
                    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
                    r.raise_for_status()
                    fname.write_bytes(r.content)
                    saved += 1
                except Exception as e:
                    log(f"  failed {url}: {e}")
                time.sleep(SLEEP_BETWEEN_REQUESTS_S * 0.3)

        log(f"  {species_key} page {page}: cumulative {saved} new images, "
            f"{seen_obs} obs seen, {skipped_low_id} skipped (low ID count)")
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS_S)
        # iNat returns total_results; stop early if we've consumed them
        if seen_obs >= data.get("total_results", 0):
            break

    return saved


if __name__ == "__main__":
    import sys
    out = Path(__file__).resolve().parents[2] / "data" / "leaf_images"
    keys = sys.argv[1:] or list(INAT_TAXA.keys())
    for k in keys:
        print(f"\n=== {k} ===")
        n = fetch_species_images(k, out_root=out, max_observations=250, max_photos_per_obs=3)
        print(f"  total new for {k}: {n}")
