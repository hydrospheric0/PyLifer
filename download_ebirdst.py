#!/usr/bin/env python3
"""
download_ebirdst.py — Bulk pre-downloader for eBird Status & Trends rasters.

Downloads occurrence_median tifs for all species the user still needs in the
specified regions, at all resolutions (3km, 9km, 27km), into the same cache
directory used by the R ebirdst package.  This means R scripts can use the
cached files without re-downloading.

Download API (reverse-engineered from R ebirdst package):
  File list : GET https://st-download.ebird.org/v1/list-obj/{year}/{code}?key={k}
  File fetch: GET https://st-download.ebird.org/v1/fetch?objKey={path}&key={k}

Requirements:
    pip install requests tqdm

Setup (once):
    Rscript export_ebirdst_runs.R   # generates ebirdst_runs.csv

Usage:
    python download_ebirdst.py                          # NL + US, all resolutions
    python download_ebirdst.py --regions NL             # NL only
    python download_ebirdst.py --resolutions 9km 27km   # skip 3km
    python download_ebirdst.py --needs global           # true lifers (global list)
    python download_ebirdst.py --dry-run                # show counts, no download
    python download_ebirdst.py --workers 8              # more parallel threads
    python download_ebirdst.py --yes                    # skip confirmation prompt

Config (.env):
    EBIRDST_KEY=...      eBird S&T download key
    EBIRD_API_KEY=...    eBird API key
    USER=...             your name (shown in map titles)
"""

import argparse
import csv
import os
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EBIRDST_VERSION   = "2023"
DOWNLOAD_BASE     = "https://st-download.ebird.org/v1"
EBIRD_API_BASE    = "https://api.ebird.org/v2"
# Store rasters in the project folder, not the hidden R per-user dir.
# R scripts read this via the EBIRDST_DATA_DIR env var (set at top of each script).
DEFAULT_CACHE_DIR = Path(__file__).parent / "data" / "ebirdst"
RESOLUTIONS       = ["3km", "9km", "27km"]
# Species excluded in R scripts due to known data issues
EXCLUDED_CODES    = {"laugul", "rocpig", "compea", "yebsap-example"}


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
def load_config(config_path: Path = Path(".env")) -> dict:
    """Parse ebirdst_key and ebird_api_key from .env."""
    import re
    if not config_path.exists():
        sys.exit(
            f".env not found at {config_path}.\n"
            "Copy .env.example to .env and fill in your keys."
        )
    keys = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, rhs = line.partition("=")
        name = name.strip()
        rhs = rhs.strip().strip('"').strip("'")
        if name in ("EBIRDST_KEY", "EBIRD_API_KEY"):
            keys[name.lower()] = rhs
    for required in ("ebirdst_key", "ebird_api_key"):
        if required not in keys or not keys[required] or "YOUR_" in keys[required]:
            sys.exit(f"Missing or placeholder value for '{required.upper()}' in .env")
    return keys


# ---------------------------------------------------------------------------
# eBird API helpers
# ---------------------------------------------------------------------------
def ebird_regional_species(region: str, api_key: str) -> list[str]:
    """Return list of species codes on the eBird regional checklist."""
    url = f"{EBIRD_API_BASE}/product/spplist/{region}"
    r = requests.get(url, headers={"X-eBirdApiToken": api_key}, timeout=30)
    r.raise_for_status()
    return r.json()


def ebird_taxonomy(api_key: str) -> dict[str, str]:
    """Return {common_name: species_code} mapping from eBird taxonomy API."""
    url = f"{EBIRD_API_BASE}/ref/taxonomy/ebird"
    r = requests.get(
        url,
        headers={"X-eBirdApiToken": api_key},
        params={"fmt": "json", "cat": "species"},
        timeout=60,
    )
    r.raise_for_status()
    return {row["comName"]: row["speciesCode"] for row in r.json()}


# ---------------------------------------------------------------------------
# User species parsing
# ---------------------------------------------------------------------------
def user_seen_codes(csv_path: Path, region: str, name_to_code: dict[str, str]) -> set[str]:
    """
    Parse MyEBirdData.csv and return species codes the user has already observed
    in the given region.  Filters by country for country-level regions (e.g. 'NL'),
    or by full state/province code for sub-national regions (e.g. 'US-NY').
    """
    country = region.split("-")[0]
    is_state = "-" in region
    seen = set()

    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — treating all regional species as needed.")
        return seen

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state_prov = row.get("State/Province", "")
            if is_state:
                if state_prov != region:
                    continue
            else:
                if not (state_prov == country or state_prov.startswith(country + "-")):
                    continue
            common = row.get("Common Name", "").strip()
            if common and common in name_to_code:
                seen.add(name_to_code[common])
    return seen


# ---------------------------------------------------------------------------
# ebirdst_runs table
# ---------------------------------------------------------------------------
def load_ebirdst_runs(csv_path: Path = Path("ebirdst_runs.csv")) -> dict[str, str]:
    """Return {species_code: common_name} for species with S&T models."""
    if not csv_path.exists():
        sys.exit(
            f"ebirdst_runs.csv not found.\n"
            "Run once:  Rscript export_ebirdst_runs.R"
        )
    modeled = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            modeled[row["species_code"]] = row["common_name"]
    return modeled


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def tif_dest(cache_dir: Path, species_code: str, resolution: str) -> Path:
    return (
        cache_dir
        / EBIRDST_VERSION
        / species_code
        / "weekly"
        / f"{species_code}_occurrence_median_{resolution}_{EBIRDST_VERSION}.tif"
    )


def tif_url(species_code: str, resolution: str, ebirdst_key: str) -> str:
    obj_key = (
        f"{EBIRDST_VERSION}/{species_code}/weekly/"
        f"{species_code}_occurrence_median_{resolution}_{EBIRDST_VERSION}.tif"
    )
    return (
        f"{DOWNLOAD_BASE}/fetch"
        f"?objKey={urllib.parse.quote(obj_key, safe='/')}"
        f"&key={ebirdst_key}"
    )


def download_one(
    species_code: str,
    resolution: str,
    ebirdst_key: str,
    cache_dir: Path,
    force: bool = False,
) -> tuple[str, str, str]:
    """Download a single tif.  Returns (status, species_code, resolution)."""
    dest = tif_dest(cache_dir, species_code, resolution)
    if dest.exists() and not force:
        return "cached", species_code, resolution

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = tif_url(species_code, resolution, ebirdst_key)

    tmp = dest.with_suffix(".tmp")
    try:
        with requests.get(url, timeout=600, stream=True) as r:
            if r.status_code == 404:
                return "missing", species_code, resolution
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
            tmp.rename(dest)
        return "downloaded", species_code, resolution
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return f"error:{exc}", species_code, resolution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk pre-download eBird S&T occurrence rasters for lifeR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--regions", nargs="+", default=["NL", "US"],
        metavar="CODE",
        help="eBird region codes (e.g. NL US US-NY MX-TAM)",
    )
    parser.add_argument(
        "--resolutions", nargs="+", default=["3km", "9km", "27km"],
        choices=["3km", "9km", "27km"],
    )
    parser.add_argument(
        "--needs", choices=["regional", "global"], default="regional",
        help="'regional' subtracts species you've seen in the region; "
             "'global' subtracts all species you've ever seen anywhere.",
    )
    parser.add_argument(
        "--ebird-csv", default="MyEBirdData.csv",
        metavar="PATH", help="Path to your MyEBirdData.csv eBird export.",
    )
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE_DIR),
        metavar="DIR", help="ebirdst cache directory (shared with R).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download threads.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without downloading anything.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt and proceed immediately.",
    )
    args = parser.parse_args()

    config    = load_config()
    cache_dir = Path(args.cache_dir)
    csv_path  = Path(args.ebird_csv)
    modeled   = load_ebirdst_runs()

    print(f"S&T modeled species: {len(modeled)}")
    print(f"Fetching eBird taxonomy for name→code lookup …")
    name_to_code = ebird_taxonomy(config["ebird_api_key"])

    # ------------------------------------------------------------------
    # Build union of needed species across all regions
    # ------------------------------------------------------------------
    all_needed: set[str] = set()

    # For "global" needs: collect everything the user has ever seen anywhere
    if args.needs == "global":
        global_seen: set[str] = set()
        if csv_path.exists():
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    common = row.get("Common Name", "").strip()
                    if common and common in name_to_code:
                        global_seen.add(name_to_code[common])
        print(f"User global life list: {len(global_seen)} species\n")

    for region in args.regions:
        print(f"--- {region} ---")
        regional_codes = set(ebird_regional_species(region, config["ebird_api_key"]))
        print(f"  eBird checklist  : {len(regional_codes)} species")

        if args.needs == "global":
            user_seen = global_seen
        else:
            user_seen = user_seen_codes(csv_path, region, name_to_code)
        print(f"  User seen ({args.needs:8s}): {len(user_seen)} species")

        needed = (
            (regional_codes - user_seen)
            & set(modeled.keys())
            - EXCLUDED_CODES
        )
        print(f"  Still needed + S&T model: {len(needed)} species")
        all_needed.update(needed)

    print(f"\nUnique species to cache: {len(all_needed)}")
    print(f"Resolutions            : {', '.join(args.resolutions)}")
    total_files = len(all_needed) * len(args.resolutions)
    print(f"Total files            : {total_files}  (~{total_files * 22 // 1024} GB estimated at 9km sizing)")

    if args.dry_run:
        print("\nDry run — no files downloaded.")
        for sp in sorted(all_needed):
            name = modeled.get(sp, sp)
            for res in args.resolutions:
                dest = tif_dest(cache_dir, sp, res)
                status = "EXISTS" if dest.exists() else "MISSING"
                print(f"  [{status}] {sp:20s} {res}  {name}")
        return

    # ------------------------------------------------------------------
    # Confirmation prompt
    # ------------------------------------------------------------------
    already_cached = sum(
        1 for sp in all_needed for res in args.resolutions
        if tif_dest(cache_dir, sp, res).exists()
    )
    to_download = total_files - already_cached
    print(f"  Already cached     : {already_cached}")
    print(f"  To download        : {to_download}")

    if to_download == 0:
        print("\nNothing to download.")
        return

    if not args.yes:
        try:
            answer = input("\nProceed with download? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # ------------------------------------------------------------------
    # Parallel download
    # ------------------------------------------------------------------
    jobs = [(sp, res) for sp in sorted(all_needed) for res in args.resolutions]
    counters: dict[str, int] = {"downloaded": 0, "cached": 0, "missing": 0, "error": 0}
    errors: list[str] = []

    print(f"\nDownloading with {args.workers} threads …")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one, sp, res, config["ebirdst_key"], cache_dir, args.force
            ): (sp, res)
            for sp, res in jobs
        }

        completed = as_completed(futures)
        if _HAS_TQDM:
            completed = tqdm(completed, total=len(jobs), unit="file", dynamic_ncols=True)

        for future in completed:
            status, sp, res = future.result()
            key = "error" if status.startswith("error") else status
            counters[key] += 1
            if key in ("error", "missing"):
                msg = f"  [{status}] {sp}  {res}"
                errors.append(msg)
                if not _HAS_TQDM:
                    print(f"\n{msg}")
            if not _HAS_TQDM:
                done = sum(counters.values())
                print(
                    f"\r  [{done}/{len(jobs)}]  "
                    f"ok={counters['downloaded']}  "
                    f"cached={counters['cached']}  "
                    f"err={counters['error']}  ",
                    end="", flush=True,
                )

    if not _HAS_TQDM:
        print()  # newline after progress

    print(
        f"\nDone.\n"
        f"  downloaded : {counters['downloaded']}\n"
        f"  cached     : {counters['cached']}\n"
        f"  missing    : {counters['missing']}  (species exist in eBird but not in S&T)\n"
        f"  errors     : {counters['error']}"
    )
    if errors:
        print("\nProblematic files:")
        for e in errors:
            print(e)


if __name__ == "__main__":
    main()
