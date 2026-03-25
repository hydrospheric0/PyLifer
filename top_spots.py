#!/usr/bin/env python3
"""
top_spots.py — Find the top birding hotspots to visit for your remaining lifers.

For each region, accumulates all 52 weekly S&T rasters at 3km to build a
(52, h, w) stack, then scores each pixel by:
  - peak   : highest species count in any single week  (default)
  - median : median species count across all 52 weeks
  - mean   : mean species count across all 52 weeks

The top-N pixel centres are reverse-geocoded via Nominatim to find the
nearest named place.

Usage:
    python top_spots.py --regions NL
    python top_spots.py --regions US US-CA --n 5 --metric median
    python top_spots.py --regions US --resolution 9km --n 20

Options:
    --regions CODE [CODE ...]   eBird region codes (default: NL US)
    --resolution RES            3km / 9km / 27km (default: 3km)
    --n N                       number of top spots to report (default: 10)
    --metric peak|median|mean   scoring metric (default: peak)
    --min-km MIN                minimum km between returned spots (default: 50)
    --offline                   skip eBird API, use all locally cached tifs
    --ebird-csv PATH            path to MyEBirdData.csv (default: MyEBirdData.csv)
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
import requests
from rasterio.transform import xy as rasterio_xy
from pyproj import Transformer

from map_lifers import (
    load_config,
    get_boundary,
    accumulate_all_weeks,
    get_taxonomy,
    ebird_regional_species,
    user_seen_codes,
)

# Nominatim endpoint — uses OSM, no key required; 1 req/s policy
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "PyLifer/1.0 (github.com/hydrospheric0/PyLifer)"}

# Mapping of region prefix → relative path to local hotspot GeoPackage
HOTSPOT_DBS: dict[str, Path] = {
    "US": Path("data/ebird_hotspots_us.gpkg"),
}

# Fixed EPSG:8857 3km global grid (matches eBird S&T rasters exactly)
_GRID_ORIGIN_X = -17226000.0
_GRID_ORIGIN_Y =   8343000.0
_GRID_CELL_M   =      3000.0


def _cell_key(x: float, y: float) -> tuple[int, int]:
    """Map an EPSG:8857 coordinate to its 3km grid cell index (col, row)."""
    col = int(round((x - _GRID_ORIGIN_X) / _GRID_CELL_M))
    row = int(round((_GRID_ORIGIN_Y - y) / _GRID_CELL_M))
    return (col, row)


# ---------------------------------------------------------------------------
# Local hotspot grid helpers
# ---------------------------------------------------------------------------
# HotspotEntry keys: locId, locName, lat, lon, spp, county (subnational2Code)
HotspotEntry = dict

# Pickle format version — bump to force a rebuild when the schema changes
_GRID_VERSION = 2


def load_hotspot_grid(path: Path) -> dict[tuple[int, int], HotspotEntry]:
    """Build (or load cached) grid dict: cell key → best hotspot in that cell.

    One entry per 3km cell: the hotspot with the highest numSpeciesAllTime.
    Includes county (subnational2Code) for deduplication.
    Cached as a versioned .pkl next to the GPKG; rebuilt when GPKG is newer
    or the schema version changes.
    """
    cache_path = path.with_suffix(".pkl")
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and cached.get("_version") == _GRID_VERSION:
            grid = cached["grid"]
            print(f"  Hotspot grid: {len(grid):,} cells (cached)")
            return grid

    mb = path.stat().st_size // 1024 ** 2
    print(f"  Hotspot grid: building from GPKG ({mb} MB) … ", end="", flush=True)
    gdf   = gpd.read_file(path)
    gdf_p = gdf.to_crs("EPSG:8857")
    xs = gdf_p.geometry.x.to_numpy()
    ys = gdf_p.geometry.y.to_numpy()

    # Vectorised cell key computation
    cols_g = np.round((xs - _GRID_ORIGIN_X) / _GRID_CELL_M).astype(np.int32)
    rows_g = np.round((_GRID_ORIGIN_Y - ys) / _GRID_CELL_M).astype(np.int32)
    spp_arr = gdf["numSpeciesAllTime"].fillna(0).to_numpy(dtype=np.float32)

    # Sort descending by spp so the first time we see a cell key it's the best
    order = np.argsort(spp_arr)[::-1]

    grid: dict[tuple[int, int], HotspotEntry] = {}
    for i in order:
        key = (int(cols_g[i]), int(rows_g[i]))
        if key in grid:
            continue  # already have the best entry for this cell
        grid[key] = {
            "locId":   str(gdf["locId"].iat[i]),
            "locName": str(gdf["locName"].iat[i]),
            "lat":     float(gdf["latitude"].iat[i]),
            "lon":     float(gdf["longitude"].iat[i]),
            "spp":     float(spp_arr[i]),
            "county":  str(gdf["subnational2Code"].iat[i] or ""),
        }

    payload = {"_version": _GRID_VERSION, "grid": grid}
    with open(cache_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(grid):,} cells")
    return grid


def best_hotspot_in_grid(
    cx: float,
    cy: float,
    grid: dict[tuple[int, int], HotspotEntry],
) -> HotspotEntry | None:
    """Return the best hotspot for (cx, cy) in EPSG:8857.

    Step 1: exact cell match (same 3km pixel).
    Step 2: if nothing found, search the 8 surrounding cells (3x3, ~9km).
    Returns the entry with the highest spp, or None.
    """
    col0, row0 = _cell_key(cx, cy)

    # Tier 1 — exact cell (3km)
    hit = grid.get((col0, row0))
    if hit is not None:
        return hit

    # Tier 2 — 3×3 ring (9km, 8 cells) — early exit if anything found
    best: HotspotEntry | None = None
    best_spp = -1.0
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == 0 and dr == 0:
                continue
            entry = grid.get((col0 + dc, row0 + dr))
            if entry is not None and entry["spp"] > best_spp:
                best_spp = entry["spp"]
                best = entry
    if best is not None:
        return best

    # Tier 3 — 9×9 ring (27km, 72 remaining cells outside the 3×3)
    for dc in range(-4, 5):
        for dr in range(-4, 5):
            if -1 <= dc <= 1 and -1 <= dr <= 1:
                continue  # already checked in tier 2
            entry = grid.get((col0 + dc, row0 + dr))
            if entry is not None and entry["spp"] > best_spp:
                best_spp = entry["spp"]
                best = entry
    return best


# ---------------------------------------------------------------------------
# Reverse geocode (fallback when no hotspot found)
# ---------------------------------------------------------------------------
def reverse_geocode(lat: float, lon: float) -> str:
    """Return a human-readable place name for (lat, lon) via Nominatim.

    Priority:
      1. Named nature area (reserve / park) — great for remote birding spots
      2. Nearest town / village / suburb
      3. County (fallback for wilderness with no named settlement)
    """
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 14},
            headers=NOMINATIM_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        addr = data.get("address", {})

        state   = addr.get("state") or addr.get("region") or ""
        country = addr.get("country_code", "").upper()

        # Tier 1 — protected / natural area name
        natural = (
            addr.get("nature_reserve")
            or addr.get("protected_area")
            or addr.get("national_park")
        )

        # Tier 2 — nearest named settlement
        town = (
            addr.get("suburb")
            or addr.get("village")
            or addr.get("hamlet")
            or addr.get("city_district")
            or addr.get("city")
            or addr.get("town")
            or addr.get("municipality")
        )

        # Tier 3 — administrative fallback
        county = addr.get("county")

        # Build output: prefer natural + town; fall back to county
        place = natural or town or county or state or "unknown"

        parts: list[str] = [place]
        if state and state != place:
            parts.append(state)
        if country and country not in parts:
            parts.append(country)

        return ", ".join(parts)
    except Exception as exc:
        return f"geocode error: {exc}"


# ---------------------------------------------------------------------------
# Minimum-distance filter  (greedy, Euclidean in raster CRS)
# ---------------------------------------------------------------------------
def filter_min_distance(
    candidates: list[tuple],   # list of (score, row, col, lat, lon, crs_x, crs_y)
    min_km: float,
    crs: str,
) -> list[tuple]:
    """Keep candidates that are at least min_km apart (in the raster CRS)."""
    if min_km <= 0:
        return candidates

    kept: list[tuple] = []
    kept_xy: list[tuple[float, float]] = []
    threshold_sq = (min_km * 1000) ** 2   # compare squared distances — no sqrt needed

    for cand in candidates:
        cx, cy = cand[5], cand[6]
        too_close = any(
            (cx - kx) ** 2 + (cy - ky) ** 2 < threshold_sq
            for kx, ky in kept_xy
        )
        if not too_close:
            kept.append(cand)
            kept_xy.append((cx, cy))
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find top lifer hotspots using S&T rasters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--regions",    nargs="+", default=["NL", "US"], metavar="CODE")
    parser.add_argument("--resolution", default="3km", choices=["3km", "9km", "27km"])
    parser.add_argument("--n",          type=int, default=10, help="Number of top spots.")
    parser.add_argument(
        "--metric", default="peak", choices=["peak", "median", "mean"],
        help="Scoring metric across the 52-week stack.",
    )
    parser.add_argument(
        "--min-km", type=float, default=50.0,
        help="Minimum km between returned spots (0 = disabled).",
    )
    parser.add_argument("--offline",   action="store_true")
    parser.add_argument("--ebird-csv", default="MyEBirdData.csv", metavar="PATH")
    args = parser.parse_args()

    cfg    = load_config()
    n_want = args.n * 20   # oversample before distance + dedup filter

    # Cache loaded hotspot DBs across regions (loaded at most once per prefix)
    _hs_cache: dict[str, tuple | None] = {}

    for region in args.regions:
        print(f"\n{'='*60}")
        print(f"  Region: {region}  |  Resolution: {args.resolution}  |  Metric: {args.metric}")
        print(f"{'='*60}")

        boundary = get_boundary(region)

        # ---- load hotspot grid for this region prefix (once) ------------
        db_prefix = region.split("-")[0]
        if db_prefix not in _hs_cache:
            db_path = HOTSPOT_DBS.get(db_prefix)
            if db_path and db_path.exists():
                _hs_cache[db_prefix] = load_hotspot_grid(db_path)
            else:
                _hs_cache[db_prefix] = None
        hs_db = _hs_cache[db_prefix]

        # ---- species list ------------------------------------------------
        if args.offline:
            from map_lifers import CACHE_DIR, EBIRDST_VERSION, tif_path
            _offline_cache = CACHE_DIR / EBIRDST_VERSION
            if not _offline_cache.exists():
                sys.exit(f"Offline mode: cache directory {_offline_cache} not found.")
            needed: set[str] = {
                d.name for d in _offline_cache.iterdir()
                if d.is_dir() and tif_path(d.name, args.resolution).exists()
            }
            print(f"  [offline] {len(needed)} cached species")
        else:
            taxonomy    = get_taxonomy(cfg["ebird_api_key"])
            regional_sp = ebird_regional_species(region, cfg["ebird_api_key"])
            seen        = user_seen_codes(Path(args.ebird_csv), region, taxonomy)
            from map_lifers import load_ebirdst_runs, EXCLUDED_CODES
            modeled = load_ebirdst_runs()
            needed = (regional_sp - seen) & set(modeled.keys()) - EXCLUDED_CODES
            print(f"  eBird checklist : {len(regional_sp)}  |  seen : {len(seen)}  |  needed : {len(needed)}")

        # ---- accumulate --------------------------------------------------
        print(f"  Accumulating {len(needed)} species …")
        stack, transform, crs_src, week_dates = accumulate_all_weeks(
            needed, boundary, args.resolution,
        )
        # stack shape: (52, H, W)   values: species count per pixel per week

        # ---- score -------------------------------------------------------
        with np.errstate(all="ignore"):
            if args.metric == "peak":
                score = np.nanmax(stack.astype(np.float32), axis=0)
            elif args.metric == "median":
                score = np.nanmedian(stack.astype(np.float32), axis=0)
            else:
                score = np.nanmean(stack.astype(np.float32), axis=0)

        # ---- top candidates ---------------------------------------------
        H, W = score.shape
        flat = score.ravel()

        # Only consider valid (non-NaN, non-zero) pixels
        valid_mask = np.isfinite(flat) & (flat > 0)
        valid_idx  = np.where(valid_mask)[0]
        if valid_idx.size == 0:
            print("  No valid pixels found — make sure tifs are cached.")
            continue

        # Sort valid pixels descending by score
        order    = np.argsort(flat[valid_idx])[::-1]
        top_flat = valid_idx[order[:n_want]]

        # Batch-convert all candidate pixel centres to WGS84 in one call each
        rows_arr, cols_arr = np.divmod(top_flat, W)
        cxs, cys = rasterio_xy(transform, rows_arr, cols_arr)
        lons, lats = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True).transform(cxs, cys)

        candidates: list[tuple] = [
            (float(flat[top_flat[i]]), int(rows_arr[i]), int(cols_arr[i]),
             float(lats[i]), float(lons[i]), float(cxs[i]), float(cys[i]))
            for i in range(len(top_flat))
        ]

        # ---- distance filter --------------------------------------------
        kept_all = filter_min_distance(candidates, args.min_km, crs_src)

        # ---- reverse geocode + print ------------------------------------
        print(f"\n  Top {args.n} spots (min {args.min_km:.0f} km apart):\n")
        print(f"  {'Rank':>4}  {'Score':>6}  {'HsSpp':>6}  {'Lat':>8}  {'Lon':>9}  Location")
        print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*30}")

        rank = 0
        seen_places: set[str] = set()
        for sc, r, c, lat, lon, cx, cy in kept_all:
            if rank >= args.n:
                break

            # --- local hotspot grid lookup (exact cell → 3x3 fallback) --
            hs = None
            if hs_db is not None:
                hs = best_hotspot_in_grid(cx, cy, hs_db)

            if hs:
                place     = hs["locName"]
                hs_spp    = str(int(hs["spp"])) if hs["spp"] == hs["spp"] else "-"
                # Deduplicate by county code so each county appears at most once
                dedup_key = hs["county"] or hs["locId"]
            else:
                place     = reverse_geocode(lat, lon)
                hs_spp    = "-"
                dedup_key = ", ".join(p.strip() for p in place.split(",")[:2])
                time.sleep(1.1)   # Nominatim 1 req/s policy

            if dedup_key in seen_places:
                continue
            seen_places.add(dedup_key)
            rank += 1
            print(f"  {rank:>4}  {sc:>6.1f}  {hs_spp:>6}  {lat:>8.4f}  {lon:>9.4f}  {place}")

        print()


if __name__ == "__main__":
    main()
