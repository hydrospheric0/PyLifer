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
    --per-unit N                max results per admin unit: 1 per state (US),
                                1 per county (US-CA). Default: 1.
    --metric peak|median|mean   scoring metric (default: peak)
    --min-km MIN                minimum km between returned spots (default: 50)
    --offline                   skip eBird API, use all locally cached tifs
    --ebird-csv PATH            path to MyEBirdData.csv (default: MyEBirdData.csv)
"""
import argparse
import hashlib
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
    tif_path,
    OCCURRENCE_THRESH,
    N_WORKERS,
)

# Nominatim endpoint — uses OSM, no key required; 1 req/s policy
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "PyLifer/1.0 (github.com/hydrospheric0/PyLifer)"}

# Mapping of region prefix → relative path to local hotspot GeoPackage
HOTSPOT_DBS: dict[str, Path] = {
    "US": Path("data/ebird_hotspots_us.gpkg"),
}

# Score-layer cache directory (keyed by region + resolution + metric + species hash)
SCORE_CACHE_DIR = Path("data") / "score_cache"

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
_GRID_VERSION = 3


def _species_hash(needed: set[str]) -> str:
    """12-char hex hash of a sorted species set — stable cache key."""
    blob = "\n".join(sorted(needed)).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


_PRECOMPUTED_VERSION = 1


def _precomputed_path(region: str, resolution: str, needed: set[str]) -> Path:
    h = _species_hash(needed)
    return SCORE_CACHE_DIR / f"{region}_{resolution}_{h}.precomputed.pkl"


def load_precomputed(region: str, resolution: str, needed: set[str]) -> dict | None:
    """Load the unified precomputed cache dict, or None on miss / version mismatch."""
    p = _precomputed_path(region, resolution, needed)
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    if d.get("_v") != _PRECOMPUTED_VERSION or d.get("species") != sorted(needed):
        return None
    p_mb = p.stat().st_size // 1024 ** 2
    print(f"  Precomputed: loaded from cache ({p.name}, {p_mb} MB)")
    return d


def _precomputed_from_arrays(
    region: str, resolution: str, needed: set[str],
    stack, transform, crs: str, week_dates, annual_richness, win,
    available: list[str], sp_packed,
) -> dict:
    """Compute score layers from an already-accumulated stack and save the cache.

    Call this when you already hold the stack in memory (e.g. after a
    map_lifers animation run) to avoid a second tif-reading pass.
    Prints per-array disk-size breakdown so the overhead is transparent.

    Returns the assembled precomputed dict.
    """
    import time as _time
    t0 = _time.perf_counter()

    with np.errstate(all="ignore"):
        fstack = stack.astype(np.float32)
        peak   = np.nanmax(fstack,    axis=0)
        mean_  = np.nanmean(fstack,   axis=0)
        median = np.nanmedian(fstack, axis=0)
        del fstack

    scores_s = _time.perf_counter() - t0

    d: dict = {
        "_v":              _PRECOMPUTED_VERSION,
        "species":         sorted(needed),
        "stack":           stack,
        "transform":       transform,
        "crs":             crs,
        "week_dates":      week_dates,
        "win":             win,
        "available":       available,
        "annual_richness": annual_richness,
        "peak":            peak,
        "mean":            mean_,
        "median":          median,
        "richness":        annual_richness,   # metric alias
        "sp_packed":       sp_packed,         # None if tracking was disabled
    }

    SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _precomputed_path(region, resolution, needed)
    t1 = _time.perf_counter()
    with open(p, "wb") as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    write_s = _time.perf_counter() - t1

    total_mb = p.stat().st_size / 1024 ** 2

    def _mb(arr) -> str:
        if arr is None: return "—"
        return f"{arr.nbytes / 1024**2:.1f} MB"

    H, W = stack.shape[1], stack.shape[2]
    n_sp = len(available)
    print(
        f"  Precomputed saved → {p.name}  ({total_mb:.0f} MB on disk)\n"
        f"    stack        {_mb(stack):>8}  ({stack.shape[0]}×{H}×{W} float32)\n"
        f"    score layers {_mb(peak):>8}  each  (peak/mean/median/richness × {H}×{W})\n"
        f"    sp_packed    {_mb(sp_packed):>8}  ({n_sp} species → {n_sp//8+1} bytes/pixel)\n"
        f"    score-compute {scores_s:.1f}s   pickle-write {write_s:.1f}s"
    )
    return d


def build_precomputed(
    region: str, resolution: str, needed: set[str], boundary,
) -> dict:
    """Read all tifs ONCE, compute all metrics + per-species presence, cache everything.

    To amortise the tif-reading cost during a map_lifers animation run, call
    _precomputed_from_arrays() directly with the already-accumulated stack.
    """
    import time as _time
    print(f"  Building precomputed cache ({region} @ {resolution}) …")
    t0 = _time.perf_counter()
    stack, transform, crs, week_dates, annual_richness, win, available, sp_packed = \
        accumulate_all_weeks(needed, boundary, resolution, track_sp_presence=True)
    accum_s = _time.perf_counter() - t0
    print(f"    accumulation: {accum_s:.1f}s")
    return _precomputed_from_arrays(
        region, resolution, needed,
        stack, transform, crs, week_dates, annual_richness, win, available, sp_packed,
    )


def _species_from_packed(
    sp_packed, n_sp: int, available: list[str], r: int, c: int,
) -> list[str]:
    """Return species codes present at raster pixel (r, c) using bitpacked array."""
    if sp_packed is None:
        return []
    bits = np.unpackbits(sp_packed[:, r, c])[:n_sp].astype(bool)
    return [available[i] for i in np.where(bits)[0]]



def _results_cache_path(
    region: str, resolution: str, metric: str,
    n: int, depth: int, needed: set[str],
) -> Path:
    h = _species_hash(needed)
    return SCORE_CACHE_DIR / f"{region}_{resolution}_{metric}_n{n}_d{depth}_{h}.results.pkl"


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
            "state":   str(gdf["subnational1Code"].iat[i] or ""),
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
# Species loader (shared between CLI and Dash app)
# ---------------------------------------------------------------------------
def load_needed_species(
    region: str,
    resolution: str,
    cfg: dict,
    offline: bool = False,
    ebird_csv: str = "MyEBirdData.csv",
) -> set[str]:
    """Return the set of species codes to analyse for a region."""
    if offline:
        from map_lifers import CACHE_DIR, EBIRDST_VERSION, tif_path
        cache_dir = CACHE_DIR / EBIRDST_VERSION
        if not cache_dir.exists():
            sys.exit(f"Offline mode: cache directory {cache_dir} not found.")
        needed: set[str] = {
            d.name for d in cache_dir.iterdir()
            if d.is_dir() and tif_path(d.name, resolution).exists()
        }
        print(f"  [offline] {len(needed)} cached species")
        return needed
    taxonomy    = get_taxonomy(cfg["ebird_api_key"])
    regional_sp = ebird_regional_species(region, cfg["ebird_api_key"])
    seen        = user_seen_codes(Path(ebird_csv), region, taxonomy)
    from map_lifers import load_ebirdst_runs, EXCLUDED_CODES
    modeled = load_ebirdst_runs()
    needed = (regional_sp - seen) & set(modeled.keys()) - EXCLUDED_CODES
    print(f"  eBird checklist : {len(regional_sp)}  |  seen : {len(seen)}  |  needed : {len(needed)}")
    return needed


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
# Species extraction (which lifers can you see at each spot?)
# ---------------------------------------------------------------------------
def _extract_species_at_spots(
    available: list[str],
    spots: list[dict],
    win,
    resolution: str,
) -> None:
    """Fill species_codes into each spot dict in-place.

    For every species in `available`, reads its tif once and checks each spot
    pixel.  A species is attributed to a spot if any week exceeds
    OCCURRENCE_THRESH.  Runs fully parallel; results are merged in the
    main thread to avoid any GIL / race issues.
    """
    from concurrent.futures import ThreadPoolExecutor

    import rasterio as _rio

    if not available or not spots:
        for s in spots:
            s.setdefault("species_codes", [])
        return

    spot_rc = [(s["row"], s["col"]) for s in spots]
    n_spots = len(spot_rc)

    def _check(args: tuple[int, str]) -> tuple[str, list[bool]]:
        i, code = args
        try:
            with _rio.open(tif_path(code, resolution)) as src:
                data = src.read(window=win)           # (52, H, W)
            present = []
            for r, c in spot_rc:
                pixel = data[:, r, c]
                present.append(bool(np.any(np.isfinite(pixel) & (pixel > OCCURRENCE_THRESH))))
            return code, present
        except Exception:
            return code, [False] * n_spots

    per_spot: list[list[str]] = [[] for _ in spots]
    workers = min(N_WORKERS, len(available))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for code, presence in pool.map(_check, enumerate(available)):
            for j, present in enumerate(presence):
                if present:
                    per_spot[j].append(code)

    for j, spot in enumerate(spots):
        spot["species_codes"] = per_spot[j]

    n_total = sum(len(s["species_codes"]) for s in spots)
    print(f"  Species extraction: {n_total} species-spot attributions across {len(spots)} spots.")


# ---------------------------------------------------------------------------
# Core computation (importable by Dash app)
# ---------------------------------------------------------------------------
def compute_top_spots(
    region: str,
    resolution: str,
    metric: str,
    n: int,
    depth: int,          # max spots per admin unit (state for country, county for state region)
    min_km: float,
    needed: set[str],
    boundary,
    hs_db: dict | None,
) -> tuple[list[dict], list[str]]:
    """Compute top-N hotspots for a region, including weekly lifer profiles.

    Returns (spots, week_dates).  Results are persisted in the score-cache
    directory; subsequent calls with identical parameters load instantly.

    Each spot dict contains:
        rank, score, lat, lon, row, col,
        loc_id, loc_name, hs_spp, county, state,
        weekly (list[float] × 52, or None if only score cache available)
    """
    rcache_path = _results_cache_path(region, resolution, metric, n, depth, needed)
    if rcache_path.exists():
        with open(rcache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("species") == sorted(needed):
            print(f"  Results: loaded from cache ({rcache_path.name})")
            return cached["spots"], cached["week_dates"]

    # ---- precomputed cache (all metrics + species presence, built once) ------
    precomp = load_precomputed(region, resolution, needed)
    if precomp is None:
        p = _precomputed_path(region, resolution, needed)
        n_sp_est = len(needed)
        print(
            f"\n  No precomputed cache found for {region} @ {resolution} "
            f"({n_sp_est} species).\n"
            f"  Building it reads all tifs once and saves all score layers to:\n"
            f"    {p}\n"
            f"  This takes ~the same time as a full animation run (can be several minutes)."
        )
        try:
            answer = input("  Build now? [Y/n] ").strip().lower()
        except EOFError:
            answer = "y"   # non-interactive (pipe / --build-cache path): proceed
        if answer not in ("", "y", "yes"):
            sys.exit("  Aborted — run with --build-cache to pre-build without prompting.")
        precomp = build_precomputed(region, resolution, needed, boundary)

    stack           = precomp["stack"]
    transform       = precomp["transform"]
    crs_src         = precomp["crs"]
    week_dates      = precomp["week_dates"]
    annual_richness = precomp["annual_richness"]
    available       = precomp["available"]
    sp_packed       = precomp["sp_packed"]
    n_sp            = len(available)

    # ---- score layer (precomputed — O(1) lookup) ---------------------
    score = precomp[metric]   # "peak" | "mean" | "median" | "richness"

    # ---- candidates: per-unit top-K pool for geographic diversity ----------
    # Approach: spatial-join ALL valid pixels once to assign admin units, then
    # cap each unit at k_per_unit candidates (sorted by score desc).  This
    # prevents high-score states (e.g. TX/LA for "peak") from monopolising the
    # global top-N pool and ensures every state with valid pixels is represented.
    H, W = score.shape
    flat = score.ravel()
    valid_mask = np.isfinite(flat) & (flat > 0)
    valid_idx  = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        del stack
        return [], week_dates or []

    use_county_unit = "-" in region

    all_rows, all_cols = np.divmod(valid_idx, W)
    all_cx, all_cy     = rasterio_xy(transform, all_rows, all_cols)
    _tr_wgs84          = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
    all_lons, all_lats = _tr_wgs84.transform(all_cx, all_cy)

    # Spatial-join all valid pixels → state codes (vectorised; results cached)
    _unit_arr = np.full(len(valid_idx), "", dtype=object)
    try:
        from map_lifers import get_us_boundary
        _states_gdf = get_us_boundary().to_crs("EPSG:4326")[["iso_3166_2", "geometry"]].copy()
        _pts_all = gpd.GeoDataFrame(
            {"_i": np.arange(len(valid_idx))},
            geometry=gpd.points_from_xy(all_lons, all_lats),
            crs="EPSG:4326",
        )
        _joined_all = gpd.sjoin(_pts_all, _states_gdf, how="left", predicate="within")
        _joined_all = _joined_all.drop_duplicates(subset=["_i"])
        _valid_join = _joined_all["iso_3166_2"].notna()
        _unit_arr[
            _joined_all.loc[_valid_join, "_i"].to_numpy(dtype=int)
        ] = _joined_all.loc[_valid_join, "iso_3166_2"].to_numpy(dtype=str)
    except Exception:
        pass  # graceful fallback: all pixels in "" unit → global distance filter

    # Per-unit top-k: keeps depth*20 candidates per state, guaranteeing that
    # every state (even with low peak scores) contributes to the candidate pool.
    k_per_unit   = max(depth, 1) * 20
    _units_uniq  = np.unique(_unit_arr)
    _selected: list[int] = []   # indices into valid_idx / all_rows / _unit_arr
    for _u in _units_uniq:
        _u_mask   = _unit_arr == _u
        _u_vi_idx = np.where(_u_mask)[0]   # positions in valid_idx
        _u_scores = flat[valid_idx[_u_vi_idx]]
        if len(_u_vi_idx) <= k_per_unit:
            _selected.extend(_u_vi_idx.tolist())
        else:
            _top_k = np.argpartition(_u_scores, -k_per_unit)[-k_per_unit:]
            _selected.extend(_u_vi_idx[_top_k].tolist())

    _selected.sort(key=lambda i: float(flat[valid_idx[i]]), reverse=True)
    _sel = np.array(_selected, dtype=np.intp)

    rows_arr = all_rows[_sel]
    cols_arr = all_cols[_sel]
    cxs      = all_cx[_sel]
    cys      = all_cy[_sel]
    lons     = all_lons[_sel]
    lats     = all_lats[_sel]

    _cand_unit: list[str] = [str(_unit_arr[i]) for i in _sel]
    candidates: list[tuple] = [
        (float(flat[valid_idx[_sel[i]]]), int(rows_arr[i]), int(cols_arr[i]),
         float(lats[i]), float(lons[i]), float(cxs[i]), float(cys[i]))
        for i in range(len(_sel))
    ]

    # ---- distance + unit dedup ----------------------------------------

    # Step 2: apply min_distance within each admin unit separately so that a
    # cluster of spots in one state cannot block candidates in other states.
    if min_km > 0:
        from collections import defaultdict as _dd
        by_unit: dict[str, list[tuple[int, tuple]]] = _dd(list)
        for i, cand in enumerate(candidates):
            by_unit[_cand_unit[i] or "__none__"].append((i, cand))

        kept_pairs: list[tuple[int, tuple]] = []   # (orig_idx, cand)
        for unit_pairs in by_unit.values():
            unit_cands = [c for _, c in unit_pairs]
            unit_kept  = {(c[1], c[2]) for c in filter_min_distance(unit_cands, min_km, crs_src)}
            for orig_i, c in unit_pairs:
                if (c[1], c[2]) in unit_kept:
                    kept_pairs.append((orig_i, c))
                    unit_kept.discard((c[1], c[2]))   # first match only

        kept_pairs.sort(key=lambda x: x[1][0], reverse=True)  # re-sort by score
        kept_all = [c for _, c in kept_pairs]
        _admin_fallback: dict[int, str] = {
            new_i: _cand_unit[orig_i]
            for new_i, (orig_i, _) in enumerate(kept_pairs)
            if _cand_unit[orig_i]
        }
    else:
        kept_all = candidates
        _admin_fallback = {i: u for i, u in enumerate(_cand_unit) if u}

    unit_counts: dict[str, int] = {}
    rank = 0
    spots: list[dict] = []

    for ci, (sc, r, c, lat, lon, cx, cy) in enumerate(kept_all):
        if rank >= n:
            break
        hs = best_hotspot_in_grid(cx, cy, hs_db) if hs_db is not None else None
        if hs:
            loc_name = hs["locName"]
            loc_id   = hs["locId"]
            hs_spp   = int(hs["spp"]) if hs["spp"] == hs["spp"] else None
            county   = hs["county"]
            state    = hs["state"]
        else:
            loc_name  = reverse_geocode(lat, lon)
            loc_id    = ""
            hs_spp    = None
            county    = ""
            state     = _admin_fallback.get(ci, "")
            time.sleep(1.1)

        # Strict admin-unit dedup key: spatial-join result first, hotspot field as backup
        if use_county_unit:
            dedup_key = county or _admin_fallback.get(ci, "(unknown)")
        else:
            dedup_key = (state or _admin_fallback.get(ci, "")) or "(unknown)"

        if unit_counts.get(dedup_key, 0) >= depth:
            continue
        unit_counts[dedup_key] = unit_counts.get(dedup_key, 0) + 1
        rank += 1

        weekly = [float(x) for x in stack[:, r, c]]
        ann_r  = float(annual_richness[r, c])
        spots.append({
            "rank":             rank,
            "score":            sc,
            "lat":              lat,
            "lon":              lon,
            "row":              r,
            "col":              c,
            "loc_id":           loc_id,
            "loc_name":         loc_name,
            "hs_spp":           hs_spp,
            "county":           county,
            "state":            state,
            "weekly":           weekly,
            "annual_richness":  ann_r,
        })

    del stack   # free before species extraction

    # ---- species attribution (in-memory, no tif re-reads) -------------
    for spot in spots:
        spot["species_codes"] = _species_from_packed(
            sp_packed, n_sp, available, spot["row"], spot["col"],
        )

    # ---- persist results ----------------------------------------------
    SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(rcache_path, "wb") as fh:
        pickle.dump(
            {"species": sorted(needed), "spots": spots, "week_dates": week_dates},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"  Results: cached → {rcache_path.name}")
    return spots, week_dates


# ---------------------------------------------------------------------------
# Greedy marginal species reranking
# ---------------------------------------------------------------------------
def _greedy_marginal_rerank(spots: list[dict]) -> list[dict]:
    """Re-order spots to maximise cumulative new-species coverage.

    Greedy set cover heuristic:
      - At each step pick the spot with the most species NOT already seen
        in any previously chosen spot.
      - Ties broken by raw score (descending).
    Each spot gets a `marginal_spp` key: number of new species it contributes.
    """
    remaining = list(spots)   # preserve original order (already score-sorted)
    ordered: list[dict] = []
    seen: set[str] = set()
    while remaining:
        best_i     = 0
        best_marg  = -1
        best_score = -1.0
        for i, s in enumerate(remaining):
            codes   = set(s.get("species_codes") or [])
            marg    = len(codes - seen)
            sc      = s["score"]
            if marg > best_marg or (marg == best_marg and sc > best_score):
                best_i, best_marg, best_score = i, marg, sc
        best = remaining.pop(best_i)
        codes = set(best.get("species_codes") or [])
        best["marginal_spp"] = len(codes - seen)
        seen |= codes
        ordered.append(best)
    for i, s in enumerate(ordered):
        s["rank"] = i + 1
    return ordered


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
    parser.add_argument("--n",          type=int, default=10,
                        help="Total spots to return (= num states × depth for US).")
    parser.add_argument(
        "--depth", type=int, default=1, metavar="N",
        help="Max spots per admin unit (state for country regions, county for state regions). "
             "depth=1 (default) = 1 per state, depth=5 = top 5 per state.",
    )
    parser.add_argument(
        "--metric", default="peak", choices=["peak", "median", "mean", "richness"],
        help=("Scoring metric: peak=best single week, median/mean=central tendency, "
              "richness=annual unique species (migration breadth)."),
    )
    parser.add_argument(
        "--min-km", type=float, default=50.0,
        help="Minimum km between returned spots (0 = disabled).",
    )
    parser.add_argument(
        "--marginal", action="store_true",
        help="Re-rank spots by greedy marginal new-species coverage (maximises total unique lifers).",
    )
    parser.add_argument("--offline",   action="store_true")
    parser.add_argument("--ebird-csv", default="MyEBirdData.csv", metavar="PATH")
    parser.add_argument(
        "--list-species", action="store_true",
        help="Print the potential lifer list for each top spot.",
    )
    parser.add_argument(
        "--build-cache", action="store_true",
        help="Pre-build precomputed cache for the given regions/resolution and exit.",
    )
    args = parser.parse_args()

    cfg = load_config()
    _hs_cache: dict[str, dict | None] = {}
    # Taxonomy for code → common name (loaded once, used for --list-species)
    _taxonomy: dict[str, str] | None = None
    if args.list_species and not args.offline:
        try:
            raw_tax = get_taxonomy(cfg["ebird_api_key"])
            # get_taxonomy returns {comName: speciesCode}; invert to {code: comName}
            _taxonomy = {v: k for k, v in raw_tax.items()}
        except Exception as exc:
            print(f"  [warn] Could not load taxonomy: {exc}")

    for region in args.regions:
        print(f"\n{'='*60}")
        print(f"  Region: {region}  |  Resolution: {args.resolution}  |  Metric: {args.metric}")
        print(f"{'='*60}")

        boundary = get_boundary(region)

        # ---- hotspot grid (once per country prefix) ---------------------
        db_prefix = region.split("-")[0]
        if db_prefix not in _hs_cache:
            db_path = HOTSPOT_DBS.get(db_prefix)
            _hs_cache[db_prefix] = load_hotspot_grid(db_path) if (db_path and db_path.exists()) else None
        hs_db = _hs_cache[db_prefix]

        # ---- species list ----------------------------------------------
        needed = load_needed_species(region, args.resolution, cfg,
                                     offline=args.offline, ebird_csv=args.ebird_csv)

        if args.build_cache:
            build_precomputed(region, args.resolution, needed, boundary)
            continue

        # ---- compute ---------------------------------------------------
        spots, week_dates = compute_top_spots(
            region, args.resolution, args.metric,
            args.n, args.depth, args.min_km,
            needed, boundary, hs_db,
        )
        if not spots:
            print("  No valid pixels found — make sure tifs are cached.")
            continue

        # ---- optional greedy rerank ------------------------------------
        if args.marginal:
            spots = _greedy_marginal_rerank(spots)

        # ---- print table -----------------------------------------------
        use_county_unit = "-" in region
        unit_label = "county" if use_county_unit else "state"
        unit_str   = f"top {args.depth} per {unit_label}" if args.depth > 1 else f"1 per {unit_label}"
        rank_hdr   = "greedy-marginal" if args.marginal else "score"
        print(f"\n  Top {args.n} spots ({unit_str}, ranked by {rank_hdr}, min {args.min_km:.0f} km apart):\n")

        mrg_col = args.marginal  # show MrgSpp column only when --marginal
        hdr = (f"  {'#':>4}  {'Score':>6}  {'AnnRch':>6}  {'HsSpp':>6}  {'Lifers':>6}"
               + (f"  {'MrgSpp':>6}" if mrg_col else "")
               + f"  {'Lat':>8}  {'Lon':>9}  {'State':<8}  {'County':<22}  Location")
        sep = (f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}"
               + (f"  {'-'*6}" if mrg_col else "")
               + f"  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*22}  {'-'*35}")
        print(hdr)
        print(sep)

        for spot in spots:
            hs_spp_s  = str(spot["hs_spp"]) if spot["hs_spp"] is not None else "-"
            ann_s     = f"{spot['annual_richness']:.0f}" if spot.get("annual_richness") is not None else "-"
            lifer_s   = str(len(spot.get("species_codes") or []))
            mrg_s     = str(spot.get("marginal_spp", "-")) if mrg_col else ""
            state_s   = spot["state"].replace("US-", "").replace("NL-", "") if spot["state"] else "-"
            county_s  = spot["county"].split("-")[-1] if spot["county"] else "-"
            url = f"https://ebird.org/hotspot/{spot['loc_id']}" if spot["loc_id"] else ""
            loc_str = (f"\033]8;;{url}\033\\{spot['loc_name']}\033]8;;\033\\"
                       if url else spot["loc_name"])
            row = (f"  {spot['rank']:>4}  {spot['score']:>6.1f}  {ann_s:>6}  {hs_spp_s:>6}  {lifer_s:>6}"
                   + (f"  {mrg_s:>6}" if mrg_col else "")
                   + f"  {spot['lat']:>8.4f}  {spot['lon']:>9.4f}  {state_s:<8}  {county_s:<22}  {loc_str}")
            print(row)
            if args.list_species and spot.get("species_codes"):
                codes = spot["species_codes"]
                names = [_taxonomy.get(c, c) if _taxonomy else c for c in codes]
                names.sort()
                for name in names:
                    print(f"       · {name}")

        print()


if __name__ == "__main__":
    main()
