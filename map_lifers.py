#!/usr/bin/env python3
"""
map_lifers.py — Generate weekly lifer maps and animated GIFs from cached
eBird Status & Trends rasters.  Output goes to results_py/

Default (all regions, all resolutions, full animation):
    .venv/bin/python map_lifers.py

Single region / resolution:
    .venv/bin/python map_lifers.py --regions NL --resolutions 27km

Single-week preview:
    .venv/bin/python map_lifers.py --regions NL --resolutions 27km --week 20

Options:
    --regions NL US         eBird region codes (default: NL US)
    --resolutions 27km 9km  one or more of 3km 9km 27km (default: all three)
    --week N                single map for week N; skips animation
    --no-animate            render week 20 only, no GIF
    --ram-gb 4.0            RAM budget per batch in GB (default 4)
    --fps 5                 GIF frames per second (default 5)
    --offline               skip API calls, use locally cached tifs
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, Window
from rasterio.transform import array_bounds
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — prevents Tkinter threading errors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from ebird.api.requests.species import get_species_list as _ebird_species_list
from ebird.api.requests.taxonomy import get_taxonomy as _ebird_taxonomy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EBIRDST_VERSION   = "2023"
CACHE_DIR         = Path("data/ebirdst")
MASK_CACHE_DIR    = CACHE_DIR / "masks"
SP_CACHE_DIR      = Path("data/sp_cache")
NE_DIR            = Path("data/naturalearth")
OUT_DIR           = Path("results_py")
EXCLUDED_CODES    = {"laugul", "rocpig", "compea", "yebsap-example"}
OCCURRENCE_THRESH = 0.01

# Leave at least 8 cores free so the system stays responsive.
_cpus     = os.cpu_count() or 4
N_WORKERS = max(4, min(_cpus * 3 // 4, _cpus - 8))

# Output frame dimensions — hi-res PNG frames are scaled to exactly _FRAME_WIDTH_PX
# pixels wide (height is aspect-driven); lo-res GIF is 25% of that.
# 1920 px → clean 2K; actual height ~1653 px (aspect-driven by CONUS map bounds).
_FRAME_WIDTH_PX  = 1920
_LORES_SCALE     = 0.25   # 480 px wide

# Dark theme — matches R: bg = viridisLite::turbo(1)
BG_COLOR = "#30123b"
FG_DARK  = "#e5e5e5"   # grey90
FG_LIGHT = "white"

# ---------------------------------------------------------------------------
# Colour scale — turbo colormap, linear norm.  Only vmax differs.
# ---------------------------------------------------------------------------
CMAP_NAME = "turbo"
SCALES: dict[str, int | None] = {
    "auto":    None,  # vmax = data max (original default)
    "compact": 35,    # saturates early — punchy colours
    "wide":    50,    # more headroom — shows higher-count hotspots
}
DEFAULT_SCALE = "auto"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: Path = Path(".env")) -> dict:
    if not path.exists():
        sys.exit(".env not found — copy .env.example and fill in your keys.")
    keys = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        # Support both shell-style KEY=value and R-style KEY <- "value"
        if "<-" in line:
            name, _, rhs = line.partition("<-")
        elif "=" in line:
            name, _, rhs = line.partition("=")
        else:
            continue
        name = name.strip().lower()
        rhs = rhs.strip().strip('"').strip("'")
        if name in ("ebirdst_key", "ebird_api_key", "user"):
            keys[name] = rhs
    return keys


def load_ebirdst_runs(path: Path = Path("ebirdst_runs.csv")) -> dict:
    if not path.exists():
        sys.exit("ebirdst_runs.csv not found — see README for download link.")
    with open(path) as f:
        return {row["species_code"]: row["common_name"]
                for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# eBird API helpers
# ---------------------------------------------------------------------------
def _tif_url(species_code: str, resolution: str, ebirdst_key: str) -> str:
    import urllib.parse
    obj_key = (
        f"{EBIRDST_VERSION}/{species_code}/weekly/"
        f"{species_code}_occurrence_median_{resolution}_{EBIRDST_VERSION}.tif"
    )
    return (
        "https://st-download.ebird.org/v1/fetch"
        f"?objKey={urllib.parse.quote(obj_key, safe='/')}"
        f"&key={ebirdst_key}"
    )


def download_missing(
    needed: set[str],
    resolution: str,
    ebirdst_key: str,
    workers: int = 4,
) -> None:
    """Download any tifs in `needed` that are not yet cached.  Prompts before starting."""
    missing = [sp for sp in sorted(needed) if not tif_path(sp, resolution).exists()]
    if not missing:
        return

    print(f"  {len(missing)} tifs not yet cached at {resolution}.")
    try:
        answer = input(f"  Download now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    if answer != "y":
        print("  Skipping download — some species may be missing from the map.")
        return

    def _fetch_one(sp: str) -> tuple[str, str]:
        dest = tif_path(sp, resolution)
        if dest.exists():   # never overwrite an already-cached tif
            return "ok", sp
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _tif_url(sp, resolution, ebirdst_key)
        try:
            with requests.get(url, timeout=600, stream=True) as r:
                if r.status_code == 404:
                    return "missing", sp
                r.raise_for_status()
                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        f.write(chunk)
                tmp.rename(dest)
            return "ok", sp
        except Exception as exc:
            return f"error:{exc}", sp

    ok = err = skip = 0
    total = len(missing)
    print(f"  Downloading {total} files ({workers} threads) …")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, sp in pool.map(_fetch_one, missing):
            done = ok + err + skip
            if status == "ok":
                ok += 1
            elif status == "missing":
                skip += 1
            else:
                err += 1
                print(f"\n    [warn] {sp}: {status}")
            done += 1
            print(f"\r  [{done}/{total}]  ok={ok}  skip={skip}  err={err}  ", end="", flush=True)
    print()  # newline after progress
    print(f"  Download complete — {ok} downloaded, {skip} not in S&T, {err} errors.")


def get_taxonomy(api_key: str) -> dict:
    return {r["comName"]: r["speciesCode"] for r in _ebird_taxonomy(api_key)}


def ebird_regional_species(region: str, api_key: str) -> set:
    return set(_ebird_species_list(api_key, region))


def user_seen_codes(csv_path: Path, region: str, name_to_code: dict) -> set:
    country  = region.split("-")[0]
    is_state = "-" in region
    seen = set()
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — treating everything as a need.")
        return seen
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sp = row.get("State/Province", "")
            if is_state:
                if sp != region:
                    continue
            else:
                if not (sp == country or sp.startswith(country + "-")):
                    continue
            name = row.get("Common Name", "").strip()
            if name in name_to_code:
                seen.add(name_to_code[name])
    return seen


# ---------------------------------------------------------------------------
# NaturalEarth boundaries
# ---------------------------------------------------------------------------
def _ne_download(url: str, dest_dir: Path) -> Path:
    stem = Path(url).stem
    shp  = dest_dir / f"{stem}.shp"
    if shp.exists():
        return shp
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading NaturalEarth: {stem} …")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    zip_path = dest_dir / Path(url).name
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    zip_path.unlink()
    if not shp.exists():
        shp = next(dest_dir.rglob("*.shp"), None)
        if shp is None:
            sys.exit(f"No .shp file found in {dest_dir} after extracting {Path(url).name}")
    return shp


def get_nl_boundary() -> gpd.GeoDataFrame:
    shp = _ne_download(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
        NE_DIR / "ne_10m_admin_0_countries",
    )
    world = gpd.read_file(shp)
    col = "ISO_A3" if "ISO_A3" in world.columns else "sov_a3"
    nl = world[world[col] == "NLD"].to_crs("EPSG:4326").copy()
    # NaturalEarth "NLD" is a multipolygon that includes Caribbean overseas territories
    # (Aruba, Curaçao, Bonaire etc.).  Explode and keep only European parts.
    nl_exp = nl.explode(index_parts=True)
    nl_proj = nl_exp.to_crs("EPSG:4087")   # World Equidistant Cylindrical — safe for centroid
    nl_exp = nl_exp[
        (nl_proj.geometry.centroid.x > -1_000_000) &   # east of ~9°W
        (nl_proj.geometry.centroid.y >  4_500_000)     # north of ~40°N
    ]
    if nl_exp.empty:
        sys.exit("Could not isolate European part of Netherlands from NaturalEarth data.")
    return nl_exp.dissolve().to_crs("EPSG:4326")


def get_us_boundary(state_code: str | None = None) -> gpd.GeoDataFrame:
    """Return CONUS boundary, or a single state when state_code is given (e.g. 'US-NY')."""
    shp = _ne_download(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
        NE_DIR / "ne_10m_admin_1_states_provinces",
    )
    states = gpd.read_file(shp)
    if state_code:
        # Single state requested — no AK/HI exclusion needed
        result = states[states["iso_3166_2"] == state_code].copy()
        if result.empty:
            sys.exit(f"No NaturalEarth polygon found for state code '{state_code}'.")
        return result.to_crs("EPSG:4326")
    # Full CONUS: drop AK and HI
    return states[
        (states["adm0_a3"] == "USA") &
        (~states["iso_3166_2"].isin(["US-HI", "US-AK"]))
    ].copy().to_crs("EPSG:4326")


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------
def tif_path(code: str, resolution: str) -> Path:
    return (
        CACHE_DIR / EBIRDST_VERSION / code / "weekly"
        / f"{code}_occurrence_median_{resolution}_{EBIRDST_VERSION}.tif"
    )


def _setup_window(ref_tif: Path, boundary_wgs84: gpd.GeoDataFrame,
                  cache_key: str | None = None):
    """Compute shared windowed extent and outside-boundary mask from one tif.

    If *cache_key* is given the result is persisted to
    MASK_CACHE_DIR/<cache_key>.npz so subsequent runs skip the
    geometry_mask() rasterisation (which can take a few seconds for
    complex polygons like the US boundary).
    """
    from affine import Affine

    if cache_key:
        cache_path = MASK_CACHE_DIR / f"{cache_key}.npz"
        if cache_path.exists():
            d = np.load(cache_path, allow_pickle=False)
            ref_crs = rasterio.crs.CRS.from_wkt(d["ref_crs_wkt"].item().decode())
            win = Window(
                col_off=int(d["win_col_off"]), row_off=int(d["win_row_off"]),
                width=int(d["win_width"]),     height=int(d["win_height"]),
            )
            ref_shape = (int(d["ref_shape"][0]), int(d["ref_shape"][1]))
            a, b, c, dd, e, f = d["transform_params"]
            ref_transform = Affine(float(a), float(b), float(c),
                                   float(dd), float(e), float(f))
            outside_mask = d["outside_mask"].astype(bool)
            week_dates   = [s.decode() if isinstance(s, bytes) else s
                            for s in d["week_dates"]]
            n_bands = int(d["n_bands"])
            return ref_crs, win, ref_shape, ref_transform, outside_mask, week_dates, n_bands

    with rasterio.open(ref_tif) as src:
        ref_crs      = src.crs
        boundary_src = boundary_wgs84.to_crs(ref_crs)
        xmin, ymin, xmax, ymax = boundary_src.total_bounds
        raw_win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
        raw_win = raw_win.round_offsets().round_lengths()
        win     = raw_win.intersection(Window(0, 0, src.width, src.height))
        ref_shape     = (int(win.height), int(win.width))
        ref_transform = src.window_transform(win)
        outside_mask  = geometry_mask(
            list(boundary_src.geometry), transform=ref_transform,
            invert=False, out_shape=ref_shape,
        )
        week_dates = list(src.descriptions)
        n_bands    = src.count

    if cache_key:
        MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            MASK_CACHE_DIR / f"{cache_key}.npz",
            ref_crs_wkt=np.bytes_(ref_crs.to_wkt()),
            win_col_off=win.col_off, win_row_off=win.row_off,
            win_width=win.width,     win_height=win.height,
            ref_shape=np.array(ref_shape, dtype=np.int32),
            transform_params=np.array(
                [ref_transform.a, ref_transform.b, ref_transform.c,
                 ref_transform.d, ref_transform.e, ref_transform.f],
                dtype=np.float64,
            ),
            outside_mask=outside_mask,
            week_dates=np.array(week_dates),
            n_bands=np.int32(n_bands),
        )

    return ref_crs, win, ref_shape, ref_transform, outside_mask, week_dates, n_bands


def _sp_cache_meta(region: str, resolution: str) -> dict | None:
    """Load sp_cache _meta.npz for (region, resolution), or None if missing/stale."""
    from affine import Affine
    mp = SP_CACHE_DIR / region / resolution / EBIRDST_VERSION / "_meta.npz"
    if not mp.exists():
        return None
    try:
        d = np.load(mp, allow_pickle=False)
        if d["version"].item().decode() != EBIRDST_VERSION:
            return None
        if float(d["threshold"]) != OCCURRENCE_THRESH:
            print(f"  [sp_cache] threshold mismatch "
                  f"(cache={float(d['threshold'])} vs current={OCCURRENCE_THRESH}) "
                  f"— falling back to tif reads")
            return None
        ref_crs = rasterio.crs.CRS.from_wkt(d["ref_crs_wkt"].item().decode())
        win = Window(
            col_off=int(d["win_col_off"]), row_off=int(d["win_row_off"]),
            width=int(d["win_width"]),     height=int(d["win_height"]),
        )
        ref_shape = (int(d["ref_shape"][0]), int(d["ref_shape"][1]))
        a, b, c, dd, e, f = d["transform_params"]
        ref_transform = Affine(float(a), float(b), float(c),
                               float(dd), float(e), float(f))
        outside_mask = d["outside_mask"].astype(bool)
        inside_idx   = d["inside_idx"].astype(np.int32)
        week_dates   = [s.decode() if isinstance(s, bytes) else s
                        for s in d["week_dates"]]
        n_bands = int(d["n_bands"])
        return dict(
            ref_crs=ref_crs, win=win, ref_shape=ref_shape,
            ref_transform=ref_transform, outside_mask=outside_mask,
            inside_idx=inside_idx, week_dates=week_dates, n_bands=n_bands,
            cache_dir=mp.parent,
        )
    except Exception as exc:
        print(f"  [sp_cache] could not load meta: {exc} — falling back to tif reads")
        return None


def accumulate_all_weeks(
    needed: set,
    boundary_wgs84: gpd.GeoDataFrame,
    resolution: str = "3km",
    ram_budget_gb: float = 4.0,
    workers: int | None = None,
    batch_size: int | None = None,
    track_sp_presence: bool = False,
    region: str = "",
) -> tuple:
    """
    Accumulate per-week species counts into a (52, H, W) int16 stack.

    Fast path  — sp_cache present (preprocess_region.py Step 2 has been run):
        Reads compact packbits .npy files (~32× smaller than float32 tifs).
        No threshold check needed (already encoded in the cache).

    Slow path  — no sp_cache:
        Reads float32 tifs, thresholds, and accumulates in one pass.
        Also writes geometry_mask to MASK_CACHE_DIR for reuse.

    Returns (result, transform, crs, week_dates, richness, win, available, sp_packed).
    """
    available = [c for c in sorted(needed) if tif_path(c, resolution).exists()]
    n_no_tif  = len(needed) - len(available)
    print(f"  {len(available):3d} / {len(needed):3d} species have {resolution} tifs"
          + (f"  ({n_no_tif} not modelled at {resolution} by S&T — skipped)" if n_no_tif else ""))
    if not available:
        raise RuntimeError(f"No {resolution} tifs found.")

    workers = max(1, min(workers or N_WORKERS, len(available)))

    # ── Try sp_cache fast path ────────────────────────────────────────────
    meta = _sp_cache_meta(region, resolution) if region else None

    if meta is not None:
        cache_dir   = meta["cache_dir"]
        ref_crs     = meta["ref_crs"]
        win         = meta["win"]
        ref_shape   = meta["ref_shape"]
        ref_transform = meta["ref_transform"]
        outside_mask  = meta["outside_mask"]
        inside_idx    = meta["inside_idx"]
        week_dates    = meta["week_dates"]
        n_bands       = meta["n_bands"]
        n_inside      = len(inside_idx)

        # Only species whose .npy file exists are in the cache (below-threshold
        # species were excluded by preprocess_region.py).
        cached_codes = [c for c in available if (cache_dir / f"{c}.npy").exists()]
        n_uncached   = len(available) - len(cached_codes)
        print(f"  [sp_cache] {len(cached_codes)} species in cache"
              + (f"  ({n_uncached} missing — will read tif)" if n_uncached else ""))

        accumulator  = np.zeros((n_bands, *ref_shape), dtype=np.int16)
        richness_acc = np.zeros(ref_shape, dtype=np.int16)
        n_sp_total   = len(cached_codes)
        sp_presence  = (np.zeros((n_sp_total, *ref_shape), dtype=np.uint8)
                        if track_sp_presence else None)
        done = 0
        available_acc: list[str] = []
        n_bits = n_bands * n_inside

        def _load_cached(code: str) -> np.ndarray | None:
            # Returns compact (n_bands, n_inside) int8 — no full-grid allocation.
            try:
                packed = np.load(cache_dir / f"{code}.npy")
                flat   = np.unpackbits(packed)[:n_bits].astype(np.int8)
                return flat.reshape(n_bands, n_inside)          # (52, n_inside)
            except Exception as exc:
                print(f"    [skip] {code}: {exc}")
                return None

        # Flat views for scatter-add — avoids re-allocating per-species full arrays.
        acc_flat  = accumulator.reshape(n_bands, -1)            # (52, H*W) view
        rich_flat = richness_acc.ravel()                        # (H*W,) view

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_load_cached, c): c for c in cached_codes}
            for fut in as_completed(futures):
                arr2d = fut.result()                            # (n_bands, n_inside)
                code  = futures[fut]
                if arr2d is not None:
                    sp_idx = len(available_acc)
                    available_acc.append(code)
                    plane = arr2d.max(axis=0)                   # (n_inside,) int8
                    acc_flat[:, inside_idx]  += arr2d
                    rich_flat[inside_idx]    += plane.astype(np.int16)
                    if sp_presence is not None:
                        sp_presence[sp_idx].ravel()[inside_idx] = plane.view(np.uint8)
                done += 1
                if done % 100 == 0 or done == len(cached_codes):
                    print(f"    {done}/{len(cached_codes)} species …")

        print(f"  Pre-filter: kept {len(available_acc)} / {len(available)} species "
              f"(from sp_cache)")
        available = available_acc
        if sp_presence is not None:
            sp_presence = sp_presence[:len(available)]

    else:
        # ── Slow path: read float32 tifs ─────────────────────────────────────
        cache_key = f"{region}_{resolution}_{EBIRDST_VERSION}" if region else None
        ref_crs, win, ref_shape, ref_transform, outside_mask, week_dates, n_bands = \
            _setup_window(tif_path(available[0], resolution), boundary_wgs84,
                          cache_key=cache_key)

        bytes_per_sp = n_bands * ref_shape[0] * ref_shape[1]
        batch_size = batch_size or max(
            4, min(workers * 2, int(ram_budget_gb * 1e9 / max(bytes_per_sp, 1)))
        )
        print(f"  {workers} threads, batch {batch_size} sp "
              f"({bytes_per_sp * batch_size / 1e9:.2f} GB/batch)")
        sp_cache_path = SP_CACHE_DIR / region / resolution / EBIRDST_VERSION if region else None
        if sp_cache_path is None or not sp_cache_path.exists():
            print(f"  [tip] run preprocess_region.py to build sp_cache and speed up future runs")

        accumulator  = np.zeros((n_bands, *ref_shape), dtype=np.int16)
        richness_acc = np.zeros(ref_shape, dtype=np.int16)
        n_sp_total   = len(available)
        sp_presence  = (np.zeros((n_sp_total, *ref_shape), dtype=np.uint8)
                        if track_sp_presence else None)
        done = 0
        n_kept = 0
        n_dropped = 0
        available_acc: list[str] = []

        def _read_and_filter(code: str) -> np.ndarray | None:
            try:
                with rasterio.open(tif_path(code, resolution)) as src:
                    data = src.read(window=win).astype(np.float32)
                data[:, outside_mask] = np.nan
                if not (np.nanmax(data) > OCCURRENCE_THRESH):
                    return None
                return np.where(
                    ~np.isfinite(data),
                    np.int8(0),
                    (data > OCCURRENCE_THRESH).astype(np.int8),
                )
            except Exception as exc:
                print(f"    [skip] {code}: {exc}")
                return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batch_start in range(0, len(available), batch_size):
                batch   = available[batch_start : batch_start + batch_size]
                futures = {pool.submit(_read_and_filter, c): c for c in batch}
                for fut in as_completed(futures):
                    arr  = fut.result()
                    code = futures[fut]
                    if arr is not None:
                        n_kept += 1
                        sp_idx = len(available_acc)
                        available_acc.append(code)
                        plane = arr.max(axis=0)
                        accumulator  += arr
                        richness_acc += plane.astype(np.int16)
                        if sp_presence is not None:
                            sp_presence[sp_idx] = plane.view(np.uint8)
                    else:
                        n_dropped += 1
                    done += 1
                if done % 50 == 0 or done == len(available):
                    print(f"    {done}/{len(available)} species …")

        print(f"  Pre-filter: kept {n_kept} / {len(available)} "
              f"species (dropped {n_dropped} below {OCCURRENCE_THRESH*100:.0f}% threshold)")
        if not available_acc:
            raise RuntimeError("No species exceed the occurrence threshold in this region.")
        available = available_acc
        if sp_presence is not None:
            sp_presence = sp_presence[:len(available)]

    # ── Finalise (common to both paths) ──────────────────────────────────────
    result = accumulator.astype(np.float32)
    result[:, outside_mask] = np.nan
    richness_acc = richness_acc.astype(np.float32)
    richness_acc[outside_mask] = np.nan
    if sp_presence is not None:
        sp_packed = np.packbits(sp_presence, axis=0)
        del sp_presence
    else:
        sp_packed = None
    return result, ref_transform, ref_crs, week_dates, richness_acc, win, available, sp_packed


def reproject_stack(
    data3d: np.ndarray,
    src_transform,
    src_crs,
    dst_crs_str: str,
    workers: int | None = None,
) -> tuple:
    """Reproject all n layers of (n, h, w) using parallel threads."""
    n, h, w = data3d.shape
    src_bounds = array_bounds(h, w, src_transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs_str, w, h, *src_bounds
    )
    dst = np.full((n, dst_h, dst_w), np.nan, dtype=np.float32)

    def _proj(i):
        out = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=rasterio.errors.NotGeoreferencedWarning
            )
            reproject(
                source=data3d[i], destination=out,
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs_str,
                resampling=Resampling.nearest,
                src_nodata=np.nan, dst_nodata=np.nan,
            )
        return i, out

    max_workers = max(1, min(workers or N_WORKERS, n))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, layer in pool.map(_proj, range(n)):
            dst[i] = layer

    return dst, dst_transform


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _raster_extent(transform, h: int, w: int) -> tuple:
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + transform.a * w
    ymin = ymax + transform.e * h
    return xmin, xmax, ymin, ymax


# ---------------------------------------------------------------------------
# State label overlay (US country-level maps only)
# ---------------------------------------------------------------------------
HOTSPOT_DB_US = Path("data/ebird_hotspots_us.gpkg")

# States too small to label in-situ on the east coast — use leader lines
_EAST_COAST_SMALL = {"ME", "NH", "VT", "MA", "RI", "CT", "NJ", "DE", "MD"}


def _load_us_hotspot_gdf_projected(target_crs: str):
    """Load the US eBird hotspot GPKG from disk (projected to target_crs).
    Returns a GeoDataFrame or None if the file is absent."""
    if not HOTSPOT_DB_US.exists():
        return None
    return gpd.read_file(HOTSPOT_DB_US).to_crs(target_crs)


def _build_state_overlay(
    score_2d: np.ndarray,
    tf_local,
    states_proj: gpd.GeoDataFrame,
    hs_gdf_proj,        # GeoDataFrame in same CRS, or None
    bx_min: float, bx_max: float, by_min: float, by_max: float,
) -> list[dict]:
    """Compute per-state label data for the map overlay.

    For each CONUS state:
      - Finds the highest-scoring pixel inside the state (annual peak)
      - Looks up the nearest eBird hotspot within 50 km of that pixel
      - Determines label position (interior for large states,
        leader line for small east coast states)

    Returns a list of dicts with keys:
        abbr, loc_name, score, spot_x, spot_y, text_x, text_y, leader
    """
    from rasterio.transform import xy as rxy

    H, W = score_2d.shape
    map_width = bx_max - bx_min

    results: list[dict] = []

    for _, state_row in states_proj.iterrows():
        abbr = str(state_row.get("iso_3166_2", "") or "").replace("US-", "").strip()
        if not abbr:
            continue
        geom = state_row.geometry
        if geom is None or geom.is_empty:
            continue

        # Rasterise state mask (True = OUTSIDE state)
        try:
            outside = geometry_mask(
                [geom], out_shape=(H, W), transform=tf_local, invert=False,
            )
        except Exception:
            continue

        local = score_2d.copy().astype(np.float32)
        local[outside] = np.nan

        valid_idx = np.where(np.isfinite(local.ravel()) & (local.ravel() > 0))[0]
        if valid_idx.size == 0:
            continue

        flat = local.ravel()
        best_flat = valid_idx[int(np.argmax(flat[valid_idx]))]
        best_r, best_c = np.divmod(best_flat, W)
        best_score = float(flat[best_flat])

        # Map-CRS coordinates of best pixel
        spot_x, spot_y = rxy(tf_local, int(best_r), int(best_c))
        spot_x, spot_y = float(spot_x), float(spot_y)

        # Hotspot name lookup
        loc_name = abbr
        if hs_gdf_proj is not None:
            from shapely.geometry import Point
            pt = Point(spot_x, spot_y)
            dists = hs_gdf_proj.geometry.distance(pt)
            near_idx = int(dists.idxmin())
            if dists.iloc[near_idx] < 50_000:
                loc_name = str(hs_gdf_proj.at[near_idx, "locName"])

        # Determine label placement
        sb = geom.bounds  # (minx, miny, maxx, maxy)
        is_small_ec = abbr in _EAST_COAST_SMALL

        if is_small_ec:
            # Text to the right of the map boundary; arrow points to spot
            text_x = bx_max + 0.055 * map_width
            text_y = float(geom.centroid.y)
            leader = True
        else:
            # Upper-left quarter of state bounding box, check interior
            from shapely.geometry import Point as _Pt
            tx = sb[0] + 0.15 * (sb[2] - sb[0])
            ty = sb[3] - 0.15 * (sb[3] - sb[1])
            if not geom.contains(_Pt(tx, ty)):
                rp = geom.representative_point()
                tx, ty = rp.x, rp.y
            text_x, text_y = float(tx), float(ty)
            leader = False

        # Shorten very long names
        display_name = loc_name if len(loc_name) <= 22 else loc_name[:20] + "…"

        results.append({
            "abbr":      abbr,
            "loc_name":  display_name,
            "score":     best_score,
            "spot_x":    spot_x,
            "spot_y":    spot_y,
            "text_x":    text_x,
            "text_y":    text_y,
            "leader":    leader,
        })

    # Redistribute east-coast leader labels so they don't collide vertically
    ec_items = [(i, r) for i, r in enumerate(results) if r["leader"]]
    if len(ec_items) > 1:
        ec_items.sort(key=lambda x: x[1]["text_y"])   # south → north
        y_lo = by_min + 0.04 * (by_max - by_min)
        y_hi = by_max - 0.04 * (by_max - by_min)
        n = len(ec_items)
        for k, (orig_idx, _) in enumerate(ec_items):
            results[orig_idx]["text_y"] = y_lo + k * (y_hi - y_lo) / (n - 1)

    return results


def _draw_state_overlay(ax, state_labels: list[dict]) -> None:
    """Draw per-state hotspot labels onto `ax` (map axes, data CRS)."""
    if not state_labels:
        return

    _TXT = dict(color="white", fontsize=4.8, fontweight="bold",
                ha="left", va="top", zorder=12, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.55),
                          ec="none", zorder=11))
    _ARROW = dict(arrowstyle="-", color="white", lw=0.45,
                  shrinkA=2, shrinkB=2, zorder=11)

    for s in state_labels:
        label = f"{s['abbr']}\n{s['loc_name']}\n{int(s['score'])} sp."
        if s["leader"]:
            ax.annotate(
                label,
                xy=(s["spot_x"], s["spot_y"]),
                xytext=(s["text_x"], s["text_y"]),
                xycoords="data", textcoords="data",
                annotation_clip=False,
                arrowprops=_ARROW,
                **_TXT,
            )
        else:
            ax.text(
                s["text_x"], s["text_y"], label,
                transform=ax.transData,
                **_TXT,
            )


# ---------------------------------------------------------------------------
# Figure geometry
# ---------------------------------------------------------------------------
def _fig_geometry(transform, boundary_local: gpd.GeoDataFrame) -> tuple:
    """Layout constants for make_map.  Aspect ratio driven by boundary bounds."""
    bx_min, by_min, bx_max, by_max = boundary_local.total_bounds
    pad_in    = 0.15
    fig_w     = 6.6
    cw_in     = fig_w - 2.0 * pad_in
    data_aspect = (bx_max - bx_min) / (by_max - by_min)
    map_h_in   = cw_in / data_aspect
    header_in  = 0.66
    cap_in     = 1.05
    fig_h      = map_h_in + header_in + cap_in
    cap_frac   = cap_in   / fig_h
    map_frac   = map_h_in / fig_h
    hdr_bot    = cap_frac + map_frac
    pad_h      = pad_in / fig_w
    pad_v      = pad_in / fig_h
    cb_h_n     = 0.053 / fig_h
    cb_y       = hdr_bot + pad_v
    y_r1       = 1.0 - pad_v
    return fig_w, fig_h, cap_in, cap_frac, map_frac, hdr_bot, pad_h, pad_v, cb_h_n, cb_y, y_r1


def _colorbar_ticks(vmax_binned: int) -> tuple[list[int], list[str]]:
    """Build readable colorbar ticks across small and very large vmax values.

    Ticks stop at the last round multiple of the step size.  If vmax
    extends beyond that, the final label shows "N+" (e.g. "100+").
    This avoids crowding the top-two labels (e.g. 100 and 103).
    """
    if vmax_binned <= 10:
        step = 1
    elif vmax_binned <= 20:
        step = 2
    elif vmax_binned <= 40:
        step = 5
    elif vmax_binned <= 80:
        step = 10
    elif vmax_binned <= 160:
        step = 20
    else:
        step = 50

    last_round = (vmax_binned // step) * step
    ticks = list(range(0, last_round + 1, step))
    if not ticks:
        ticks = [0]

    labels = [str(t) for t in ticks]
    # Highest label always shows "> N sp." — the scale is clipped/saturated there
    labels[-1] = f"> {last_round} sp."
    return ticks, labels


def _render_legend_bar(
    vmax_binned: int,
    bar_w: int = 400,
    bar_h: int = 18,
    font_size: int = 14,
) -> "Image.Image":
    """Render a standalone legend bar as a PIL Image with transparent background.

    Returns an RGBA image: a linear turbo-ramp strip with tick labels below,
    ready to be pasted onto any frame.
    """
    from PIL import Image, ImageDraw, ImageFont

    cmap = matplotlib.colormaps[CMAP_NAME]

    # Build a linear colour ramp as a numpy array
    ramp_arr = (cmap(np.linspace(0, 1, bar_w))[:, :3] * 255).astype(np.uint8)
    ramp_arr = np.tile(ramp_arr[np.newaxis, :, :], (bar_h, 1, 1))
    bar_img = Image.fromarray(ramp_arr, mode="RGB")

    # Composite onto a canvas with space for tick labels below
    tick_pos, tick_labels = _colorbar_ticks(vmax_binned)
    label_h = font_size + 8
    canvas_h = bar_h + 4 + label_h   # bar + tick marks + labels
    canvas = Image.new("RGBA", (bar_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(bar_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Tick positions proportional to vmax_binned (matches R exactly)
    for val, label in zip(tick_pos, tick_labels):
        x = int(val / vmax_binned * (bar_w - 1)) if vmax_binned > 0 else 0
        # Tick mark
        draw.line([(x, bar_h), (x, bar_h + 3)], fill=(229, 229, 229, 255), width=1)
        # Label — clamp to bar edges
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        lx = max(0, min(x - tw // 2, bar_w - tw))
        draw.text((lx, bar_h + 4), label, fill=(229, 229, 229, 255), font=font)

    return canvas


def make_map(
    data: np.ndarray,
    transform,
    boundary_local: gpd.GeoDataFrame,
    week_date: str,
    region: str,
    resolution: str,
    out_path: Path,
    vmax_binned: int,
    username: str = "unknown",
) -> None:
    fig_w, fig_h, cap_in, cap_frac, map_frac, hdr_bot, pad_h, pad_v, cb_h_n, cb_y, y_r1 = \
        _fig_geometry(transform, boundary_local)
    h, w = data.shape
    xmin, xmax, ymin, ymax = _raster_extent(transform, h, w)
    # Use boundary bounds for axis limits so small regions fill the canvas.
    bx_min, by_min, bx_max, by_max = boundary_local.total_bounds

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    # Map axes: pad_in on each side so the map content has uniform side margins
    ax  = fig.add_axes([pad_h, cap_frac, 1.0 - 2*pad_h, map_frac])
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    # Turbo colormap, linear scaling — same as R.
    cmap = matplotlib.colormaps[CMAP_NAME].with_extremes(bad="none")
    norm = mcolors.Normalize(vmin=0, vmax=vmax_binned, clip=True)

    ax.imshow(data, extent=[xmin, xmax, ymin, ymax], origin="upper",
              cmap=cmap, norm=norm, interpolation="nearest")

    boundary_local.boundary.plot(ax=ax, color=FG_LIGHT, alpha=0.3, linewidth=0.5)
    ax.set_aspect("equal")   # re-assert after geopandas (which resets it)
    ax.set_xlim(bx_min, bx_max)
    ax.set_ylim(by_min, by_max)



    # Title (row 1, left — white) — pad_in from figure top
    fig.text(pad_h, y_r1, "Lifer finder: mapping the birds you've yet to meet",
             color=FG_LIGHT, fontsize=8, ha="left", va="top")

    # Legend label (row 1, right) — pad_in from figure right
    fig.text(1.0 - pad_h, y_r1, "My regional needs",
             color=FG_DARK, fontsize=9, fontweight="bold", ha="right", va="top")

    # Date (row 2, left — just above colorbar)
    try:
        date_label = datetime.strptime(week_date, "%Y-%m-%d").strftime("%b-%d")
    except ValueError:
        date_label = week_date
    fig.text(pad_h, cb_y + cb_h_n + 0.025 / fig_h, date_label,
             color=FG_LIGHT, fontsize=9, fontweight="bold", ha="left", va="bottom")

    # Caption (bottom band) — starts just below map, ends pad_in from figure bottom
    occ_pct = int(OCCURRENCE_THRESH * 100)
    caption = (
        f'eBird life list of {username}.\n'
        "Inspired by original code from Sam Safran.\n\n"
        f"A candidate lifer is considered `possible` if the species has a >{occ_pct}% modeled occurrence probability at the location and date.\n\n"
        "Data from 2023 eBird Status & Trends products (https://ebird.org/science/status-and-trends): Fink, D., T. Auer, A. Johnston, M. Strimas-Mackey, S. Ligocki,\n"
        "O. Robinson, W. Hochachka, L. Jaromczyk, C. Crowley, K. Dunham, A. Stillman, I. Davies, A. Rodewald, V. Ruiz-Gutierrez, C. Wood. 2024. eBird Status and\n"
        "Trends, Data Version: 2023; Released: 2024. Cornell Lab of Ornithology, Ithaca, New York. https://doi.org/10.2173/ebirdst.2023. This material uses data from\n"
        "the eBird Status and Trends Project at the Cornell Lab of Ornithology, eBird.org. Any opinions, findings, and conclusions or recommendations expressed in this\n"
        "material are those of the author(s) and do not necessarily reflect the views of the Cornell Lab of Ornithology."
    )
    # Caption top = cap_in minus a small separator from map bottom (0.07 in)
    cap_text_y = (cap_in - 0.07) / fig_h
    fig.text(pad_h, cap_text_y, caption,
             color=FG_DARK, fontsize=5.0, ha="left", va="top", alpha=0.8)

    # ── Inline colorbar — drawn inside the figure, just like R's rasterGrob ──
    # This avoids any PIL round-trip; matplotlib writes one single image.
    tick_pos, tick_labels = _colorbar_ticks(vmax_binned)
    cb_left  = 0.50
    cb_right = 1.0 - 3 * pad_h
    cb_w     = cb_right - cb_left

    # imshow of a 1-row linear gradient (same as R's rasterGrob with interpolate=TRUE)
    cb_ax = fig.add_axes([cb_left, cb_y, cb_w, cb_h_n])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cb_ax.imshow(gradient, aspect="auto", cmap=matplotlib.colormaps[CMAP_NAME],
                 extent=[0, vmax_binned, 0, 1], origin="lower")
    cb_ax.set_xlim(0, vmax_binned)
    cb_ax.set_ylim(0, 1)
    cb_ax.set_xticks(tick_pos)
    cb_ax.set_xticklabels(tick_labels, fontsize=7, color=FG_LIGHT)
    cb_ax.tick_params(axis="x", length=2, width=0.5, color=FG_LIGHT,
                      pad=1, bottom=True, top=False, labelbottom=True)
    cb_ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in cb_ax.spines.values():
        spine.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Scale DPI so the saved PNG is exactly _FRAME_WIDTH_PX pixels wide.
    out_dpi = _FRAME_WIDTH_PX / fig_w
    fig.savefig(out_path, dpi=out_dpi, facecolor=BG_COLOR)
    plt.close(fig)


_FRAME_RENDER_CONTEXT: dict = {}


def _render_frame_worker(i: int) -> int:
    """Fork worker for parallel frame rendering (Linux fork path)."""
    ctx = _FRAME_RENDER_CONTEXT
    wd = ctx["week_dates"][i]
    out_path = ctx["weekly_dir"] / f"{ctx['region']}_{wd}.png"
    if out_path.exists() and not ctx.get("overwrite_frames", False):
        return 0
    make_map(
        ctx["stack_local"][i],
        ctx["tf_local"],
        ctx["boundary_local"],
        week_date=wd,
        region=ctx["region"],
        resolution=ctx["resolution"],
        out_path=out_path,
        vmax_binned=ctx["vmax_binned"],
        username=ctx["username"],
    )
    return 1


def make_gif_lores(frame_dir: Path, gif_path: Path, fps: int = 5, scale: float = _LORES_SCALE,
                   workers: int = 4) -> None:
    """Assemble a downscaled GIF at `scale` fraction of original size.

    Frames already contain a baked-in PIL legend (from make_map).
    Reuses the same global-palette strategy as make_gif.
    Frame loading, resize, and per-frame quantization are parallelised with threads.
    """
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    jpgs = sorted(frame_dir.glob("*.png"))
    if not jpgs:
        print("  [warn] no frames found for lo-res GIF")
        return

    first = Image.open(jpgs[0])
    new_w = round(first.width  * scale)
    new_h = round(first.height * scale)
    first.close()

    n_threads = max(1, min(workers, len(jpgs)))

    def _load_resize(p: Path) -> Image.Image:
        src = Image.open(p)
        img = src.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
        src.close()
        return img

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        frames = list(ex.map(_load_resize, jpgs))

    # Build palette from sampled frames + a tall ramp strip so the quantizer
    # reserves enough palette entries for the full colour gradient.
    sample = frames[::4]
    ramp_w = 256
    ramp_h = 256  # tall strip → heavy weight in the palette
    ramp_1d = (matplotlib.colormaps[CMAP_NAME](np.linspace(0, 1, ramp_w))[:, :3] * 255).astype(np.uint8)
    ramp_arr = np.tile(ramp_1d[np.newaxis, :, :], (ramp_h, 1, 1))
    ramp_img = Image.fromarray(ramp_arr, mode="RGB")

    combined = Image.new("RGB", (new_w, new_h * len(sample) + ramp_h))
    for idx, img in enumerate(sample):
        combined.paste(img, (0, idx * new_h))
    combined.paste(ramp_img.resize((new_w, ramp_h), Image.NEAREST), (0, new_h * len(sample)))
    palette_img = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0)
    del combined

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        quantized = list(ex.map(
            lambda f: f.quantize(palette=palette_img, dither=Image.Dither.NONE), frames
        ))

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        gif_path, save_all=True, append_images=quantized[1:],
        loop=0, duration=max(20, 1000 // fps), optimize=False,
    )
    print(f"  Lo-res GIF ({len(frames)} frames, "
          f"{gif_path.stat().st_size / 1e6:.1f} MB) → {gif_path}")


def make_gif(frame_dir: Path, gif_path: Path, fps: int = 5, workers: int = 4) -> None:
    """Assemble frames into a GIF using a single global palette.

    Frames already contain a baked-in PIL legend (from make_map).
    The global-palette strategy ensures colours are preserved across frames.
    Frame loading and per-frame quantization are parallelised with threads.
    """
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    jpgs = sorted(frame_dir.glob("*.png"))
    if not jpgs:
        print("  [warn] no frames found for GIF")
        return

    n_threads = max(1, min(workers, len(jpgs)))

    def _load(p: Path) -> Image.Image:
        src = Image.open(p)
        img = src.convert("RGB")
        src.close()
        return img

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        frames = list(ex.map(_load, jpgs))

    # Build palette from sampled frames + a tall ramp strip so the quantizer
    # reserves enough palette entries for the full colour gradient.
    sample_pixels: list[Image.Image] = frames[::4]
    combined_w = frames[0].width
    ramp_h = 256  # tall strip → heavy weight in the palette
    combined_h = frames[0].height * len(sample_pixels) + ramp_h
    combined = Image.new("RGB", (combined_w, combined_h))
    for idx, img in enumerate(sample_pixels):
        combined.paste(img, (0, idx * frames[0].height))
    ramp_1d = (matplotlib.colormaps[CMAP_NAME](np.linspace(0, 1, combined_w))[:, :3] * 255).astype(np.uint8)
    ramp_arr = np.tile(ramp_1d[np.newaxis, :, :], (ramp_h, 1, 1))
    combined.paste(Image.fromarray(ramp_arr, mode="RGB"), (0, frames[0].height * len(sample_pixels)))
    palette_img = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0)
    del combined

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        quantized = list(ex.map(
            lambda f: f.quantize(palette=palette_img, dither=Image.Dither.NONE), frames
        ))

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        gif_path, save_all=True, append_images=quantized[1:],
        loop=0, duration=max(20, 1000 // fps), optimize=False,
    )
    print(f"  GIF ({len(frames)} frames, "
          f"{gif_path.stat().st_size / 1e6:.1f} MB) → {gif_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TARGET_CRS = {"NL": "EPSG:28992", "US": "EPSG:5070"}


def get_boundary(region: str) -> gpd.GeoDataFrame:
    if region == "NL":
        return get_nl_boundary()
    if region == "US":
        return get_us_boundary()
    if region.startswith("US-"):
        return get_us_boundary(state_code=region)
    raise NotImplementedError(f"No boundary loader for '{region}' yet.")


def _target_crs(region: str) -> str:
    """Return the plotting CRS for a region code."""
    if region == "NL":
        return TARGET_CRS["NL"]
    if region == "US" or region.startswith("US-"):
        return TARGET_CRS["US"]
    return "EPSG:4326"


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------
def _lap(label: str, t0: float) -> float:
    """Print elapsed seconds since t0, return new reference time."""
    print(f"  ✓ {label}: {time.perf_counter() - t0:.1f}s")
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--regions",     nargs="+", default=["NL", "US"], metavar="CODE")
    parser.add_argument("--resolutions", nargs="+", default=["3km", "9km", "27km"],
                        choices=["3km", "9km", "27km"], metavar="RES",
                        help="Resolutions to process (default: all three).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--week",       type=int, metavar="N", choices=range(1, 53),
                      help="Single map for week N (1-52).")
    mode.add_argument("--animate",    action="store_true",
                      help="All 52 frames + animated GIF (default; explicit flag accepted for compatibility).")
    mode.add_argument("--no-animate", action="store_true",
                      help="Skip animation; render week 20 only.")
    mode.add_argument("--accumulate-only", action="store_true",
                      help="Accumulate rasters and exit — no reprojection, no rendering. "
                           "Useful for benchmarking the sp_cache / tif read pipeline.")
    parser.add_argument("--fps",    type=int,   default=5,   help="GIF fps.")
    parser.add_argument("--ram-gb", type=float, default=4.0, help="RAM budget per batch (GB).")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help="Max worker threads for accumulation/reprojection.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Species per accumulation batch; overrides RAM-derived size.")
    parser.add_argument("--overwrite-frames", action="store_true",
                        help="Re-render JPEG frames even if they already exist.")
    parser.add_argument(
        "--scale", default=DEFAULT_SCALE, choices=list(SCALES),
        help="Colour-scale preset: auto (data max), compact (35+), wide (75+). Default: %(default)s.",
    )
    parser.add_argument(
        "--vmax", type=int, default=None, metavar="N",
        help="Override colour-scale max (e.g. --vmax 10 for small regions).",
    )
    parser.add_argument("--ebird-csv", default="MyEBirdData.csv", metavar="PATH")
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip all eBird API calls. Use every locally cached tif as 'needed'. "
             "Useful for timing tests and offline development.",
    )
    parser.add_argument(
        "--build-cache", action="store_true",
        help="After accumulation, also build the top_spots precomputed cache "
             "(all score layers + species presence). Adds score-compute + pickle-write "
             "overhead; accumulation itself is shared with the animation pass.",
    )
    args = parser.parse_args()
    worker_cap = max(1, args.workers)

    # Default mode is full animation; --no-animate / --week / --accumulate-only override
    accumulate_only = getattr(args, "accumulate_only", False)
    animate = not args.no_animate and args.week is None and not accumulate_only
    single_week = args.week if args.week is not None else (None if animate else 20)

    t_wall = time.perf_counter()

    config   = load_config()
    modeled  = load_ebirdst_runs()
    username = config.get("user", "unknown")

    if args.offline:
        print("[offline] Skipping eBird API calls — using locally cached tifs.")
        name_to_code = {}
        api_key      = None
    else:
        api_key = config["ebird_api_key"]
        t = time.perf_counter()
        print("Fetching eBird taxonomy …")
        name_to_code = get_taxonomy(api_key)
        t = _lap("taxonomy", t)

    for region in args.regions:
        print(f"\n{'═'*60}")
        print(f"Region: {region}")
        t_region = time.perf_counter()

        # ------------------------------------------------------------------
        # Build needed species once per region (shared across resolutions)
        # ------------------------------------------------------------------
        if args.offline:
            regional = None
            user_seen = set()
        else:
            print("  Fetching eBird checklist …")
            t = time.perf_counter()
            regional  = ebird_regional_species(region, api_key)
            t = _lap("checklist fetch", t)
            user_seen = user_seen_codes(Path(args.ebird_csv), region, name_to_code)
            needed_base = (regional - user_seen) & set(modeled) - EXCLUDED_CODES
            print(f"  Checklist: {len(regional)}, seen: {len(user_seen)}, "
                  f"needed+modeled: {len(needed_base)}")
            t = _lap("user-list diff", t)

        boundary   = get_boundary(region)
        target_crs = _target_crs(region)

        for resolution in args.resolutions:
            print(f"\n{'─'*60}")
            print(f"  Resolution: {resolution}")
            t = time.perf_counter()

            if args.offline:
                needed = {
                    c for c in modeled
                    if c not in EXCLUDED_CODES
                    and tif_path(c, resolution).exists()
                }
                print(f"  [offline] {len(needed)} species with local {resolution} tifs")
                t = _lap("species selection (offline)", t)
            else:
                needed = needed_base
                download_missing(needed, resolution, config["ebirdst_key"], workers=worker_cap)
                t = _lap("download check", t)

            t = _lap("boundary load", t)

            # ------------------------------------------------------------------
            # Determine vmax *before* accumulation so we can skip the full
            # 52-week pass when only a single week is needed.
            # ------------------------------------------------------------------
            scale_vmax  = SCALES.get(args.scale, SCALES[DEFAULT_SCALE])
            if args.vmax is not None:
                vmax_binned = args.vmax
            elif scale_vmax is not None:
                vmax_binned = scale_vmax
            else:
                vmax_binned = None  # need global_max — must accumulate all

            need_all_weeks = animate or vmax_binned is None or accumulate_only

            if need_all_weeks:
                stack, tf_src, crs_src, week_dates, annual_richness, win, available, sp_packed = \
                    accumulate_all_weeks(
                        needed, boundary, resolution, args.ram_gb,
                        workers=worker_cap, batch_size=args.batch_size,
                        track_sp_presence=args.build_cache,
                        region=region,
                    )
                global_max = float(np.nanmax(stack))
                if vmax_binned is None:
                    vmax_binned = max(10, int(global_max))
                print(f"  Global max: {global_max:.0f}  →  vmax: {vmax_binned}  scale: {args.scale}")
                t = _lap("raster accumulation", t)

                if accumulate_only:
                    print(f"  [accumulate-only] done — skipping reproject + render.")
                    del stack
                    continue

                if args.build_cache:
                    try:
                        from top_spots import _precomputed_from_arrays
                        needed_set = set(needed) if not isinstance(needed, set) else needed
                        _precomputed_from_arrays(
                            region, resolution, needed_set,
                            stack, tf_src, crs_src, week_dates,
                            annual_richness, win, available, sp_packed,
                        )
                    except Exception as _exc:
                        print(f"  [build-cache] skipped: {_exc}")
                    t = _lap("build-cache (score+pickle overhead)", t)

                print(f"  Reprojecting 52 layers to {target_crs} …")
                stack_local, tf_local = reproject_stack(
                    stack, tf_src, crs_src, target_crs, workers=worker_cap
                )
                del stack
            else:
                # Fast path: accumulate only the single needed week.
                stack, tf_src, crs_src, week_dates, annual_richness, win, available, sp_packed = \
                    accumulate_all_weeks(
                        needed, boundary, resolution, args.ram_gb,
                        workers=worker_cap, batch_size=args.batch_size,
                        track_sp_presence=args.build_cache,
                        region=region,
                    )
                week_idx = single_week - 1
                if not (0 <= week_idx < len(week_dates)):
                    sys.exit(f"--week {single_week} out of range (valid: 1-{len(week_dates)})")

                if args.build_cache:
                    try:
                        from top_spots import _precomputed_from_arrays
                        needed_set = set(needed) if not isinstance(needed, set) else needed
                        _precomputed_from_arrays(
                            region, resolution, needed_set,
                            stack, tf_src, crs_src, week_dates,
                            annual_richness, win, available, sp_packed,
                        )
                    except Exception as _exc:
                        print(f"  [build-cache] skipped: {_exc}")
                    t = _lap("build-cache (score+pickle overhead)", t)

                single_layer = stack[week_idx:week_idx+1]  # keep 3-D for reproject
                del stack
                print(f"  vmax: {vmax_binned}  scale: {args.scale}  (single-week fast path)")
                t = _lap("raster accumulation", t)

                print(f"  Reprojecting 1 layer to {target_crs} …")
                stack_local, tf_local = reproject_stack(
                    single_layer, tf_src, crs_src, target_crs, workers=worker_cap
                )
                del single_layer

            boundary_local = boundary.to_crs(target_crs)
            t = _lap("reprojection", t)

            if animate:
                weekly_dir = OUT_DIR / region / resolution / f"Weekly_maps_{args.scale}"

                n_new = 0
                frame_workers = max(1, min(worker_cap, len(week_dates)))
                use_fork_pool = (
                    frame_workers > 1
                    and os.name == "posix"
                    and "fork" in mp.get_all_start_methods()
                )

                if use_fork_pool:
                    global _FRAME_RENDER_CONTEXT
                    _FRAME_RENDER_CONTEXT = {
                        "stack_local": stack_local,
                        "tf_local": tf_local,
                        "boundary_local": boundary_local,
                        "week_dates": week_dates,
                        "weekly_dir": weekly_dir,
                        "region": region,
                        "resolution": resolution,
                        "vmax_binned": vmax_binned,
                        "username": username,
                        "overwrite_frames": args.overwrite_frames,
                    }
                    done = 0
                    with mp.get_context("fork").Pool(processes=frame_workers) as pool:
                        for created in pool.imap_unordered(_render_frame_worker, range(len(week_dates)), chunksize=1):
                            done += 1
                            n_new += int(created)
                            if n_new > 0 and n_new % 10 == 0:
                                print(f"    {n_new} frames saved …")
                    _FRAME_RENDER_CONTEXT = {}
                else:
                    for i, wd in enumerate(week_dates):
                        out_path = weekly_dir / f"{region}_{wd}.png"
                        if out_path.exists() and not args.overwrite_frames:
                            continue
                        make_map(stack_local[i], tf_local, boundary_local,
                                 week_date=wd, region=region, resolution=resolution,
                                 out_path=out_path, vmax_binned=vmax_binned,
                                 username=username)
                        n_new += 1
                        if n_new % 10 == 0:
                            print(f"    {n_new} frames saved …")
                print(f"  {n_new} new frames saved." if n_new else "  All frames already exist.")
                del stack_local
                t = _lap("frame rendering", t)

                gif_path = (OUT_DIR / region / resolution / "Animated_map"
                            / f"{region}_{resolution}_{args.scale}_animated.gif")
                gif_lores_path = (OUT_DIR / region / resolution / "Animated_map"
                                  / f"{region}_{resolution}_{args.scale}_animated_lores.gif")
                print("  Assembling GIF …")
                make_gif(weekly_dir, gif_path, fps=args.fps, workers=worker_cap)
                t = _lap("GIF assembly", t)

                print(f"  Assembling lo-res GIF ({int(_LORES_SCALE * 100)}%) …")
                make_gif_lores(weekly_dir, gif_lores_path, fps=args.fps, workers=worker_cap)
                t = _lap("lo-res GIF assembly", t)

            else:
                # Single-week preview
                if need_all_weeks:
                    # Full stack was built — index into it
                    week_idx = single_week - 1
                    if not (0 <= week_idx < len(week_dates)):
                        sys.exit(f"--week {single_week} out of range (valid: 1-{len(week_dates)})")
                    layer = stack_local[week_idx]
                else:
                    # Fast path — stack_local has exactly 1 layer
                    week_idx = single_week - 1
                    layer = stack_local[0]
                week_date = week_dates[week_idx]
                out_path  = OUT_DIR / region / resolution / f"{region}_{resolution}_week{single_week:02d}_{args.scale}.png"
                make_map(layer, tf_local, boundary_local,
                         week_date=week_date, region=region,
                         resolution=resolution, out_path=out_path,
                         vmax_binned=vmax_binned, username=username)
                del stack_local
                print(f"  Saved → {out_path}")
                t = _lap("map render", t)

    total = time.perf_counter() - t_wall
    print(f"\nDone — output in {OUT_DIR}/  (total {total:.1f}s)")


if __name__ == "__main__":
    main()
