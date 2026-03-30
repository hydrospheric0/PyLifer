#!/usr/bin/env python3
"""
preprocess_region.py — Convert raw eBird S&T float32 tifs into compact
per-species binary (int8) caches cropped to a region boundary.

This is Step 2 of the PyLifer pipeline.  It must be run AFTER tifs have
been downloaded (Step 1) and BEFORE map_lifers.py or top_spots.py are run
(Steps 3–5).

Data written
------------
data/sp_cache/<region>/<resolution>/<version>/
    _meta.npz             — window, transform, CRS, outside_mask, week_dates,
                            threshold, version string
    <code>.npy            — (52,) * H_inside flat int8, packed bits
                            i.e. np.packbits of the (52, n_inside_pixels) binary array
                            Pixels are stored in row-major order, inside-mask only.

Only species whose max occurrence within the region exceeds OCCURRENCE_THRESH
are written (others are silently skipped — they contribute nothing to maps
or scores).

Usage
-----
    python preprocess_region.py                        # US, all resolutions
    python preprocess_region.py --regions US NL
    python preprocess_region.py --resolutions 3km 9km
    python preprocess_region.py --threshold 0.05       # override default
    python preprocess_region.py --workers 16
    python preprocess_region.py --force                # overwrite existing cache
    python preprocess_region.py --dry-run              # report counts, no writes
"""

import argparse
import csv
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window

# ---------------------------------------------------------------------------
# Shared constants (must stay in sync with map_lifers.py)
# ---------------------------------------------------------------------------
EBIRDST_VERSION   = "2023"
CACHE_DIR         = Path("data/ebirdst")
SP_CACHE_DIR      = Path("data/sp_cache")    # ← new preprocessed store
NE_DIR            = Path("data/naturalearth")
OCCURRENCE_THRESH = 0.01

_cpus     = os.cpu_count() or 4
N_WORKERS = max(4, min(_cpus * 3 // 4, _cpus - 8))

EBIRD_CSV = Path("MyEBirdData.csv")

# ---------------------------------------------------------------------------
# Boundary helpers (duplicated from map_lifers.py to keep this self-contained)
# ---------------------------------------------------------------------------
def _ne_download(url: str, dest_dir: Path) -> Path:
    stem = Path(url).stem
    shp  = dest_dir / f"{stem}.shp"
    if shp.exists():
        return shp
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading NaturalEarth: {stem} …")
    import requests
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


def _get_us_boundary(state_code: str | None = None) -> gpd.GeoDataFrame:
    shp = _ne_download(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
        NE_DIR / "ne_10m_admin_1_states_provinces",
    )
    states = gpd.read_file(shp)
    if state_code:
        return states[states["iso_3166_2"] == state_code].copy().to_crs("EPSG:4326")
    return states[
        (states["adm0_a3"] == "USA") &
        (~states["iso_3166_2"].isin(["US-HI", "US-AK"]))
    ].copy().to_crs("EPSG:4326")


def _get_nl_boundary() -> gpd.GeoDataFrame:
    shp = _ne_download(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
        NE_DIR / "ne_10m_admin_0_countries",
    )
    world = gpd.read_file(shp)
    col = "ISO_A3" if "ISO_A3" in world.columns else "sov_a3"
    nl = world[world[col] == "NLD"].to_crs("EPSG:4326").copy()
    # NaturalEarth "NLD" includes Caribbean overseas territories (Aruba, Curaçao,
    # Bonaire etc.).  Explode and keep only the European mainland part so the
    # sp_cache window and boundary mask match map_lifers.py exactly.
    nl_exp  = nl.explode(index_parts=True)
    nl_proj = nl_exp.to_crs("EPSG:4087")   # World Equidistant Cylindrical
    nl_exp  = nl_exp[
        (nl_proj.geometry.centroid.x > -1_000_000) &   # east of ~9°W
        (nl_proj.geometry.centroid.y >  4_500_000)     # north of ~40°N
    ]
    if nl_exp.empty:
        sys.exit("Could not isolate European part of Netherlands from NaturalEarth data.")
    return nl_exp.dissolve().to_crs("EPSG:4326")


def get_boundary(region: str) -> gpd.GeoDataFrame:
    if region == "NL":
        return _get_nl_boundary()
    if region == "US":
        return _get_us_boundary()
    if region.startswith("US-"):
        return _get_us_boundary(state_code=region)
    raise NotImplementedError(f"No boundary loader for '{region}'")


# ---------------------------------------------------------------------------
# tif path helper
# ---------------------------------------------------------------------------
def tif_path(code: str, resolution: str) -> Path:
    return (
        CACHE_DIR / EBIRDST_VERSION / code / "weekly"
        / f"{code}_occurrence_median_{resolution}_{EBIRDST_VERSION}.tif"
    )


def sp_cache_dir(region: str, resolution: str) -> Path:
    return SP_CACHE_DIR / region / resolution / EBIRDST_VERSION


def meta_path(region: str, resolution: str) -> Path:
    return sp_cache_dir(region, resolution) / "_meta.npz"


def sp_cache_path(region: str, resolution: str, code: str) -> Path:
    return sp_cache_dir(region, resolution) / f"{code}.npy"


# ---------------------------------------------------------------------------
# Cache validity check
# ---------------------------------------------------------------------------
def cache_is_valid(region: str, resolution: str, threshold: float) -> bool:
    """Return True if _meta.npz exists and threshold/version match."""
    mp = meta_path(region, resolution)
    if not mp.exists():
        return False
    try:
        d = np.load(mp, allow_pickle=False)
        return (
            d["version"].item().decode() == EBIRDST_VERSION
            and float(d["threshold"]) == threshold
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Build window + outside_mask from the first available tif
# ---------------------------------------------------------------------------
def build_window_and_mask(ref_tif: Path, boundary_wgs84: gpd.GeoDataFrame):
    from affine import Affine
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
    inside_idx = np.flatnonzero(~outside_mask.ravel())   # 1-D indices of valid pixels
    return ref_crs, win, ref_shape, ref_transform, outside_mask, inside_idx, week_dates, n_bands


# ---------------------------------------------------------------------------
# Per-species preprocessing
# ---------------------------------------------------------------------------
def preprocess_species(
    code: str,
    resolution: str,
    win: "Window",
    outside_mask: np.ndarray,
    inside_idx: np.ndarray,
    threshold: float,
    out_dir: Path,
    force: bool,
) -> str:
    """
    Read species tif, threshold, extract inside pixels, packbits, save.

    Returns one of: 'cached' | 'written' | 'below_threshold' | 'error:<msg>'
    """
    out_path = out_dir / f"{code}.npy"
    if out_path.exists() and not force:
        return "cached"
    try:
        with rasterio.open(tif_path(code, resolution)) as src:
            data = src.read(window=win).astype(np.float32)   # (52, H, W)
        # mask outside pixels → NaN
        data[:, outside_mask] = np.nan
        if not (np.nanmax(data) > threshold):
            return "below_threshold"
        # binarise: 1 if above threshold, 0 otherwise (NaN/nodata → 0)
        binary = np.where(
            ~np.isfinite(data),
            np.int8(0),
            (data > threshold).astype(np.int8),
        )                                                     # (52, H, W) int8
        # Extract only inside-mask pixels → (52, n_inside) then flatten to 1-D
        # Shape: (52 * n_inside,) int8   → packed to ceil(52*n_inside/8) bytes
        inside_flat = binary.reshape(binary.shape[0], -1)[:, inside_idx]  # (52, n_inside)
        packed = np.packbits(inside_flat.ravel())
        np.save(out_path, packed)
        return "written"
    except Exception as exc:
        return f"error:{exc}"


# ---------------------------------------------------------------------------
# Main preprocess function for one (region, resolution) pair
# ---------------------------------------------------------------------------
def preprocess(
    region: str,
    resolution: str,
    species_codes: list[str],
    threshold: float,
    workers: int,
    force: bool,
    dry_run: bool,
) -> None:
    print(f"\n{'═'*62}")
    print(f"  Region: {region}   Resolution: {resolution}")
    print(f"{'═'*62}")

    boundary = get_boundary(region)

    # Find available tifs
    available = [c for c in sorted(species_codes) if tif_path(c, resolution).exists()]
    n_missing = len(species_codes) - len(available)
    print(f"  Tifs found: {len(available)} / {len(species_codes)}"
          + (f"  ({n_missing} not modelled at {resolution})" if n_missing else ""))
    if not available:
        print("  [skip] no tifs found")
        return

    if dry_run:
        print(f"  [dry-run] would preprocess {len(available)} species")
        return

    # Check cache validity (skip if already complete + not forced)
    if not force and cache_is_valid(region, resolution, threshold):
        existing = list(sp_cache_dir(region, resolution).glob("*.npy"))
        print(f"  Cache already valid ({len(existing)} species files). "
              f"Use --force to rebuild.")
        return

    t0 = time.perf_counter()

    # Build window + mask from first tif
    print(f"  Building window and boundary mask …")
    ref_crs, win, ref_shape, ref_transform, outside_mask, inside_idx, week_dates, n_bands = \
        build_window_and_mask(tif_path(available[0], resolution), boundary)

    n_inside = len(inside_idx)
    bytes_per_sp = int(np.ceil(n_bands * n_inside / 8))
    print(f"  Window: {ref_shape[1]}×{ref_shape[0]} px  "
          f"Inside-mask: {n_inside:,} px  "
          f"Per-species cache: {bytes_per_sp/1024:.1f} KB")

    # Save _meta.npz
    out_dir = sp_cache_dir(region, resolution)
    out_dir.mkdir(parents=True, exist_ok=True)

    from affine import Affine
    tf = ref_transform
    np.savez_compressed(
        meta_path(region, resolution),
        version=np.bytes_(EBIRDST_VERSION),
        threshold=np.float32(threshold),
        ref_crs_wkt=np.bytes_(ref_crs.to_wkt()),
        win_col_off=np.int32(win.col_off),
        win_row_off=np.int32(win.row_off),
        win_width=np.int32(win.width),
        win_height=np.int32(win.height),
        ref_shape=np.array(ref_shape, dtype=np.int32),
        transform_params=np.array(
            [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f], dtype=np.float64
        ),
        outside_mask=outside_mask,
        inside_idx=inside_idx.astype(np.int32),
        week_dates=np.array(week_dates),
        n_bands=np.int32(n_bands),
    )
    print(f"  Saved _meta.npz")

    # Parallel species processing
    print(f"  Processing {len(available)} species with {workers} workers …")
    counts = {"cached": 0, "written": 0, "below_threshold": 0, "error": 0}
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                preprocess_species,
                code, resolution, win, outside_mask, inside_idx,
                threshold, out_dir, force,
            ): code
            for code in available
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result.startswith("error:"):
                counts["error"] += 1
                print(f"    [error] {futures[fut]}: {result[6:]}")
            else:
                counts[result] += 1
            done += 1
            if done % 100 == 0 or done == len(available):
                print(f"    {done}/{len(available)}  "
                      f"written={counts['written']}  "
                      f"cached={counts['cached']}  "
                      f"skipped={counts['below_threshold']}  "
                      f"errors={counts['error']}")

    elapsed = time.perf_counter() - t0
    total_size = sum(p.stat().st_size for p in out_dir.glob("*.npy"))
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Written: {counts['written']}  Cached: {counts['cached']}  "
          f"Below-threshold: {counts['below_threshold']}  Errors: {counts['error']}")
    print(f"  Cache size: {total_size / 1e6:.1f} MB  →  {out_dir}")


# ---------------------------------------------------------------------------
# Species discovery (mirrors map_lifers.py logic)
# ---------------------------------------------------------------------------
def load_all_species() -> list[str]:
    """Return all species codes available in the tif store."""
    d = CACHE_DIR / EBIRDST_VERSION
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess eBird S&T tifs into compact binary sp_cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--regions", nargs="+", default=["US"],
                        metavar="CODE",
                        help="Region codes to preprocess (default: US)")
    parser.add_argument("--resolutions", nargs="+", default=["3km", "9km", "27km"],
                        metavar="RES")
    parser.add_argument("--threshold", type=float, default=OCCURRENCE_THRESH,
                        help=f"Occurrence threshold (default: {OCCURRENCE_THRESH})")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Parallel workers (default: {N_WORKERS})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing cache files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be done without writing")
    args = parser.parse_args()

    species = load_all_species()
    if not species:
        sys.exit(f"ERROR: No species tifs found in {CACHE_DIR / EBIRDST_VERSION}. "
                 f"Run Step 1 (download_ebirdst.py) first.")

    print(f"PyLifer — preprocess_region.py")
    print(f"  Tif store:    {CACHE_DIR / EBIRDST_VERSION}  ({len(species)} species)")
    print(f"  sp_cache:     {SP_CACHE_DIR}")
    print(f"  Threshold:    {args.threshold}")
    print(f"  Regions:      {' '.join(args.regions)}")
    print(f"  Resolutions:  {' '.join(args.resolutions)}")
    print(f"  Workers:      {args.workers}")
    if args.dry_run:
        print(f"  [DRY RUN — no files will be written]")

    for region in args.regions:
        for resolution in args.resolutions:
            preprocess(
                region=region,
                resolution=resolution,
                species_codes=species,
                threshold=args.threshold,
                workers=args.workers,
                force=args.force,
                dry_run=args.dry_run,
            )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
