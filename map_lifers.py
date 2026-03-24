#!/usr/bin/env python3
"""
map_lifers.py — Generate weekly lifer maps and animated GIF from cached
eBird Status & Trends rasters.  Output goes to results_py/

Single-week test:
    .venv/bin/python map_lifers.py --week 20

Full 52-week animation (NL + US, 3km):
    .venv/bin/python map_lifers.py --animate

Options:
    --regions NL US     eBird region codes
    --resolution 3km    3km | 9km | 27km
    --week N            single map for week N (1-52)
    --animate           all 52 frames + GIF
    --ram-gb 4.0        RAM budget per batch in GB (default 4)
    --fps 5             GIF frames per second (default 5)
"""

import argparse
import csv
import os
import re
import sys
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
NE_DIR            = Path("data/naturalearth")
OUT_DIR           = Path("results_py")
EXCLUDED_CODES    = {"laugul", "rocpig", "compea", "yebsap-example"}
OCCURRENCE_THRESH = 0.01

# Leave at least 8 cores free so the system stays responsive.
_cpus     = os.cpu_count() or 4
N_WORKERS = max(4, min(_cpus * 3 // 4, _cpus - 8))

# Dark theme — matches R: bg = viridisLite::turbo(1)
BG_COLOR = "#30123b"
FG_DARK  = "#e5e5e5"   # grey90
FG_LIGHT = "white"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: Path = Path("config_local.R")) -> dict:
    if not path.exists():
        sys.exit("config_local.R not found — copy config_local.R.example and fill in your keys.")
    keys = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "<-" not in line:
            continue
        name, _, rhs = line.partition("<-")
        m = re.search(r'["\']([^"\']+)["\']', rhs)
        if m:
            keys[name.strip()] = m.group(1)
    return keys


def load_ebirdst_runs(path: Path = Path("ebirdst_runs.csv")) -> dict:
    if not path.exists():
        sys.exit("ebirdst_runs.csv not found — run:  Rscript export_ebirdst_runs.R")
    return {row["species_code"]: row["common_name"]
            for row in csv.DictReader(open(path))}


# ---------------------------------------------------------------------------
# eBird API helpers
# ---------------------------------------------------------------------------
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
        shp = next(dest_dir.rglob("*.shp"))
    return shp


def get_nl_boundary() -> gpd.GeoDataFrame:
    shp = _ne_download(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
        NE_DIR / "ne_110m_admin_0_countries",
    )
    world = gpd.read_file(shp)
    col = "ISO_A3" if "ISO_A3" in world.columns else "sov_a3"
    return world[world[col] == "NLD"].to_crs("EPSG:4326").copy()


def get_us_boundary() -> gpd.GeoDataFrame:
    shp = _ne_download(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_1_states_provinces.zip",
        NE_DIR / "ne_110m_admin_1_states_provinces",
    )
    states = gpd.read_file(shp)
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


def _setup_window(ref_tif: Path, boundary_wgs84: gpd.GeoDataFrame):
    """Compute shared windowed extent and outside-boundary mask from one tif."""
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
    return ref_crs, win, ref_shape, ref_transform, outside_mask, week_dates, n_bands


def accumulate_all_weeks(
    needed: set,
    boundary_wgs84: gpd.GeoDataFrame,
    resolution: str = "3km",
    ram_budget_gb: float = 4.0,
) -> tuple:
    """
    Read every species tif ONCE (all 52 bands), accumulate into (52, h, w).

    Phase 1 — serial: compute window + outside-boundary mask from first file.
    Phase 2 — parallel batched reads: each thread returns (52, h, w) int8.
               Batch size is set so peak RAM ≈ ram_budget_gb.
    Phase 3 — apply NaN mask.

    Returns (result, transform, crs, week_dates).
    """
    available = [c for c in sorted(needed) if tif_path(c, resolution).exists()]
    print(f"  {len(available):3d} / {len(needed):3d} species have {resolution} tifs")
    if not available:
        raise RuntimeError(f"No {resolution} tifs found.")

    ref_crs, win, ref_shape, ref_transform, outside_mask, week_dates, n_bands = \
        _setup_window(tif_path(available[0], resolution), boundary_wgs84)

    bytes_per_sp = n_bands * ref_shape[0] * ref_shape[1]   # int8
    batch_size   = max(4, min(N_WORKERS * 2,
                              int(ram_budget_gb * 1e9 / max(bytes_per_sp, 1))))
    print(f"  {min(N_WORKERS, len(available))} threads, "
          f"batch {batch_size} sp "
          f"({bytes_per_sp * batch_size / 1e9:.2f} GB/batch)")

    accumulator = np.zeros((n_bands, *ref_shape), dtype=np.int16)
    done = 0

    def _read_all_bands(code: str):
        try:
            with rasterio.open(tif_path(code, resolution)) as src:
                data = src.read(window=win).astype(np.float32)
            return np.where(
                ~np.isfinite(data) | outside_mask[np.newaxis],
                np.int8(0),
                (data > OCCURRENCE_THRESH).astype(np.int8),
            )
        except Exception as exc:
            print(f"    [skip] {code}: {exc}")
            return None

    workers = min(N_WORKERS, len(available))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch_start in range(0, len(available), batch_size):
            batch   = available[batch_start : batch_start + batch_size]
            futures = {pool.submit(_read_all_bands, c): c for c in batch}
            for fut in as_completed(futures):
                arr = fut.result()
                if arr is not None:
                    accumulator += arr
                done += 1
            if done % 50 == 0 or done == len(available):
                print(f"    {done}/{len(available)} species …")

    result = accumulator.astype(np.float32)
    result[:, outside_mask] = np.nan
    return result, ref_transform, ref_crs, week_dates


def reproject_stack(
    data3d: np.ndarray,
    src_transform,
    src_crs,
    dst_crs_str: str,
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
        reproject(
            source=data3d[i], destination=out,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs_str,
            resampling=Resampling.nearest,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
        return i, out

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
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


def make_map(
    data: np.ndarray,
    transform,
    boundary_local: gpd.GeoDataFrame,
    week_date: str,
    region: str,
    resolution: str,
    out_path: Path,
    vmax_binned: int,
) -> None:
    h, w = data.shape
    xmin, xmax, ymin, ymax = _raster_extent(transform, h, w)

    data_aspect = (xmax - xmin) / (ymax - ymin)
    fig_w = 6.6
    fig_h = min(max(fig_w / data_aspect / 0.77, 3.5), 8.0)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    ax  = fig.add_axes([0.0, 0.08, 1.0, 0.72])
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # Discrete binned colormap: equal 5-species bins, boundaries fixed to global vmax.
    # Values of 0 (no species reachable) fall below 0.5 → "under" → BG_COLOR.
    step = 5
    bin_edges = [0.5] + list(range(step, vmax_binned + 1, step))
    n_bins    = len(bin_edges) - 1
    norm      = mcolors.BoundaryNorm(bin_edges, ncolors=n_bins)
    cmap_disc = mcolors.LinearSegmentedColormap.from_list(
        "turbo_binned", plt.cm.turbo(np.linspace(0, 1, n_bins)), N=n_bins
    )
    cmap_disc.set_bad(color="none")
    cmap_disc.set_under(color=BG_COLOR)

    im = ax.imshow(data, extent=[xmin, xmax, ymin, ymax], origin="upper",
                   cmap=cmap_disc, norm=norm,
                   interpolation="nearest", aspect="equal")

    boundary_local.boundary.plot(ax=ax, color=FG_LIGHT, alpha=0.3, linewidth=0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Legend title above colourbar
    fig.text(0.975, 0.935, "My regional needs",
             color=FG_DARK, fontsize=9, fontweight="bold", ha="right", va="bottom")

    # Colourbar — ticks at bin edges, fixed across all frames
    cbar_ax = fig.add_axes([0.50, 0.895, 0.465, 0.025])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="neither")
    cb.ax.xaxis.set_tick_params(color=FG_DARK, labelcolor=FG_DARK, labelsize=8)
    cb.outline.set_edgecolor(FG_DARK)
    tick_pos    = list(range(step, vmax_binned + 1, step))  # [5, 10, ..., vmax]
    tick_labels = [str(t) for t in tick_pos[:-1]] + [f"{tick_pos[-1]} sp."]
    cb.set_ticks(tick_pos)
    cb.set_ticklabels(tick_labels)

    # Date tag
    try:
        date_label = datetime.strptime(week_date, "%Y-%m-%d").strftime("%b-%d")
    except ValueError:
        date_label = week_date
    fig.text(0.02, 0.955, date_label,
             color=FG_LIGHT, fontsize=11, fontweight="bold", ha="left", va="top")

    # Title
    fig.text(0.02, 0.925, "Lifer finder: mapping the birds you've yet to meet",
             color=FG_DARK, fontsize=8, ha="left", va="top")

    # Caption
    fig.text(0.02, 0.068,
             f"Region: {region}  |  Resolution: {resolution}\n"
             "Data: eBird Status & Trends 2023 (Cornell Lab of Ornithology, "
             "ebird.org/science/status-and-trends). "
             "A species is counted as 'possible' if its modelled occurrence "
             "probability > 1% at that location and date.",
             color=FG_DARK, fontsize=4.5, ha="left", va="top", alpha=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def make_gif(frame_dir: Path, gif_path: Path, fps: int = 5) -> None:
    from PIL import Image
    jpgs = sorted(frame_dir.glob("*.jpg"))
    if not jpgs:
        print("  [warn] no frames found for GIF")
        return
    frames = [Image.open(p).convert("RGB") for p in jpgs]
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
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
    raise NotImplementedError(f"No boundary loader for '{region}' yet.")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--regions",    nargs="+", default=["NL", "US"], metavar="CODE")
    parser.add_argument("--resolution", default="3km", choices=["3km", "9km", "27km"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--week",    type=int, metavar="N",
                      help="Single map for week N (1-52). Default 20.")
    mode.add_argument("--animate", action="store_true",
                      help="All 52 frames + animated GIF.")
    parser.add_argument("--fps",    type=int,   default=5,   help="GIF fps.")
    parser.add_argument("--ram-gb", type=float, default=4.0, help="RAM budget per batch (GB).")
    parser.add_argument("--ebird-csv", default="MyEBirdData.csv", metavar="PATH")
    args = parser.parse_args()

    if not args.animate and args.week is None:
        args.week = 20

    config       = load_config()
    modeled      = load_ebirdst_runs()
    api_key      = config["ebird_api_key"]

    print("Fetching eBird taxonomy …")
    name_to_code = get_taxonomy(api_key)

    for region in args.regions:
        print(f"\n{'─'*60}")
        print(f"Region: {region}  |  resolution: {args.resolution}")

        print("  Fetching eBird checklist …")
        regional  = ebird_regional_species(region, api_key)
        user_seen = user_seen_codes(Path(args.ebird_csv), region, name_to_code)
        needed    = (regional - user_seen) & set(modeled) - EXCLUDED_CODES
        print(f"  Checklist: {len(regional)}, seen: {len(user_seen)}, "
              f"needed+modeled: {len(needed)}")

        boundary   = get_boundary(region)
        target_crs = TARGET_CRS.get(region, "EPSG:4326")

        # ------------------------------------------------------------------
        # Always accumulate all 52 weeks — required for a global vmax that
        # is consistent across every frame (both --animate and --week modes).
        # ------------------------------------------------------------------
        stack, tf_src, crs_src, week_dates = accumulate_all_weeks(
            needed, boundary, args.resolution, args.ram_gb,
        )
        global_max  = float(np.nanmax(stack))
        vmax_binned = max(5, int(np.ceil(global_max / 5)) * 5)
        print(f"  Global max: {global_max:.0f}  →  vmax (binned @ 5 sp): {vmax_binned}")

        print(f"  Reprojecting 52 layers to {target_crs} …")
        stack_local, tf_local = reproject_stack(stack, tf_src, crs_src, target_crs)
        del stack
        boundary_local = boundary.to_crs(target_crs)

        if args.animate:
            weekly_dir = OUT_DIR / region / args.resolution / "Weekly_maps"
            n_new = 0
            for i, wd in enumerate(week_dates):
                out_path = weekly_dir / f"{region}_{wd}.jpg"
                if out_path.exists():
                    continue
                make_map(stack_local[i], tf_local, boundary_local,
                         week_date=wd, region=region, resolution=args.resolution,
                         out_path=out_path, vmax_binned=vmax_binned)
                n_new += 1
                if n_new % 10 == 0:
                    print(f"    {n_new} frames saved …")
            print(f"  {n_new} new frames saved." if n_new else "  All frames already exist.")
            del stack_local

            gif_path = (OUT_DIR / region / args.resolution / "Animated_map"
                        / f"{region}_{args.resolution}_animated.gif")
            print("  Assembling GIF …")
            make_gif(weekly_dir, gif_path, fps=args.fps)

        else:
            # Single-week preview — vmax still derived from the full time series
            week_idx  = args.week - 1
            week_date = week_dates[week_idx] if week_idx < len(week_dates) else f"week{args.week}"
            out_path  = OUT_DIR / f"{region}_{args.resolution}_week{args.week:02d}.jpg"
            make_map(stack_local[week_idx], tf_local, boundary_local,
                     week_date=week_date, region=region,
                     resolution=args.resolution, out_path=out_path,
                     vmax_binned=vmax_binned)
            del stack_local
            print(f"  Saved → {out_path}")

    print(f"\nDone — output in {OUT_DIR}/")


if __name__ == "__main__":
    main()
