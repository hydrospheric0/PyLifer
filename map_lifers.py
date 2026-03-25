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
import time
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
# Colormap: gist_rainbow with PowerNorm — bright rainbow, high values pop.
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
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
        NE_DIR / "ne_10m_admin_0_countries",
    )
    world = gpd.read_file(shp)
    col = "ISO_A3" if "ISO_A3" in world.columns else "sov_a3"
    return world[world[col] == "NLD"].to_crs("EPSG:4326").copy()


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

    # ------------------------------------------------------------------
    # Pre-filter: drop species whose max occurrence within the masked
    # region never exceeds OCCURRENCE_THRESH (mirrors R's
    # filter_rasters_to_sp_above_threshold called after crop + after mask).
    # Each thread reads all bands for the window then checks the masked max.
    # This is cheap because we short-circuit to np.max without accumulating.
    # ------------------------------------------------------------------
    def _exceeds_threshold(code: str) -> bool:
        try:
            with rasterio.open(tif_path(code, resolution)) as src:
                data = src.read(window=win).astype(np.float32)
            # set outside-boundary pixels to NaN, then check max
            data[:, outside_mask] = np.nan
            return bool(np.nanmax(data) > OCCURRENCE_THRESH)
        except Exception:
            return False

    workers = min(N_WORKERS, len(available))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        keep_flags = list(pool.map(_exceeds_threshold, available))

    available_filtered = [c for c, keep in zip(available, keep_flags) if keep]
    n_dropped = len(available) - len(available_filtered)
    print(f"  Pre-filter: kept {len(available_filtered)} / {len(available)} "
          f"species (dropped {n_dropped} below {OCCURRENCE_THRESH*100:.0f}% threshold)")
    available = available_filtered

    if not available:
        raise RuntimeError("No species exceed the occurrence threshold in this region.")

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


def _fig_geometry(data: np.ndarray, transform) -> tuple:
    """Layout constants for make_map (figure geometry from raster shape/transform)."""
    h, w = data.shape
    xmin, xmax, ymin, ymax = _raster_extent(transform, h, w)
    pad_in    = 0.15                          # uniform margin, all four sides (inches)
    fig_w     = 6.6
    cw_in     = fig_w - 2.0 * pad_in         # content width for the map (6.3 in)
    data_aspect = (xmax - xmin) / (ymax - ymin)
    map_h_in   = cw_in / data_aspect          # map height: data fills the padded width
    header_in  = 0.66                         # header band: top-margin + two rows + row-gap
    cap_in     = 1.05                         # caption band: sep + text + bottom-margin
    fig_h      = map_h_in + header_in + cap_in
    cap_frac   = cap_in   / fig_h
    map_frac   = map_h_in / fig_h
    hdr_bot    = cap_frac + map_frac
    pad_h      = pad_in / fig_w               # normalized horizontal pad
    pad_v      = pad_in / fig_h               # normalized vertical pad
    cb_h_n     = 0.053 / fig_h               # colorbar bar height (normalized)
    # Row 2 (date + colorbar): pad_in above map top
    cb_y       = hdr_bot + pad_v
    # Row 1 (title + legend label): pad_in below figure top
    y_r1       = 1.0 - pad_v
    return fig_w, fig_h, cap_in, cap_frac, map_frac, hdr_bot, pad_h, pad_v, cb_h_n, cb_y, y_r1


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
        _fig_geometry(data, transform)
    h, w = data.shape
    xmin, xmax, ymin, ymax = _raster_extent(transform, h, w)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    # Map axes: pad_in on each side so the map content has uniform side margins
    ax  = fig.add_axes([pad_h, cap_frac, 1.0 - 2*pad_h, map_frac])
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # turbo colormap, blue (low) → red (high), linear scale.
    cmap = plt.cm.turbo.with_extremes(bad="none")
    norm = mcolors.Normalize(vmin=0, vmax=vmax_binned)

    im = ax.imshow(data, extent=[xmin, xmax, ymin, ymax], origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest")

    boundary_local.boundary.plot(ax=ax, color=FG_LIGHT, alpha=0.3, linewidth=0.5)
    ax.set_aspect("equal")   # re-assert after geopandas (which resets it)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Title (row 1, left — white) — pad_in from figure top
    fig.text(pad_h, y_r1, "Lifer finder: mapping the birds you've yet to meet",
             color=FG_LIGHT, fontsize=8, ha="left", va="top")

    # Legend label (row 1, right) — pad_in from figure right
    fig.text(1.0 - pad_h, y_r1, "My regional needs",
             color=FG_DARK, fontsize=9, fontweight="bold", ha="right", va="top")

    # Colorbar (row 2, right) — right edge inset by 3× pad so last tick label fits
    step = 5
    tick_pos = list(range(0, vmax_binned, step))
    if not tick_pos or tick_pos[-1] != vmax_binned:
        tick_pos.append(vmax_binned)
    tick_labels = [str(t) for t in tick_pos[:-1]] + [f"{tick_pos[-1]} sp."]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_left  = 0.50
    cbar_right = 1.0 - 3 * pad_h
    cbar_ax = fig.add_axes([cbar_left, cb_y, cbar_right - cbar_left, cb_h_n])
    cbar_ax.set_facecolor(BG_COLOR)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="neither")
    cb.ax.tick_params(which="both", top=False, bottom=True, length=3,
                      color=FG_DARK, labelcolor=FG_DARK, labelsize=7)
    cb.outline.set_visible(False)
    cb.set_ticks(tick_pos)
    cb.set_ticklabels(tick_labels)
    for tp in tick_pos:
        cbar_ax.axvline(tp, color=BG_COLOR, linewidth=0.6, ymin=0, ymax=1)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)


def make_gif_lores(frame_dir: Path, gif_path: Path, fps: int = 5, scale: float = 0.38) -> None:
    """Assemble a downscaled GIF at `scale` fraction of original size.

    Reuses the same global-palette strategy as make_gif so the colorbar
    encodes identically across frames.
    """
    from PIL import Image

    jpgs = sorted(frame_dir.glob("*.jpg"))
    if not jpgs:
        print("  [warn] no frames found for lo-res GIF")
        return

    first = Image.open(jpgs[0])
    new_w = round(first.width  * scale)
    new_h = round(first.height * scale)
    first.close()

    frames = [
        Image.open(p).convert("RGB").resize((new_w, new_h), Image.LANCZOS)
        for p in jpgs
    ]

    sample = frames[::4]
    combined = Image.new("RGB", (new_w, new_h * len(sample)))
    for idx, img in enumerate(sample):
        combined.paste(img, (0, idx * new_h))
    palette_img = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0)
    del combined

    quantized = [f.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
                 for f in frames]

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        gif_path, save_all=True, append_images=quantized[1:],
        loop=0, duration=max(20, 1000 // fps), optimize=False,
    )
    print(f"  Lo-res GIF ({len(frames)} frames, "
          f"{gif_path.stat().st_size / 1e6:.1f} MB) → {gif_path}")


def make_gif(frame_dir: Path, gif_path: Path, fps: int = 5) -> None:
    """Assemble frames into a GIF using a single global palette.

    PIL's default behaviour quantizes each frame to 256 colours independently,
    which produces different dithering patterns in the colorbar (a smooth turbo
    gradient) on every frame and causes visible flicker.  R's magick/gifski
    backend avoids this by building one global palette across the whole sequence.
    We replicate that here: derive the palette from all frames combined, then
    apply that one palette to every frame so the colorbar encodes identically.
    """
    from PIL import Image

    jpgs = sorted(frame_dir.glob("*.jpg"))
    if not jpgs:
        print("  [warn] no frames found for GIF")
        return

    frames = [Image.open(p).convert("RGB") for p in jpgs]

    # ------------------------------------------------------------------ #
    # Build a global 256-colour palette from a representative sample of
    # the whole sequence (every 4th frame keeps memory reasonable).
    # ------------------------------------------------------------------ #
    sample_pixels: list[Image.Image] = frames[::4]
    combined_w = frames[0].width
    combined_h = frames[0].height * len(sample_pixels)
    combined = Image.new("RGB", (combined_w, combined_h))
    for idx, img in enumerate(sample_pixels):
        combined.paste(img, (0, idx * frames[0].height))
    palette_img = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0)
    del combined

    # Apply the same palette to every frame (Floyd-Steinberg for quality;
    # deterministic for the colorbar because its pixels are identical each frame).
    quantized = [f.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
                 for f in frames]

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
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip all eBird API calls. Use every locally cached tif as 'needed'. "
             "Useful for timing tests and offline development.",
    )
    args = parser.parse_args()

    if not args.animate and args.week is None:
        args.week = 20

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
        print(f"\n{'─'*60}")
        print(f"Region: {region}  |  resolution: {args.resolution}")
        t = time.perf_counter()

        if args.offline:
            # Use every modeled species that has a tif at the requested resolution
            needed = {
                c for c in modeled
                if c not in EXCLUDED_CODES
                and tif_path(c, args.resolution).exists()
            }
            print(f"  [offline] {len(needed)} species with local {args.resolution} tifs")
            t = _lap("species selection (offline)", t)
        else:
            print("  Fetching eBird checklist …")
            regional  = ebird_regional_species(region, api_key)
            t = _lap("checklist fetch", t)
            user_seen = user_seen_codes(Path(args.ebird_csv), region, name_to_code)
            needed    = (regional - user_seen) & set(modeled) - EXCLUDED_CODES
            print(f"  Checklist: {len(regional)}, seen: {len(user_seen)}, "
                  f"needed+modeled: {len(needed)}")
            t = _lap("user-list diff", t)

        boundary   = get_boundary(region)
        target_crs = _target_crs(region)
        t = _lap("boundary load", t)

        # ------------------------------------------------------------------
        # Always accumulate all 52 weeks — required for a global vmax that
        # is consistent across every frame (both --animate and --week modes).
        # ------------------------------------------------------------------
        stack, tf_src, crs_src, week_dates = accumulate_all_weeks(
            needed, boundary, args.resolution, args.ram_gb,
        )
        global_max  = float(np.nanmax(stack))
        vmax_binned = 35
        print(f"  Global max: {global_max:.0f}  →  vmax: {vmax_binned}")
        t = _lap("raster accumulation", t)

        print(f"  Reprojecting 52 layers to {target_crs} …")
        stack_local, tf_local = reproject_stack(stack, tf_src, crs_src, target_crs)
        del stack
        boundary_local = boundary.to_crs(target_crs)
        t = _lap("reprojection", t)

        if args.animate:
            weekly_dir = OUT_DIR / region / args.resolution / "Weekly_maps"

            n_new = 0
            for i, wd in enumerate(week_dates):
                out_path = weekly_dir / f"{region}_{wd}.jpg"
                if out_path.exists():
                    continue
                make_map(stack_local[i], tf_local, boundary_local,
                         week_date=wd, region=region, resolution=args.resolution,
                         out_path=out_path, vmax_binned=vmax_binned,
                         username=username)
                n_new += 1
                if n_new % 10 == 0:
                    print(f"    {n_new} frames saved …")
            print(f"  {n_new} new frames saved." if n_new else "  All frames already exist.")
            del stack_local
            t = _lap("frame rendering", t)

            gif_path = (OUT_DIR / region / args.resolution / "Animated_map"
                        / f"{region}_{args.resolution}_animated.gif")
            gif_lores_path = (OUT_DIR / region / args.resolution / "Animated_map"
                              / f"{region}_{args.resolution}_animated_lores.gif")
            print("  Assembling GIF …")
            make_gif(weekly_dir, gif_path, fps=args.fps)
            t = _lap("GIF assembly", t)

            print("  Assembling lo-res GIF (38%) …")
            make_gif_lores(weekly_dir, gif_lores_path, fps=args.fps)
            t = _lap("lo-res GIF assembly", t)

        else:
            # Single-week preview — vmax still derived from the full time series
            week_idx  = args.week - 1
            week_date = week_dates[week_idx] if week_idx < len(week_dates) else f"week{args.week}"
            out_path  = OUT_DIR / f"{region}_{args.resolution}_week{args.week:02d}.jpg"
            make_map(stack_local[week_idx], tf_local, boundary_local,
                     week_date=week_date, region=region,
                     resolution=args.resolution, out_path=out_path,
                     vmax_binned=vmax_binned, username=username)
            del stack_local
            print(f"  Saved → {out_path}")
            _lap("map render", t)

    total = time.perf_counter() - t_wall
    print(f"\nDone — output in {OUT_DIR}/  (total {total:.1f}s)")


if __name__ == "__main__":
    main()
