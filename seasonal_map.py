#!/usr/bin/env python3
"""
seasonal_map.py — Four-season lifer opportunity composite map.

For each pixel the peak weekly lifer-candidate count in each season is
computed, then blended into a single RGBA image where hue encodes the
dominant season and brightness encodes intensity (log-scaled).

Season colour key:
  ● Blue   = Winter  (Dec 21 – Feb 21,  weeks 51-52 + 1-8,   idx 0-7, 50-51)
  ● Green  = Spring  (Feb 22 – Jun 5,   weeks 9-23,           idx 8-22)
  ● Red    = Summer  (Jun 6  – Aug 15,  weeks 24-33,          idx 23-32)
  ● Yellow = Fall    (Aug 16 – Dec 20,  weeks 34-50,          idx 33-49)

Style: white background · gray land fill · white NaturalEarth boundary lines.

Usage:
    .venv/bin/python seasonal_map.py
    .venv/bin/python seasonal_map.py --resolution 9km
    .venv/bin/python seasonal_map.py --regions US --resolution 3km
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import geopandas as gpd

from map_lifers import (
    load_config,
    load_ebirdst_runs,
    get_taxonomy,
    ebird_regional_species,
    user_seen_codes,
    get_us_boundary,
    get_nl_boundary,
    accumulate_all_weeks,
    reproject_stack,
    EXCLUDED_CODES,
    _FRAME_WIDTH_PX,
    OUT_DIR,
)

# ---------------------------------------------------------------------------
# Season definitions — 0-indexed into the 52-week stack
# Week 1 ≈ Jan 4; Week 52 ≈ Dec 27.  Dec 21 ≈ week 51; Feb 21 ≈ week 8.
# ---------------------------------------------------------------------------
SEASONS: dict[str, list[int]] = {
    "winter": list(range(0, 8)) + list(range(50, 52)),  # wks  1-8  + 51-52
    "spring": list(range(8, 23)),                         # wks  9-23
    "summer": list(range(23, 33)),                        # wks 24-33
    "fall":   list(range(33, 50)),                        # wks 34-50
}

# Saturated linear-light RGB so colours stay vivid after log-brightness scaling
SEASON_RGB: dict[str, tuple[float, float, float]] = {
    "winter": (0.12, 0.38, 0.95),   # blue
    "spring": (0.12, 0.80, 0.22),   # green
    "summer": (0.92, 0.14, 0.10),   # red
    "fall":   (0.95, 0.78, 0.04),   # yellow
}

SEASON_LABEL: dict[str, str] = {
    "winter": "Winter\nDec 21 – Feb 21",
    "spring": "Spring\nFeb 22 – Jun 5",
    "summer": "Summer\nJun 6 – Aug 15",
    "fall":   "Fall\nAug 16 – Dec 20",
}

_LAND_COLOR = "#BEBEBE"   # medium gray land fill
_BG_COLOR   = "white"
_LINE_COLOR = "white"


# ---------------------------------------------------------------------------
# CRS / boundary helpers
# ---------------------------------------------------------------------------
def _target_crs(region: str) -> str:
    if region == "NL":
        return "EPSG:28992"
    return "EPSG:5070"   # CONUS Albers


def _get_boundary(region: str) -> gpd.GeoDataFrame:
    if region == "NL":
        return get_nl_boundary()
    if region == "US":
        return get_us_boundary()
    if region.startswith("US-"):
        return get_us_boundary(state_code=region)
    raise NotImplementedError(f"No boundary loader for '{region}'")


def _raster_extent(transform, h: int, w: int) -> tuple:
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + transform.a * w
    ymin = ymax + transform.e * h
    return xmin, xmax, ymin, ymax


# ---------------------------------------------------------------------------
# Seasonal RGBA composite
# ---------------------------------------------------------------------------
def build_seasonal_rgba(stack: np.ndarray) -> np.ndarray:
    """Convert (52, H, W) weekly float32 stack → (H, W, 4) RGBA composite.

    Per-pixel colour = weighted average of the four season colours where the
    weight for each season is its peak weekly count at that pixel.
    Brightness is log-scaled from the sum of all four season peaks.
    Alpha = 1 where any season has a positive count, 0 elsewhere.
    """
    peaks = {
        name: np.nanmax(stack[idx], axis=0).astype(np.float32)
        for name, idx in SEASONS.items()
    }
    total = sum(peaks.values())           # (H, W), NaN propagated
    max_total = float(np.nanmax(total))
    if max_total <= 0:
        H, W = stack.shape[1:]
        return np.zeros((H, W, 4), dtype=np.float32)

    # Log brightness [0, 1]
    brightness = (np.log1p(total) / np.log1p(max_total)).astype(np.float32)

    # Safe denominator for proportional colour blending
    denom = np.where(total > 0, total, 1.0).astype(np.float32)

    # Weighted-average season colours per pixel
    H, W = total.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for name, peak in peaks.items():
        weight = (peak / denom)[..., np.newaxis]       # (H, W, 1)
        rgb += weight * np.array(SEASON_RGB[name], dtype=np.float32)

    # Apply log brightness
    rgb *= brightness[..., np.newaxis]
    np.clip(rgb, 0.0, 1.0, out=rgb)

    # Alpha: 1 wherever at least one season has a positive count
    alpha = np.where(
        np.isfinite(total) & (total > 0), 1.0, 0.0
    ).astype(np.float32)

    return np.concatenate([rgb, alpha[..., np.newaxis]], axis=-1)


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------
def make_seasonal_map(
    rgba: np.ndarray,           # (H, W, 4)
    transform,
    states_local: gpd.GeoDataFrame,
    out_path: Path,
    region: str,
    resolution: str,
    username: str,
) -> None:
    h, w = rgba.shape[:2]
    xmin, xmax, ymin, ymax = _raster_extent(transform, h, w)
    bx_min, by_min, bx_max, by_max = states_local.total_bounds

    # ── Figure geometry ──────────────────────────────────────────────────
    fig_w_in   = _FRAME_WIDTH_PX / 160.0   # e.g. 1920/160 = 12 in
    map_aspect = (bx_max - bx_min) / (by_max - by_min)
    map_h_in   = (fig_w_in - 2 * 0.15) / map_aspect
    pad_in     = 0.15
    header_in  = 0.60    # title + subtitle above map
    legend_in  = 0.80    # season swatches below map
    fig_h_in   = header_in + map_h_in + legend_in + 2 * pad_in

    pad_h   = pad_in    / fig_w_in
    pad_v   = pad_in    / fig_h_in
    leg_f   = legend_in / fig_h_in
    map_f   = map_h_in  / fig_h_in
    hdr_f   = header_in / fig_h_in
    map_bot = leg_f + pad_v

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor=_BG_COLOR)

    # ── Map axes ─────────────────────────────────────────────────────────
    ax = fig.add_axes([pad_h, map_bot, 1.0 - 2 * pad_h, map_f])
    ax.set_facecolor(_BG_COLOR)
    ax.axis("off")

    # Gray land fill (behind everything)
    states_local.plot(ax=ax, color=_LAND_COLOR, linewidth=0, zorder=1)

    # Seasonal RGBA composite — transparent outside boundary so land shows through
    ax.imshow(rgba, extent=[xmin, xmax, ymin, ymax],
              origin="upper", interpolation="nearest", zorder=2)

    # Internal state/province boundary lines
    states_local.boundary.plot(ax=ax, color=_LINE_COLOR, linewidth=0.35, zorder=3)
    # Outer boundary slightly bolder
    states_local.dissolve().boundary.plot(ax=ax, color=_LINE_COLOR, linewidth=0.80, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(bx_min, bx_max)
    ax.set_ylim(by_min, by_max)

    # ── Header (above map) ───────────────────────────────────────────────
    hdr_bot_f = map_bot + map_f
    y_title   = hdr_bot_f + pad_v + hdr_f * 0.62
    y_sub     = hdr_bot_f + pad_v + hdr_f * 0.18

    fig.text(pad_h, y_title,
             "Seasonal Lifer Opportunities",
             color="#111111", fontsize=11, fontweight="bold",
             ha="left", va="bottom", transform=fig.transFigure)
    fig.text(pad_h, y_sub,
             f"{username}  ·  {region}  ·  {resolution}  ·  eBird Status & Trends 2023",
             color="#555555", fontsize=7.5,
             ha="left", va="bottom", transform=fig.transFigure)

    # ── Season legend (below map) ────────────────────────────────────────
    n = len(SEASONS)
    col_w    = (1.0 - 2 * pad_h) / n
    sw_h     = legend_in * 0.30 / fig_h_in
    sw_w     = col_w * 0.20
    lbl_y    = pad_v * 1.1              # bottom of text block
    swatch_y = lbl_y + legend_in * 0.36 / fig_h_in   # centre of swatch

    for i, (name, color) in enumerate(SEASON_RGB.items()):
        cx = pad_h + (i + 0.5) * col_w

        rect = mpatches.FancyBboxPatch(
            (cx - sw_w / 2, swatch_y - sw_h / 2),
            sw_w, sw_h,
            boxstyle="round,pad=0.003",
            transform=fig.transFigure, figure=fig,
            facecolor=color, edgecolor="none", zorder=5,
        )
        fig.add_artist(rect)

        fig.text(cx, lbl_y, SEASON_LABEL[name],
                 color="#333333", fontsize=7.5, ha="center", va="bottom",
                 transform=fig.transFigure, linespacing=1.4)

    # ── Save ─────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dpi = _FRAME_WIDTH_PX / fig_w_in
    fig.savefig(out_path, dpi=out_dpi, facecolor=_BG_COLOR)
    plt.close(fig)
    print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seasonal lifer opportunity composite map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--regions",    nargs="+", default=["US"], metavar="CODE")
    parser.add_argument("--resolution", default="27km",
                        choices=["3km", "9km", "27km"])
    parser.add_argument("--ebird-csv",  default="MyEBirdData.csv", metavar="PATH")
    parser.add_argument("--ram-gb",     type=float, default=4.0)
    parser.add_argument("--workers",    type=int,   default=None)
    args = parser.parse_args()

    config   = load_config()
    modeled  = load_ebirdst_runs()
    username = config.get("user", "unknown")
    api_key  = config["ebird_api_key"]

    t0 = time.perf_counter()
    print("Fetching eBird taxonomy …")
    name_to_code = get_taxonomy(api_key)

    for region in args.regions:
        print(f"\n{'═'*60}")
        print(f"Region: {region}  Resolution: {args.resolution}")
        t = time.perf_counter()

        regional  = ebird_regional_species(region, api_key)
        user_seen = user_seen_codes(Path(args.ebird_csv), region, name_to_code)
        needed    = ((regional - user_seen) & set(modeled)) - EXCLUDED_CODES
        print(f"  Checklist: {len(regional)}, seen: {len(user_seen)}, needed: {len(needed)}")

        boundary   = _get_boundary(region)
        target_crs = _target_crs(region)

        stack, tf_src, crs_src, week_dates, _, _, available, _ = accumulate_all_weeks(
            needed, boundary, args.resolution, args.ram_gb,
            workers=args.workers, region=region,
        )
        print(f"  Accumulated {len(week_dates)} weeks, {len(available)} species "
              f"({time.perf_counter() - t:.1f}s)")
        t = time.perf_counter()

        print(f"  Reprojecting to {target_crs} …")
        stack_local, tf_local = reproject_stack(
            stack, tf_src, crs_src, target_crs, workers=args.workers,
        )
        del stack
        print(f"  Reprojection done ({time.perf_counter() - t:.1f}s)")
        t = time.perf_counter()

        # Project the states/boundary GDF for gray fill + white lines
        states_local = boundary.to_crs(target_crs)

        print("  Building seasonal RGBA composite …")
        rgba = build_seasonal_rgba(stack_local)
        del stack_local
        print(f"  Composite built ({time.perf_counter() - t:.1f}s)")

        out_path = (
            OUT_DIR / region / args.resolution
            / f"{region}_{args.resolution}_seasonal.png"
        )
        make_seasonal_map(
            rgba, tf_local, states_local,
            out_path, region, args.resolution, username,
        )

    print(f"\nDone ({time.perf_counter() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
