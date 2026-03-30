#!/usr/bin/env bash
# pipeline.sh — PyLifer step-gated pipeline
#
# Pipeline steps (each is a checkpoint — skipped if already complete):
#
#   STEP 1 — Download tifs       download_ebirdst.py
#   STEP 2 — Preprocess cache    preprocess_region.py  (float32 tif → packbits sp_cache)
#   STEP 3 — Generate maps       map_lifers.py          (animate / weekly frames)
#   STEP 4 — Build score cache   top_spots.py --build-cache  (user-specific lifer pickle)
#
# Completion is tracked by stamp files in .pipeline/<region>/<resolution>/:
#   step1.done   step2.done   step3.done   step4.done
#
# Usage:
#   ./pipeline.sh                                  # US 27km, all steps
#   ./pipeline.sh --regions US --resolutions 3km
#   ./pipeline.sh --from-step 2                    # skip step 1
#   ./pipeline.sh --only-step 2                    # run only step 2
#   ./pipeline.sh --force-step 2                   # re-run step 2 even if stamped
#   ./pipeline.sh --workers 16
#   ./pipeline.sh --no-animate                     # step 3: single-week preview only
#   ./pipeline.sh -h

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Locate Python venv
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in \
    ".venv/bin/python" \
    "../lifeR/.venv/bin/python" \
    "$(command -v python3 2>/dev/null || true)"
do
    if [[ -x "$candidate" ]]; then
        PYTHON="$candidate"
        break
    fi
done
[[ -z "$PYTHON" ]] && { echo "ERROR: no Python found"; exit 1; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REGIONS="US"
RESOLUTIONS="27km"
WORKERS=0            # 0 = let each script use its own default
FROM_STEP=1
ONLY_STEP=""
FORCE_STEP=""
ANIMATE=1
RAM_GB=4.0
THRESHOLD=0.01
EBIRD_CSV="MyEBirdData.csv"
EXTRA_MAP_ARGS=""

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
usage() {
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --regions)
            shift; REGIONS=""
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                REGIONS="${REGIONS:+$REGIONS }$1"; shift
            done ;;
        -r|--resolutions|--resolution)
            shift; RESOLUTIONS=""
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                RESOLUTIONS="${RESOLUTIONS:+$RESOLUTIONS }$1"; shift
            done ;;
        --workers)        shift; WORKERS="$1"; shift ;;
        --from-step)      shift; FROM_STEP="$1"; shift ;;
        --only-step)      shift; ONLY_STEP="$1"; shift ;;
        --force-step)     shift; FORCE_STEP="$1"; shift ;;
        --no-animate)     ANIMATE=0; shift ;;
        --ram-gb)         shift; RAM_GB="$1"; shift ;;
        --threshold)      shift; THRESHOLD="$1"; shift ;;
        --ebird-csv)      shift; EBIRD_CSV="$1"; shift ;;
        -h|--help)        usage ;;
        --) shift; EXTRA_MAP_ARGS="$*"; break ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Normalise: accept space-separated or comma-separated region/resolution lists
REGIONS="${REGIONS//,/ }"
RESOLUTIONS="${RESOLUTIONS//,/ }"

WORKERS_ARG=""
[[ "$WORKERS" -gt 0 ]] && WORKERS_ARG="--workers $WORKERS"

# ---------------------------------------------------------------------------
# Stamp helpers
# ---------------------------------------------------------------------------
STAMP_ROOT=".pipeline"

stamp_dir()  { echo "$STAMP_ROOT/$1/$2"; }   # region resolution
stamp_file() { echo "$STAMP_ROOT/$1/$2/step${3}.done"; }

is_done() {
    local region="$1" res="$2" step="$3"
    [[ -f "$(stamp_file "$region" "$res" "$step")" ]]
}

mark_done() {
    local region="$1" res="$2" step="$3"
    mkdir -p "$(stamp_dir "$region" "$res")"
    date -Iseconds > "$(stamp_file "$region" "$res" "$step")"
}

clear_stamp() {
    local region="$1" res="$2" step="$3"
    rm -f "$(stamp_file "$region" "$res" "$step")"
}

should_run() {
    local step="$1" region="$2" res="$3"
    # --only-step: run nothing else
    [[ -n "$ONLY_STEP" && "$ONLY_STEP" != "$step" ]] && return 1
    # --from-step: skip earlier steps
    [[ "$step" -lt "$FROM_STEP" ]] && return 1
    # --force-step: always run this step
    [[ "$FORCE_STEP" == "$step" ]] && { clear_stamp "$region" "$res" "$step"; return 0; }
    # already stamped
    is_done "$region" "$res" "$step" && return 1
    return 0
}

require_done() {
    local step="$1" region="$2" res="$3"
    if ! is_done "$region" "$res" "$step"; then
        echo ""
        echo "ERROR: Step $step has not completed for $region/$res."
        echo "       Run pipeline.sh --from-step $step --regions $region --resolutions $res"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "════════════════════════════════════════════════════════"
echo "PyLifer pipeline"
echo "  Regions:     $REGIONS"
echo "  Resolutions: $RESOLUTIONS"
echo "  Workers:     ${WORKERS:-auto}"
echo "  Threshold:   $THRESHOLD"
echo "  Animate:     $ANIMATE"
echo "════════════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# Step 1 — Download tifs
# ---------------------------------------------------------------------------
for REGION in $REGIONS; do
    for RES in $RESOLUTIONS; do
        if should_run 1 "$REGION" "$RES"; then
            echo ""
            echo "──── Step 1: Download tifs  [$REGION/$RES] ────"
            # download_ebirdst.py is region/resolution-aware
            # Pass --yes to skip interactive confirmation
            $PYTHON download_ebirdst.py \
                --regions "$REGION" \
                --resolutions "$RES" \
                ${WORKERS_ARG} \
                --yes
            mark_done "$REGION" "$RES" 1
            echo "  ✓ Step 1 complete"
        else
            echo "  ✓ Step 1 [$REGION/$RES] already done — skipping"
        fi
    done
done

# ---------------------------------------------------------------------------
# Step 2 — Preprocess tifs → sp_cache (packbits int8, inside-mask pixels only)
# ---------------------------------------------------------------------------
for REGION in $REGIONS; do
    for RES in $RESOLUTIONS; do
        require_done 1 "$REGION" "$RES"
        if should_run 2 "$REGION" "$RES"; then
            echo ""
            echo "──── Step 2: Preprocess sp_cache  [$REGION/$RES] ────"
            $PYTHON preprocess_region.py \
                --regions "$REGION" \
                --resolutions "$RES" \
                --threshold "$THRESHOLD" \
                ${WORKERS_ARG}
            mark_done "$REGION" "$RES" 2
            echo "  ✓ Step 2 complete"
        else
            echo "  ✓ Step 2 [$REGION/$RES] already done — skipping"
        fi
    done
done

# ---------------------------------------------------------------------------
# Step 3 — Generate maps (frames + GIF animation)
# ---------------------------------------------------------------------------
for REGION in $REGIONS; do
    for RES in $RESOLUTIONS; do
        require_done 2 "$REGION" "$RES"
        if should_run 3 "$REGION" "$RES"; then
            echo ""
            echo "──── Step 3: Generate maps  [$REGION/$RES] ────"
            ANIMATE_ARG="--animate"
            [[ "$ANIMATE" -eq 0 ]] && ANIMATE_ARG="--no-animate"
            $PYTHON map_lifers.py \
                --regions "$REGION" \
                --resolutions "$RES" \
                --offline \
                --ram-gb "$RAM_GB" \
                ${WORKERS_ARG} \
                $ANIMATE_ARG \
                --ebird-csv "$EBIRD_CSV" \
                $EXTRA_MAP_ARGS
            mark_done "$REGION" "$RES" 3
            echo "  ✓ Step 3 complete"
        else
            echo "  ✓ Step 3 [$REGION/$RES] already done — skipping"
        fi
    done
done

# ---------------------------------------------------------------------------
# Step 4 — Build score cache (top_spots user-specific precomputed pickle)
#
# NOTE: This step calls the eBird API to build the user-specific lifer set
# (regional_species - seen - excluded).  Requires internet access.
# Skip with --from-step 3 --only-step 3 to stay fully offline; top_spots.py
# will prompt to build the cache on first use.
# ---------------------------------------------------------------------------
for REGION in $REGIONS; do
    for RES in $RESOLUTIONS; do
        require_done 3 "$REGION" "$RES"
        if should_run 4 "$REGION" "$RES"; then
            echo ""
            echo "──── Step 4: Build score cache  [$REGION/$RES] ────"
            $PYTHON top_spots.py \
                --build-cache \
                --regions "$REGION" \
                --resolution "$RES" \
                --ebird-csv "$EBIRD_CSV"
            mark_done "$REGION" "$RES" 4
            echo "  ✓ Step 4 complete"
        else
            echo "  ✓ Step 4 [$REGION/$RES] already done — skipping"
        fi
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "════════════════════════════════════════════════════════"
echo "Pipeline complete"
echo ""
echo "Step stamps:"
for REGION in $REGIONS; do
    for RES in $RESOLUTIONS; do
        for S in 1 2 3 4; do
            if is_done "$REGION" "$RES" "$S"; then
                STAMP=$(cat "$(stamp_file "$REGION" "$RES" "$S")" 2>/dev/null || echo "done")
                echo "  [$REGION/$RES] step $S ✓  $STAMP"
            else
                echo "  [$REGION/$RES] step $S ✗"
            fi
        done
    done
done
echo ""
echo "Data layout:"
echo "  data/ebirdst/2023/<code>/weekly/*.tif   ← source tifs (step 1)"
echo "  data/sp_cache/<region>/<res>/2023/      ← preprocessed cache (step 2)"
echo "    _meta.npz                               window, mask, transform, threshold"
echo "    <code>.npy                              packbits int8, inside-mask pixels"
echo "  data/score_cache/                       ← top_spots pickle (step 4)"
echo "  results_py/<region>/<res>/              ← map output (step 3)"
