#!/usr/bin/env bash
# bench_workers.sh — adaptive worker-count benchmark for map_lifers.py
#
# Usage:
#   ./bench_workers.sh [RESOLUTION] [--mode accumulate-only|no-animate|animate]
#   ./bench_workers.sh 3km --mode accumulate-only   # fastest: pure sp_cache/tif read
#   ./bench_workers.sh 3km --mode no-animate         # accumulate + reproject + 1 frame (default)
#   ./bench_workers.sh 3km --mode animate            # full 52-frame pipeline
#
# Modes:
#   accumulate-only  — exits right after accumulate_all_weeks(); no reproject, no render.
#                      Best for isolating sp_cache / tif read thread scaling.
#   no-animate       — accumulate + reproject + render week 20 (default; original behaviour).
#   animate          — full 52-week animation + GIF.
#
# Strategy:
#   Phase 1 — probe 8 and 36
#   Phase 2 — always probe 24, then one guided probe:
#               36 won, 24 lost  → probe 40  (explore above 36)
#               36 won, 24 won   → skip guided probe, enter Phase 3
#               8  won, 24 won   → probe 16  (narrow 8..24)
#               8  won, 24 lost  → probe 30  (check mid of 24..36)
#   Phase 3 — general binary-search narrowing until both neighbours are
#              within MIN_STEP workers of the current best (optimum confirmed).

set -euo pipefail
cd "$(dirname "$0")"

# ── argument parsing ───────────────────────────────────────────────────────────
RES="9km"
MODE="no-animate"   # default

_positional=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --mode=*)     MODE="${1#--mode=}"; shift ;;
        --resolution) RES="$2"; shift 2 ;;
        --resolution=*) RES="${1#--resolution=}"; shift ;;
        -*)           echo "Unknown flag: $1"; exit 1 ;;
        *)
            if [[ $_positional -eq 0 ]]; then
                RES="$1"; _positional=1
            fi
            shift ;;
    esac
done
# Normalise: "3" → "3km"
[[ "$RES" =~ ^[0-9]+$ ]] && RES="${RES}km"

# Validate mode
case "$MODE" in
    accumulate-only|no-animate|animate) ;;
    *) echo "Unknown --mode '$MODE'. Choose: accumulate-only | no-animate | animate"; exit 1 ;;
esac

# Map mode → map_lifers.py flags
case "$MODE" in
    accumulate-only) MODE_FLAGS="--accumulate-only" ;;
    no-animate)      MODE_FLAGS="--no-animate" ;;
    animate)         MODE_FLAGS="" ;;
esac

MIN_WORKERS=4
MAX_WORKERS=64
MIN_STEP=4        # stop narrowing when gap between tested neighbours ≤ this
MAX_REFINE=8      # safety cap on Phase-3 probes

LOGDIR="benchmark_results/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

declare -A TIMES  # worker_count → wall_seconds (float string)

# ── helpers ────────────────────────────────────────────────────────────────────

# Parse "h:mm:ss.s" or "m:ss.s"  →  total seconds (float)
wall_to_sec() {
    local t="$1"
    if [[ "$t" =~ ^([0-9]+):([0-9]+):([0-9]+\.?[0-9]*)$ ]]; then
        awk "BEGIN{printf \"%.3f\", ${BASH_REMATCH[1]}*3600+${BASH_REMATCH[2]}*60+${BASH_REMATCH[3]}}"
    elif [[ "$t" =~ ^([0-9]+):([0-9]+\.?[0-9]*)$ ]]; then
        awk "BEGIN{printf \"%.3f\", ${BASH_REMATCH[1]}*60+${BASH_REMATCH[2]}}"
    else
        echo "99999"
    fi
}

# Return 0 (true) if float $1 < float $2
is_lt() { awk "BEGIN{exit !($1 < $2)}"; }

# Echo the worker count with the lowest wall time
find_best() {
    local bw="" bt="99999"
    for w in "${!TIMES[@]}"; do
        is_lt "${TIMES[$w]}" "$bt" && { bt="${TIMES[$w]}"; bw="$w"; }
    done
    printf '%s' "$bw"
}

# Run one benchmark at worker count $1, store result in TIMES[$1]
run_probe() {
    local W="$1"
    local LOG="$LOGDIR/workers_${W}.txt"
    local tlist
    tlist=$(printf '%s\n' "${!TIMES[@]}" | sort -n | tr '\n' ' ')

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    printf "  Probing workers=%-4s  (tested so far: %s)\n" "$W" "$tlist"
    echo "──────────────────────────────────────────────────────────────"

    rm -rf "results_py/US/$RES/Weekly_maps/"*.png \
             "results_py/US/$RES/Weekly_maps_compact/"*.png \
             "results_py/US/$RES/Animated_map/" 2>/dev/null || true
    sync

    # Build the command as an array so MODE_FLAGS (which may be empty) expands cleanly
    local CMD=(
        /usr/bin/time -v .venv/bin/python -u map_lifers.py
        --regions US
        --resolutions "$RES"
        --offline
        --workers "$W"
        --batch-size 40
        --overwrite-frames
    )
    [[ -n "$MODE_FLAGS" ]] && CMD+=("$MODE_FLAGS")

    "${CMD[@]}" 2>&1 | tee "$LOG"

    local ELAPSED
    ELAPSED=$(grep "Elapsed (wall clock)" "$LOG" | grep -oP '[\d:\.]+$' || true)
    TIMES[$W]=$(wall_to_sec "${ELAPSED:-}")

    echo "  ✓ workers=$W  wall=${TIMES[$W]}s"
    sleep 5
}

# ── Banner ─────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "Adaptive worker benchmark   resolution=$RES   cores=$(nproc)"
echo "Mode: $MODE"
echo "Phase 1: 8 36  |  Phase 2: 24 + guided probe  |  Phase 3: binary narrowing"
echo "Logs: $LOGDIR"
echo "════════════════════════════════════════════════════════════"

# ── Phase 1: two anchor probes ─────────────────────────────────────────────────
echo ""
echo "════ Phase 1 ════════════════════════════════════════════════"
run_probe 8
run_probe 36
P1_WINNER=8
is_lt "${TIMES[36]}" "${TIMES[8]}" && P1_WINNER=36
echo ""
echo "  Phase-1 winner: $P1_WINNER workers  (8=${TIMES[8]}s  36=${TIMES[36]}s)"

# ── Phase 2: probe midpoint (24), then one guided probe ───────────────────────
echo ""
echo "════ Phase 2 ════════════════════════════════════════════════"
run_probe 24
BEST=$(find_best)
echo "  Best after probing 24: $BEST workers"

if [[ "$P1_WINNER" -eq 36 ]]; then
    if [[ "$BEST" -ne 24 ]]; then
        # 36 won phase 1, and 24 didn't beat it → look higher
        echo "  → 36 fastest, 24 lost — probing 40"
        run_probe 40
    else
        echo "  → 36 won phase 1 but 24 is now fastest — entering Phase 3"
    fi
else
    # 8 won phase 1
    if [[ "$BEST" -eq 24 ]]; then
        # 24 beat 8 → narrow between 8 and 24
        echo "  → 8 won phase 1 but 24 is faster — probing 16"
        run_probe 16
    else
        # 8 still best, 24 lost → check middle of 24..36
        echo "  → 8 fastest, 24 lost — probing 30 (mid of 24..36)"
        run_probe 30
    fi
fi

# ── Phase 3: general binary-search narrowing ───────────────────────────────────
echo ""
echo "════ Phase 3 — adaptive narrowing ══════════════════════════"

for (( R=1; R<=MAX_REFINE; R++ )); do
    BEST=$(find_best)
    mapfile -t S < <(printf '%s\n' "${!TIMES[@]}" | sort -n)
    N=${#S[@]}

    # Locate BEST in the sorted array
    IDX=-1
    for i in "${!S[@]}"; do [[ "${S[$i]}" -eq "$BEST" ]] && IDX=$i && break; done

    # Gap to nearest tested neighbours
    if [[ $IDX -lt $((N-1)) ]]; then GAP_UP=$(( ${S[$((IDX+1))]} - BEST )); else GAP_UP=99999; fi
    if [[ $IDX -gt 0 ]];        then GAP_DN=$(( BEST - ${S[$((IDX-1))]} )); else GAP_DN=99999; fi

    # A side is confirmed when its gap to the tested neighbour is ≤ MIN_STEP
    # (or we've hit the hard boundary on that side)
    CONF_UP=0; CONF_DN=0
    if   [[ $IDX -eq $((N-1)) && $BEST -ge $MAX_WORKERS ]]; then CONF_UP=1
    elif [[ $IDX -lt $((N-1)) && $GAP_UP -le $MIN_STEP  ]]; then CONF_UP=1; fi
    if   [[ $IDX -eq 0        && $BEST -le $MIN_WORKERS ]]; then CONF_DN=1
    elif [[ $IDX -gt 0        && $GAP_DN -le $MIN_STEP  ]]; then CONF_DN=1; fi

    if [[ $CONF_UP -eq 1 && $CONF_DN -eq 1 ]]; then
        echo "  → Optimum confirmed: $BEST workers (both neighbours within ${MIN_STEP}w)"
        break
    fi

    # Pick the unconfirmed side with the larger gap
    NEXT=""
    PROBE_UP=0; PROBE_DN=0
    [[ $CONF_UP -eq 0 ]] && PROBE_UP=1
    [[ $CONF_DN -eq 0 ]] && PROBE_DN=1
    # If both sides open, pick the bigger gap
    if [[ $PROBE_UP -eq 1 && $PROBE_DN -eq 1 ]]; then
        is_lt "$GAP_UP" "$GAP_DN" && PROBE_UP=0 || PROBE_DN=0
    fi

    if [[ $PROBE_UP -eq 1 ]]; then
        if [[ $IDX -eq $((N-1)) ]]; then
            C=$((BEST + 4))
            [[ $C -le $MAX_WORKERS && -z "${TIMES[$C]+x}" ]] && NEXT=$C || CONF_UP=1
        else
            MID=$(( (BEST + ${S[$((IDX+1))]}) / 2 ))
            [[ $MID -ne $BEST && -z "${TIMES[$MID]+x}" ]] && NEXT=$MID || CONF_UP=1
        fi
    elif [[ $PROBE_DN -eq 1 ]]; then
        if [[ $IDX -eq 0 ]]; then
            C=$((BEST - 4))
            [[ $C -ge $MIN_WORKERS && -z "${TIMES[$C]+x}" ]] && NEXT=$C || CONF_DN=1
        else
            MID=$(( (${S[$((IDX-1))]} + BEST) / 2 ))
            [[ $MID -ne $BEST && -z "${TIMES[$MID]+x}" ]] && NEXT=$MID || CONF_DN=1
        fi
    fi

    if [[ -z "$NEXT" ]]; then
        echo "  → No useful probe remaining. Optimum: $BEST workers"
        break
    fi

    GAP_UP_D=$GAP_UP; [[ $GAP_UP -eq 99999 ]] && GAP_UP_D="open"
    GAP_DN_D=$GAP_DN; [[ $GAP_DN -eq 99999 ]] && GAP_DN_D="open"
    echo "  → Round $R: best=$BEST (up-gap=${GAP_UP_D}w dn-gap=${GAP_DN_D}w) → probing $NEXT"
    run_probe "$NEXT"

    NEW_BEST=$(find_best)
    if [[ "$NEW_BEST" != "$BEST" ]]; then
        echo "  → Improved: $BEST → $NEW_BEST workers"
    else
        echo "  → $NEXT did not beat $BEST workers"
    fi
done

# ── Summary ────────────────────────────────────────────────────────────────────
FINAL=$(find_best)
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Results (sorted by worker count)"
echo "════════════════════════════════════════════════════════════"
mapfile -t ALL < <(printf '%s\n' "${!TIMES[@]}" | sort -n)
for W in "${ALL[@]}"; do
    TAG=""; [[ "$W" -eq "$FINAL" ]] && TAG="  ◀ OPTIMAL"
    RSS=$(grep "Maximum resident" "$LOGDIR/workers_${W}.txt" | grep -oP '\d+' || echo "?")
    printf "  workers=%-4s  wall=%ss  maxRSS=%skB%s\n" "$W" "${TIMES[$W]}" "$RSS" "$TAG"
done
echo ""
echo "Optimal worker count: $FINAL"
echo "Full logs: $LOGDIR/"

