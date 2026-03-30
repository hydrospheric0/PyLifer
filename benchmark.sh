#!/usr/bin/env bash
# benchmark.sh — Full multi-resolution benchmark: map_lifers.py (PyLifer) vs
#                map_lifers.R (R port).
#
# Runs 27km → 9km → 3km for a single region.
# Each resolution is timed independently (one Python call + one R call).
# Default mode: full 52-week animation + both GIFs.
# QUICK=1: week 20 only, no GIF (fast ad-hoc check).
#
# Usage:
#   ./benchmark.sh [REGION] [RAMP]
#   QUICK=1 ./benchmark.sh US colOriginal
#
# Defaults: US      colOriginal

set -euo pipefail

CHILD_PID=""

cleanup_child() {
  if [ -n "${CHILD_PID}" ] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    # Kill the whole process group for the active timed command.
    kill -TERM -- "-${CHILD_PID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${CHILD_PID}" 2>/dev/null || true
  fi
}

on_interrupt() {
  echo ""
  echo "[interrupt] Ctrl+C received — stopping benchmark..."
  cleanup_child
  exit 130
}

trap on_interrupt INT TERM

REGION="${1:-US}"
RAMP="${2:-colOriginal}"
RESOLUTIONS=("27km" "9km" "3km")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
QUICK="${QUICK:-0}"

OUT_LOG="benchmark_results/${TIMESTAMP}_${REGION}_${RAMP}"
TIME_BIN="/usr/bin/time"
RESULTS_MD="BENCHMARK_RESULTS.md"
TIMEOUT_BIN="$(command -v timeout || true)"

# Safety defaults: prevent hangs and runaway resource spikes.
R_MAX_RSS_GB="${R_MAX_RSS_GB:-192}"
BENCH_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC:-7200}"
PY_RAM_GB="${PY_RAM_GB:-4}"
PY_OVERWRITE_FRAMES="${PY_OVERWRITE_FRAMES:-1}"

if ! [ -x "$TIME_BIN" ]; then
  echo "ERROR: /usr/bin/time not found."
  exit 1
fi

mkdir -p "$OUT_LOG"

run_timed_command() {
  local stdout_log="$1"
  local time_log="$2"
  shift 2

  # New session => child PID is also process-group ID for clean signal fanout.
  setsid "$@" > "$stdout_log" 2> "$time_log" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  local rc=$?
  CHILD_PID=""
  return "$rc"
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
parse_wall() {
  local log="$1"
  grep -i "Elapsed (wall clock)" "$log" 2>/dev/null | awk '{print $NF}' | head -1 || echo "N/A"
}
wall_to_sec() {
  local t="$1"
  [[ "$t" == "N/A" ]] && echo "N/A" && return
  IFS=: read -r -a parts <<< "$t"
  local n=${#parts[@]}
  if   [[ $n -eq 3 ]]; then echo $(( parts[0]*3600 + parts[1]*60 + ${parts[2]%.*} ))
  elif [[ $n -eq 2 ]]; then echo $(( parts[0]*60   + ${parts[1]%.*} ))
  else echo "${parts[0]%.*}"
  fi
}
parse_rss_mb() {
  local log="$1"
  local kb
  kb=$(grep -i "Maximum resident set size" "$log" 2>/dev/null | awk '{print $NF}' | head -1) || true
  [[ -z "$kb" || "$kb" == "N/A" ]] && echo "N/A" && return
  awk "BEGIN {printf \"%.0f MB\", $kb / 1024}"
}
parse_phases() {
  local log="$1"
  grep -E "✓|✔" "$log" 2>/dev/null | sed 's/^[[:space:]]*/  /' || echo "  (none)"
}
summarise_dir() {
  local dir="$1"
  [ -d "$dir" ] || { echo "missing"; return; }
  local n size
  n=$(find "$dir" -maxdepth 1 -name "*.jpg" | wc -l)
  size=$(du -sh "$dir" 2>/dev/null | cut -f1)
  echo "${n} JPEGs, ${size}"
}
copy_sample() {
  local src_dir="$1" dest="$2" label="$3"
  local f
  f=$(find "$src_dir" -maxdepth 2 -name "*.jpg" 2>/dev/null | sort | sed -n '20p') || true
  [ -n "$f" ] && cp "$f" "${dest}/${label}_sample.jpg" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Clear previous results for this region (both code bases)
# ---------------------------------------------------------------------------
echo "Clearing results_py/${REGION}  results_r/${REGION} …"
rm -rf "results_py/${REGION}" "results_r/${REGION}"

echo "╔══════════════════════════════════════════════════════════╗"
printf "║  map_lifers benchmark  •  %-31s║\n" "$(date '+%Y-%m-%d %H:%M')  "
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Region:   $REGION"
echo "  Ramp:     $RAMP"
echo "  Mode:     $([ "$QUICK" = "1" ] && echo "QUICK (week 20 only)" || echo "FULL (52 weeks + GIF)")"
echo "  Safety:   R max RSS ${R_MAX_RSS_GB}GB, timeout ${BENCH_TIMEOUT_SEC}s"
echo "  Log dir:  $OUT_LOG"
echo ""

declare -a RES_ROWS=()
MD_ROWS_FILE="${OUT_LOG}/md_rows.tmp"
: > "$MD_ROWS_FILE"

# ===========================================================================
# Loop over all three resolutions
# ===========================================================================
for RES in "${RESOLUTIONS[@]}"; do
  RES_LOG="${OUT_LOG}/${RES}"
  mkdir -p "$RES_LOG"

  PY_BASE="results_py/${REGION}/${RES}"
  R_BASE="results_r/${REGION}/${RES}"
  PY_WEEKLY="${PY_BASE}/Weekly_maps"
  R_WEEKLY="${R_BASE}/Weekly_maps"
  mkdir -p "$PY_WEEKLY" "$R_WEEKLY"

  PY_STDOUT="${RES_LOG}/python_stdout.txt"
  PY_TIMELOG="${RES_LOG}/python_time.txt"
  R_STDOUT="${RES_LOG}/r_stdout.txt"
  R_TIMELOG="${RES_LOG}/r_time.txt"

  [[ "$QUICK" == "1" ]] && PY_EXTRA="--no-animate" || PY_EXTRA=""
  [[ "$QUICK" == "1" ]] && R_EXTRA="--no-animate"  || R_EXTRA=""
  [[ "$PY_OVERWRITE_FRAMES" == "1" ]] && PY_FRAME_FLAG="--overwrite-frames" || PY_FRAME_FLAG=""

  case "$RES" in
    3km)
      BENCH_WORKERS="${BENCH_WORKERS_3KM:-40}"
      BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE_3KM:-40}"
      ;;
    9km)
      BENCH_WORKERS="${BENCH_WORKERS_9KM:-40}"
      BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE_9KM:-40}"
      ;;
    *)
      BENCH_WORKERS="${BENCH_WORKERS_27KM:-40}"
      BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE_27KM:-40}"
      ;;
  esac

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  ${RES}  —  Python"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ -n "$TIMEOUT_BIN" ]; then
    # shellcheck disable=SC2086
    run_timed_command "$PY_STDOUT" "$PY_TIMELOG" \
      "$TIME_BIN" -v \
      "$TIMEOUT_BIN" --foreground --signal=TERM --kill-after=30s "${BENCH_TIMEOUT_SEC}s" \
      .venv/bin/python map_lifers.py \
        --regions "$REGION" \
        --resolutions "$RES" \
        --ramp "$RAMP" \
        --color-scale custom \
        --offline \
        --workers "$BENCH_WORKERS" \
        --batch-size "$BENCH_BATCH_SIZE" \
        --ram-gb "$PY_RAM_GB" \
        $PY_FRAME_FLAG \
        $PY_EXTRA
  else
    # shellcheck disable=SC2086
    run_timed_command "$PY_STDOUT" "$PY_TIMELOG" \
      "$TIME_BIN" -v \
      .venv/bin/python map_lifers.py \
        --regions "$REGION" \
        --resolutions "$RES" \
        --ramp "$RAMP" \
        --color-scale custom \
        --offline \
        --workers "$BENCH_WORKERS" \
        --batch-size "$BENCH_BATCH_SIZE" \
        --ram-gb "$PY_RAM_GB" \
        $PY_FRAME_FLAG \
        $PY_EXTRA
  fi
  cat "$PY_TIMELOG" >> "$PY_STDOUT"
  cat "$PY_STDOUT"

  PY_WALL=$(parse_wall "$PY_TIMELOG")
  PY_SEC=$(wall_to_sec "$PY_WALL")
  PY_RSS=$(parse_rss_mb "$PY_TIMELOG")
  PY_FILES=$(summarise_dir "$PY_WEEKLY")
  copy_sample "$PY_BASE" "$RES_LOG" "python"
  echo "  Python  wall=$PY_WALL  RSS=${PY_RSS}  output: $PY_FILES"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  ${RES}  —  R"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ -n "$TIMEOUT_BIN" ]; then
    # shellcheck disable=SC2086
    run_timed_command "$R_STDOUT" "$R_TIMELOG" \
      "$TIME_BIN" -v \
      "$TIMEOUT_BIN" --foreground --signal=TERM --kill-after=30s "${BENCH_TIMEOUT_SEC}s" \
      Rscript map_lifers.R \
        --region "$REGION" \
        --resolution "$RES" \
        --ramp "$RAMP" \
        --color-scale custom \
        --offline \
        --workers "$BENCH_WORKERS" \
        --batch-size "$BENCH_BATCH_SIZE" \
        --max-rss-gb "$R_MAX_RSS_GB" \
        $R_EXTRA
  else
    # shellcheck disable=SC2086
    run_timed_command "$R_STDOUT" "$R_TIMELOG" \
      "$TIME_BIN" -v \
      Rscript map_lifers.R \
        --region "$REGION" \
        --resolution "$RES" \
        --ramp "$RAMP" \
        --color-scale custom \
        --offline \
        --workers "$BENCH_WORKERS" \
        --batch-size "$BENCH_BATCH_SIZE" \
        --max-rss-gb "$R_MAX_RSS_GB" \
        $R_EXTRA
  fi
  cat "$R_TIMELOG" >> "$R_STDOUT"
  cat "$R_STDOUT"

  R_WALL=$(parse_wall "$R_TIMELOG")
  R_SEC=$(wall_to_sec "$R_WALL")
  R_RSS=$(parse_rss_mb "$R_TIMELOG")
  R_FILES=$(summarise_dir "$R_WEEKLY")
  copy_sample "$R_BASE" "$RES_LOG" "r"
  echo "  R       wall=$R_WALL  RSS=${R_RSS}  output: $R_FILES"

  SPEED_LABEL="N/A"
  if [[ "$PY_SEC" =~ ^[0-9]+$ ]] && [[ "$R_SEC" =~ ^[0-9]+$ ]] && \
     [ "$R_SEC" -gt 0 ] && [ "$PY_SEC" -gt 0 ]; then
    SPEED_LABEL=$(awk -v p="$PY_SEC" -v r="$R_SEC" 'BEGIN {
      if (p < r) printf "Py %.2fx faster", r/p
      else       printf "R  %.2fx faster", p/r
    }')
  fi
  RES_ROWS+=("  ${RES}  Python: ${PY_WALL} (${PY_SEC}s) ${PY_RSS}   R: ${R_WALL} (${R_SEC}s) ${R_RSS}   ${SPEED_LABEL}")
  printf '| %s | %s | %s | %s | %s | %s |\n' "$RES" "Python" "$PY_WALL" "$PY_SEC" "$PY_RSS" "$PY_FILES" >> "$MD_ROWS_FILE"
  printf '| %s | %s | %s | %s | %s | %s |\n' "$RES" "R" "$R_WALL" "$R_SEC" "$R_RSS" "$R_FILES" >> "$MD_ROWS_FILE"

  echo ""
done

# ===========================================================================
# Final summary
# ===========================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY — all resolutions                                       ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
for row in "${RES_ROWS[@]}"; do
  printf "║  %-64s║\n" "$row"
done
echo "╚══════════════════════════════════════════════════════════════════╝"

{
  echo ""
  echo "## Run: ${TIMESTAMP} — ${REGION} / ramp:${RAMP} / $([ "$QUICK" = "1" ] && echo "QUICK" || echo "FULL 52-week + GIF")"
  echo ""
  echo "| Resolution | Lang | Wall time | Seconds | Peak RSS | Output |"
  echo "|-----------|------|----------|---------|----------|--------|"
  cat "$MD_ROWS_FILE"
  echo ""
  for RES in "${RESOLUTIONS[@]}"; do
    echo "**${RES} Python phases:**"
    parse_phases "${OUT_LOG}/${RES}/python_stdout.txt" 2>/dev/null
    echo ""
    echo "**${RES} R phases:**"
    parse_phases "${OUT_LOG}/${RES}/r_stdout.txt" 2>/dev/null
    echo ""
  done
  echo "---"
} >> "$RESULTS_MD"

echo ""
echo "Results appended to $RESULTS_MD"
echo "Logs + sample frames in $OUT_LOG/"
