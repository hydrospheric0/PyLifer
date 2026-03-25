#!/usr/bin/env bash
# run.sh — PyLifer convenience runner
#
# Defaults: --regions US  --resolution 27km  (dark theme is always on)
# Override any default by passing the flag explicitly.
#
# Usage:
#   ./run.sh                          # US, 27km, full animation (all 52 frames + GIF)
#   ./run.sh --animate                # US, 27km, full animation
#   ./run.sh --week 30                # US, 27km, week 30
#   ./run.sh --resolution 3km         # US, 3km, single map
#   ./run.sh --regions NL US          # NL + US, 27km
#   ./run.sh --resolution 9km --animate --fps 8
#   ./run.sh -h                       # this help + map_lifers.py flags

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Locate Python
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in \
    "$SCRIPT_DIR/.venv/bin/python" \
    "$SCRIPT_DIR/../lifeR/.venv/bin/python" \
    "$(command -v python3 2>/dev/null || true)"
do
    if [[ -x "$candidate" ]]; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: no Python found. Set up a venv or install from requirements.txt." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
PyLifer — lifer map generator

Usage: ./run.sh [OPTIONS] [-- EXTRA_ARGS]

Options (defaults shown):
  --regions    CODE [CODE ...]   eBird region codes     (default: US)
  --resolution 3km|9km|27km      raster resolution      (default: 27km)
  --week       N                 single map, week 1-52  (default: 20)
  --animate                      full 52-frame animation + GIF
  --fps        N                 GIF frames per second   (default: 5)
  --ram-gb     F                 RAM budget per batch GB (default: 4.0)
  --ebird-csv  PATH              path to MyEBirdData.csv
  -h, --help                     show this message then map_lifers.py --help

Dark theme is always on (hardcoded in map_lifers.py).
Output goes to results_py/<region>/<resolution>/.

Data check:
  python check_data.py

Download missing tifs:
  python download_ebirdst.py --regions US --resolutions 27km
EOF
    echo ""
    "$PYTHON" "$SCRIPT_DIR/map_lifers.py" --help
    exit 0
}

# ---------------------------------------------------------------------------
# Parse args — inject defaults if flags are absent
# ---------------------------------------------------------------------------
ARGS=("$@")

has_flag() {
    local flag="$1"
    for a in "${ARGS[@]}"; do [[ "$a" == "$flag" ]] && return 0; done
    return 1
}

if has_flag "-h" || has_flag "--help"; then
    usage
fi

# Inject defaults only when flags are not already present
EXTRA=()
has_flag "--regions"    || EXTRA+=(--regions US)
has_flag "--resolution" || EXTRA+=(--resolution 27km)
has_flag "--animate" || has_flag "--week" || EXTRA+=(--animate)

# cd to script dir so relative paths (config_local.R, data/, etc.) resolve
cd "$SCRIPT_DIR"

echo "Python  : $PYTHON"
echo "Command : map_lifers.py ${EXTRA[*]:-} ${ARGS[*]:-}"
echo ""

"$PYTHON" "$SCRIPT_DIR/map_lifers.py" "${EXTRA[@]}" "${ARGS[@]}"
