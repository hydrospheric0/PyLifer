# PyLifer

## About the tool

PyLifer generates animated personal lifer maps from eBird Status & Trends occurrence data.
It is a fully Python port of [lifeR](https://github.com/smsfrn/lifeR) — no R required.

For each week of the year, it stacks occurrence rasters for every species you still need, producing weekly heatmaps and a 52-frame animated GIF showing where your best lifer opportunities are.

## Features

- Downloads eBird S&T rasters in bulk at 27km, 9km, or 3km resolution
- Accumulates weekly occurrence probability across all needed species
- Renders weekly heatmaps and assembles a full-year animated GIF
- Produces a lo-res GIF alongside the full-res version for easy sharing
- Parallel raster processing with configurable RAM cap
- Fully offline after initial data download (`--offline` flag)
- Minimal dependencies — pure Python stack, no R

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

System libraries required (Ubuntu/Debian):
```bash
sudo apt-get install -y libgdal-dev libgeos-dev libproj-dev
```

### 2. API keys

You need two keys:

| Key | Purpose | Where to request |
|-----|---------|-----------------|
| eBird Status & Trends key | Download occurrence rasters | https://ebird.org/st/request |
| eBird API key | Fetch your regional needs list | https://ebird.org/api/keygen |

Copy the example config and fill in your keys:

```bash
cp config_local.R.example config_local.R
```

`config_local.R` is gitignored and will never be committed.

### 3. Species model list

Both scripts need `ebirdst_runs.csv` — a list of all species that have eBird S&T models.
Download a pre-built copy from the [releases](https://github.com/hydrospheric0/PyLifer/releases) page, or generate it from the [lifeR repo](https://github.com/smsfrn/lifeR).

Place it in the project root. It is gitignored.

### 4. Your eBird data

Export your life list from [ebird.org/downloadMyData](https://ebird.org/downloadMyData) and save the file as `MyEBirdData.csv` in the project root. It is gitignored.

## Workflow

```
1. Fill in config_local.R with your API keys
2. Place MyEBirdData.csv in the project root
3. Place ebirdst_runs.csv in the project root
4. python download_ebirdst.py    # download the rasters you need
5. python map_lifers.py --animate
```

## Usage

### Download rasters

```bash
# US + NL, all resolutions
python download_ebirdst.py

# US only, skip 3km
python download_ebirdst.py --regions US --resolutions 9km 27km

# True global lifers (not just regional needs)
python download_ebirdst.py --needs global

# Preview counts without downloading
python download_ebirdst.py --dry-run
```

### Generate maps

```bash
# Single week
python map_lifers.py --week 20

# Full 52-week animation (hi-res + lo-res GIFs)
python map_lifers.py --animate

# 3km animation, US only, custom frame rate
python map_lifers.py --regions US --resolution 3km --animate --fps 5

# Re-render from cached rasters (no API calls)
python map_lifers.py --animate --offline
```

Output goes to `results_py/<region>/<resolution>/Weekly_maps/` and `Animated_map/`.

## Data

- **eBird Status & Trends 2023** — Cornell Lab of Ornithology, [ebird.org/science/status-and-trends](https://ebird.org/science/status-and-trends)
- **NaturalEarth** — country and state/province boundaries, auto-downloaded on first run

## Credits

Built on [lifeR](https://github.com/smsfrn/lifeR) by **Sam Safran** — the original R implementation this tool is ported from.
The [hydrospheric0/lifeR fork](https://github.com/hydrospheric0/lifeR) provided additional refinements used during development.

## Support this project

If you find this tool useful, please consider supporting its development:

<a href="https://buymeacoffee.com/bartg">
	<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="180" />
</a>
