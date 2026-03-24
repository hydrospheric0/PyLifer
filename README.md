# PyLifer

Python tools for generating personal lifer maps from eBird Status & Trends data.

Inspired by [smsfrn/lifeR](https://github.com/smsfrn/lifeR). The workflow here is fully Python — no R required.

---

## Scripts

### `download_ebirdst.py`
Bulk pre-downloads eBird Status & Trends occurrence rasters (`.tif` files) for all species you still need in the specified regions.

```bash
# NL + US, all resolutions (3km, 9km, 27km)
python download_ebirdst.py

# NL only
python download_ebirdst.py --regions NL

# Skip 3km
python download_ebirdst.py --resolutions 9km 27km

# True lifers (subtract global life list, not just regional)
python download_ebirdst.py --needs global

# Preview counts without downloading
python download_ebirdst.py --dry-run

# More parallel threads
python download_ebirdst.py --workers 8
```

### `map_lifers.py`
Generates weekly lifer maps and an animated GIF from the cached rasters.

```bash
# Single week
python map_lifers.py --week 20

# Full 52-week animation
python map_lifers.py --animate

# Options
python map_lifers.py --regions NL US --resolution 3km --animate --fps 5 --ram-gb 4.0
```

Output goes to `results_py/<region>/<resolution>/`.

---

## Setup

### 1. Dependencies

```bash
pip install -r requirements.txt
```

System libraries required for rasterio/geopandas (Ubuntu/Debian):
```bash
sudo apt-get install -y libgdal-dev libgeos-dev libproj-dev
```

### 2. API keys

You need two keys:

| Key | Purpose | Request at |
|-----|---------|------------|
| eBird Status & Trends key | Download occurrence rasters | https://ebird.org/st/request |
| eBird API key | Fetch regional checklists | https://ebird.org/api/keygen |

Copy the example config and fill in your keys:

```bash
cp config_local.R.example config_local.R
```

`config_local.R` is gitignored and will never be committed.

### 3. Species model list

`download_ebirdst.py` and `map_lifers.py` both need `ebirdst_runs.csv` — a list of all species that have eBird S&T models. Generate it once:

```bash
Rscript export_ebirdst_runs.R   # if you have the lifeR R scripts
```

Or download a pre-built copy from the [lifeR repo](https://github.com/smsfrn/lifeR).

### 4. Your eBird data

Export your life list from [ebird.org/downloadMyData](https://ebird.org/downloadMyData) and save the file as `MyEBirdData.csv` in the project root. This is gitignored.

---

## Workflow

```
1. Set up config_local.R with your API keys
2. Place MyEBirdData.csv in the project root
3. python download_ebirdst.py    # cache the rasters you need
4. python map_lifers.py --animate
```

---

## Data

- **eBird Status & Trends 2023** — Cornell Lab of Ornithology, [ebird.org/science/status-and-trends](https://ebird.org/science/status-and-trends)
- **NaturalEarth** — country and state boundaries, auto-downloaded on first run
