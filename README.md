# PyLifer

**A complete Python rewrite by [Bart Wickel](https://github.com/hydrospheric0)**

PyLifer is a standalone Python tool for generating animated personal lifer maps from eBird Status & Trends occurrence data. It was inspired by the concept of [lifeR](https://github.com/smsfrn/lifeR) by Sam Safran but shares no code with it — this is a full ground-up Python implementation with a single entry point, RAM-aware parallel processing, and a compact sp_cache for fast re-runs.

For each week of the year it stacks occurrence rasters for every species you still need, producing weekly heatmaps and a 52-frame animated GIF showing where your best lifer opportunities are throughout the year.

## How it works

Everything runs through a single script — `PyLifer.py` — which chains the full pipeline automatically:

1. **Workspace setup** — finds your eBird export zip, extracts `MyEBirdData.csv`
2. **Model table** — generates `ebirdst_runs.csv` from the S&T API if absent
3. **Download** — fetches missing species tifs from eBird Status & Trends
4. **Preprocess** — builds a compact packbits sp_cache (~32× smaller than raw tifs)
5. **Render** — weekly heatmaps + 52-frame animated GIF

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

| Key | Purpose | Where to get it |
|-----|---------|-----------------|
| eBird Status & Trends key | Download occurrence rasters | https://ebird.org/st/request |
| eBird API key | Fetch your regional checklist | https://ebird.org/api/keygen |

Create a `.env` file in the project root:

```ini
EBIRDST_KEY=your_ebirdst_key
EBIRD_API_KEY=your_ebird_api_key
USER=Your Name
```

### 3. Your eBird data

Export your life list from [ebird.org/downloadMyData](https://ebird.org/downloadMyData) and drop the downloaded `.zip` file into the project root. PyLifer extracts it automatically on first run.

## Usage

```bash
./run.sh                                        # US, all resolutions, full animation
./run.sh --regions US-CA                        # California only
./run.sh --regions US NL                        # multiple regions
./run.sh --week 20                              # single-week preview
./run.sh --no-animate                           # week 20 only, no GIF
./run.sh --offline                              # no API calls, use cached tifs
./run.sh --generate-runs-csv                    # refresh ebirdst_runs.csv and exit
./run.sh --skip-preprocess                      # skip sp_cache build
./run.sh --force-preprocess                     # rebuild sp_cache from scratch
./run.sh --ram-gb 8                             # cap RAM usage
./run.sh --scale compact                        # colour scale preset (auto/compact/wide)
./run.sh -y                                     # auto-confirm download prompts
```

Or call Python directly:

```bash
python PyLifer.py --regions US --animate
```

Output goes to `results_py/<region>/<resolution>/`.

## Credits

Inspired by [lifeR](https://github.com/smsfrn/lifeR) by **Sam Safran**.

## Support

<a href="https://buymeacoffee.com/bartg">
	<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="180" />
</a>
