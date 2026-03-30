#!/usr/bin/env python3
"""
dash_spots.py — Interactive lifer-hotspot explorer.

Layout (all sizes are css-calculated so the page fills the viewport exactly):

  ┌── topbar (3vh) ─────────────────────────────────────────────────────────┐
  ├── nav sidebar (10vw) ──┬── main content (90vw) ────────────────────────┤
  │  filters & controls   │  MAP  (50vh)                                   │
  │  metric / n / min-km  ├────────────────────────┬───────────────────────┤
  │  export buttons       │  CHART (50vh – topbar) │  TABLE  (50vh)        │
  └───────────────────────┴────────────────────────┴───────────────────────┘

Interactions:
  • Sidebar controls → live-recalculate top-N spots (from precomputed cache)
  • Click marker or table row → highlight spot, update chart + species list
  • Week slider (inside chart panel) → update weekly-raster image overlay
  • Export buttons → download KML or CSV (with per-spot species checklist)

Usage:
    python dash_spots.py --regions US --n 20
    python dash_spots.py --regions US-CA --n 15 --per-unit 3
    python dash_spots.py --regions US --offline --port 8051
"""
import argparse
import io
import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import dash
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, dash_table
from dash_extensions.javascript import assign as js_assign

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive map of top lifer hotspots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--regions",    nargs="+", default=["US"], metavar="CODE")
    p.add_argument("--resolution", default="3km", choices=["3km", "9km", "27km"])
    p.add_argument("--n",          type=int,   default=10)
    p.add_argument("--per-unit",   type=int,   default=3,    metavar="N",
                   help="Max spots per state (country region) or county (state region).")
    p.add_argument("--metric",     default="peak", choices=["peak", "median", "mean"])
    p.add_argument("--min-km",     type=float, default=50.0)
    p.add_argument("--offline",    action="store_true")
    p.add_argument("--ebird-csv",  default="MyEBirdData.csv", metavar="PATH")
    p.add_argument("--port",       type=int,   default=8050)
    p.add_argument("--debug",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colour ramp  (low → blue, mid → cyan/green, high → orange/red)
# ---------------------------------------------------------------------------

def _score_colour(score: float, score_max: float) -> str:
    t = min(1.0, score / max(score_max, 1.0))
    if t < 0.25:
        r, g, b = 20,  60,  180
    elif t < 0.5:
        f = (t - 0.25) / 0.25
        r, g, b = int(20 + f * 0),   int(60  + f * 195), int(180 - f * 60)
    elif t < 0.75:
        f = (t - 0.5) / 0.25
        r, g, b = int(0  + f * 255), int(255 - f * 100), int(120 - f * 120)
    else:
        f = (t - 0.75) / 0.25
        r, g, b = 255, int(155 - f * 155), 0
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Results directory (served as /results/)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results_py"


def _raster_url(region: str, resolution: str, week_date: str) -> str | None:
    """Return the URL path for a weekly raster PNG, or None if it doesn't exist."""
    fname = f"{region}_{week_date}.png"
    p = RESULTS_DIR / region / resolution / "Weekly_maps_auto" / fname
    if p.exists():
        return f"/results/{region}/{resolution}/Weekly_maps_auto/{fname}"
    return None


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _spots_to_kml(spots: list[dict]) -> str:
    """Build a KML string from spots list."""
    root = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc  = ET.SubElement(root, "Document")
    ET.SubElement(doc, "name").text = "PyLifer Top Spots"

    for s in spots:
        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = f"#{s['rank']} {s['loc_name']}"
        desc_lines = [
            f"Score: {s['score']:.1f}",
            f"eBird spp: {s['hs_spp'] or '—'}",
            f"State: {s['state'] or '—'}",
            f"County: {s['county'] or '—'}",
        ]
        if s.get("species_codes"):
            desc_lines.append(f"Lifer species: {', '.join(s['species_codes'])}")
        ET.SubElement(pm, "description").text = "\n".join(desc_lines)
        pt = ET.SubElement(pm, "Point")
        ET.SubElement(pt, "coordinates").text = f"{s['lon']},{s['lat']},0"
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def _spots_to_csv(spots: list[dict], week_dates: list[str]) -> str:
    rows = []
    for s in spots:
        base = {
            "rank":      s["rank"],
            "loc_name":  s["loc_name"],
            "loc_id":    s["loc_id"] or "",
            "lat":       round(s["lat"], 5),
            "lon":       round(s["lon"], 5),
            "score":     round(s["score"], 2),
            "hs_spp":    s["hs_spp"] or "",
            "state":     s["state"] or "",
            "county":    s["county"] or "",
            "species_checklist": " | ".join(s.get("species_codes") or []),
        }
        # append weekly columns
        for i, w in enumerate(s.get("weekly") or []):
            label = week_dates[i] if i < len(week_dates) else f"w{i}"
            base[label] = round(w, 2) if w == w else ""
        rows.append(base)
    return pd.DataFrame(rows).to_csv(index=False)


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

_DARK_BG   = "#0e1117"
_PANEL_BG  = "#161b22"
_GRID_COL  = "#21262d"
_TEXT_COL  = "#c9d1d9"
_ACCENT    = "#58a6ff"
_PEAK_COL  = "#f78166"
_NAV_BG    = "#0d1117"
_BORDER    = "1px solid #21262d"
_BTN_STYLE = {
    "width": "100%", "marginBottom": "6px",
    "background": "#21262d", "color": _TEXT_COL,
    "border": "1px solid #30363d", "borderRadius": "4px",
    "padding": "6px 0", "cursor": "pointer", "fontSize": "12px",
}


def _blank_chart(msg: str = "Click a hotspot to see the 52-week lifer profile") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
        font_color=_TEXT_COL,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#484f58"),
        )],
    )
    return fig


def _weekly_chart(spot: dict, week_dates: list[str], highlight_week: int | None = None) -> go.Figure:
    y = spot.get("weekly")
    if not y or not week_dates:
        return _blank_chart()

    labels: list[str] = []
    for ds in week_dates:
        try:
            d = date.fromisoformat(ds)
            labels.append(d.strftime("%-d %b"))
        except Exception:
            labels.append(ds)

    peak_i = int(max(range(len(y)), key=lambda i: y[i]))
    colours = []
    for i in range(len(y)):
        if i == highlight_week:
            colours.append("#ffd700")
        elif i == peak_i:
            colours.append(_PEAK_COL)
        else:
            colours.append(_ACCENT)

    ebird_url = f"https://ebird.org/hotspot/{spot['loc_id']}" if spot["loc_id"] else None
    title_txt = (
        f"<a href='{ebird_url}' target='_blank' style='color:{_ACCENT}'>{spot['loc_name']}</a>"
        if ebird_url else spot["loc_name"]
    )
    state_s  = spot["state"].replace("US-", "") if spot["state"] else ""
    county_s = spot["county"].split("-")[-1] if spot["county"] else ""
    subtitle = "  |  ".join(filter(None, [
        state_s, county_s,
        f"Score {spot['score']:.1f}",
        f"eBird spp {spot['hs_spp']}" if spot["hs_spp"] else None,
    ]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=y,
        marker_color=colours,
        hovertemplate="%{x}<br><b>%{y:.0f} lifers</b><extra></extra>",
    ))
    fig.add_annotation(
        x=labels[peak_i], y=y[peak_i],
        text=f"Peak week<br>{y[peak_i]:.0f} lifers",
        showarrow=True, arrowhead=2, arrowcolor=_PEAK_COL,
        font=dict(color=_PEAK_COL, size=11),
        bgcolor="rgba(0,0,0,0.65)", bordercolor=_PEAK_COL, borderpad=4,
        ay=-45,
    )
    fig.update_layout(
        title=dict(
            text=f"{title_txt}<br><sup style='color:#888'>{subtitle}</sup>",
            font=dict(size=14, color=_TEXT_COL), x=0.01,
        ),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PANEL_BG,
        font_color=_TEXT_COL,
        xaxis=dict(
            title="Week", tickangle=-50, tickfont=dict(size=8),
            gridcolor=_GRID_COL, showline=False,
            tickmode="array", tickvals=labels[::4], ticktext=labels[::4],
        ),
        yaxis=dict(title="Potential lifers", gridcolor=_GRID_COL, zeroline=False),
        margin=dict(l=55, r=20, t=60, b=70),
        bargap=0.1,
    )
    return fig


# ---------------------------------------------------------------------------
# Leaflet JS helpers
# ---------------------------------------------------------------------------
_POINT_TO_LAYER = js_assign("""
function(feature, latlng, context) {
    const p = feature.properties;
    const icon = L.divIcon({
        className: '',
        html: `<div style="
            background: ${p.colour};
            color: #fff; font-weight: bold; font-size: 11px;
            width: 26px; height: 26px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            border: 2px solid rgba(255,255,255,0.25);
            box-shadow: 0 2px 6px rgba(0,0,0,0.6); cursor: pointer;">${p.rank}</div>`,
        iconSize: [26, 26], iconAnchor: [13, 13],
    });
    return L.marker(latlng, {icon});
}
""")

_ON_EACH_FEATURE = js_assign("""
function(feature, layer, context) {
    const p = feature.properties;
    const url = p.loc_id ? `https://ebird.org/hotspot/${p.loc_id}` : null;
    const nameHtml = url
        ? `<a href="${url}" target="_blank" style="color:#58a6ff">${p.loc_name}</a>`
        : p.loc_name;
    layer.bindPopup(
        `<div style="font-family:sans-serif;min-width:180px">
          <b>#${p.rank}</b> ${nameHtml}<br/>
          <span style="color:#999">Score: ${p.score.toFixed(1)}
          &nbsp;|&nbsp; eBird spp: ${p.hs_spp ?? '—'}</span><br/>
          <span style="color:#aaa;font-size:0.85em">
            ${p.state || '—'} &nbsp;/&nbsp; ${p.county || '—'}
          </span>
        </div>`
    );
}
""")


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------
def build_app(
    all_spots: list[dict],
    week_dates: list[str],
    region: str = "US",
    resolution: str = "3km",
) -> dash.Dash:
    app = dash.Dash(
        __name__,
        title="PyLifer — Hotspot Explorer",
        suppress_callback_exceptions=True,
    )

    # Serve results directory as /results/
    import flask
    @app.server.route("/results/<path:filepath>")
    def serve_results(filepath):
        return flask.send_from_directory(str(RESULTS_DIR), filepath)

    app.index_string = app.index_string.replace(
        "</head>",
        "<style>"
        "*, *::before, *::after {box-sizing: border-box}"
        "body,html{margin:0;padding:0;background:#0e1117;color:#c9d1d9;font-family:sans-serif}"
        ".leaflet-popup-content-wrapper{background:#161b22;color:#c9d1d9;border:1px solid #30363d}"
        ".leaflet-popup-tip{background:#161b22}"
        ".dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,"
        ".dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th"
        "{background:#161b22 !important;color:#c9d1d9 !important;border-color:#21262d !important;font-size:12px}"
        ".dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td"
        "{background:#21262d !important;cursor:pointer}"
        "</style></head>",
    )

    if not all_spots:
        app.layout = html.Div(
            "No spots computed. Run top_spots.py first to build the cache.",
            style={"padding": "2rem", "color": _TEXT_COL},
        )
        return app

    score_max = max(s["score"] for s in all_spots)

    def _make_geojson(spots: list[dict]) -> dict:
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id":   s["loc_id"] or f"_spot_{s['rank']}",
                    "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                    "properties": {
                        "rank":     s["rank"],
                        "score":    s["score"],
                        "loc_name": s["loc_name"],
                        "loc_id":   s["loc_id"],
                        "state":    s["state"],
                        "county":   s["county"],
                        "hs_spp":   s["hs_spp"],
                        "colour":   _score_colour(s["score"], score_max),
                    },
                }
                for s in spots
            ],
        }

    mean_lat = sum(s["lat"] for s in all_spots) / len(all_spots)
    mean_lon = sum(s["lon"] for s in all_spots) / len(all_spots)

    # ── week slider marks (every 4 weeks) ──────────────────────────────
    slider_marks: dict[int, str] = {}
    for i, ds in enumerate(week_dates):
        if i % 4 == 0:
            try:
                slider_marks[i] = date.fromisoformat(ds).strftime("%-d %b")
            except Exception:
                slider_marks[i] = ds
    week_max = max(len(week_dates) - 1, 0)

    # ── table columns ───────────────────────────────────────────────────
    table_cols = [
        {"name": "#",         "id": "rank"},
        {"name": "Hotspot",   "id": "loc_name"},
        {"name": "State",     "id": "state"},
        {"name": "Score",     "id": "score"},
        {"name": "eBird spp", "id": "hs_spp"},
        {"name": "Peak wk",   "id": "peak_week"},
        {"name": "Lifers",    "id": "n_lifers"},
    ]

    def _table_rows(spots: list[dict]) -> list[dict]:
        rows = []
        for s in spots:
            wy = s.get("weekly") or []
            peak_i = int(max(range(len(wy)), key=lambda i: wy[i])) if wy else 0
            peak_label = ""
            if peak_i < len(week_dates):
                try:
                    peak_label = date.fromisoformat(week_dates[peak_i]).strftime("%-d %b")
                except Exception:
                    peak_label = week_dates[peak_i]
            rows.append({
                "rank":      s["rank"],
                "loc_name":  s["loc_name"],
                "state":     (s["state"] or "").replace("US-", ""),
                "score":     round(s["score"], 1),
                "hs_spp":    s["hs_spp"] or "—",
                "peak_week": peak_label,
                "n_lifers":  len(s.get("species_codes") or []),
            })
        return rows

    # ── layout ──────────────────────────────────────────────────────────
    TOPBAR_H = "3vh"
    NAV_W    = "10vw"
    MAIN_W   = "90vw"
    MAP_H    = "50vh"
    BOT_H    = f"calc(97vh - {MAP_H})"   # 97vh = 100vh - topbar 3vh

    app.layout = html.Div(
        style={"height": "100vh", "width": "100vw", "display": "flex",
               "flexDirection": "column", "overflow": "hidden", "background": _DARK_BG},
        children=[

            # ── Top bar ──────────────────────────────────────────────────
            html.Div(
                style={"height": TOPBAR_H, "minHeight": TOPBAR_H, "width": "100%",
                       "background": "#010409", "borderBottom": _BORDER,
                       "display": "flex", "alignItems": "center",
                       "padding": "0 14px", "gap": "18px", "flexShrink": "0"},
                children=[
                    html.Span("🦆 PyLifer", style={"fontWeight": "bold", "fontSize": "14px",
                                                    "color": _ACCENT, "letterSpacing": "0.5px"}),
                    html.Span(f"{region} · {resolution}",
                              style={"fontSize": "12px", "color": "#666"}),
                    html.Span(id="topbar-status",
                              style={"fontSize": "11px", "color": "#555", "marginLeft": "auto"}),
                ],
            ),

            # ── Body row (nav + main) ─────────────────────────────────────
            html.Div(
                style={"flex": "1", "display": "flex", "overflow": "hidden"},
                children=[

                    # ── Left nav sidebar ───────────────────────────────────
                    html.Div(
                        style={"width": NAV_W, "minWidth": NAV_W, "background": _NAV_BG,
                               "borderRight": _BORDER, "display": "flex",
                               "flexDirection": "column", "padding": "10px 8px",
                               "overflowY": "auto", "gap": "4px"},
                        children=[
                            html.Div("Filters", style={"fontSize": "11px", "color": "#555",
                                                        "textTransform": "uppercase",
                                                        "letterSpacing": "1px", "marginBottom": "6px"}),

                            html.Label("Metric", style={"fontSize": "11px", "color": "#888"}),
                            dcc.Dropdown(
                                id="ctrl-metric",
                                options=[{"label": m, "value": m} for m in ("peak", "median", "mean")],
                                value="peak",
                                clearable=False,
                                style={"fontSize": "12px", "background": _PANEL_BG,
                                       "color": _TEXT_COL, "border": "none", "marginBottom": "8px"},
                            ),

                            html.Label("Top N", style={"fontSize": "11px", "color": "#888"}),
                            dcc.Slider(
                                id="ctrl-n",
                                min=5, max=50, step=5, value=len(all_spots),
                                marks={5: "5", 25: "25", 50: "50"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Div(style={"marginBottom": "8px"}),

                            html.Label("Min km apart", style={"fontSize": "11px", "color": "#888"}),
                            dcc.Slider(
                                id="ctrl-minkm",
                                min=0, max=200, step=10, value=50,
                                marks={0: "0", 100: "100", 200: "200"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Div(style={"marginBottom": "10px"}),

                            html.Hr(style={"borderColor": "#21262d", "margin": "6px 0"}),
                            html.Div("Export", style={"fontSize": "11px", "color": "#555",
                                                       "textTransform": "uppercase",
                                                       "letterSpacing": "1px", "marginBottom": "6px"}),

                            html.Button("⬇ KML", id="btn-kml", n_clicks=0, style=_BTN_STYLE),
                            dcc.Download(id="dl-kml"),

                            html.Button("⬇ CSV", id="btn-csv", n_clicks=0, style=_BTN_STYLE),
                            dcc.Download(id="dl-csv"),

                            html.Hr(style={"borderColor": "#21262d", "margin": "6px 0"}),
                            html.Div("Week raster", style={"fontSize": "11px", "color": "#555",
                                                            "textTransform": "uppercase",
                                                            "letterSpacing": "1px", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="ctrl-week",
                                min=0, max=week_max, step=1, value=19,
                                marks=slider_marks,
                                vertical=True,
                                verticalHeight=220,
                                tooltip={"placement": "right", "always_visible": False},
                            ),
                        ],
                    ),

                    # ── Main content ─────────────────────────────────────
                    html.Div(
                        style={"flex": "1", "display": "flex", "flexDirection": "column",
                               "overflow": "hidden"},
                        children=[

                            # Map row (50vh)
                            html.Div(
                                style={"height": MAP_H, "position": "relative", "flexShrink": "0"},
                                children=[
                                    dl.Map(
                                        id="spots-map",
                                        center={"lat": mean_lat, "lng": mean_lon},
                                        zoom=4,
                                        style={"height": "100%", "width": "100%"},
                                        children=[
                                            dl.TileLayer(
                                                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                                                attribution=(
                                                    '&copy; <a href="https://www.openstreetmap.org/copyright">'
                                                    'OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
                                                ),
                                                maxZoom=19,
                                            ),
                                            # Raster image overlay (weekly lifer map)
                                            dl.ImageOverlay(
                                                id="raster-overlay",
                                                url="",
                                                bounds=[[24.4, -125.0], [49.4, -66.9]],
                                                opacity=0.55,
                                            ),
                                            dl.GeoJSON(
                                                id="spots-layer",
                                                data=_make_geojson(all_spots),
                                                options={
                                                    "pointToLayer":  _POINT_TO_LAYER,
                                                    "onEachFeature": _ON_EACH_FEATURE,
                                                },
                                                zoomToBoundsOnClick=False,
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            # Bottom row: chart (50%) + table (50%)
                            html.Div(
                                style={"flex": "1", "display": "flex", "overflow": "hidden"},
                                children=[
                                    # Chart panel
                                    html.Div(
                                        style={"width": "50%", "borderRight": _BORDER,
                                               "display": "flex", "flexDirection": "column"},
                                        children=[
                                            dcc.Graph(
                                                id="weekly-chart",
                                                style={"flex": "1"},
                                                config={"displayModeBar": False},
                                                figure=_blank_chart(),
                                            ),
                                        ],
                                    ),
                                    # Table panel
                                    html.Div(
                                        style={"width": "50%", "display": "flex",
                                               "flexDirection": "column", "overflow": "hidden"},
                                        children=[
                                            html.Div(
                                                style={"padding": "6px 10px", "borderBottom": _BORDER,
                                                       "fontSize": "11px", "color": "#555",
                                                       "flexShrink": "0"},
                                                children=[
                                                    html.Span("Top Spots  ", style={"textTransform": "uppercase", "letterSpacing": "1px"}),
                                                    html.Span(id="species-badge",
                                                              style={"color": _ACCENT, "marginLeft": "8px"}),
                                                ],
                                            ),
                                            dash_table.DataTable(
                                                id="spots-table",
                                                columns=table_cols,
                                                data=_table_rows(all_spots),
                                                row_selectable="single",
                                                selected_rows=[],
                                                sort_action="native",
                                                filter_action="native",
                                                page_action="native",
                                                page_size=20,
                                                style_table={"overflowY": "auto", "flex": "1",
                                                             "background": _DARK_BG},
                                                style_header={"background": "#010409",
                                                              "color": "#888",
                                                              "fontWeight": "normal",
                                                              "fontSize": "11px",
                                                              "border": "none",
                                                              "textTransform": "uppercase",
                                                              "letterSpacing": "0.5px"},
                                                style_cell={"background": _DARK_BG,
                                                            "color": _TEXT_COL,
                                                            "border": "none",
                                                            "borderBottom": _BORDER,
                                                            "fontSize": "12px",
                                                            "padding": "6px 8px"},
                                                style_data_conditional=[{
                                                    "if": {"state": "selected"},
                                                    "background": "#1f2937 !important",
                                                    "border": f"1px solid {_ACCENT} !important",
                                                }],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Hidden stores
            dcc.Store(id="store-spots",      data=all_spots),
            dcc.Store(id="store-week-dates", data=week_dates),
            dcc.Store(id="store-selected",   data=None),  # loc_id or _spot_N of selected spot
            dcc.Store(id="store-region",     data={"region": region, "resolution": resolution}),
        ],
    )

    # ── Callbacks ──────────────────────────────────────────────────────────

    # 1.  Marker click OR table row select → update store-selected
    @app.callback(
        Output("store-selected", "data"),
        Input("spots-layer", "clickData"),
        Input("spots-table", "selected_rows"),
        State("store-spots", "data"),
        prevent_initial_call=True,
    )
    def _select_spot(click_data, selected_rows, spots):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        trigger = ctx.triggered[0]["prop_id"]
        if "spots-layer" in trigger and click_data:
            props = click_data.get("properties") or {}
            return click_data.get("id") or props.get("loc_id") or None
        if "spots-table" in trigger and selected_rows and spots:
            idx = selected_rows[0]
            if 0 <= idx < len(spots):
                s = spots[idx]
                return s["loc_id"] or f"_spot_{s['rank']}"
        return dash.no_update

    # 2.  store-selected + ctrl-week → weekly chart
    @app.callback(
        Output("weekly-chart", "figure"),
        Output("species-badge", "children"),
        Input("store-selected", "data"),
        Input("ctrl-week", "value"),
        State("store-spots", "data"),
        State("store-week-dates", "data"),
    )
    def _update_chart(selected_id, week_idx, spots, wdates):
        if not selected_id or not spots:
            return _blank_chart(), ""
        spot = next(
            (s for s in spots if (s["loc_id"] or f"_spot_{s['rank']}") == selected_id),
            None,
        )
        if not spot:
            return _blank_chart(), ""
        n_sp = len(spot.get("species_codes") or [])
        badge = f"{n_sp} potential lifers" if n_sp else ""
        return _weekly_chart(spot, wdates or [], highlight_week=week_idx), badge

    # 3.  ctrl-week + store-region → raster image overlay URL
    @app.callback(
        Output("raster-overlay", "url"),
        Input("ctrl-week", "value"),
        State("store-week-dates", "data"),
        State("store-region", "data"),
    )
    def _update_raster(week_idx, wdates, reg_info):
        if not wdates or week_idx is None or week_idx >= len(wdates) or not reg_info:
            return ""
        url = _raster_url(reg_info["region"], reg_info["resolution"], wdates[week_idx])
        return url or ""

    # 4.  ctrl-week → sync table row selection to peak-week spot (optional: noop)
    #     (we keep table row in sync when user clicks a marker)
    @app.callback(
        Output("spots-table", "selected_rows"),
        Input("store-selected", "data"),
        State("store-spots", "data"),
    )
    def _sync_table_row(selected_id, spots):
        if not selected_id or not spots:
            return []
        for i, s in enumerate(spots):
            if (s["loc_id"] or f"_spot_{s['rank']}") == selected_id:
                return [i]
        return []

    # 5.  topbar status
    @app.callback(
        Output("topbar-status", "children"),
        Input("store-spots", "data"),
    )
    def _topbar(spots):
        if not spots:
            return ""
        return f"{len(spots)} spots loaded"

    # 6.  KML download
    @app.callback(
        Output("dl-kml", "data"),
        Input("btn-kml", "n_clicks"),
        State("store-spots", "data"),
        prevent_initial_call=True,
    )
    def _dl_kml(_, spots):
        if not spots:
            return dash.no_update
        kml_str = _spots_to_kml(spots)
        return dcc.send_string(kml_str, filename="pylifer_spots.kml")

    # 7.  CSV download
    @app.callback(
        Output("dl-csv", "data"),
        Input("btn-csv", "n_clicks"),
        State("store-spots", "data"),
        State("store-week-dates", "data"),
        prevent_initial_call=True,
    )
    def _dl_csv(_, spots, wdates):
        if not spots:
            return dash.no_update
        csv_str = _spots_to_csv(spots, wdates or [])
        return dcc.send_string(csv_str, filename="pylifer_spots.csv")

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from top_spots import (
        compute_top_spots,
        load_config,
        load_needed_species,
        load_hotspot_grid,
        HOTSPOT_DBS,
    )
    from map_lifers import get_boundary

    cfg = load_config()
    _hs_cache: dict[str, dict | None] = {}
    all_spots:  list[dict] = []
    week_dates: list[str]  = []
    primary_region     = args.regions[0]
    primary_resolution = args.resolution

    for region in args.regions:
        print(f"\nPreparing region {region} …")
        boundary = get_boundary(region)

        db_prefix = region.split("-")[0]
        if db_prefix not in _hs_cache:
            db_path = HOTSPOT_DBS.get(db_prefix)
            _hs_cache[db_prefix] = (
                load_hotspot_grid(db_path) if (db_path and db_path.exists()) else None
            )
        hs_db = _hs_cache[db_prefix]

        needed = load_needed_species(
            region, args.resolution, cfg,
            offline=args.offline, ebird_csv=args.ebird_csv,
        )

        spots, wdates = compute_top_spots(
            region, args.resolution, args.metric,
            args.n, args.per_unit, args.min_km,
            needed, boundary, hs_db,
        )
        all_spots.extend(spots)
        if wdates:
            week_dates = wdates

    if not all_spots:
        sys.exit("No spots found — make sure tifs are cached.")

    app = build_app(all_spots, week_dates,
                    region=primary_region, resolution=primary_resolution)
    print(f"\n  Dash app running → http://localhost:{args.port}/\n")
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()



def _weekly_chart(spot: dict, week_dates: list[str]) -> go.Figure:
    y = spot.get("weekly")
    if not y or not week_dates:
        return _blank_chart()

    labels: list[str] = []
    for ds in week_dates:
        try:
            d = date.fromisoformat(ds)
            labels.append(d.strftime("%-d %b"))
        except Exception:
            labels.append(ds)

    peak_i = int(max(range(len(y)), key=lambda i: y[i]))
    colours = [_PEAK_COL if i == peak_i else _ACCENT for i in range(len(y))]

    ebird_url = f"https://ebird.org/hotspot/{spot['loc_id']}" if spot["loc_id"] else None
    title_txt = (
        f"<a href='{ebird_url}' target='_blank' style='color:{_ACCENT}'>{spot['loc_name']}</a>"
        if ebird_url else spot["loc_name"]
    )
    state_s  = spot["state"].replace("US-", "") if spot["state"] else ""
    county_s = spot["county"].split("-")[-1] if spot["county"] else ""
    subtitle = "  |  ".join(filter(None, [state_s, county_s,
                                           f"Score {spot['score']:.1f}",
                                           f"eBird spp {spot['hs_spp']}" if spot["hs_spp"] else None]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=y,
        marker_color=colours,
        hovertemplate="%{x}<br><b>%{y:.0f} lifers</b><extra></extra>",
    ))
    fig.add_annotation(
        x=labels[peak_i], y=y[peak_i],
        text=f"Peak week<br>{y[peak_i]:.0f} lifers",
        showarrow=True, arrowhead=2, arrowcolor=_PEAK_COL,
        font=dict(color=_PEAK_COL, size=11),
        bgcolor="rgba(0,0,0,0.65)", bordercolor=_PEAK_COL, borderpad=4,
        ay=-45,
    )
    fig.update_layout(
        title=dict(
            text=f"{title_txt}<br><sup style='color:#888'>{subtitle}</sup>",
            font=dict(size=15, color=_TEXT_COL), x=0.01,
        ),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PANEL_BG,
        font_color=_TEXT_COL,
        xaxis=dict(
            title="Week", tickangle=-50, tickfont=dict(size=8),
            gridcolor=_GRID_COL, showline=False,
            tickmode="array",
            tickvals=labels[::4],
            ticktext=labels[::4],
        ),
        yaxis=dict(title="Potential lifers", gridcolor=_GRID_COL, zeroline=False),
        margin=dict(l=55, r=20, t=65, b=80),
        bargap=0.1,
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app builder
# ---------------------------------------------------------------------------

# Inline JS assigned via dl.assign — dash_leaflet pattern for pointToLayer
_POINT_TO_LAYER = js_assign("""
function(feature, latlng, context) {
    const p = feature.properties;
    const icon = L.divIcon({
        className: '',
        html: `<div style="
            background: ${p.colour};
            color: #fff;
            font-weight: bold;
            font-size: 11px;
            width: 26px; height: 26px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            border: 2px solid rgba(255,255,255,0.25);
            box-shadow: 0 2px 6px rgba(0,0,0,0.6);
            cursor: pointer;">${p.rank}</div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13],
    });
    return L.marker(latlng, {icon});
}
""")

_ON_EACH_FEATURE = js_assign("""
function(feature, layer, context) {
    const p = feature.properties;
    const url = p.loc_id ? `https://ebird.org/hotspot/${p.loc_id}` : null;
    const nameHtml = url
        ? `<a href="${url}" target="_blank" style="color:#58a6ff">${p.loc_name}</a>`
        : p.loc_name;
    layer.bindPopup(
        `<div style="font-family:sans-serif;min-width:180px">
          <b>#${p.rank}</b> ${nameHtml}<br/>
          <span style="color:#999">Score: ${p.score.toFixed(1)}
          &nbsp;|&nbsp; eBird spp: ${p.hs_spp ?? '—'}</span><br/>
          <span style="color:#aaa; font-size:0.85em">
            ${p.state || '—'} &nbsp;/&nbsp; ${p.county || '—'}
          </span>
        </div>`
    );
}
""")


def build_app(all_spots: list[dict], week_dates: list[str]) -> dash.Dash:
    app = dash.Dash(__name__, title="PyLifer — Top Spots")
    app.index_string = app.index_string.replace(
        "</head>",
        "<style>body,html{margin:0;padding:0;background:#0e1117;color:#c9d1d9}"
        ".leaflet-popup-content-wrapper{background:#161b22;color:#c9d1d9;border:1px solid #30363d}"
        ".leaflet-popup-tip{background:#161b22}</style></head>",
    )

    if not all_spots:
        app.layout = html.Div(
            "No spots computed. Run top_spots.py first to build the cache.",
            style={"padding": "2rem", "color": _TEXT_COL, "fontFamily": "sans-serif"},
        )
        return app

    score_max = max(s["score"] for s in all_spots)

    # Build GeoJSON feature collection
    features = [
        {
            "type": "Feature",
            "id":   spot["loc_id"] or f"_spot_{spot['rank']}",
            "geometry": {"type": "Point", "coordinates": [spot["lon"], spot["lat"]]},
            "properties": {
                "rank":     spot["rank"],
                "score":    spot["score"],
                "loc_name": spot["loc_name"],
                "loc_id":   spot["loc_id"],
                "state":    spot["state"],
                "county":   spot["county"],
                "hs_spp":   spot["hs_spp"],
                "colour":   _score_colour(spot["score"], score_max),
            },
        }
        for spot in all_spots
    ]
    geojson_data = {"type": "FeatureCollection", "features": features}

    # Mean centre for initial map view
    mean_lat = sum(s["lat"] for s in all_spots) / len(all_spots)
    mean_lon = sum(s["lon"] for s in all_spots) / len(all_spots)

    app.layout = html.Div(
        style={"display": "flex", "height": "100vh", "width": "100vw",
               "background": _DARK_BG, "overflow": "hidden"},
        children=[
            # ── Left sidebar (10 %) ──────────────────────────────────────
            html.Div(
                style={"width": "10%", "minHeight": "100vh",
                       "background": "#0d1117", "borderRight": "1px solid #21262d"},
            ),

            # ── Main content (90 %) ─────────────────────────────────────
            html.Div(
                style={"width": "90%", "display": "flex", "flexDirection": "column"},
                children=[

                    # Map — top 50 vh
                    dl.Map(
                        id="spots-map",
                        center={"lat": mean_lat, "lng": mean_lon},
                        zoom=4,
                        style={"height": "50vh", "width": "100%"},
                        children=[
                            dl.TileLayer(
                                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                                attribution=(
                                    '&copy; <a href="https://www.openstreetmap.org/copyright">'
                                    'OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
                                ),
                                maxZoom=19,
                            ),
                            dl.GeoJSON(
                                id="spots-layer",
                                data=geojson_data,
                                options={
                                    "pointToLayer":  _POINT_TO_LAYER,
                                    "onEachFeature": _ON_EACH_FEATURE,
                                },
                                zoomToBoundsOnClick=False,
                            ),
                        ],
                    ),

                    # Chart — bottom 50 vh
                    dcc.Graph(
                        id="weekly-chart",
                        style={"height": "50vh", "width": "100%"},
                        config={"displayModeBar": False},
                        figure=_blank_chart(),
                    ),
                ],
            ),
        ],
    )

    # ---- callback: marker click → chart --------------------------------
    @app.callback(
        Output("weekly-chart", "figure"),
        Input("spots-layer", "clickData"),
        prevent_initial_call=True,
    )
    def update_chart(click_data: dict | None) -> go.Figure:
        if not click_data:
            return _blank_chart()
        props = click_data.get("properties") or {}
        fid   = click_data.get("id") or props.get("loc_id") or ""
        spot  = next(
            (s for s in all_spots if (s["loc_id"] or f"_spot_{s['rank']}") == fid),
            None,
        )
        return _weekly_chart(spot, week_dates) if spot else _blank_chart()

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Import computation helpers from top_spots
    from top_spots import (
        compute_top_spots,
        load_config,
        load_needed_species,
        load_hotspot_grid,
        HOTSPOT_DBS,
    )
    from map_lifers import get_boundary

    cfg = load_config()
    _hs_cache: dict[str, dict | None] = {}
    all_spots:  list[dict] = []
    week_dates: list[str]  = []

    for region in args.regions:
        print(f"\nPreparing region {region} …")
        boundary = get_boundary(region)

        db_prefix = region.split("-")[0]
        if db_prefix not in _hs_cache:
            db_path = HOTSPOT_DBS.get(db_prefix)
            _hs_cache[db_prefix] = (
                load_hotspot_grid(db_path) if (db_path and db_path.exists()) else None
            )
        hs_db = _hs_cache[db_prefix]

        needed = load_needed_species(
            region, args.resolution, cfg,
            offline=args.offline, ebird_csv=args.ebird_csv,
        )

        spots, wdates = compute_top_spots(
            region, args.resolution, args.metric,
            args.n, args.per_unit, args.min_km,
            needed, boundary, hs_db,
        )
        all_spots.extend(spots)
        if wdates:
            week_dates = wdates   # same for all regions

    if not all_spots:
        sys.exit("No spots found — make sure tifs are cached.")

    app = build_app(all_spots, week_dates)
    print(f"\n  Dash app running → http://localhost:{args.port}/\n")
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
