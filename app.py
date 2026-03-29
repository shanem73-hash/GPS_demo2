from __future__ import annotations

import json
from datetime import datetime, timezone
import base64
from pathlib import Path
import logging
from typing import Iterable

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from skyfield.api import EarthSatellite, load

GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
BEIDOU_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"
EARTH_TEXTURE_PATH = Path(__file__).parent / "assets" / "earth_beauty.jpg"
FUTURE_STEP_SECONDS = 30
FUTURE_STEPS = 21
logger = logging.getLogger(__name__)


def parse_tle_lines(lines: Iterable[str]) -> list[tuple[str, str, str]]:
    cleaned = [ln.strip() for ln in lines if ln.strip()]
    sats: list[tuple[str, str, str]] = []
    i = 0
    while i + 2 < len(cleaned):
        name, l1, l2 = cleaned[i], cleaned[i + 1], cleaned[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            sats.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return sats


@st.cache_data(ttl=300)
def fetch_tles(url: str) -> list[tuple[str, str, str]]:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch TLEs from %s: %s", url, e)
        return []

    return parse_tle_lines(resp.text.splitlines())


def build_payload(show_gps: bool, show_beidou: bool, max_trails: int, orbit_points: int) -> dict:
    ts = load.timescale()
    t0 = ts.now()

    all_tles = []
    if show_gps:
        all_tles += [("GPS",) + t for t in fetch_tles(GPS_TLE_URL)]
    if show_beidou:
        all_tles += [("BeiDou",) + t for t in fetch_tles(BEIDOU_TLE_URL)]

    satellites = [(sys, EarthSatellite(l1, l2, name, ts)) for sys, name, l1, l2 in all_tles]

    sat_points = []
    for sys, sat in satellites:
        x, y, z = sat.at(t0).position.km
        sat_points.append({"sys": sys, "name": sat.name, "x": float(x * 1000), "y": float(y * 1000), "z": float(z * 1000)})

    # complete-orbit trails by period from mean motion
    trails = []
    for sys, sat in satellites[:max_trails]:
        n = float(getattr(sat.model, "no_kozai", 0.0))
        if n <= 0:
            continue
        period_min = 2 * np.pi / n
        mins = np.linspace(0, period_min, orbit_points)
        tt = t0.tt + mins / 1440.0
        t_arr = ts.tt_jd(tt)
        p = sat.at(t_arr).position.km.T
        trails.append(
            {
                "sys": sys,
                "name": sat.name,
                "positions": [[float(px * 1000), float(py * 1000), float(pz * 1000)] for px, py, pz in p],
            }
        )

    # future positions for smooth client-side updates every 30s
    future = []
    for k in range(FUTURE_STEPS):
        t = ts.tt_jd(t0.tt + (k * FUTURE_STEP_SECONDS) / 86400.0)
        frame = []
        for sys, sat in satellites:
            x, y, z = sat.at(t).position.km
            frame.append({"name": sat.name, "x": float(x * 1000), "y": float(y * 1000), "z": float(z * 1000)})
        future.append(frame)

    return {
        "satellites": sat_points,
        "trails": trails,
        "future": future,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    }


def _earth_texture_data_url() -> str | None:
    if not EARTH_TEXTURE_PATH.exists():
        return None
    raw = EARTH_TEXTURE_PATH.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def cesium_html(payload: dict, use_world_terrain: bool, earth_data_url: str | None) -> str:
    data = json.dumps(payload)
    terrain_js = "Cesium.createWorldTerrainAsync()" if use_world_terrain else "new Cesium.EllipsoidTerrainProvider()"

    return f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <script src=\"https://cesium.com/downloads/cesiumjs/releases/1.122/Build/Cesium/Cesium.js\"></script>
  <link href=\"https://cesium.com/downloads/cesiumjs/releases/1.122/Build/Cesium/Widgets/widgets.css\" rel=\"stylesheet\" />
  <style>
    html, body, #cesiumContainer {{ width:100%; height:100%; margin:0; padding:0; overflow:hidden; background:#000; }}
  </style>
</head>
<body>
<div id=\"cesiumContainer\"></div>
<script>
(async function() {{
  const payload = {data};
  const earthDataUrl = {json.dumps(earth_data_url)};

  let terrainProvider;
  try {{
    terrainProvider = await {terrain_js};
  }} catch (e) {{
    terrainProvider = new Cesium.EllipsoidTerrainProvider();
  }}

  const viewer = new Cesium.Viewer('cesiumContainer', {{
    shouldAnimate: true,
    baseLayerPicker: false,
    timeline: false,
    animation: false,
    geocoder: false,
    homeButton: true,
    sceneModePicker: true,
    navigationHelpButton: true,
    infoBox: true,
    terrainProvider: terrainProvider
  }});

  // Ensure at least one imagery layer is present so Earth is visible.
  let imagerySet = false;

  if (earthDataUrl) {{
    try {{
      const single = new Cesium.SingleTileImageryProvider({{
        url: earthDataUrl,
        rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90)
      }});
      viewer.imageryLayers.removeAll();
      viewer.imageryLayers.addImageryProvider(single);
      imagerySet = true;
    }} catch(e) {{}}
  }}

  if (!imagerySet) {{
    try {{
      const arcgis = await Cesium.ArcGisMapServerImageryProvider.fromUrl(
        'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
      );
      viewer.imageryLayers.removeAll();
      viewer.imageryLayers.addImageryProvider(arcgis);
      imagerySet = true;
    }} catch(e) {{}}
  }}

  if (!imagerySet) {{
    try {{
      const osm = new Cesium.OpenStreetMapImageryProvider({{ url: 'https://tile.openstreetmap.org/' }});
      viewer.imageryLayers.removeAll();
      viewer.imageryLayers.addImageryProvider(osm);
      imagerySet = true;
    }} catch(e) {{}}
  }}

  viewer.scene.globe.baseColor = Cesium.Color.DARKSLATEGRAY;
  viewer.scene.globe.enableLighting = true;
  viewer.scene.skyAtmosphere.show = true;
  viewer.scene.fog.enabled = true;

  // Guaranteed Earth body (independent of globe imagery providers)
  viewer.entities.add({{
    name: 'EarthBody',
    position: Cesium.Cartesian3.ZERO,
    ellipsoid: {{
      radii: new Cesium.Cartesian3(6378137.0, 6378137.0, 6356752.3),
      material: Cesium.Color.CORNFLOWERBLUE.withAlpha(1.0),
      outline: true,
      outlineColor: Cesium.Color.WHITE.withAlpha(0.35),
      outlineWidth: 1
    }}
  }});

  const satByName = new Map();
  const colorBySys = {{"GPS": Cesium.Color.GOLD, "BeiDou": Cesium.Color.CYAN}};

  for (const sat of payload.satellites) {{
    const ent = viewer.entities.add({{
      name: sat.name,
      position: new Cesium.Cartesian3(sat.x, sat.y, sat.z),
      point: {{ pixelSize: 7, color: colorBySys[sat.sys] || Cesium.Color.WHITE, outlineColor: Cesium.Color.WHITE, outlineWidth: 1 }},
      label: {{ text: sat.name, font: '12px sans-serif', show: false, fillColor: Cesium.Color.WHITE, pixelOffset: new Cesium.Cartesian2(0, -16) }},
      description: `<b>${{sat.name}}</b><br/>${{sat.sys}}`
    }});
    satByName.set(sat.name, ent);
  }}

  const trailColorBySys = {{"GPS": Cesium.Color.YELLOW.withAlpha(0.7), "BeiDou": Cesium.Color.CYAN.withAlpha(0.7)}};
  for (const tr of payload.trails) {{
    const pts = tr.positions.map(p => new Cesium.Cartesian3(p[0], p[1], p[2]));
    viewer.entities.add({{
      name: `Orbit: ${{tr.name}}`,
      polyline: {{ positions: pts, width: 1.6, material: trailColorBySys[tr.sys] || Cesium.Color.LIME.withAlpha(0.7) }}
    }});
  }}

  viewer.clock.currentTime = Cesium.JulianDate.now();
  // Force camera toward Earth center so the globe is always in view.
  viewer.camera.setView({{
    destination: new Cesium.Cartesian3(2.1e7, 2.1e7, 1.4e7)
  }});

  // Smooth update every 30s (no full redraw)
  let idx = 0;
  setInterval(() => {{
    idx = (idx + 1) % payload.future.length;
    for (const p of payload.future[idx]) {{
      const ent = satByName.get(p.name);
      if (ent) ent.position = new Cesium.Cartesian3(p.x, p.y, p.z);
    }}
  }}, 30000);
}})();
</script>
</body>
</html>
"""


def main():
    st.set_page_config(page_title="GPS Demo 2", page_icon="🌍", layout="wide")
    st.title("🌍 GPS_demo2 — Google-Earth-like GNSS Viewer (Cesium)")

    with st.sidebar:
        show_gps = st.checkbox("Show GPS", value=True)
        show_beidou = st.checkbox("Show BeiDou", value=True)
        use_world_terrain = st.checkbox("Use world terrain (if available)", value=False)
        max_trails = st.slider("Max orbit trails", 10, 120, 50)
        orbit_points = st.slider("Orbit points", 120, 400, 220)
        height_px = st.slider("Viewer height", 700, 1200, 980)
        if st.button("Refresh data now"):
            st.cache_data.clear()
            st.rerun()

    try:
        payload = build_payload(show_gps, show_beidou, max_trails, orbit_points)
    except Exception as e:
        st.error(f"Failed to build satellite view: {e}")
        st.stop()

    sat_count = len(payload["satellites"])
    st.caption(f"UTC data epoch: {payload['timestamp']} | Satellites: {sat_count}")

    if sat_count == 0 and (show_gps or show_beidou):
        st.warning("No satellites loaded. Data source may be temporarily unavailable; try Refresh.")

    earth_data_url = _earth_texture_data_url()
    html = cesium_html(payload, use_world_terrain=use_world_terrain, earth_data_url=earth_data_url)
    components.html(html, height=height_px + 8)


if __name__ == "__main__":
    main()
