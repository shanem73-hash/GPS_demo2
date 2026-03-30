from __future__ import annotations

import json
from datetime import datetime, timezone
import base64
from pathlib import Path
import logging
import os
import threading
import http.server
import socketserver
import time
import hashlib
from typing import Iterable

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from skyfield.api import EarthSatellite, load, wgs84

GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
BEIDOU_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"
STATIONS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"
STARLINK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
ACTIVE_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
EARTH_TEXTURE_PATH = Path(__file__).parent / "assets" / "earth_beauty.jpg"
FUTURE_STEP_SECONDS = 30
FUTURE_STEPS = 21
logger = logging.getLogger(__name__)

_EPHEMERIS = None


def _get_ephemeris():
    global _EPHEMERIS
    if _EPHEMERIS is None:
        _EPHEMERIS = load("de421.bsp")
    return _EPHEMERIS


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


def _tle_cache_path(url: str) -> Path:
    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return TLE_CACHE_DIR / f"{key}.tle"


def _fallback_tle_path(url: str) -> Path | None:
    try:
        if "GROUP=" in url:
            group = url.split("GROUP=", 1)[1].split("&", 1)[0].strip().lower()
            p = FALLBACK_TLE_DIR / f"{group}.txt"
            if p.exists():
                return p
    except Exception:
        return None
    return None


def _tle_candidate_urls(url: str) -> list[str]:
    # Primary endpoint + text fallback endpoint used by CelesTrak.
    out = [url]
    try:
        if "GROUP=" in url and "gp.php" in url:
            group = url.split("GROUP=", 1)[1].split("&", 1)[0].strip().lower()
            out.append(f"https://celestrak.org/NORAD/elements/{group}.txt")
    except Exception:
        pass
    # de-dup preserve order
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


@st.cache_data(ttl=300)
def fetch_tles(url: str) -> list[tuple[str, str, str]]:
    cache_path = _tle_cache_path(url)
    headers = {
        "User-Agent": "GPS_demo2/1.0 (+streamlit-community)",
        "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
    }

    for candidate in _tle_candidate_urls(url):
        for attempt in range(3):
            try:
                resp = requests.get(candidate, timeout=20, headers=headers)
                resp.raise_for_status()
                text = resp.text
                sats = parse_tle_lines(text.splitlines())
                if sats:
                    cache_path.write_text(text, encoding="utf-8")
                    return sats
                raise requests.RequestException(f"No valid TLE triplets from {candidate}")
            except requests.RequestException as e:
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
                else:
                    logger.warning("Failed to fetch TLEs from %s after retries: %s", candidate, e)

    if cache_path.exists():
        try:
            cached = cache_path.read_text(encoding="utf-8")
            logger.warning("Using cached TLE data for %s", url)
            sats = parse_tle_lines(cached.splitlines())
            if sats:
                return sats
        except Exception as e:
            logger.warning("Failed to read cached TLEs for %s: %s", url, e)

    fallback = _fallback_tle_path(url)
    if fallback is not None:
        try:
            txt = fallback.read_text(encoding="utf-8")
            sats = parse_tle_lines(txt.splitlines())
            if sats:
                logger.warning("Using bundled fallback TLE data for %s", url)
                return sats
        except Exception as e:
            logger.warning("Failed to read bundled fallback TLEs for %s: %s", url, e)

    return []


@st.cache_data(ttl=300)
def fetch_station_tles() -> dict[str, tuple[str, str, str] | None]:
    result: dict[str, tuple[str, str, str] | None] = {"iss": None, "tiangong": None}
    for name, l1, l2 in fetch_tles(STATIONS_TLE_URL):
        n = name.upper()
        if result["iss"] is None and "ISS" in n:
            result["iss"] = (name, l1, l2)
        if result["tiangong"] is None and ("TIANGONG" in n or "CSS" in n):
            result["tiangong"] = (name, l1, l2)
    return result


@st.cache_data(ttl=300)
def fetch_earth_resource_tles(limit: int) -> list[tuple[str, str, str]]:
    keys = ("LANDSAT", "SENTINEL")
    out: list[tuple[str, str, str]] = []
    for name, l1, l2 in fetch_tles(ACTIVE_TLE_URL):
        n = name.upper()
        if any(k in n for k in keys):
            out.append((name, l1, l2))
            if len(out) >= limit:
                break
    return out


def _assign_orbit_groups(entries: list[dict], group_count: int) -> None:
    valid = [e for e in entries if e.get("raan") is not None]
    valid.sort(key=lambda e: e["raan"])
    n = len(valid)
    if n == 0:
        return
    for i, e in enumerate(valid):
        grp = min(group_count - 1, int(i * group_count / n))
        e["orbit_group"] = grp


def build_payload(
    show_gps: bool,
    show_beidou: bool,
    show_starlink: bool,
    show_earth_resource: bool,
    show_iss: bool,
    show_tiangong: bool,
    starlink_limit: int,
    earth_resource_limit: int,
    max_trails: int,
    orbit_points: int,
    include_history: bool,
    history_days: int,
    history_step_seconds: int,
) -> dict:
    ts = load.timescale()
    t0 = ts.now()

    all_tles = []
    if show_gps:
        all_tles += [("GPS",) + t for t in fetch_tles(GPS_TLE_URL)]
    if show_beidou:
        all_tles += [("BeiDou",) + t for t in fetch_tles(BEIDOU_TLE_URL)]
    if show_starlink:
        starlink_tles = fetch_tles(STARLINK_TLE_URL)
        if starlink_limit > 0:
            starlink_tles = starlink_tles[:starlink_limit]
        all_tles += [("Starlink",) + t for t in starlink_tles]

    if show_earth_resource:
        resource_tles = fetch_earth_resource_tles(earth_resource_limit)
        all_tles += [("EarthResource",) + t for t in resource_tles]

    sat_entries = []
    for sys, name, l1, l2 in all_tles:
        sat = EarthSatellite(l1, l2, name, ts)
        parts = l2.split()
        raan = None
        if len(parts) >= 4:
            try:
                raan = float(parts[3])
            except ValueError:
                raan = None

        subtype = ""
        if sys == "EarthResource":
            n = name.upper()
            if "LANDSAT" in n:
                subtype = "Landsat"
            elif "SENTINEL" in n:
                subtype = "Sentinel"
            else:
                subtype = "EarthResource"

        sat_entries.append({"sys": sys, "subtype": subtype, "sat": sat, "raan": raan, "orbit_group": 0})

    _assign_orbit_groups([e for e in sat_entries if e["sys"] == "GPS"], group_count=6)
    _assign_orbit_groups([e for e in sat_entries if e["sys"] == "BeiDou"], group_count=3)
    _assign_orbit_groups([e for e in sat_entries if e["sys"] == "Starlink"], group_count=12)
    _assign_orbit_groups([e for e in sat_entries if e["sys"] == "EarthResource"], group_count=6)

    sat_points = []
    for e in sat_entries:
        sat = e["sat"]
        g = sat.at(t0)
        g_prev = sat.at(ts.tt_jd(t0.tt - FUTURE_STEP_SECONDS / 86400.0))
        x, y, z = g.position.km
        vx, vy, vz = g.velocity.km_per_s
        sp = wgs84.subpoint(g)
        sat_points.append(
            {
                "sys": e["sys"],
                "name": sat.name,
                "orbit_group": int(e.get("orbit_group", 0)),
                "subtype": e.get("subtype", ""),
                "x": float(x * 1000),
                "y": float(y * 1000),
                "z": float(z * 1000),
                "px": float(g_prev.position.km[0] * 1000),
                "py": float(g_prev.position.km[1] * 1000),
                "pz": float(g_prev.position.km[2] * 1000),
                "lat": float(sp.latitude.degrees),
                "lon": float(sp.longitude.degrees),
                "alt_km": float(sp.elevation.km),
                "speed_km_s": float(np.sqrt(vx * vx + vy * vy + vz * vz)),
            }
        )

    trails = []
    for e in sat_entries[:max_trails]:
        sat = e["sat"]
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
                "sys": e["sys"],
                "name": sat.name,
                "orbit_group": int(e.get("orbit_group", 0)),
                "subtype": e.get("subtype", ""),
                "period_sec": float(period_min * 60.0),
                "positions": [[float(px * 1000), float(py * 1000), float(pz * 1000)] for px, py, pz in p],
            }
        )

    future = []
    for k in range(FUTURE_STEPS):
        t = ts.tt_jd(t0.tt + (k * FUTURE_STEP_SECONDS) / 86400.0)
        frame = []
        for e in sat_entries:
            sat = e["sat"]
            x, y, z = sat.at(t).position.km
            frame.append({"name": sat.name, "x": float(x * 1000), "y": float(y * 1000), "z": float(z * 1000)})
        future.append(frame)

    history_days = max(1, int(history_days))
    history_step_seconds = max(60, int(history_step_seconds))
    history_steps = int((history_days * 24 * 3600) / history_step_seconds) + 1
    history24 = []
    if include_history:
        total_seconds = history_days * 24 * 3600
        for k in range(history_steps):
            t = ts.tt_jd(t0.tt - (total_seconds - k * history_step_seconds) / 86400.0)
            frame = []
            for e in sat_entries:
                sat = e["sat"]
                x, y, z = sat.at(t).position.km
                frame.append({"name": sat.name, "x": float(x * 1000), "y": float(y * 1000), "z": float(z * 1000)})
            history24.append(frame)

    specials = []
    station_tles = fetch_station_tles() if (show_iss or show_tiangong) else {}
    for kind in ("iss", "tiangong"):
        if kind == "iss" and not show_iss:
            continue
        if kind == "tiangong" and not show_tiangong:
            continue
        tle = station_tles.get(kind)
        if tle is None:
            continue
        s_name, s_l1, s_l2 = tle
        s_sat = EarthSatellite(s_l1, s_l2, s_name, ts)
        sg = s_sat.at(t0)
        sg_prev = s_sat.at(ts.tt_jd(t0.tt - FUTURE_STEP_SECONDS / 86400.0))
        x, y, z = sg.position.km
        svx, svy, svz = sg.velocity.km_per_s
        ssp = wgs84.subpoint(sg)
        s_future = []
        for k in range(FUTURE_STEPS):
            t = ts.tt_jd(t0.tt + (k * FUTURE_STEP_SECONDS) / 86400.0)
            fx, fy, fz = s_sat.at(t).position.km
            s_future.append({"x": float(fx * 1000), "y": float(fy * 1000), "z": float(fz * 1000)})

        s_history24 = []
        if include_history:
            total_seconds = history_days * 24 * 3600
            for k in range(history_steps):
                t = ts.tt_jd(t0.tt - (total_seconds - k * history_step_seconds) / 86400.0)
                hx, hy, hz = s_sat.at(t).position.km
                s_history24.append({"x": float(hx * 1000), "y": float(hy * 1000), "z": float(hz * 1000)})
        specials.append(
            {
                "kind": kind,
                "name": s_name,
                "x": float(x * 1000),
                "y": float(y * 1000),
                "z": float(z * 1000),
                "px": float(sg_prev.position.km[0] * 1000),
                "py": float(sg_prev.position.km[1] * 1000),
                "pz": float(sg_prev.position.km[2] * 1000),
                "future": s_future,
                "history24": s_history24,
                "lat": float(ssp.latitude.degrees),
                "lon": float(ssp.longitude.degrees),
                "alt_km": float(ssp.elevation.km),
                "speed_km_s": float(np.sqrt(svx * svx + svy * svy + svz * svz)),
            }
        )

    return {
        "satellites": sat_points,
        "trails": trails,
        "future": future,
        "history24": history24,
        "history_days": history_days,
        "history_step_seconds": history_step_seconds,
        "specials": specials,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    }


def build_solar_payload() -> dict:
    ts = load.timescale()
    t0 = ts.now()
    eph = _get_ephemeris()
    sun = eph["sun"]
    earth = eph["earth"]
    moon = eph["moon"]

    step_seconds = 600
    span_days = 14
    frame_count = int((span_days * 2 * 24 * 3600) / step_seconds) + 1

    frames = []
    for i in range(frame_count):
        dt_sec = -span_days * 24 * 3600 + i * step_seconds
        t = ts.tt_jd(t0.tt + dt_sec / 86400.0)
        e = sun.at(t).observe(earth).position.km
        m_rel = earth.at(t).observe(moon).position.km
        m = e + m_rel
        frames.append(
            {
                "t_unix": float(t.utc_datetime().timestamp()),
                "earth": [float(e[0] * 1000), float(e[1] * 1000), float(e[2] * 1000)],
                "moon": [float(m[0] * 1000), float(m[1] * 1000), float(m[2] * 1000)],
            }
        )

    # Geometry from real ephemeris samples
    earth_orbit = []
    for i in range(0, 366):
        t = ts.tt_jd(t0.tt - 182.5 / 365.25 + i / 365.25)
        e = sun.at(t).observe(earth).position.km
        earth_orbit.append([float(e[0] * 1000), float(e[1] * 1000), float(e[2] * 1000)])

    moon_rel_orbit = []
    for i in range(0, 145):
        t = ts.tt_jd(t0.tt - 13.5 / 27.32 + (i / 144.0) * 27.32)
        mr = earth.at(t).observe(moon).position.km
        moon_rel_orbit.append([float(mr[0] * 1000), float(mr[1] * 1000), float(mr[2] * 1000)])

    return {
        "solar": {
            "frames": frames,
            "step_seconds": step_seconds,
            "span_days": span_days,
            "earth_orbit": earth_orbit,
            "moon_rel_orbit": moon_rel_orbit,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        }
    }


def _earth_texture_data_url() -> str | None:
    if not EARTH_TEXTURE_PATH.exists():
        return None
    raw = EARTH_TEXTURE_PATH.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


VIEWER_PORT = 8603
VIEWER_DIR = Path(__file__).parent / ".viewer_site"
TLE_CACHE_DIR = Path(__file__).parent / ".tle_cache"
FALLBACK_TLE_DIR = Path(__file__).parent / "fallback_tles"
DEFAULT_VIEWER_HOST = os.getenv("GPS_VIEWER_HOST", "192.168.50.20")


def _viewer_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>GPS_demo2 Viewer</title>
  <script src=\"https://cesium.com/downloads/cesiumjs/releases/1.122/Build/Cesium/Cesium.js\"></script>
  <link href=\"https://cesium.com/downloads/cesiumjs/releases/1.122/Build/Cesium/Widgets/widgets.css\" rel=\"stylesheet\" />
  <style>
    html, body, #cesiumContainer { width:100%; height:100%; margin:0; padding:0; overflow:hidden; background:#000; }
    #status { position:fixed; left:8px; top:8px; z-index:99; color:#fff; background:rgba(0,0,0,.55); padding:5px 8px; border-radius:4px; font:12px sans-serif; }
    #simtime { position:fixed; left:8px; top:36px; z-index:99; color:#fff; background:rgba(0,0,0,.55); padding:4px 8px; border-radius:4px; font:12px monospace; display:none; }
  </style>
</head>
<body>
<div id=\"status\">Loading…</div>
<div id=\"simtime\"></div>
<div id=\"cesiumContainer\"></div>
<script>
(async function() {
  const statusEl = document.getElementById('status');
  const simtimeEl = document.getElementById('simtime');
  try {
    const payloadRes = await fetch('./payload.json?ts=' + Date.now());
    const cfgRes = await fetch('./config.json?ts=' + Date.now());
    const payload = await payloadRes.json();
    const cfg = await cfgRes.json();

    if (cfg.ionToken && cfg.ionToken.trim()) {
      Cesium.Ion.defaultAccessToken = cfg.ionToken.trim();
    }

    let terrainProvider = new Cesium.EllipsoidTerrainProvider();
    if (cfg.useWorldTerrain) {
      try { terrainProvider = await Cesium.createWorldTerrainAsync(); } catch (e) {}
    }

    const viewer = new Cesium.Viewer('cesiumContainer', {
      shouldAnimate: true,
      baseLayerPicker: false,
      timeline: false,
      animation: false,
      geocoder: false,
      homeButton: true,
      sceneModePicker: true,
      navigationHelpButton: true,
      infoBox: true,
      terrainProvider,
      baseLayer: false
    });

    if (cfg.viewMode === 'Solar System Lab') {
      viewer.scene.globe.show = false;
      viewer.scene.skyAtmosphere.show = false;
      viewer.scene.sun.show = false;
      viewer.scene.moon.show = false;

      const solar = payload.solar || {};
      const frames = Array.isArray(solar.frames) ? solar.frames : [];
      if (frames.length < 2) {
        statusEl.textContent = 'Solar data unavailable';
        return;
      }

      const earthTex = cfg.solarEarthTextureUrl || cfg.earthDataUrl;
      const realStartMs = Date.now();
      const DIST_SCALE = 1.0e9 / 1.495978707e11;
      const firstT = Number(frames[0].t_unix || 0);
      const lastT = Number(frames[frames.length - 1].t_unix || firstT + 1);
      const midT = 0.5 * (firstT + lastT);

      function simUnixNow() {
        if (cfg.cartoonMode) {
          const speedDaysPerSec = Math.max(0.01, Number(cfg.solarSpeedDaysPerSec || 5.0));
          const sim = midT + ((Date.now() - realStartMs) / 1000.0) * speedDaysPerSec * 86400.0;
          const span = Math.max(1, lastT - firstT);
          return firstT + (((sim - firstT) % span) + span) % span;
        }
        // realtime mode
        const now = Date.now() / 1000.0;
        return Math.max(firstT, Math.min(lastT, now));
      }

      function lerp3(a, b, r) {
        return new Cesium.Cartesian3(
          (a[0] + (b[0] - a[0]) * r) * DIST_SCALE,
          (a[1] + (b[1] - a[1]) * r) * DIST_SCALE,
          (a[2] + (b[2] - a[2]) * r) * DIST_SCALE
        );
      }

      function sampleBodies(tUnix) {
        let lo = 0, hi = frames.length - 1;
        while (lo + 1 < hi) {
          const mid = (lo + hi) >> 1;
          if ((frames[mid].t_unix || 0) <= tUnix) lo = mid;
          else hi = mid;
        }
        const f0 = frames[lo];
        const f1 = frames[Math.min(lo + 1, frames.length - 1)];
        const t0 = Number(f0.t_unix || tUnix);
        const t1 = Number(f1.t_unix || tUnix + 1);
        const r = t1 > t0 ? (tUnix - t0) / (t1 - t0) : 0.0;
        return {
          earth: lerp3(f0.earth, f1.earth, r),
          moon: lerp3(f0.moon, f1.moon, r),
        };
      }

      function polyFromList(list) {
        if (!Array.isArray(list)) return [];
        const out = [];
        for (const p of list) {
          if (!Array.isArray(p) || p.length < 3) continue;
          out.push(new Cesium.Cartesian3(Number(p[0]) * DIST_SCALE, Number(p[1]) * DIST_SCALE, Number(p[2]) * DIST_SCALE));
        }
        return out;
      }

      // Sun
      viewer.entities.add({
        name: 'Sun',
        position: Cesium.Cartesian3.ZERO,
        point: { pixelSize: 22, color: Cesium.Color.YELLOW, outlineColor: Cesium.Color.ORANGE, outlineWidth: 2 },
        label: {
          text: 'Sun', font: '14px sans-serif', fillColor: Cesium.Color.YELLOW,
          outlineColor: Cesium.Color.BLACK, outlineWidth: 2, style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(14, -12),
        }
      });

      const earthOrbit = polyFromList(solar.earth_orbit);
      if (earthOrbit.length > 1) {
        viewer.entities.add({
          name: 'Orbit: Earth',
          polyline: { positions: earthOrbit, width: 3.2, material: Cesium.Color.WHITE.withAlpha(0.9) }
        });
        for (let i = 0; i < earthOrbit.length; i += 12) {
          viewer.entities.add({
            position: earthOrbit[i],
            point: { pixelSize: 4, color: Cesium.Color.WHITE.withAlpha(0.95) }
          });
        }
      }

      // Earth 3D sphere
      const EARTH_R = 7.0e5;
      const earthGeom = new Cesium.SphereGeometry({
        radius: EARTH_R,
        vertexFormat: Cesium.MaterialAppearance.MaterialSupport.TEXTURED.vertexFormat,
        stackPartitions: 12,
        slicePartitions: 12,
      });
      const earthPrim = viewer.scene.primitives.add(new Cesium.Primitive({
        geometryInstances: new Cesium.GeometryInstance({ geometry: earthGeom }),
        appearance: new Cesium.MaterialAppearance({
          material: earthTex
            ? new Cesium.Material({ fabric: { type: 'Image', uniforms: { image: earthTex } } })
            : Cesium.Material.fromType('Color', { color: Cesium.Color.DODGERBLUE }),
          faceForward: true,
          closed: true,
        }),
        asynchronous: false,
      }));

      viewer.entities.add({
        name: 'Earth',
        position: new Cesium.CallbackProperty(() => sampleBodies(simUnixNow()).earth, false),
        label: {
          text: 'Earth', font: '13px sans-serif', fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK, outlineWidth: 2, style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(12, -10),
        }
      });

      // Moon orbit ring from real relative geometry, centered on Earth each frame
      const moonRel = polyFromList(solar.moon_rel_orbit);
      const moonRingPts = moonRel.map(() => new Cesium.Cartesian3());
      if (moonRel.length > 1) {
        viewer.entities.add({
          name: 'Orbit: Moon',
          polyline: {
            positions: new Cesium.CallbackProperty(() => {
              const e = sampleBodies(simUnixNow()).earth;
              const out = new Array(moonRel.length);
              for (let i = 0; i < moonRel.length; i++) {
                moonRingPts[i].x = e.x + moonRel[i].x;
                moonRingPts[i].y = e.y + moonRel[i].y;
                moonRingPts[i].z = e.z + moonRel[i].z;
                out[i] = Cesium.Cartesian3.clone(moonRingPts[i]);
              }
              return out;
            }, false),
            width: 2.4,
            material: Cesium.Color.LIGHTGRAY.withAlpha(0.95),
          }
        });

        // extra markers so moon orbit is always visible
        for (let i = 0; i < moonRel.length; i += 12) {
          viewer.entities.add({
            position: new Cesium.CallbackProperty(() => {
              const e = sampleBodies(simUnixNow()).earth;
              return new Cesium.Cartesian3(e.x + moonRel[i].x, e.y + moonRel[i].y, e.z + moonRel[i].z);
            }, false),
            point: { pixelSize: 3, color: Cesium.Color.LIGHTGRAY.withAlpha(0.95) }
          });
        }
      }

      viewer.entities.add({
        name: 'Moon',
        position: new Cesium.CallbackProperty(() => sampleBodies(simUnixNow()).moon, false),
        point: { pixelSize: 10, color: Cesium.Color.LIGHTGRAY, outlineColor: Cesium.Color.BLACK, outlineWidth: 1 },
        label: {
          text: 'Moon', font: '11px sans-serif', fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK, outlineWidth: 2, style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(8, -6),
        }
      });

      viewer.scene.preRender.addEventListener(() => {
        const e = sampleBodies(simUnixNow()).earth;
        const spin = 2.0 * Math.PI * ((simUnixNow() - midT) / 86400.0);
        const t = Cesium.Matrix4.fromTranslation(e);
        const r = Cesium.Matrix4.fromRotationTranslation(Cesium.Matrix3.fromRotationZ(spin));
        earthPrim.modelMatrix = Cesium.Matrix4.multiply(t, r, new Cesium.Matrix4());
      });

      const focus = sampleBodies(simUnixNow()).earth;
      viewer.camera.setView({ destination: new Cesium.Cartesian3(focus.x, focus.y - 6.5e8, focus.z + 3.0e8) });
      simtimeEl.style.display = 'block';
      setInterval(() => {
        const d = new Date(simUnixNow() * 1000);
        simtimeEl.textContent = `Sim UTC ${d.toISOString().replace('T', ' ').replace('.000Z', 'Z')}`;
      }, 200);
      statusEl.textContent = `OK · Solar real-data ${cfg.cartoonMode ? 'cartoon' : 'realtime'}`;
      return;
    }

    let imagerySet = false;
    if (cfg.earthDataUrl) {
      try {
        const single = new Cesium.SingleTileImageryProvider({
          url: cfg.earthDataUrl,
          rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90)
        });
        viewer.imageryLayers.removeAll();
        viewer.imageryLayers.addImageryProvider(single);
        imagerySet = true;
      } catch (e) {}
    }
    if (!imagerySet) {
      try {
        const arcgis = await Cesium.ArcGisMapServerImageryProvider.fromUrl(
          'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
        );
        viewer.imageryLayers.removeAll();
        viewer.imageryLayers.addImageryProvider(arcgis);
        imagerySet = true;
      } catch (e) {}
    }

    // Real Earth model (Cesium globe pipeline)
    viewer.scene.globe.show = true;
    viewer.scene.globe.baseColor = Cesium.Color.DARKSLATEGRAY;
    viewer.scene.globe.translucency.enabled = false;
    viewer.scene.globe.enableLighting = true;

    const eciMode = !!cfg.eciView;
    if (eciMode) {
      // ECI: keep stars/Sun/Moon in inertial frame; user can still move camera.
      viewer.scene.sun.show = true;
      viewer.scene.moon.show = true;
      viewer.scene.globe.enableLighting = true;
    }

    viewer.scene.moon.show = true;
    viewer.scene.sun.show = true;
    viewer.scene.skyAtmosphere.show = true;

    if (eciMode) {
      let lastUserInputMs = Date.now();
      const input = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
      const markInput = () => { lastUserInputMs = Date.now(); };
      input.setInputAction(markInput, Cesium.ScreenSpaceEventType.LEFT_DOWN);
      input.setInputAction(markInput, Cesium.ScreenSpaceEventType.MIDDLE_DOWN);
      input.setInputAction(markInput, Cesium.ScreenSpaceEventType.RIGHT_DOWN);
      input.setInputAction(markInput, Cesium.ScreenSpaceEventType.WHEEL);
      input.setInputAction(markInput, Cesium.ScreenSpaceEventType.PINCH_START);

      let lastSunWin = null;
      let anchorSunWin = null;
      let lastAnchorResetMs = 0;
      let wasIdle = false;
      let eciOffset = new Cesium.Cartesian3(2.3e7, 6.0e6, 3.5e6);
      function updateSunDriftDiagnostics(time, justBecameIdle) {
        if (!Cesium.Simon1994PlanetaryPositions || !Cesium.Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame) return;
        const inertial = Cesium.Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(
          time,
          new Cesium.Cartesian3()
        );
        const icrfToFixed = Cesium.Transforms.computeIcrfToFixedMatrix(time);
        if (!Cesium.defined(icrfToFixed)) return;
        const world = Cesium.Matrix3.multiplyByVector(icrfToFixed, inertial, new Cesium.Cartesian3());
        const win = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, world);
        if (win) {
          const nowMs = Date.now();
          if (!anchorSunWin || justBecameIdle || (nowMs - lastUserInputMs < 1200 && nowMs - lastAnchorResetMs > 400)) {
            anchorSunWin = { x: win.x, y: win.y };
            lastAnchorResetMs = nowMs;
          }
          lastSunWin = { x: win.x, y: win.y };
        }
      }

      viewer.scene.postUpdate.addEventListener(function(scene, time) {
        const icrfToFixed = Cesium.Transforms.computeIcrfToFixedMatrix(time);
        if (!Cesium.defined(icrfToFixed)) {
          updateSunDriftDiagnostics(time, false);
          return;
        }

        const transform = Cesium.Matrix4.fromRotationTranslation(icrfToFixed);
        const inv = Cesium.Matrix4.inverseTransformation(transform, new Cesium.Matrix4());
        const now = Date.now();
        const isIdle = (now - lastUserInputMs > 1200);

        if (isIdle) {
          viewer.camera.lookAtTransform(transform, eciOffset);
        } else {
          // User moved camera: capture new inertial offset for subsequent idle lock.
          eciOffset = Cesium.Matrix4.multiplyByPoint(inv, viewer.camera.positionWC, new Cesium.Cartesian3());
        }

        const justBecameIdle = isIdle && !wasIdle;
        wasIdle = isIdle;
        updateSunDriftDiagnostics(time, justBecameIdle);
      });
    }

    // Moon label (computed from ephemeris each frame)
    viewer.entities.add({
      name: 'Moon',
      position: new Cesium.CallbackProperty(function(time, result) {
        const inertial = Cesium.Simon1994PlanetaryPositions.computeMoonPositionInEarthInertialFrame(
          time,
          new Cesium.Cartesian3()
        );
        const icrfToFixed = Cesium.Transforms.computeIcrfToFixedMatrix(time);
        if (Cesium.defined(icrfToFixed)) {
          return Cesium.Matrix3.multiplyByVector(icrfToFixed, inertial, result || new Cesium.Cartesian3());
        }
        return inertial;
      }, false),
      point: {
        pixelSize: 6,
        color: Cesium.Color.LIGHTGRAY,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 1
      },
      label: {
        text: 'Moon',
        font: '13px sans-serif',
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(12, -10)
      }
    });

    const gpsPalette = [
      Cesium.Color.fromCssColorString('#FFF176'), // light yellow
      Cesium.Color.fromCssColorString('#FFD54F'),
      Cesium.Color.fromCssColorString('#FFCA28'),
      Cesium.Color.fromCssColorString('#FFB300'),
      Cesium.Color.fromCssColorString('#FB8C00'),
      Cesium.Color.fromCssColorString('#F57C00'), // deep orange
    ];
    const beidouPalette = [
      Cesium.Color.fromCssColorString('#64B5F6'), // light blue
      Cesium.Color.fromCssColorString('#26C6DA'), // cyan
      Cesium.Color.fromCssColorString('#7E57C2'), // purple
    ];
    const starlinkPalette = [
      Cesium.Color.fromCssColorString('#D7DCE2'),
      Cesium.Color.fromCssColorString('#B9C2CC'),
      Cesium.Color.fromCssColorString('#9AA7B8'),
      Cesium.Color.fromCssColorString('#C7CED6'),
    ];
    const landsatPalette = [
      Cesium.Color.fromCssColorString('#5DA9FF'),
      Cesium.Color.fromCssColorString('#2FD6FF'),
      Cesium.Color.fromCssColorString('#8A63FF'),
      Cesium.Color.fromCssColorString('#4F8CFF'),
      Cesium.Color.fromCssColorString('#29BEEA'),
      Cesium.Color.fromCssColorString('#7A55E8'),
    ];
    const sentinelPalette = [
      Cesium.Color.fromCssColorString('#FFE082'),
      Cesium.Color.fromCssColorString('#FFCA28'),
      Cesium.Color.fromCssColorString('#FFA726'),
      Cesium.Color.fromCssColorString('#FF8F00'),
      Cesium.Color.fromCssColorString('#F4511E'),
      Cesium.Color.fromCssColorString('#E53935'),
    ];
    function toneColor(sys, orbitGroup, subtype) {
      if (sys === 'GPS') return gpsPalette[(orbitGroup || 0) % gpsPalette.length];
      if (sys === 'BeiDou') return beidouPalette[(orbitGroup || 0) % beidouPalette.length];
      if (sys === 'Starlink') return starlinkPalette[(orbitGroup || 0) % starlinkPalette.length];
      if (sys === 'EarthResource') {
        if (subtype === 'Landsat') return landsatPalette[(orbitGroup || 0) % landsatPalette.length];
        if (subtype === 'Sentinel') return sentinelPalette[(orbitGroup || 0) % sentinelPalette.length];
        return Cesium.Color.LIME;
      }
      return Cesium.Color.WHITE;
    }
    function darkenColor(c, factor) {
      const f = Math.max(0.0, Math.min(1.0, factor));
      return new Cesium.Color(c.red * f, c.green * f, c.blue * f, c.alpha);
    }

    function normLon(lon) {
      let x = lon;
      while (x > 180) x -= 360;
      while (x < -180) x += 360;
      return x;
    }

    function isValidLonLatArray(a) {
      if (!Array.isArray(a) || a.length < 6 || (a.length % 2) !== 0) return false;
      for (let i = 0; i < a.length; i += 2) {
        const lon = a[i], lat = a[i + 1];
        if (!Number.isFinite(lon) || !Number.isFinite(lat)) return false;
        if (lat < -90 || lat > 90) return false;
      }
      return true;
    }

    function footprintLonLat(latDeg, lonDeg, headingRad, widthKm, lengthKm) {
      const halfW = (widthKm * 1000) / 2.0;
      const halfL = (lengthKm * 1000) / 2.0;
      const cosH = Math.cos(headingRad);
      const sinH = Math.sin(headingRad);
      const cosLat = Math.max(0.2, Math.cos(Cesium.Math.toRadians(latDeg)));
      const mPerDegLat = 111320.0;
      const mPerDegLon = mPerDegLat * cosLat;

      function offsetToLonLat(eastM, northM) {
        const dLon = eastM / mPerDegLon;
        const dLat = northM / mPerDegLat;
        return [normLon(lonDeg + dLon), latDeg + dLat];
      }

      function corner(fwd, right) {
        const east = fwd * sinH + right * cosH;
        const north = fwd * cosH - right * sinH;
        return offsetToLonLat(east, north);
      }

      const c1 = corner(+halfL, -halfW);
      const c2 = corner(+halfL, +halfW);
      const c3 = corner(-halfL, +halfW);
      const c4 = corner(-halfL, -halfW);
      const arr = [c1[0], c1[1], c2[0], c2[1], c3[0], c3[1], c4[0], c4[1]];

      if (!isValidLonLatArray(arr)) return null;
      const lons = [arr[0], arr[2], arr[4], arr[6]];
      const span = Math.max(...lons) - Math.min(...lons);
      // Dateline crossing can explode arc tessellation in ground polygons.
      if (span > 170) return null;
      return arr;
    }

    const isEarthResourceMode = (cfg.viewMode === 'Earth Resource Lab');
    const coverageFocusSet = new Set((cfg.coverageFocusNames || []).map((x) => String(x)));
    const extremeMode = !!cfg.extremeMode;

    let lastIcrfToFixed = null;
    function eciToFixed(cart, time) {
      if (!eciMode) return cart;
      const m = Cesium.Transforms.computeIcrfToFixedMatrix(time || viewer.clock.currentTime);
      if (Cesium.defined(m)) lastIcrfToFixed = m;
      // Fallback: don't freeze updates if matrix is temporarily unavailable.
      if (!lastIcrfToFixed) return cart;
      return Cesium.Matrix3.multiplyByVector(lastIcrfToFixed, cart, new Cesium.Cartesian3());
    }
    const cartoonMode = !!cfg.cartoonMode;
    const satByName = new Map();
    const satTrackByName = new Map();
    const satTrackPosByName = new Map();
    const satPrevByName = new Map();
    const enableTracks = !!cfg.showTracks;
    const erFootprintByName = new Map();
    const erCoverageCounterByName = new Map();
    const trailByName = new Map((payload.trails || []).map(t => [t.name, t]));
    const filterERInCartoon = isEarthResourceMode && cartoonMode && coverageFocusSet.size > 0;

    for (const sat of payload.satellites) {
      const c = toneColor(sat.sys, sat.orbit_group, sat.subtype);
      const curInertial = new Cesium.Cartesian3(sat.x, sat.y, sat.z);
      const cur = eciToFixed(curInertial, viewer.clock.currentTime) || curInertial;
      const isStarlink = sat.sys === 'Starlink';
      const isER = sat.sys === 'EarthResource';
      if (filterERInCartoon && isER && !coverageFocusSet.has(String(sat.name))) continue;
      const subtype = sat.subtype || 'EarthResource';
      const ent = viewer.entities.add({
        name: sat.name,
        position: cur,
        point: {
          pixelSize: (extremeMode && isStarlink) ? 3 : 7,
          color: c,
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: (extremeMode && isStarlink) ? 0 : 1
        },
        label: isER ? {
          text: String(sat.name || '').replace(/\\s+/g, '_'),
          font: '12px sans-serif',
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(0, -14),
          showBackground: true,
          backgroundColor: Cesium.Color.BLACK.withAlpha(0.35),
        } : undefined,
        description: `<b>${sat.name}</b><br/>${subtype} · Orbit group ${sat.orbit_group}`
      });
      satByName.set(sat.name, ent);
      satPrevByName.set(sat.name, cur);

      if (isEarthResourceMode && isER) {
        const liveFov = { lonLat: null };
        const fovColor = subtype === 'Landsat'
          ? Cesium.Color.fromCssColorString('#5DA9FF').withAlpha(0.22)
          : Cesium.Color.fromCssColorString('#FFA726').withAlpha(0.22);
        viewer.entities.add({
          name: `FOV: ${sat.name}`,
          polygon: {
            hierarchy: new Cesium.CallbackProperty(() => {
              if (!isValidLonLatArray(liveFov.lonLat)) return new Cesium.PolygonHierarchy([]);
              return new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(liveFov.lonLat));
            }, false),
            material: fovColor,
            outline: true,
            outlineColor: darkenColor(fovColor, 0.7),
            perPositionHeight: false,
            heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          }
        });
        erFootprintByName.set(sat.name, { liveFov, subtype });
        erCoverageCounterByName.set(sat.name, 0);
      }

      if (enableTracks && !extremeMode) {
        const trackPositions = [];
        satTrackPosByName.set(sat.name, trackPositions);
        const trackColor = (cfg.cartoonMode && sat.sys === 'Starlink')
          ? Cesium.Color.WHITE.withAlpha(0.38)
          : darkenColor(c, 0.75).withAlpha(0.95);
        const track = viewer.entities.add({
          name: `Track: ${sat.name}`,
          polyline: {
            positions: new Cesium.CallbackProperty(() => trackPositions, false),
            width: 6.4,
            material: trackColor,
            clampToGround: false,
          }
        });
        satTrackByName.set(sat.name, track);
      }
    }

    // Orbit lines (always shown, thinner than live tracks)
    const orbitEntities3D = [];
    const orbitEntities2D = [];
    const orbitBaseMaterialByName = new Map();

    if (!extremeMode) for (const tr of payload.trails) {
      if (filterERInCartoon && tr.sys === 'EarthResource' && !coverageFocusSet.has(String(tr.name))) continue;
      const c = (tr.sys === 'Starlink')
        ? Cesium.Color.YELLOW.withAlpha(0.35)
        : toneColor(tr.sys, tr.orbit_group, tr.subtype).withAlpha(0.65);
      const pts3d = tr.positions.map((p) => new Cesium.Cartesian3(p[0], p[1], p[2]));

      // 3D orbital path (space orbit)
      const e3d = viewer.entities.add({
        name: `Orbit3D: ${tr.name}`,
        polyline: {
          positions: new Cesium.CallbackProperty((time) => {
            if (!eciMode) return pts3d;
            return pts3d.map((p) => eciToFixed(p, time));
          }, false),
          width: 1.8,
          material: c,
          clampToGround: false,
          zIndex: 100
        }
      });
      orbitEntities3D.push(e3d);
      orbitBaseMaterialByName.set(e3d.name, c);

      // 2D fallback path (ground track) to avoid disappearing lines after map redraw
      const lonLat = [];
      for (const p of pts3d) {
        const carto = Cesium.Cartographic.fromCartesian(p);
        if (!carto) continue;
        lonLat.push(normLon(Cesium.Math.toDegrees(carto.longitude)), Cesium.Math.toDegrees(carto.latitude));
      }
      if (isValidLonLatArray(lonLat)) {
        const e2d = viewer.entities.add({
          name: `Orbit2D: ${tr.name}`,
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArray(lonLat),
            width: 2.8,
            material: c,
            clampToGround: true,
            zIndex: 120
          },
          show: false
        });
        orbitEntities2D.push(e2d);
      }
    }

    function applyOrbitStyleForMode() {
      const mode = viewer.scene.mode;
      const is2DLike = mode === Cesium.SceneMode.SCENE2D || mode === Cesium.SceneMode.COLUMBUS_VIEW;

      for (const oe of orbitEntities3D) {
        if (!oe || !oe.polyline) continue;
        oe.show = !is2DLike;
        oe.polyline.width = 1.8;
        oe.polyline.material = orbitBaseMaterialByName.get(oe.name) || Cesium.Color.YELLOW.withAlpha(0.65);
      }
      for (const oe of orbitEntities2D) {
        if (!oe || !oe.polyline) continue;
        oe.show = is2DLike;
        oe.polyline.width = 2.8;
        const base = orbitBaseMaterialByName.get(oe.name.replace('Orbit2D:', 'Orbit3D:'));
        if (base) oe.polyline.material = base;
      }
    }

    // 2D fallback: use ground tracks instead of space orbits.
    viewer.scene.morphComplete.addEventListener(applyOrbitStyleForMode);
    applyOrbitStyleForMode();

    const specialEntities = new Map();
    const specialTrackPosByKind = new Map();
    for (const s of (payload.specials || [])) {
      const isIss = s.kind === 'iss';
      const labelText = isIss ? 'ISS' : 'Tiangong';
      const color = isIss ? Cesium.Color.WHITE : Cesium.Color.LIME;
      const outline = isIss ? Cesium.Color.RED : Cesium.Color.DARKGREEN;
      const desc = isIss ? 'International Space Station' : 'Chinese Space Station (Tiangong)';
      const curInertial = new Cesium.Cartesian3(s.x, s.y, s.z);
      const cur = eciToFixed(curInertial, viewer.clock.currentTime) || curInertial;

      const ent = viewer.entities.add({
        name: s.name,
        position: cur,
        point: {
          pixelSize: 9,
          color: color,
          outlineColor: outline,
          outlineWidth: 2,
        },
        label: {
          text: labelText,
          font: '13px sans-serif',
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(10, -10),
        },
        description: `<b>${s.name}</b><br/>${desc}`,
      });
      if (enableTracks && !extremeMode) {
        const tpos = [];
        specialTrackPosByKind.set(s.kind, tpos);
        viewer.entities.add({
          name: `Track: ${labelText}`,
          polyline: {
            positions: new Cesium.CallbackProperty(() => tpos, false),
            width: 6.8,
            material: darkenColor(color, 0.75).withAlpha(0.95)
          }
        });
      }
      specialEntities.set(s.kind, { entity: ent, data: s });
    }

    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(-98.0, 39.5, 2.3e7),
      duration: 0.8,
    });

    const cartoonSpeedMs = Math.max(40, Number(cfg.cartoonSpeedMs || 120));
    const historyStepSeconds = Math.max(60, Number(payload.history_step_seconds || 900));
    const trailHours = Math.max(1, Number(cfg.trackTrailHours || 24));
    const historyDays = Math.max(1, Number(payload.history_days || cfg.cartoonDays || 1));
    const nowEpochMs = Date.parse(payload.timestamp);
    let idx = 0;

    function updateSimTimeLabel(frameIdx) {
      if (!simtimeEl) return null;
      if (!cartoonMode) {
        simtimeEl.style.display = 'none';
        return null;
      }
      const simEpochMs = nowEpochMs - (historyDays * 86400 * 1000) + (frameIdx * historyStepSeconds * 1000);
      const d = new Date(simEpochMs);
      simtimeEl.style.display = 'block';
      simtimeEl.textContent = 'Sim time: ' + d.toISOString().replace('T', ' ').replace('.000Z', ' UTC');
      return d;
    }

    function enableInertialObserverMode() {
      // Disabled: replaced by explicit Earth Resource ECI visual mode above.
      return;
    }
    enableInertialObserverMode();

    const trackStartMs = Date.now() + 2000;
    function pushTrack(trackMap, key, cart) {
      if (Date.now() < trackStartMs) return;
      const arr = trackMap.get(key);
      if (!arr) return;
      arr.push(cart);
      const maxPts = cartoonMode
        ? Math.max(2, Math.floor((trailHours * 3600) / historyStepSeconds))
        : 220;
      if (arr.length > maxPts) arr.shift();
    }

    function updateEarthResourceFovAndCoverage(name, cartCur) {
      if (!isEarthResourceMode) return;
      const fp = erFootprintByName.get(name);
      const prev = satPrevByName.get(name);
      if (!fp || !prev) return;

      const cc = Cesium.Cartographic.fromCartesian(cartCur);
      const cp = Cesium.Cartographic.fromCartesian(prev);
      if (!cc || !cp) return;

      const lat = Cesium.Math.toDegrees(cc.latitude);
      const lon = Cesium.Math.toDegrees(cc.longitude);
      const dLon = Cesium.Math.toDegrees(cc.longitude - cp.longitude);
      const dLat = Cesium.Math.toDegrees(cc.latitude - cp.latitude);
      const heading = Math.atan2(dLon, dLat);

      const widthKm = fp.subtype === 'Landsat' ? 185 : 290;
      const lengthKm = fp.subtype === 'Landsat' ? 185 : 290;
      fp.liveFov.lonLat = footprintLonLat(lat, lon, heading, widthKm, lengthKm);
      if (!isValidLonLatArray(fp.liveFov.lonLat)) return;

      const focused = coverageFocusSet.has(String(name));
      const n = (erCoverageCounterByName.get(name) || 0) + 1;
      erCoverageCounterByName.set(name, n);
      const emitEvery = focused ? 1 : 4;

      function emitCoverage(lonLat, alphaScale) {
        const covColor = fp.subtype === 'Landsat'
          ? Cesium.Color.fromCssColorString('#5DA9FF').withAlpha((focused ? 0.20 : 0.28) * alphaScale)
          : Cesium.Color.fromCssColorString('#FFA726').withAlpha((focused ? 0.20 : 0.28) * alphaScale);
        const covOutline = fp.subtype === 'Landsat'
          ? Cesium.Color.fromCssColorString('#2D6FC0').withAlpha(0.9)
          : Cesium.Color.fromCssColorString('#B35A00').withAlpha(0.9);
        viewer.entities.add({
          name: `Coverage: ${name}`,
          polygon: {
            hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(lonLat)),
            material: covColor,
            outline: true,
            outlineColor: covOutline,
            perPositionHeight: false,
            heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          }
        });
      }

      if (n % emitEvery === 0) {
        // Base footprint at current epoch.
        emitCoverage(fp.liveFov.lonLat, 1.0);

        // For focused satellites, interpolate extra in-between footprints
        // to make strips visually continuous during fast cartoon playback.
        if (focused && cp && cc) {
          const sub = 4;
          for (let k = 1; k <= sub; k++) {
            const f = k / (sub + 1);
            const ilat = Cesium.Math.toDegrees(cp.latitude + (cc.latitude - cp.latitude) * f);
            const ilon = normLon(Cesium.Math.toDegrees(cp.longitude + (cc.longitude - cp.longitude) * f));
            const lonLatMid = footprintLonLat(ilat, ilon, heading, widthKm, lengthKm);
            if (isValidLonLatArray(lonLatMid)) {
              emitCoverage(lonLatMid, 0.75);
            }
          }
        }
      }
    }

    if (cartoonMode && Array.isArray(payload.history24) && payload.history24.length > 1) {
      const d0 = updateSimTimeLabel(0);
      if (d0) viewer.clock.currentTime = Cesium.JulianDate.fromDate(d0);

      const totalSimSec = historyDays * 86400;
      const cartoonSimStepSec = (isEarthResourceMode && coverageFocusSet.size > 0) ? 120 : historyStepSeconds;
      const simSpeed = cartoonSimStepSec / Math.max(0.04, cartoonSpeedMs / 1000.0);
      const realStartMs = Date.now();
      let lastCoverageSec = -1;

      setInterval(() => {
        const simElapsedSec = (((Date.now() - realStartMs) / 1000.0) * simSpeed) % totalSimSec;
        idx = Math.floor(simElapsedSec / historyStepSeconds) % payload.history24.length;

        const simEpochMs = nowEpochMs - (historyDays * 86400 * 1000) + (simElapsedSec * 1000);
        const simDate = new Date(simEpochMs);
        if (simtimeEl) {
          simtimeEl.style.display = 'block';
          simtimeEl.textContent = 'Sim time: ' + simDate.toISOString().replace('T', ' ').replace('.000Z', ' UTC');
        }
        viewer.clock.currentTime = Cesium.JulianDate.fromDate(simDate);

        // Cartoon in normal-motion logic: fixed orbit geometry + Earth clock progression.
        for (const [name, ent] of satByName) {
          const tr = trailByName.get(name);
          if (tr && Array.isArray(tr.positions) && tr.positions.length > 1 && tr.period_sec > 0) {
            const frac = (simElapsedSec % tr.period_sec) / tr.period_sec;
            const j = Math.floor(frac * (tr.positions.length - 1));
            const q = tr.positions[j];
            const cartInertial = new Cesium.Cartesian3(q[0], q[1], q[2]);
            const cart = eciToFixed(cartInertial, viewer.clock.currentTime);
            if (!cart) continue;
            ent.position = cart;

            // Avoid over-emitting coverage every frame; ~1 simulated minute cadence.
            if (Math.floor(simElapsedSec / 60) !== lastCoverageSec) {
              pushTrack(satTrackPosByName, name, cart);
              updateEarthResourceFovAndCoverage(name, cart);
              satPrevByName.set(name, cart);
            }
          }
        }

        for (const [kind, obj] of specialEntities) {
          const sp = obj?.data?.history24?.[idx];
          if (sp) {
            const cartInertial = new Cesium.Cartesian3(sp.x, sp.y, sp.z);
            const cart = eciToFixed(cartInertial, viewer.clock.currentTime);
            if (!cart) continue;
            obj.entity.position = cart;
            if (Math.floor(simElapsedSec / 60) !== lastCoverageSec) {
              pushTrack(specialTrackPosByKind, kind, cart);
            }
          }
        }

        lastCoverageSec = Math.floor(simElapsedSec / 60);
      }, 100);
      statusEl.textContent = `OK · cartoon mode (${historyDays}d)${extremeMode ? ' · extreme' : ''} · sats=${payload.satellites.length}`;
    } else {
      updateSimTimeLabel(0);
      viewer.clock.currentTime = Cesium.JulianDate.now();

      // Smooth realtime motion using interpolation between 30s frames
      const stepMs = Math.max(1000, historyStepSeconds ? 30000 : 30000);
      let segStartMs = Date.now();
      let idxA = 0;
      let idxB = 1 % payload.future.length;

      function frameMap(arr) {
        const m = new Map();
        for (const p of arr) m.set(p.name, p);
        return m;
      }

      let mapA = frameMap(payload.future[idxA] || []);
      let mapB = frameMap(payload.future[idxB] || []);

      setInterval(() => {
        viewer.clock.currentTime = Cesium.JulianDate.now();
        const now = Date.now();
        let t = (now - segStartMs) / stepMs;
        if (t >= 1.0) {
          idxA = idxB;
          idxB = (idxB + 1) % payload.future.length;
          mapA = frameMap(payload.future[idxA] || []);
          mapB = frameMap(payload.future[idxB] || []);
          segStartMs = now;
          t = 0.0;
        }

        for (const [name, pa] of mapA) {
          const pb = mapB.get(name) || pa;
          const x = pa.x + (pb.x - pa.x) * t;
          const y = pa.y + (pb.y - pa.y) * t;
          const z = pa.z + (pb.z - pa.z) * t;
          const ent = satByName.get(name);
          if (ent) {
            const cartInertial = new Cesium.Cartesian3(x, y, z);
            const cart = eciToFixed(cartInertial, viewer.clock.currentTime);
            if (!cart) continue;
            ent.position = cart;
          }
        }

        // Update tracks/coverage at coarse cadence (once per segment)
        if (t < 0.05) {
          for (const [name, pa] of mapA) {
            const pb = mapB.get(name) || pa;
            const x = pa.x + (pb.x - pa.x) * t;
            const y = pa.y + (pb.y - pa.y) * t;
            const z = pa.z + (pb.z - pa.z) * t;
            const cartInertial = new Cesium.Cartesian3(x, y, z);
            const cart = eciToFixed(cartInertial, viewer.clock.currentTime);
            if (!cart) continue;
            pushTrack(satTrackPosByName, name, cart);
            updateEarthResourceFovAndCoverage(name, cart);
            satPrevByName.set(name, cart);
          }

          for (const [kind, obj] of specialEntities) {
            const spa = obj?.data?.future?.[idxA];
            const spb = obj?.data?.future?.[idxB] || spa;
            if (spa) {
              const x = spa.x + (spb.x - spa.x) * t;
              const y = spa.y + (spb.y - spa.y) * t;
              const z = spa.z + (spb.z - spa.z) * t;
              const cartInertial = new Cesium.Cartesian3(x, y, z);
              const cart = eciToFixed(cartInertial, viewer.clock.currentTime);
              if (!cart) continue;
              obj.entity.position = cart;
              pushTrack(specialTrackPosByKind, kind, cart);
            }
          }
        }
      }, 500);

      statusEl.textContent = `OK · realtime mode${extremeMode ? ' · extreme' : ''} · sats=${payload.satellites.length} · imagery=${imagerySet}`;
    }
    setTimeout(() => { statusEl.style.display = 'none'; }, 2500);
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Render failed: ' + (err?.message || err);
  }
})();
</script>
</body>
</html>
"""


def _viewer_html_inline(payload: dict, cfg: dict) -> str:
    html = _viewer_html()
    inline_bootstrap = (
        "    const payload = " + json.dumps(payload) + ";\n"
        "    const cfg = " + json.dumps(cfg) + ";"
    )
    html = html.replace(
        "    const payloadRes = await fetch('./payload.json?ts=' + Date.now());\n"
        "    const cfgRes = await fetch('./config.json?ts=' + Date.now());\n"
        "    const payload = await payloadRes.json();\n"
        "    const cfg = await cfgRes.json();",
        inline_bootstrap,
    )
    return html


def _write_viewer_data(
    payload: dict,
    use_world_terrain: bool,
    earth_data_url: str | None,
    ion_token: str,
    cartoon_mode: bool,
    cartoon_speed_ms: int,
    extreme_mode: bool,
    cartoon_days: int,
    track_trail_hours: int,
    show_tracks: bool,
    spin_earth: bool,
    spin_deg_per_sec: float,
    view_mode: str,
    eci_view: bool,
    coverage_focus_names: list[str],
    solar_speed_days_per_sec: float,
) -> dict:
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    (VIEWER_DIR / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    solar_earth_texture_url = None
    if EARTH_TEXTURE_PATH.exists():
        try:
            tex_dst = VIEWER_DIR / "earth_texture.jpg"
            tex_dst.write_bytes(EARTH_TEXTURE_PATH.read_bytes())
            solar_earth_texture_url = f"earth_texture.jpg?ts={int(time.time())}"
        except Exception:
            solar_earth_texture_url = None

    cfg = {
        "useWorldTerrain": bool(use_world_terrain),
        "earthDataUrl": earth_data_url,
        "solarEarthTextureUrl": solar_earth_texture_url,
        "ionToken": ion_token,
        "cartoonMode": bool(cartoon_mode),
        "cartoonSpeedMs": int(cartoon_speed_ms),
        "extremeMode": bool(extreme_mode),
        "cartoonDays": int(cartoon_days),
        "trackTrailHours": int(track_trail_hours),
        "showTracks": bool(show_tracks),
        "spinEarth": bool(spin_earth),
        "spinDegPerSec": float(spin_deg_per_sec),
        "viewMode": view_mode,
        "eciView": bool(eci_view),
        "coverageFocusNames": coverage_focus_names,
        "solarSpeedDaysPerSec": float(solar_speed_days_per_sec),
    }
    (VIEWER_DIR / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def main():
    st.set_page_config(page_title="Shane’s Space Traffic Lab", page_icon="🌍", layout="wide")
    st.title("🌍 Shane’s Space Traffic Lab")

    view_mode = st.radio("Mode", ["GNSS Lab", "Starlink Lab", "Earth Resource Lab", "Solar System Lab"], horizontal=True)

    with st.sidebar:
        if view_mode == "GNSS Lab":
            show_gps = st.checkbox("Show GPS", value=True)
            show_beidou = st.checkbox("Show BeiDou", value=False)
            show_starlink = False
            show_earth_resource = False
            show_iss = False
            show_tiangong = False
            starlink_limit = 0
            earth_resource_limit = 0
            max_trails = st.slider("Max orbit trails", 10, 220, 160)
            orbit_points = st.slider("Orbit points", 120, 400, 220)
            cartoon_mode = st.checkbox("Cartoon mode (24h fast playback)", value=False)
            cartoon_speed_ms = st.slider("Cartoon frame interval (ms)", 40, 500, 120, 10)
            cartoon_days = 1
            track_trail_hours = 24
            if cartoon_mode:
                show_tracks = False
                st.caption("Tracks disabled in cartoon mode.")
            else:
                show_tracks = st.checkbox("Show tracks", value=True)
            spin_earth = False
            spin_deg_per_sec = 0.0
            eci_view = True
            extreme_mode = False
            solar_speed_days_per_sec = 5.0

        elif view_mode == "Starlink Lab":
            show_gps = False
            show_beidou = False
            show_starlink = True
            show_earth_resource = False
            show_iss = False
            show_tiangong = False
            earth_resource_limit = 0
            total_starlink = len(fetch_tles(STARLINK_TLE_URL))

            st.metric("Total Starlink now", f"{total_starlink}")
            starlink_limit = st.slider("Show Starlink satellites", 50, max(300, total_starlink), 100, 50)

            if starlink_limit < 300:
                show_starlink_orbits = st.checkbox("Show Starlink orbits", value=True)
                max_trails = starlink_limit if show_starlink_orbits else 0
                cartoon_mode = st.checkbox("Cartoon mode (24h fast playback)", value=False)
                cartoon_speed_ms = st.slider("Cartoon frame interval (ms)", 40, 500, 120, 10)
                cartoon_days = 1
                track_trail_hours = 1
            else:
                max_trails = 0
                cartoon_mode = False
                cartoon_speed_ms = 120
                cartoon_days = 1
                track_trail_hours = 1
                st.caption("Orbits/cartoon disabled at 300+ for performance.")

            if cartoon_mode:
                show_tracks = False
                st.caption("Tracks disabled in cartoon mode.")
            else:
                show_tracks = st.checkbox("Show tracks", value=True)
            orbit_points = 180
            spin_earth = False
            spin_deg_per_sec = 0.0
            eci_view = True
            extreme_mode = False
            solar_speed_days_per_sec = 5.0

        elif view_mode == "Earth Resource Lab":
            show_gps = False
            show_beidou = False
            show_starlink = False
            show_earth_resource = True
            show_iss = st.checkbox("Show ISS", value=False)
            show_tiangong = st.checkbox("Show Tiangong", value=False)
            starlink_limit = 0
            earth_resource_limit = st.slider("Show Landsat + Sentinel satellites", 10, 300, 80, 10)
            max_trails = st.slider("Max orbit trails", 0, 300, 80, 10)
            orbit_points = st.slider("Orbit points", 120, 400, 220)
            cartoon_mode = st.checkbox("Cartoon mode (8-day fast playback)", value=False)
            cartoon_speed_ms = st.slider("Cartoon frame interval (ms)", 40, 500, 120, 10)
            cartoon_days = 8
            track_trail_hours = 72
            if cartoon_mode:
                show_tracks = False
                st.caption("Tracks disabled in cartoon mode.")
            else:
                show_tracks = st.checkbox("Show tracks", value=False)
            spin_earth = False
            spin_deg_per_sec = 0.0
            eci_view = True
            st.caption("Earth Resource cartoon mode replays 8 days; track tail keeps last 72h.")
            extreme_mode = False
            solar_speed_days_per_sec = 5.0

        else:
            # Solar System Lab
            show_gps = False
            show_beidou = False
            show_starlink = False
            show_earth_resource = False
            show_iss = False
            show_tiangong = False
            starlink_limit = 0
            earth_resource_limit = 0
            max_trails = 0
            orbit_points = 0
            cartoon_mode = st.checkbox("Cartoon mode (accelerated)", value=False)
            cartoon_speed_ms = 120
            cartoon_days = 1
            track_trail_hours = 24
            show_tracks = False
            spin_earth = False
            spin_deg_per_sec = 0.0
            eci_view = False
            extreme_mode = False
            solar_speed_days_per_sec = st.slider("Solar playback speed (sim days/sec)", 0.1, 50.0, 5.0, 0.1)
            if not cartoon_mode:
                st.caption("Realtime mode uses current ephemeris time (no acceleration).")

        use_world_terrain = st.checkbox("Use world terrain (if available)", value=False)
        height_px = st.slider("Viewer height", 700, 1200, 980)
        if st.button("Refresh data now"):
            st.cache_data.clear()
            st.rerun()

    try:
        if view_mode == "Solar System Lab":
            payload = build_solar_payload()
        else:
            payload = build_payload(
                show_gps,
                show_beidou,
                show_starlink,
                show_earth_resource,
                show_iss,
                show_tiangong,
                starlink_limit,
                earth_resource_limit,
                max_trails,
                orbit_points,
                include_history=cartoon_mode,
                history_days=cartoon_days,
                history_step_seconds=1800 if view_mode == "Earth Resource Lab" else 900,
            )
    except Exception as e:
        st.error(f"Failed to build satellite view: {e}")
        st.stop()

    if view_mode == "Solar System Lab":
        solar_ts = payload.get("solar", {}).get("timestamp", "-")
        st.caption(f"UTC data epoch: {solar_ts} | Bodies: Sun, Earth, Moon")
    else:
        sat_count = len(payload["satellites"])
        st.caption(f"UTC data epoch: {payload['timestamp']} | Satellites: {sat_count}")

        if sat_count == 0 and (show_gps or show_beidou or show_starlink or show_earth_resource):
            st.warning("No satellites loaded. Data source may be temporarily unavailable; try Refresh.")

    coverage_focus_names: list[str] = []
    if view_mode == "Earth Resource Lab":
        er_names = sorted([s["name"] for s in payload.get("satellites", []) if s.get("sys") == "EarthResource"])
        coverage_focus_names = st.multiselect(
            "Coverage focus satellites (smooth FOV strips in cartoon)",
            options=er_names,
            default=[],
            help="Select one or more satellites to generate denser, more continuous FOV coverage strips in 8-day cartoon mode.",
        )

    with st.expander("Live telemetry panel", expanded=False):
        rows = []
        for s in payload.get("satellites", []):
            rows.append(
                {
                    "name": s["name"],
                    "system": s["sys"],
                    "orbit_group": s.get("orbit_group", 0),
                    "lat": round(float(s.get("lat", 0.0)), 3),
                    "lon": round(float(s.get("lon", 0.0)), 3),
                    "alt_km": round(float(s.get("alt_km", 0.0)), 1),
                    "speed_km_s": round(float(s.get("speed_km_s", 0.0)), 3),
                }
            )
        for sp in payload.get("specials", []):
            rows.append(
                {
                    "name": sp["name"],
                    "system": "ISS" if sp.get("kind") == "iss" else "Tiangong",
                    "orbit_group": "-",
                    "lat": round(float(sp.get("lat", 0.0)), 3),
                    "lon": round(float(sp.get("lon", 0.0)), 3),
                    "alt_km": round(float(sp.get("alt_km", 0.0)), 1),
                    "speed_km_s": round(float(sp.get("speed_km_s", 0.0)), 3),
                }
            )

        c1, c2 = st.columns([1, 2])
        system_filter = c1.selectbox("System", ["All", "GPS", "BeiDou", "Starlink", "EarthResource", "ISS", "Tiangong"], index=0)
        name_query = c2.text_input("Search name", value="").strip().lower()

        filtered = rows
        if system_filter != "All":
            filtered = [r for r in filtered if r["system"] == system_filter]
        if name_query:
            filtered = [r for r in filtered if name_query in r["name"].lower()]

        st.dataframe(filtered, use_container_width=True, height=320)

    earth_data_url = _earth_texture_data_url()
    cfg = _write_viewer_data(
        payload,
        use_world_terrain=use_world_terrain,
        earth_data_url=earth_data_url,
        ion_token=os.getenv("CESIUM_ION_TOKEN", ""),
        cartoon_mode=cartoon_mode,
        cartoon_speed_ms=cartoon_speed_ms,
        extreme_mode=False,
        cartoon_days=cartoon_days,
        track_trail_hours=track_trail_hours,
        show_tracks=show_tracks,
        spin_earth=spin_earth,
        spin_deg_per_sec=spin_deg_per_sec,
        view_mode=view_mode,
        eci_view=eci_view,
        coverage_focus_names=coverage_focus_names,
        solar_speed_days_per_sec=solar_speed_days_per_sec,
    )
    components.html(_viewer_html_inline(payload, cfg), height=height_px + 8, scrolling=False)


if __name__ == "__main__":
    main()
