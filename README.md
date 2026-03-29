# GPS_demo2

Google-Earth-like GNSS viewer built with **CesiumJS** inside Streamlit.

## Features
- 3D globe with smooth camera/navigation (Cesium)
- GPS + BeiDou constellations
- Full-orbit trails
- Smooth satellite position updates every 30s (client-side, no full redraw)
- Optional world terrain and high-quality imagery layer

## Run locally
```bash
cd ~/.openclaw/workspace-coder/GPS_demo2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Quick test
```bash
cd ~/.openclaw/workspace-coder/GPS_demo2
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## Notes
- Positions and trails are from live TLE + SGP4 propagation.
- Real-time-ish cadence is 30s update steps.
