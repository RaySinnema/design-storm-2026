"""Add radar rainfall to the turbidity spikes found by find_turbidity_spikes.py.

Usage:
    python add_radar_rain.py [--top 10] [--hours-before 72]

For each of the top spikes in spikes.csv, downloads NCEP MRMS 1-hour precipitation images
from the Iowa State Mesonet archive for the hours leading up to the turbidity peak, and
measures rain over three areas of the South Platte basin above Strontia Springs:

    gage       the pixel at USGS 06707525
    upper      the basin above the Trumbull gage (06701900), the far upstream part
    lower      the rest of the basin above Strontia, i.e. the land between the two gages

The window is 72 hours by default. 12 hours was too short: on Aug 6, 2024 and Jul 28, 2026
the wettest hour was 14 to 15 hours before the turbidity peak. Totals over 72 hours can
include an earlier, unrelated shower, so the lag is measured from the wettest hour.

Writes data/radar_rain_hourly.csv and data/radar_rain_events.csv.

CAUTION: the images store rain as an 8-bit index, not inches. IEM does not publish the
scale on the pages I found, so decode() below is inferred: 0.25 mm per step up to 25 mm,
then 1 mm per step. I checked it against the NOAA gauge at Strontia Springs Dam
(USC00058022) on 7 rainy days; it matched 5 of them within about 1 to 5 mm and understated
the 3.2 in day of 2023-05-12 (radar 59 mm, gauge 81 mm). Treat the numbers as approximate,
and as unreliable above 25 mm per hour. Indexes above 175 are left blank.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path as MplPath
from PIL import Image

HERE = Path(__file__).parent
BASINS = HERE.parent.parent / "strontia-brief" / "basins"
DATA = HERE / "data"
CACHE = HERE / ".cache"
URL = "https://mesonet.agron.iastate.edu/archive/data/{t:%Y/%m/%d}/GIS/mrms/p1h_{t:%Y%m%d%H}00.png"

# Grid from the .wld file: 0.01 degree pixels, upper-left corner of pixel (0, 0).
X0, Y0, STEP = -129.995, 54.995, 0.01
GAGE = (-105.15106, 39.4164)  # USGS 06707525, from strontia-brief/places.json
BBOX = (-106.25, -105.0, 38.7, 39.65)  # lon min/max, lat min/max around the basin


def decode(index):
    """Inferred, see module docstring. Returns mm, NaN where unknown."""
    index = index.astype(float)
    mm = np.where(index <= 100, index * 0.25, 25 + (index - 100))
    return np.where(index > 175, np.nan, mm)


def px(lon, lat):
    return (lon - X0) / STEP, (Y0 - lat) / STEP


def outer_ring(name):
    geom = json.load(open(BASINS / name))["features"][0]["geometry"]
    ring = geom["coordinates"][0]
    return np.array(ring if geom["type"] == "Polygon" else ring[0])


def mask(ring, cols, rows):
    """Boolean mask over a crop: pixel centres inside the polygon."""
    lon = X0 + (cols + 0.5) * STEP
    lat = Y0 - (rows + 0.5) * STEP
    lons, lats = np.meshgrid(lon, lat)
    pts = np.column_stack([lons.ravel(), lats.ravel()])
    return MplPath(ring).contains_points(pts).reshape(lons.shape)


def crop_for(t):
    """Decoded rain (mm) over the basin bounding box for the hour ending at t, or None."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{t:%Y%m%d%H}.npy"
    if f.exists():
        a = np.load(f)
        return None if a.size == 1 else a
    r = requests.get(URL.format(t=t), timeout=180)
    if r.status_code == 404:
        np.save(f, np.zeros(1))
        return None
    r.raise_for_status()
    idx = np.array(Image.open(io.BytesIO(r.content)))
    c0, r1 = px(BBOX[0], BBOX[3])
    c1, r0 = px(BBOX[1], BBOX[2])
    a = decode(idx[int(r1):int(r0), int(c0):int(c1)])
    np.save(f, a)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--hours-before", type=int, default=72)
    a = p.parse_args()

    events = pd.read_csv(DATA / "spikes.csv").head(a.top)
    events["peak_time"] = pd.to_datetime(events["peak_time"], utc=True)

    c0, r1 = px(BBOX[0], BBOX[3])
    c1, r0 = px(BBOX[1], BBOX[2])
    cols = np.arange(int(c0), int(c1))
    rows = np.arange(int(r1), int(r0))
    strontia = mask(outer_ring("south-platte-above-strontia-06707525.json"), cols, rows)
    upper = mask(outer_ring("south-platte-above-trumbull-06701900.json"), cols, rows)
    lower = strontia & ~upper
    gx, gy = px(*GAGE)
    gage_rc = (int(gy) - rows[0], int(gx) - cols[0])
    print(f"basin pixels: upper {upper.sum()}, lower {lower.sum()}")

    hourly, summary = [], []
    for i, ev in events.iterrows():
        end = ev["peak_time"].floor("h")
        print(f"event {i}: peak {ev['peak_time']} ({ev['peak_ntu']} NTU)")
        rows_ev = []
        for h in range(a.hours_before, -1, -1):
            t = end - pd.Timedelta(hours=h)
            img = crop_for(t.to_pydatetime())
            if img is None:
                rows_ev.append({"event": i, "hour_utc": t, "hours_before_peak": h})
                continue
            rows_ev.append({
                "event": i, "hour_utc": t, "hours_before_peak": h,
                "gage_mm": img[gage_rc],
                "upper_mean_mm": np.nanmean(img[upper]), "upper_max_mm": np.nanmax(img[upper]),
                "lower_mean_mm": np.nanmean(img[lower]), "lower_max_mm": np.nanmax(img[lower]),
            })
        ev_df = pd.DataFrame(rows_ev)
        hourly.append(ev_df)
        got = ev_df["lower_mean_mm"].notna().sum()
        wet = ev_df.loc[ev_df["lower_mean_mm"].idxmax()] if got else None
        wet_up = ev_df.loc[ev_df["upper_mean_mm"].idxmax()] if got else None
        summary.append({
            "event": i, "peak_time": ev["peak_time"], "peak_ntu": ev["peak_ntu"],
            "hours_with_radar": got,
            "upper_total_mm": ev_df["upper_mean_mm"].sum(),
            "lower_total_mm": ev_df["lower_mean_mm"].sum(),
            "lower_max_pixel_mm_hr": ev_df["lower_max_mm"].max(),
            "lower_wettest_hr_before_peak": None if wet is None else int(wet["hours_before_peak"]),
            "lower_wettest_hr_mm": None if wet is None else wet["lower_mean_mm"],
            "upper_wettest_hr_before_peak": None if wet_up is None else int(wet_up["hours_before_peak"]),
            "gauge_day_of_in": ev.get("rain_in_day_of"),
        })

    pd.concat(hourly).to_csv(DATA / "radar_rain_hourly.csv", index=False)
    out = pd.DataFrame(summary)
    out.to_csv(DATA / "radar_rain_events.csv", index=False)
    print("\nBasin-mean rain (mm) in the window before each spike:\n")
    print(out.round({c: 1 for c in out.select_dtypes("float").columns}).to_string(index=False))


if __name__ == "__main__":
    main()
