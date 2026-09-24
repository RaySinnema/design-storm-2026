"""Look closely at the three spikes with almost no rain nearby.

Usage:
    python investigate_mystery_spikes.py

For each event, pulls 15-minute USGS readings from 3 days before to 2 days after the
turbidity peak (turbidity, conductance, temperature, dissolved oxygen, pH at 06707525,
and flow at Trumbull 06701900), plus 72 hours of radar rain and the NOAA gauge days
around it. Writes data/mystery_readings.csv, data/mystery_radar_hourly.csv and prints a summary of each event.
USGS values are provisional.
"""
import numpy as np
import pandas as pd
import requests

from add_radar_rain import DATA, crop_for, mask, outer_ring, px, BBOX

EVENTS = {"2024-08-06": "Aug 6, 2024", "2026-07-28": "Jul 28, 2026", "2024-04-16": "Apr 16, 2024"}
PARAMS = {"63680": "turbidity", "00095": "conductance", "00010": "temp_c", "00300": "do_mg_l",
          "00400": "ph", "00060": "trumbull_cfs"}
SITES = {"turbidity": "06707525", "conductance": "06707525", "temp_c": "06707525",
         "do_mg_l": "06707525", "ph": "06707525", "trumbull_cfs": "06701900"}


def fetch(peak):
    out = {}
    for code, name in PARAMS.items():
        r = requests.get("https://waterservices.usgs.gov/nwis/iv/", timeout=120, params={
            "format": "json", "sites": SITES[name], "parameterCd": code,
            "startDT": (peak - pd.Timedelta(days=3)).isoformat(),
            "endDT": (peak + pd.Timedelta(days=2)).isoformat()})
        r.raise_for_status()
        ts = r.json()["value"]["timeSeries"]
        if not ts:
            continue
        v = pd.DataFrame(ts[0]["values"][0]["value"])
        s = pd.Series(pd.to_numeric(v["value"]).values, index=pd.to_datetime(v["dateTime"], utc=True))
        out[name] = s[s > -100]
    return pd.DataFrame(out).sort_index()


def main():
    spikes = pd.read_csv(DATA / "spikes.csv")
    spikes["peak_time"] = pd.to_datetime(spikes["peak_time"], utc=True).dt.tz_convert("America/Denver")
    noaa = pd.read_csv(DATA.parent.parent.parent / "data" / "USC00058022.csv", parse_dates=["DATE"]).set_index("DATE")
    frames = []
    for day, name in EVENTS.items():
        row = spikes[spikes["peak_time"].dt.strftime("%Y-%m-%d") == day].iloc[0]
        peak = row["peak_time"]
        df = fetch(peak)
        df.index = df.index.tz_convert("America/Denver")
        df["event"] = name
        frames.append(df.reset_index(names="time"))
        t = df["turbidity"]
        rise = t[t > 20]
        print(f"\n=== {name}: peak {row['peak_ntu']:.0f} at {peak:%H:%M}")
        print(f"  turbidity before/after peak (6 h median): "
              f"{t[peak - pd.Timedelta(hours=8):peak - pd.Timedelta(hours=2)].median():.1f} / "
              f"{t[peak + pd.Timedelta(hours=2):peak + pd.Timedelta(hours=8)].median():.1f}")
        print(f"  first reading > 20: {rise.index.min():%m-%d %H:%M}, last: {rise.index.max():%m-%d %H:%M}")
        w = df[peak - pd.Timedelta(hours=1):peak + pd.Timedelta(hours=1)]
        b = df[peak - pd.Timedelta(hours=8):peak - pd.Timedelta(hours=4)]
        for c in ("conductance", "temp_c", "do_mg_l", "ph", "trumbull_cfs"):
            if c in df:
                print(f"  {c:13s} baseline {b[c].median():7.1f}   around peak {w[c].median():7.1f}")
        d0 = pd.Timestamp(peak.date())
        print("  gauge rain (in) day-3..day+1:", [float(noaa["PRCP"].get(d0 + pd.Timedelta(days=k), np.nan)) for k in range(-3, 2)],
              " tmax/tmin day:", noaa.loc[d0, ["TMAX", "TMIN"]].tolist())
    pd.concat(frames).to_csv(DATA / "mystery_readings.csv", index=False)

    # 72 hours of radar rain over the lower basin
    c0, r1 = px(BBOX[0], BBOX[3]); c1, r0 = px(BBOX[1], BBOX[2])
    cols, rows = np.arange(int(c0), int(c1)), np.arange(int(r1), int(r0))
    strontia = mask(outer_ring("south-platte-above-strontia-06707525.json"), cols, rows)
    upper = mask(outer_ring("south-platte-above-trumbull-06701900.json"), cols, rows)
    lower = strontia & ~upper
    radar_rows = []
    print("\nRadar rain, basin-mean mm per hour, last 72 h before the peak (lower basin | upper basin):")
    for day, name in EVENTS.items():
        row = spikes[spikes["peak_time"].dt.strftime("%Y-%m-%d") == day].iloc[0]
        end = row["peak_time"].tz_convert("UTC").floor("h")
        lo, up = [], []
        for h in range(72, -1, -1):
            img = crop_for((end - pd.Timedelta(hours=h)).to_pydatetime())
            lo.append(np.nan if img is None else np.nanmean(img[lower]))
            up.append(np.nan if img is None else np.nanmean(img[upper]))
        lo, up = np.array(lo), np.array(up)
        for k, h in enumerate(range(72, -1, -1)):
            radar_rows.append({"event": name, "time": (end - pd.Timedelta(hours=h)).tz_convert("America/Denver"),
                               "lower_mm": lo[k], "upper_mm": up[k]})
        print(f"  {name}: lower total {np.nansum(lo):.1f} mm, upper total {np.nansum(up):.1f} mm; "
              f"missing hours {int(np.isnan(lo).sum())}; wettest lower hour {72 - int(np.nanargmax(lo))} h before peak "
              f"({np.nanmax(lo):.1f} mm)")
    pd.DataFrame(radar_rows).to_csv(DATA / "mystery_radar_hourly.csv", index=False)


if __name__ == "__main__":
    main()
