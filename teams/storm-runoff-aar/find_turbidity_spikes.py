"""Pull 15-minute turbidity at the Strontia gage (USGS 06707525) and list the biggest spikes.

Usage:
    python find_turbidity_spikes.py [--start 2022-04-01] [--end 2026-08-19] [--threshold 50]

Writes data/turbidity_15min.csv (cache; delete it to refetch) and data/spikes.csv, and prints the
top spikes next to the daily rain from data/USC00058022.csv so you can see whether the
single NOAA gauge caught them.

USGS values are provisional and can differ from a later pull.
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
RAW = HERE.parent.parent / "data"
DATA = HERE / "data"
SITE = "06707525"
TURBIDITY = "63680"  # FNU, same parameter as strontia-brief/series
URL = "https://waterservices.usgs.gov/nwis/iv/"
CHUNK_DAYS = 90


def fetch(start, end):
    frames = []
    for s in pd.date_range(start, end, freq=f"{CHUNK_DAYS}D"):
        e = min(s + pd.Timedelta(days=CHUNK_DAYS - 1), pd.Timestamp(end))
        r = requests.get(URL, timeout=120, params={
            "format": "json", "sites": SITE, "parameterCd": TURBIDITY,
            "startDT": s.strftime("%Y-%m-%d"), "endDT": e.strftime("%Y-%m-%d"),
        })
        r.raise_for_status()
        for ts in r.json()["value"]["timeSeries"]:
            rows = [(v["dateTime"], v["value"]) for v in ts["values"][0]["value"]]
            frames.append(pd.DataFrame(rows, columns=["time", "turbidity"]))
        print(f"fetched {s.date()} to {e.date()}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Denver")
    df["turbidity"] = pd.to_numeric(df["turbidity"], errors="coerce")
    df = df[df["turbidity"] > -100].drop_duplicates("time")  # drop -999999 sentinels
    return df.sort_values("time").reset_index(drop=True)


def find_events(df, threshold, merge_hours=6):
    """Group readings above threshold; readings within merge_hours join one event."""
    hot = df[df["turbidity"] >= threshold]
    if hot.empty:
        return pd.DataFrame()
    gap = hot["time"].diff() > pd.Timedelta(hours=merge_hours)
    hot = hot.assign(event=gap.cumsum())
    out = hot.groupby("event").apply(lambda g: pd.Series({
        "start": g["time"].iloc[0],
        "peak_time": g.loc[g["turbidity"].idxmax(), "time"],
        "peak_ntu": g["turbidity"].max(),
        "hours_above": len(g) * 0.25,
    }), include_groups=False)
    return out.sort_values("peak_ntu", ascending=False).reset_index(drop=True)


def add_rain(events):
    rain = pd.read_csv(RAW / "USC00058022.csv", parse_dates=["DATE"]).set_index("DATE")["PRCP"]
    day = events["peak_time"].dt.tz_localize(None).dt.normalize()
    events["rain_in_day_of"] = day.map(rain)
    events["rain_in_day_before"] = (day - pd.Timedelta(days=1)).map(rain)
    return events


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-04-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--threshold", type=float, default=50, help="NTU/FNU that counts as a spike")
    a = p.parse_args()

    cache = DATA / "turbidity_15min.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Denver")
    else:
        df = fetch(a.start, a.end)
        df.to_csv(cache, index=False)
    print(f"{len(df)} readings, {df['time'].min()} to {df['time'].max()}")

    events = add_rain(find_events(df, a.threshold))
    events.to_csv(DATA / "spikes.csv", index=False)
    print(f"\n{len(events)} spikes >= {a.threshold}; top 15 by peak:\n")
    print(events.head(15).to_string())


if __name__ == "__main__":
    main()
