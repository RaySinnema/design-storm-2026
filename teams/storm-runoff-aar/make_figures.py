"""Draw pictures of the CSVs in data/ into figures/.

Usage:
    python make_figures.py

Reads data/turbidity_15min.csv, data/spikes.csv, data/radar_rain_events.csv and
data/radar_rain_hourly.csv. Rain amounts are approximate radar estimates (see the
caution in add_radar_rain.py); turbidity is provisional USGS data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
TZ = "America/Denver"
IN_TO_MM = 25.4

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1 to 3

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})

NOTE = "USGS turbidity is provisional. Rain is an approximate radar estimate."


def local(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ)


def load():
    ts = pd.read_csv(DATA / "turbidity_15min.csv")
    ts["time"] = local(ts["time"])
    spikes = pd.read_csv(DATA / "spikes.csv")
    for c in ("start", "peak_time"):
        spikes[c] = local(spikes[c])
    ev = pd.read_csv(DATA / "radar_rain_events.csv")
    ev["peak_time"] = local(ev["peak_time"])
    hourly = pd.read_csv(DATA / "radar_rain_hourly.csv")
    return ts, spikes, ev, hourly


def label(row):
    return f"{row['peak_time']:%b %-d, %Y}"


def footnote(fig, text=NOTE):
    fig.text(0.01, 0.005, text, fontsize=8, color=INK2, ha="left", va="bottom")


def timeline(ts, spikes):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    # Break the line where readings are missing (winter gaps) instead of drawing across them.
    gap = ts["time"].diff() > pd.Timedelta(hours=6)
    seg = ts.assign(turbidity=ts["turbidity"].clip(lower=0.1))
    breaks = seg[gap].assign(turbidity=np.nan, time=lambda d: d["time"] - pd.Timedelta(minutes=1))
    seg = pd.concat([seg, breaks]).sort_values("time")
    ax.plot(seg["time"], seg["turbidity"], color=BLUE, lw=0.6)
    ax.set_yscale("log")
    ax.axhline(50, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(ts["time"].iloc[0], 56, "spike threshold, 50", fontsize=8, color=INK2)
    for i, r in spikes.head(5).iterrows():
        ax.annotate(f"{r['peak_ntu']:.0f}", (r["peak_time"], r["peak_ntu"]), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("Turbidity (FNU, log scale)")
    ax.set_title("Turbidity above Strontia Springs, 2022 to 2026")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("")
    footnote(fig, "USGS 06707525, 15-minute readings. Labels mark the five largest peaks. " + NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "1_turbidity_timeline.png", dpi=150)


def spike_bars(spikes):
    s = spikes.sort_values("peak_ntu")
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(s))
    ax.barh(y, s["peak_ntu"], color=BLUE, height=0.62)
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.text(r["peak_ntu"] + 6, yi, f"{r['peak_ntu']:.0f}   ({r['hours_above']:.1f} h above 50)",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [label(r) for _, r in s.iterrows()])
    ax.set_xlim(0, s["peak_ntu"].max() * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Peak turbidity (FNU)")
    ax.set_title(f"All {len(s)} spikes at or above 50 FNU, ranked by peak")
    footnote(fig, "Spikes are readings at or above 50 FNU, merged if within 6 hours. " + NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "2_spikes_ranked.png", dpi=150)


def rain_by_zone(ev):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(ev))
    w = 0.26
    series = [
        ("Upper basin (above Trumbull), radar, 12 h", ev["upper_total_mm"], BLUE),
        ("Lower basin (between the gages), radar, 12 h", ev["lower_total_mm"], ORANGE),
        ("Strontia Dam rain gauge, whole day", ev["gauge_day_of_in"] * IN_TO_MM, AQUA),
    ]
    for k, (name, vals, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * (w + 0.02), vals, w, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f"{label(r)}\n{r['peak_ntu']:.0f} FNU" for _, r in ev.iterrows()], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Rain (mm)")
    ax.set_title("Rain before each of the ten biggest spikes, by area")
    ax.legend(loc="upper right", fontsize=8.5)
    footnote(fig, "Gauge inches converted to mm. Radar covers the 12 hours before the peak; the gauge covers a full day. " + NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "3_rain_by_area.png", dpi=150)


def hourly_panels(ev, hourly):
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8), sharex=True, sharey=True)
    for ax, (i, r) in zip(axes.ravel(), ev.iterrows()):
        h = hourly[hourly["event"] == i]
        ax.bar(-h["hours_before_peak"], h["lower_mean_mm"], width=0.8, color=ORANGE)
        ax.set_title(f"{label(r)}\n{r['peak_ntu']:.0f} FNU", fontsize=9, loc="left", fontweight="normal")
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Hours before turbidity peak", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rain (mm per hour)", fontsize=8)
    fig.suptitle("Hour-by-hour rain over the lower basin before each spike", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    footnote(fig, "Average of radar pixels between the Trumbull and Strontia gages. Same scale on every panel. " + NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "4_hourly_rain_per_event.png", dpi=150)


def scatter(ev):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.scatter(ev["lower_total_mm"], ev["peak_ntu"], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    dry = ev["lower_total_mm"] < 3
    below = {"Apr 16, 2024": (8, -12)}  # keeps it clear of the Jul 28, 2026 label
    for _, r in ev.iterrows():
        dx, dy = below.get(label(r), (8, 4))
        ax.annotate(label(r), (r["lower_total_mm"], r["peak_ntu"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlabel("Radar rain over the lower basin in the 12 hours before the peak (mm)")
    ax.set_ylabel("Peak turbidity (FNU)")
    ax.set_title("More rain does not always mean muddier water")
    ax.text(0.02, 0.97, f"{dry.sum()} of {len(ev)} spikes had under 3 mm of radar rain nearby",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK2)
    footnote(fig, "Ten biggest spikes only. " + NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "5_rain_vs_turbidity.png", dpi=150)


def mystery_events():
    """Three spikes with almost no rain in the 12 hours before them: what else moved?"""
    if not (DATA / "mystery_readings.csv").exists():
        return
    rd = pd.read_csv(DATA / "mystery_readings.csv")
    rd["time"] = local(rd["time"])
    rr = pd.read_csv(DATA / "mystery_radar_hourly.csv")
    rr["time"] = local(rr["time"])
    names = ["Aug 6, 2024", "Jul 28, 2026", "Apr 16, 2024"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex="col")
    for j, n in enumerate(names):
        d = rd[rd["event"] == n].set_index("time")
        r = rr[rr["event"] == n].set_index("time")
        pk = d["turbidity"].idxmax()
        lo, hi = pk - pd.Timedelta(hours=36), pk + pd.Timedelta(hours=24)
        a0, a1, a2 = axes[:, j]
        a0.bar(r.index, r["lower_mm"], width=0.035, color=ORANGE)
        a0.set_xlim(lo, hi)
        a0.set_title(f"{n}, {d['turbidity'].max():.0f} FNU", loc="left", fontsize=11)
        a1.plot(d.index, d["trumbull_cfs"], color=BLUE, lw=1.4)
        a2.plot(d.index, d["turbidity"], color=BLUE, lw=1.2)
        a2.set_yscale("log")
        for a in axes[:, j]:
            a.axvline(pk, color=INK2, lw=1, ls=(0, (4, 3)))
        a2.xaxis.set_major_formatter(mdates.DateFormatter("%-d %Hh", tz=TZ))
        a2.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=TZ))
    axes[0, 0].set_ylabel("Lower-basin rain\n(mm per hour, radar)", fontsize=9)
    axes[1, 0].set_ylabel("Flow at Trumbull\n(cfs)", fontsize=9)
    axes[2, 0].set_ylabel("Turbidity at Strontia\n(FNU, log)", fontsize=9)
    fig.suptitle("Three spikes with little rain in the 12 hours before them", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    footnote(fig, "Dashed line marks the turbidity peak. Local time. " + NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / "6_mystery_events.png", dpi=150)


def main():
    ts, spikes, ev, hourly = load()
    timeline(ts, spikes)
    spike_bars(spikes)
    rain_by_zone(ev)
    hourly_panels(ev, hourly)
    scatter(ev)
    mystery_events()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")


if __name__ == "__main__":
    main()
