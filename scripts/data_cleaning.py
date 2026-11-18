"""Data cleaning checks per trip: shifted routes, time gaps, speed jumps, stationary points.
Writes results/cleaning/trip_flags.csv and the list of trips we exclude from training."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathsnap.data import trip_ids, load_trip, prepare

OUT = "results/cleaning"
os.makedirs(OUT, exist_ok=True)
MAX_OFFSET_M = 50     # median GPS-to-route distance above this = route and GPS do not agree
MAX_GAP_S = 60        # gaps above this split the trip into separate pieces
MAX_SPEED = 70        # m/s, faster than this between two points = GPS jump

inv = pd.read_csv("results/eda/inventory.csv", dtype={"trip": str})
rows, speeds = [], []
for part in ["train", "student_test"]:
    for t in trip_ids(part):
        tr = prepare(load_trip(part, t))
        g, ts = tr["gps"], tr["t"]
        dt = np.diff(ts)
        v = np.hypot(*np.diff(g, axis=0).T) / dt
        if part == "train":
            speeds.append(v)
        off = inv[(inv.partition == part) & (inv.trip == t)].route_offset_median_m.iloc[0]
        rows.append({"partition": part, "trip": t, "n_points": len(g),
                     "route_offset_median_m": off, "gaps_over_60s": int((dt > MAX_GAP_S).sum()),
                     "pieces": int((dt > MAX_GAP_S).sum()) + 1, "speed_jumps": int((v > MAX_SPEED).sum()),
                     "stationary_share": float(np.mean(v < 0.5)), "median_speed": float(np.median(v)),
                     "exclude": bool(off > MAX_OFFSET_M)})
f = pd.DataFrame(rows)
f.to_csv(f"{OUT}/trip_flags.csv", index=False)
f[f.exclude][["partition", "trip"]].to_csv(f"{OUT}/excluded_trips.csv", index=False)
print(f.groupby("partition")[["gaps_over_60s", "speed_jumps"]].sum())
print("trips split by gaps:", (f.pieces > 1).sum(), "| stationary share (median):", f.stationary_share.median().round(3))
print("excluded:", f[f.exclude][["partition", "trip", "route_offset_median_m"]].to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(np.clip(np.concatenate(speeds), 0, 50), bins=50, log=True)
ax[0].set_title("Speed between consecutive GPS points (train, clipped at 50 m/s)"); ax[0].set_xlabel("m/s")
s = f.sort_values("route_offset_median_m")
ax[1].bar(range(len(s)), s.route_offset_median_m, color=np.where(s.exclude, "tab:red", "tab:blue"))
ax[1].axhline(MAX_OFFSET_M, c="k", ls="--"); ax[1].set_yscale("log")
ax[1].set_title("Median GPS-to-route distance per trip (red = excluded)"); ax[1].set_ylabel("metres")
plt.tight_layout(); plt.savefig(f"{OUT}/cleaning.png", dpi=120); plt.close()
