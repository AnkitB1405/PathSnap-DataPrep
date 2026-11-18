"""Checks the per-point labels and picks the candidate search radius.
Label = route arc matched to each GPS point (monotone alignment in pathsnap/candidates.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathsnap.data import trip_ids, load_trip, prepare
from pathsnap.candidates import ArcIndex, align_route

OUT = "results/cleaning"
os.makedirs(OUT, exist_ok=True)
RADII = [15, 25, 50, 75, 100]
excluded = set(pd.read_csv(f"{OUT}/excluded_trips.csv", dtype={"trip": str}).trip)
rng = np.random.RandomState(42)
train = [t for t in trip_ids("train") if t not in excluded]
sample = [("train", t) for t in sorted(rng.choice(train, 20, replace=False))] + \
         [("student_test", t) for t in trip_ids("student_test")]

rows, dists = [], []
for part, t in sample:
    tr = prepare(load_trip(part, t))
    lab, d = align_route(tr)
    dists.append(d)
    idx = ArcIndex(tr)
    for r in RADII:
        c = idx.query(tr["gps"], r, 50)
        miss = np.mean([l not in ids for l, (ids, _, _) in zip(lab, c)])
        rows.append({"partition": part, "trip": t, "radius_m": r, "miss_rate": miss,
                     "mean_candidates": np.mean([len(x[0]) for x in c])})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/radius_check.csv", index=False)
summ = res.groupby(["partition", "radius_m"])[["miss_rate", "mean_candidates"]].mean()
summ.to_csv(f"{OUT}/radius_summary.csv")
print(summ.round(4).to_string())
d = np.concatenate(dists)
print("point-to-label distance percentiles 50/90/99:", np.percentile(d, [50, 90, 99]).round(1))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(np.clip(d, 0, 60), bins=60)
ax[0].set_title("Distance from GPS point to its label arc (sampled trips)"); ax[0].set_xlabel("metres (clipped at 60)")
for p in ["train", "student_test"]:
    s = summ.loc[p]
    ax[1].plot(s.index, 100 * s.miss_rate, "o-", label=p)
ax[1].set_xlabel("search radius (m)"); ax[1].set_ylabel("candidate miss rate (%)"); ax[1].legend()
ax[1].set_title("Label arc not among candidates")
plt.tight_layout(); plt.savefig(f"{OUT}/label_quality.png", dpi=120); plt.close()
