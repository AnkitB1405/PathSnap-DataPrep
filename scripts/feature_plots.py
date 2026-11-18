"""Feature exploration on the train candidate table: true arc vs other candidates."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathsnap.features import FEATURES

OUT = "results/features"
os.makedirs(OUT, exist_ok=True)
df = pd.read_pickle("results/features_train.pkl")
excluded = set(pd.read_csv("results/cleaning/excluded_trips.csv", dtype={"trip": str}).trip)
df = df[~df.trip.isin(excluded)]
print("rows", len(df), "| GPS points", df.groupby(["trip", "pt"]).ngroups, "| positive share", round(df.y.mean(), 4))
print("true arc is the nearest candidate:", round(df[df.y == 1].dist_rank.eq(0).mean(), 4))
m = df.groupby("y")[FEATURES].mean().T
m.columns = ["other candidates", "true arc"]
m.to_csv(f"{OUT}/feature_means.csv")
print(m.round(3).to_string())

show = ["dist", "dist_rank", "head_diff", "ctx_dist", "ctx_head_diff", "speed"]
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for ax, f in zip(axs.ravel(), show):
    hi = df[f].quantile(0.99)
    for y, c, n in [(0, "tab:gray", "other candidates"), (1, "tab:green", "true arc")]:
        ax.hist(df.loc[df.y == y, f].clip(upper=hi), bins=40, density=True, alpha=.6, color=c, label=n)
    ax.set_title(f); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/feature_distributions.png", dpi=110); plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(df[FEATURES + ["y"]].corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(FEATURES) + 1), FEATURES + ["y"], rotation=90); plt.yticks(range(len(FEATURES) + 1), FEATURES + ["y"])
plt.colorbar(); plt.title("Feature correlation (train)"); plt.tight_layout()
plt.savefig(f"{OUT}/feature_correlation.png", dpi=110); plt.close()
