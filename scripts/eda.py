"""Dataset exploration: inventory, data quality checks, network connectivity and plots."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.data import DATA_DIR, trip_ids, load_trip, prepare
from pathsnap.candidates import align_route

OUT = "results/eda"
os.makedirs(OUT, exist_ok=True)


def inspect(part, tid):
    base = os.path.join(DATA_DIR, part, tid, tid)
    sizes = {e: os.path.getsize(f"{base}.{e}") if os.path.exists(f"{base}.{e}") else 0
             for e in ["track", "nodes", "arcs", "route"]}
    tr = load_trip(part, tid)
    trk, nodes, arcs, route = tr["track"], tr["nodes"], tr["arcs"], tr["route"]
    dt = np.diff(trk[:, 2])
    bad = (~np.isfinite(trk).all(1)) | (np.abs(trk[:, 0]) > 180) | (np.abs(trk[:, 1]) > 90)
    n = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    ncomp, comp = connected_components(g, directed=True, connection="weak")
    used = np.zeros(n, bool)
    used[arcs.ravel()] = True
    prepare(tr)
    _, off = align_route(tr)
    ext = tr["gps"].max(0) - tr["gps"].min(0)
    return {
        "partition": part, "trip": tid, "n_points": len(trk), "n_nodes": n, "n_arcs": len(arcs),
        "n_route_arcs": len(route), **{f"size_{k}_kb": v / 1024 for k, v in sizes.items()},
        "invalid_points": int(bad.sum()), "duration_s": trk[-1, 2] - trk[0, 2],
        "non_increasing_t": int((dt <= 0).sum()), "share_dt_1s": float(np.mean(np.isclose(dt, 1, atol=0.01))),
        "max_gap_s": dt.max(), "gaps_over_10s": int((dt > 10).sum()),
        "extent_x_km": ext[0] / 1000, "extent_y_km": ext[1] / 1000,
        "lon_c": trk[:, 0].mean(), "lat_c": trk[:, 1].mean(),
        "weak_components": ncomp, "largest_comp_share": np.bincount(comp).max() / n,
        "isolated_nodes": int((~used).sum()),
        "route_connected_share": float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])),
        "route_offset_median_m": float(np.median(off)), "route_offset_p95_m": float(np.percentile(off, 95)),
    }


if __name__ == "__main__":
    rows = [inspect(p, t) for p in ["train", "student_test"] for t in trip_ids(p)]
    inv = pd.DataFrame(rows)
    inv.to_csv(f"{OUT}/inventory.csv", index=False)

    summ = inv.groupby("partition").agg(trips=("trip", "count"), points=("n_points", "sum"),
                                        median_points=("n_points", "median"), invalid=("invalid_points", "sum"),
                                        trips_with_gap_10s=("gaps_over_10s", lambda s: (s > 0).sum()),
                                        median_nodes=("n_nodes", "median"), median_arcs=("n_arcs", "median"))
    summ.to_csv(f"{OUT}/summary.csv")
    print(summ.to_string())
    print(inv.describe().T[["mean", "min", "50%", "max"]].round(3).to_string())
    print("anomalies (route offset > 50 m):")
    print(inv[inv.route_offset_median_m > 50][["partition", "trip", "n_points", "route_offset_median_m"]])

    # sampling intervals (train)
    dts = np.concatenate([np.diff(load_trip("train", t)["track"][:, 2]) for t in trip_ids("train")])
    pd.Series(dts).describe(percentiles=[.5, .9, .97, .99, .999]).to_csv(f"{OUT}/sampling_intervals_train.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(inv[inv.partition == "train"].n_points, bins=30, color="tab:blue")
    ax[0].set_title("GPS points per trip (train)"); ax[0].set_xlabel("points")
    ax[1].hist(np.clip(dts, 0, 10), bins=40, color="tab:orange", log=True)
    ax[1].set_title("Time between GPS points (train, clipped at 10 s)"); ax[1].set_xlabel("seconds")
    ax[2].hist(np.clip(inv.route_offset_median_m, 0, 30), bins=30, color="tab:green")
    ax[2].set_title("Median GPS-to-route distance per trip (all)"); ax[2].set_xlabel("metres (clipped at 30)")
    plt.tight_layout(); plt.savefig(f"{OUT}/distributions.png", dpi=120); plt.close()

    # where the trips are
    plt.figure(figsize=(10, 5))
    for p, c in [("train", "tab:blue"), ("student_test", "tab:red")]:
        s = inv[inv.partition == p]
        plt.scatter(s.lon_c, s.lat_c, s=25, c=c, label=p)
    plt.xlabel("longitude"); plt.ylabel("latitude"); plt.legend(); plt.title("Trip locations")
    plt.grid(alpha=.3); plt.savefig(f"{OUT}/trip_locations.png", dpi=120); plt.close()

    # GPS trajectories of a few trips
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, t in zip(axs.ravel(), trip_ids("train")[:8]):
        trk = load_trip("train", t)["track"]
        ax.plot(trk[:, 0], trk[:, 1], ".", ms=1)
        ax.set_title(f"train {t} ({len(trk)} pts)"); ax.set_aspect(1 / np.cos(np.radians(trk[:, 1].mean())))
        ax.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/gps_trajectories.png", dpi=110); plt.close()

    # road network around one trip with GPS and route
    tr = prepare(load_trip("train", "00000097"))
    g = tr["gps"]; lo, hi = g.min(0) - 300, g.max(0) + 300
    A, B = tr["A"], tr["B"]
    m = ((A >= lo) & (A <= hi)).all(1)
    plt.figure(figsize=(9, 9))
    for a, b in zip(A[m], B[m]):
        plt.plot([a[0], b[0]], [a[1], b[1]], c="0.75", lw=.6)
    for r in tr["route"]:
        plt.plot([A[r, 0], B[r, 0]], [A[r, 1], B[r, 1]], c="tab:green", lw=2.5)
    plt.plot(g[:, 0], g[:, 1], ".", c="tab:blue", ms=2)
    plt.axis("equal"); plt.title("train 00000097: road network (grey), route (green), GPS (blue)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.savefig(f"{OUT}/network_00000097.png", dpi=120); plt.close()

    # zoomed noise view
    plt.figure(figsize=(7, 7))
    c = g[len(g) // 2]
    m2 = ((A >= c - 250) & (A <= c + 250)).all(1)
    for a, b in zip(A[m2], B[m2]):
        plt.plot([a[0], b[0]], [a[1], b[1]], c="0.6", lw=1)
    for r in tr["route"]:
        plt.plot([A[r, 0], B[r, 0]], [A[r, 1], B[r, 1]], c="tab:green", lw=3)
    gm = (np.abs(g - c) < 250).all(1)
    plt.plot(g[gm, 0], g[gm, 1], "o", c="tab:blue", ms=3)
    plt.xlim(c[0] - 250, c[0] + 250); plt.ylim(c[1] - 250, c[1] + 250); plt.gca().set_aspect(1)
    plt.title("train 00000097 zoomed (500 m box)")
    plt.savefig(f"{OUT}/network_zoom_00000097.png", dpi=120); plt.close()
