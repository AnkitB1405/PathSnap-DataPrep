"""Feature extraction: one row per (GPS point, candidate arc)."""
import numpy as np
import pandas as pd
from .candidates import ArcIndex, align_route, seg_dist

RADIUS = 50.0   # candidate search radius in metres
K = 8           # max candidates per point
WIN = 5         # context window (points either side)

FEATURES = ["dist", "dist_rank", "dist_gap", "n_cand", "head_diff", "head_ok", "speed",
            "alen", "t_proj", "ctx_dist", "ctx_head_diff", "out_deg", "dt_prev"]


def gps_heading(P, w=2):
    """Heading of travel at each point from the points w steps before/after."""
    n = len(P)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    d = P[i1] - P[i0]
    move = np.hypot(d[:, 0], d[:, 1])
    head = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    return head, move


def ang_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def trip_features(trip, labels=True, radius=RADIUS, k=K):
    P, T = trip["gps"], trip["t"]
    n = len(P)
    idx = ArcIndex(trip)
    cands = idx.query(P, radius, k)
    head, move = gps_heading(P)
    head_ok = (move > 3.0).astype(float)
    dt = np.diff(T, prepend=T[0])
    step = np.hypot(*np.diff(P, axis=0, prepend=P[:1]).T)
    speed = np.where(dt > 0, step / np.where(dt > 0, dt, 1), 0)
    speed = pd.Series(speed).rolling(5, center=True, min_periods=1).median().values
    out_deg = np.bincount(trip["arcs"][:, 0], minlength=len(trip["nodes"]))
    to_node = trip["arcs"][:, 1]

    rows = []
    for i, (ids, d, t) in enumerate(cands):
        if len(ids) == 0:
            continue
        lo, hi = max(0, i - WIN), min(n, i + WIN + 1)
        ctx = P[lo:hi]
        # mean distance of neighbouring points to each candidate arc (as a segment)
        cd = np.array([seg_dist(ctx, np.broadcast_to(trip["A"][a], ctx.shape),
                                np.broadcast_to(trip["B"][a], ctx.shape))[0].mean() for a in ids])
        hd = ang_diff(head[i], trip["abear"][ids])
        r = np.radians(head[lo:hi])
        ctx_mean = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360
        ctx_h = ang_diff(ctx_mean, trip["abear"][ids])
        rows.append(pd.DataFrame({
            "trip": trip["id"], "pt": i, "arc": ids,
            "dist": d, "dist_rank": np.arange(len(ids)), "dist_gap": d - d[0], "n_cand": len(ids),
            "head_diff": np.where(head_ok[i] > 0, hd, 90.0), "head_ok": head_ok[i], "speed": speed[i],
            "alen": trip["alen"][ids], "t_proj": t, "ctx_dist": cd,
            "ctx_head_diff": np.where(head_ok[i] > 0, ctx_h, 90.0),
            "out_deg": out_deg[to_node[ids]], "dt_prev": dt[i],
        }))
    df = pd.concat(rows, ignore_index=True)
    info = {"n_points": n, "n_no_cand": sum(len(c[0]) == 0 for c in cands)}
    if labels and "route" in trip:
        lab, ldist = align_route(trip)
        df["ref_arc"] = lab[df["pt"].values]
        df["y"] = (df["arc"] == df["ref_arc"]).astype(int)
        info["ref_arc"] = lab
        info["ref_dist"] = ldist
        hit = np.zeros(n, bool)
        hit[df.loc[df.y == 1, "pt"].values] = True
        info["miss_rate"] = 1 - hit.mean()
    return df, info, cands
