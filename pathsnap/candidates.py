"""Candidate road segments for each GPS point, and per-point reference labels."""
import numpy as np
from scipy.spatial import cKDTree

STEP = 10.0  # metres between sample points along an arc in the spatial index


def seg_dist(P, A, B):
    """Distance from points P to segments A-B (all Nx2, broadcast) and projection fraction t."""
    d = B - A
    L2 = (d ** 2).sum(-1)
    t = np.where(L2 > 0, ((P - A) * d).sum(-1) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    C = A + t[..., None] * d
    return np.linalg.norm(P - C, axis=-1), t


class ArcIndex:
    """KD-tree over points sampled every STEP metres along arcs near the track."""

    def __init__(self, trip, margin=500.0):
        g = trip["gps"]
        lo, hi = g.min(0) - margin, g.max(0) + margin
        A, B = trip["A"], trip["B"]
        keep = ((np.maximum(A, B) >= lo).all(1) & (np.minimum(A, B) <= hi).all(1))
        self.arc_ids = np.where(keep)[0]
        pts, owner = [], []
        for i in self.arc_ids:
            n = max(1, int(np.ceil(trip["alen"][i] / STEP)))
            s = np.linspace(0, 1, n + 1)[:, None]
            pts.append(A[i] + s * (B[i] - A[i]))
            owner.append(np.full(n + 1, i))
        self.tree = cKDTree(np.vstack(pts))
        self.owner = np.concatenate(owner)
        self.trip = trip

    def query(self, P, radius, k):
        """For each point: up to k nearest arcs within radius -> (ids, dists, ts) lists."""
        A, B = self.trip["A"], self.trip["B"]
        hits = self.tree.query_ball_point(P, radius + STEP)
        out = []
        for p, h in zip(P, hits):
            ids = np.unique(self.owner[h]) if h else np.empty(0, int)
            if len(ids) == 0:
                out.append((ids, np.empty(0), np.empty(0)))
                continue
            d, t = seg_dist(np.broadcast_to(p, (len(ids), 2)), A[ids], B[ids])
            m = d <= radius
            ids, d, t = ids[m], d[m], t[m]
            o = np.argsort(d)[:k]
            out.append((ids[o], d[o], t[o]))
        return out


def align_route(trip):
    """Reference arc for each GPS point.
    The .route file only lists arcs in order, not per point, so we assign every point to a route
    arc with a monotone (never going back along the route) alignment that minimises total distance."""
    route = trip["route"]
    P = trip["gps"]
    A, B = trip["A"][route], trip["B"][route]
    N, R = len(P), len(route)
    cost = np.empty((N, R))
    for j in range(R):
        cost[:, j], _ = seg_dist(P, np.broadcast_to(A[j], P.shape), np.broadcast_to(B[j], P.shape))
    back = np.empty((N, R), dtype=np.int32)
    best = cost[0].copy()
    back[0] = np.arange(R)
    for i in range(1, N):
        # running prefix minimum over j' <= j
        idx = np.arange(R)
        m = np.minimum.accumulate(best)
        arg = np.maximum.accumulate(np.where(best == m, idx, 0))
        back[i] = arg
        best = cost[i] + m
    j = int(np.argmin(best))
    lab = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        lab[i] = j
        j = back[i][j] if i > 0 else j
    ridx = lab
    return route[ridx], cost[np.arange(N), ridx]
