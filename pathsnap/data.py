"""Reading raw trip files and projecting lon/lat to local metres."""
import os
import glob
import numpy as np

DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "Student folder"))
R_EARTH = 6371008.8


def trip_ids(partition):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, partition, "*")))


def load_trip(partition, trip_id, with_route=True):
    base = os.path.join(DATA_DIR, partition, trip_id, trip_id)
    t = {"id": trip_id}
    t["track"] = np.loadtxt(base + ".track", ndmin=2)            # lon, lat, time
    t["nodes"] = np.loadtxt(base + ".nodes", ndmin=2)            # lon, lat
    t["arcs"] = np.loadtxt(base + ".arcs", dtype=np.int64, ndmin=2)  # from, to
    if with_route and os.path.exists(base + ".route"):
        t["route"] = np.loadtxt(base + ".route", dtype=np.int64, ndmin=1)
    return t


class Projector:
    """Local equirectangular projection around the trip centre.
    Trips cover a few km to a few tens of km, so the error is well below GPS noise."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.radians(1) * R_EARTH * np.cos(np.radians(lat0))
        self.ky = np.radians(1) * R_EARTH

    def fwd(self, lon, lat):
        return np.column_stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky])

    def inv(self, xy):
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def prepare(trip):
    """Adds projected coordinates and arc geometry to a loaded trip."""
    tr = trip["track"]
    proj = Projector(tr[:, 0].mean(), tr[:, 1].mean())
    trip["proj"] = proj
    trip["gps"] = proj.fwd(tr[:, 0], tr[:, 1])
    trip["t"] = tr[:, 2]
    nxy = proj.fwd(trip["nodes"][:, 0], trip["nodes"][:, 1])
    trip["nxy"] = nxy
    a, b = trip["arcs"][:, 0], trip["arcs"][:, 1]
    trip["A"], trip["B"] = nxy[a], nxy[b]
    d = trip["B"] - trip["A"]
    trip["alen"] = np.hypot(d[:, 0], d[:, 1])
    trip["abear"] = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360   # compass bearing
    return trip
