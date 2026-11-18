"""Builds candidate feature tables for the train and student_test partitions."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathsnap.data import trip_ids, load_trip, prepare
from pathsnap.features import trip_features


def one(part, tid):
    df, info, _ = trip_features(prepare(load_trip(part, tid)))
    return df, {"partition": part, "trip": tid, "n_points": info["n_points"], "n_no_cand": info["n_no_cand"],
                "miss_rate": info["miss_rate"], "ref_dist_median": float(np.median(info["ref_dist"])), "n_rows": len(df)}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    stats = []
    for part in ["train", "student_test"]:
        t0 = time.time()
        res = Parallel(n_jobs=8)(delayed(one)(part, t) for t in trip_ids(part))
        pd.concat([r[0] for r in res]).to_pickle(f"results/features_{part}.pkl")
        stats += [r[1] for r in res]
        print(part, len(res), "trips", round(time.time() - t0, 1), "s")
    s = pd.DataFrame(stats)
    s.to_csv("results/candidate_stats.csv", index=False)
    print(s.groupby("partition")[["n_points", "n_no_cand", "n_rows"]].sum())
    print(s.groupby("partition")["miss_rate"].describe())
