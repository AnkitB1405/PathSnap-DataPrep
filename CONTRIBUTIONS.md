# Team roles - data processing and cleaning phase

This phase covers everything before model training: reading the raw files, checking the data, cleaning it, making labels and building features.

Overall flow:

```
raw trip files -> load + project to metres -> EDA -> cleaning -> candidates + labels -> features -> dataset
   (Ankit)            (Ankit)              (Ankit)  (Gaganaraj)      (Aaron)            (Aaron)
                                                     label check (Gaganaraj), map + feature plots (Ananya)
```

---

## Ankit B - data loading and EDA

**Files:** `pathsnap/data.py`, `scripts/eda.py`, `results/eda/`

**What I did**

- Wrote the loader for the 4 files of every trip:
  - `.track` = lon, lat, time
  - `.nodes` = lon, lat
  - `.arcs` = from node, to node
  - `.route` = list of arc ids
- Lon/lat are angles, so distances in degrees are wrong. I convert each trip to metres with a simple local projection centred on that trip. The trips are all over the world, so one fixed projection would not work.
- Wrote the EDA script. It checks every trip and saves `inventory.csv` and `summary.csv`.

**What I found**

| | train | student_test |
|---|---|---|
| Trips | 80 | 10 |
| GPS points | 208,417 | 19,160 |
| Missing / invalid rows | 0 | 0 |
| Trips with a gap over 10 s | 40 | 2 |

- Timestamps always go forward. About 97% of steps are 1 s, but one trip has a gap of about 42 hours.
- Each trip has its own road network (median about 9,600 nodes and 19,500 arcs in train).
- The networks have several disconnected pieces, but the biggest piece has about 98% of the nodes.
- The route file does not say which GPS point is on which arc. So we have to create the per-point labels ourselves (Aaron's part).

**How to explain it:** "I made sure we read the data correctly and measured distances in metres. Then I checked the data for missing values, time gaps and network connectivity before anyone built on it."

---

## Aaron Immanuel - candidates, labels and features

**Files:** `pathsnap/candidates.py`, `pathsnap/features.py`, `scripts/build_dataset.py`, `results/candidate_stats.csv`

**What I did**

1. **Candidate arcs.** For every GPS point we need a short list of roads it could be on. I sample points every 10 m along each arc and put them in a KD-tree, so the search is fast. Then for each GPS point I keep the arcs within 50 m, at most 8.
2. **Per-point labels.** The `.route` file is only an ordered list of arcs. I match every GPS point to one arc of that list, using dynamic programming that:
   - keeps the total distance as small as possible
   - never goes backwards along the route

   This gives one "correct arc" for every GPS point.
3. **Features.** Each row is one (GPS point, candidate arc) pair, with 13 features:
   - distance to the arc, distance rank, gap to the closest candidate
   - direction of travel vs arc direction (arcs are one-way, so this separates the two directions of a road)
   - average distance of the 5 points before and after (so one noisy point doesn't decide alone)
   - speed, time gap, arc length, position along the arc, number of roads at the next junction
4. `build_dataset.py` runs this for all trips in parallel and saves the tables.

**Results**

- Train has 1,033,029 rows after cleaning, and 17.4% of them are the correct arc.
- The correct arc is in the candidate list for over 99% of points on student_test.

**How to explain it:** "I turned the problem into a table. For every GPS point we list the nearby roads, mark which one is correct, and describe each road with features a model can learn from."

---

## Gaganaraj - data cleaning and label quality

**Files:** `scripts/data_cleaning.py`, `scripts/label_quality.py`, `results/cleaning/`

**What I did**

1. **Cleaning checks for every trip** (`trip_flags.csv`):
   - how far the GPS is from its own reference route
   - time gaps over 60 s
   - jumps faster than 70 m/s
   - share of points where the car is standing still
2. **Excluded trips.** Train trips 33 and 54 have GPS sitting 122 m and 140 m (median) from their route. For a normal trip that distance is about 5 m. The GPS or the route is shifted in these two trips, so their labels would be wrong. We drop them from training (`excluded_trips.csv`).
3. **Other issues.** Not removed, but noted for later:
   - 69 gaps over 60 s in train, which split 23 trips into pieces. The route model will need to restart at these gaps.
   - 147 GPS jumps over 70 m/s.
4. **Label check and search radius.** I checked how far points are from their label arc: median 5 m, 99% under 33 m. I also tested search radii of 15, 25, 50, 75 and 100 m:

| Radius | Miss rate (train) | Miss rate (student_test) | Avg candidates |
|---|---|---|---|
| 15 m | 8.6% | 6.6% | about 3 |
| 25 m | 2.6% | 1.4% | about 5 |
| 50 m | 0.6% | 0.07% | about 10-12 |
| 100 m | 0.2% | 0.04% | about 22-26 |

We picked 50 m. Going further adds many more candidates but almost no extra correct ones.

**How to explain it:** "I found the trips and points we can't trust, and removed or flagged them. I also checked that our labels make sense, and chose the 50 m search radius from the numbers instead of guessing."

---

## Ananya Dharani - maps and feature exploration

**Files:** `scripts/make_maps.py`, `maps/`, `scripts/feature_plots.py`, `results/features/`

**What I did**

1. **Folium map.** Interactive map of train trip 00000097 with GPS points (blue), reference route (green) and nearby road network (grey). It has a legend and a layer control, and there is a full view and a zoomed view. This lets us actually see the GPS noise around the road.
2. **Feature exploration.** I compared the correct arc against the other candidates for each feature (`feature_distributions.png`, `feature_means.csv`) and made a correlation plot.

**What I found**

| Feature (mean) | Other candidates | Correct arc |
|---|---|---|
| Distance to arc | 19.7 m | 6.2 m |
| Heading difference | 100 deg | 23 deg |
| Neighbour distance (ctx_dist) | 32.2 m | 16.0 m |

- The correct arc is the nearest one only **57%** of the time, so just snapping to the nearest road is not good enough. That is why we need a model.
- Heading and neighbour distance separate the classes clearly, so they should be useful.
- Some features are strongly related (dist, dist_gap, ctx_dist). This matters for models like Naive Bayes that assume independence.

**How to explain it:** "I made the data visible on a real map, and showed which features actually tell the correct road apart from the wrong ones."

---

## Next phase

Model training (each member trains and validates models), Bayesian tuning, HMM / Viterbi route reconstruction, API, UI and report.
