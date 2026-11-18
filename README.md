# PathSnap - data processing and cleaning

Map-matching and route reconstruction for noisy GPS trajectories (UE24CS352A Machine Learning hackathon).
This repo has our data processing and cleaning stage. Problem statement and guidelines are in `docs/`.

## Team

- Ankit B (AnkitB1405) - data loading, EDA
- Aaron Immanuel (aaronimmanuel) - candidate generation, labels, feature extraction
- Gaganaraj (gaganaraj) - data cleaning checks, label quality, search radius
- Ananya Dharani (Ananya-Dharani) - Folium map, feature exploration

Who did what, with details for each member: [CONTRIBUTIONS.md](CONTRIBUTIONS.md)

## What is done

- Reading all 4 file types for every trip, local metric projection per trip
- EDA: inventory, missing / invalid values, time gaps, extents, network connectivity (`results/eda/`)
- Cleaning: 2 train trips (33, 54) have GPS 120-140 m away from their route, so they are excluded. 69 gaps over 60 s in train, which split 23 trips into pieces. 147 speed jumps over 70 m/s (`results/cleaning/`)
- Per-point labels: every GPS point is matched to one arc of the reference route (monotone alignment). Median point-to-label distance is 5 m
- Candidates: arcs within 50 m, at most 8 per point. Miss rate at 50 m is 0.07% on student_test and 0.6% on train
- 13 features per (GPS point, candidate) pair: 1,033,029 train rows, 17.4% positives. The true arc is the nearest candidate only 57% of the time, so distance alone is not enough
- Folium map of a train trip (`maps/`)

## Next

Model training and comparison, Bayesian tuning, HMM / Viterbi, API, UI, report.

## Running

The dataset is not committed. Put `train/` and `student_test/` in `Student folder/` next to this repo, or set `PATHSNAP_DATA`.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/eda.py
python scripts/data_cleaning.py
python scripts/label_quality.py
python scripts/make_maps.py
python scripts/build_dataset.py
python scripts/feature_plots.py
```
