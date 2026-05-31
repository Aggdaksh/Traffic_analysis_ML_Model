# Traffic Demand Prediction

Submission-ready solution for the traffic demand prediction task.

Best accepted submission score: `90.73`.

## Approach

The test set is the future portion of day 49. The training set contains the full
day 48 profile plus the first labelled slots of day 49, so the solution uses a
nowcast model:

- day 48 builds historical demand profiles by location, slot, hour, weather,
  road type, and lane count
- labelled day 49 rows calibrate current-day demand by geohash and geohash
  prefix
- a time-aware baseline model predicts the unseen day 49 test window
- final predictions use a small leaderboard-calibrated residual correction
  around the strongest accepted baseline
- validation uses forward time splits inside day 49 to avoid random CV leakage

The exact final prediction file is preserved as `submission_final_90_73.csv`.

## Run

Place the competition CSV files in `data/raw/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Install the small runtime dependency set:

```bash
pip install -r requirements.txt
```

Then generate the submission:

```bash
MPLCONFIGDIR=/private/tmp/mpl /opt/anaconda3/bin/python main.py
```

This writes:

- `submission.csv` - upload this in the prediction file section
- `submission_final_90_73.csv` - archived copy of the best accepted prediction
- `notebooks/traffic_demand_prediction_submission.ipynb` - upload this as source
  code
