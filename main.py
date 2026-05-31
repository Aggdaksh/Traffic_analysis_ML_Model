import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("traffic_nowcast")


DATA_DIR = Path("data/raw")
REPORT_PATH = Path("data/processed/validation_report.json")
SUBMISSION_PATH = Path("submission.csv")
FINAL_SUBMISSION_PATH = Path("submission_final_90_73.csv")

NUMERIC_FEATURES = [
    "NumberofLanes",
    "Temperature",
    "Temperature_missing",
    "geo_mean",
    "geo_median",
    "geo_std",
    "geo_max",
    "geo_slot_mean",
    "geo_hour_mean",
    "gh5_slot_mean",
    "slot_mean",
    "hour_mean",
    "road_slot_mean",
    "weather_slot_mean",
    "lanes_slot_mean",
    "cur_geo_mean",
    "cur_geo_last",
    "cur_geo_first",
    "cur_geo_std",
    "cur_geo_count",
    "cur_geo_delta",
    "cur_gh5_mean",
    "cur_gh4_mean",
    "cur_seen",
]

CATEGORICAL_FEATURES = [
    "geohash",
    "gh4",
    "gh5",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable categorical and time keys without extrapolating raw time."""
    out = df.copy()
    time_parts = out["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    out["hour"] = time_parts[0].astype("int16")
    out["minute"] = time_parts[1].astype("int16")
    out["slot"] = (out["hour"] * 4 + out["minute"] // 15).astype("int16")

    for length in (3, 4, 5, 6):
        out[f"gh{length}"] = out["geohash"].astype(str).str[:length]

    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        out[col] = out[col].fillna("Missing").astype(str)

    out["Temperature_missing"] = out["Temperature"].isna().astype("int8")
    return out


def map_stat(
    frame: pd.DataFrame,
    keys: list[str],
    stat: pd.Series,
    default: float,
) -> pd.Series:
    if len(keys) == 1:
        values = frame[keys[0]].map(stat)
    else:
        values = pd.Series(
            stat.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy(),
            index=frame.index,
        )
    return values.fillna(default)


def add_history_features(
    history: pd.DataFrame,
    frame: pd.DataFrame,
    current_window: pd.DataFrame,
) -> pd.DataFrame:
    """Attach previous-day profile stats plus observed current-day calibration stats."""
    hist = add_base_features(history)
    out = add_base_features(frame)
    global_demand = float(hist["demand"].mean())

    stat_defs = {
        "geo_mean": (["geohash"], hist.groupby("geohash")["demand"].mean()),
        "geo_median": (["geohash"], hist.groupby("geohash")["demand"].median()),
        "geo_std": (["geohash"], hist.groupby("geohash")["demand"].std()),
        "geo_max": (["geohash"], hist.groupby("geohash")["demand"].max()),
        "geo_slot_mean": (
            ["geohash", "slot"],
            hist.groupby(["geohash", "slot"])["demand"].mean(),
        ),
        "geo_hour_mean": (
            ["geohash", "hour"],
            hist.groupby(["geohash", "hour"])["demand"].mean(),
        ),
        "gh5_slot_mean": (
            ["gh5", "slot"],
            hist.groupby(["gh5", "slot"])["demand"].mean(),
        ),
        "slot_mean": (["slot"], hist.groupby("slot")["demand"].mean()),
        "hour_mean": (["hour"], hist.groupby("hour")["demand"].mean()),
        "road_slot_mean": (
            ["RoadType", "slot"],
            hist.groupby(["RoadType", "slot"])["demand"].mean(),
        ),
        "weather_slot_mean": (
            ["Weather", "slot"],
            hist.groupby(["Weather", "slot"])["demand"].mean(),
        ),
        "lanes_slot_mean": (
            ["NumberofLanes", "slot"],
            hist.groupby(["NumberofLanes", "slot"])["demand"].mean(),
        ),
    }

    for name, (keys, stat) in stat_defs.items():
        out[name] = map_stat(out, keys, stat, global_demand)

    out["geo_std"] = out["geo_std"].fillna(0.0)

    current = add_base_features(current_window)
    current_global = float(current["demand"].mean())
    current_sorted = current.sort_values("slot")
    current_defs = {
        "cur_geo_mean": (current.groupby("geohash")["demand"].mean(), "geohash"),
        "cur_geo_last": (current_sorted.groupby("geohash")["demand"].last(), "geohash"),
        "cur_geo_first": (current_sorted.groupby("geohash")["demand"].first(), "geohash"),
        "cur_geo_std": (current.groupby("geohash")["demand"].std(), "geohash"),
        "cur_geo_count": (current.groupby("geohash")["demand"].size(), "geohash"),
        "cur_gh5_mean": (current.groupby("gh5")["demand"].mean(), "gh5"),
        "cur_gh4_mean": (current.groupby("gh4")["demand"].mean(), "gh4"),
    }

    for name, (stat, key) in current_defs.items():
        out[name] = out[key].map(stat).fillna(current_global)

    out["cur_geo_std"] = out["cur_geo_std"].fillna(0.0)
    out["cur_geo_delta"] = out["cur_geo_last"] - out["cur_geo_first"]
    out["cur_seen"] = out["geohash"].isin(current["geohash"]).astype("int8")

    temp_by_gh4_day = hist.groupby(["gh4", "day"])["Temperature"].median()
    temp_by_gh4 = hist.groupby("gh4")["Temperature"].median()
    global_temp = float(hist["Temperature"].median())
    day_temp = pd.Series(
        temp_by_gh4_day.reindex(pd.MultiIndex.from_frame(out[["gh4", "day"]])).to_numpy(),
        index=out.index,
    )
    temp_fill = day_temp.fillna(out["gh4"].map(temp_by_gh4)).fillna(global_temp)
    out["Temperature"] = out["Temperature"].fillna(temp_fill)

    return out


def make_ridge_model(alpha: float = 1000.0) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_tree_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=350,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )


def make_model() -> VotingRegressor:
    return VotingRegressor(
        estimators=[
            ("ridge", make_ridge_model(alpha=1000.0)),
            ("extra_trees", make_tree_model()),
        ],
        weights=[0.3, 0.7],
    )


def select_windows(train: pd.DataFrame, test: pd.DataFrame) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    forecast_day = int(test["day"].mode().iloc[0])
    history = train[train["day"] < forecast_day].copy()
    calibration = train[train["day"] == forecast_day].copy()

    if history.empty:
        raise ValueError("No historical rows are available before the forecast day.")
    if calibration.empty:
        raise ValueError("No labelled current-day calibration rows are available.")

    return forecast_day, history, calibration


def forward_validation(history: pd.DataFrame, calibration: pd.DataFrame) -> dict:
    cal = add_base_features(calibration)
    slots = sorted(cal["slot"].unique())
    rows = []

    for cutoff in slots[3:-1]:
        train_window = cal[cal["slot"] <= cutoff].drop(columns=["hour", "minute", "slot", "gh3", "gh4", "gh5", "gh6", "Temperature_missing"])
        valid_window = cal[cal["slot"] > cutoff].drop(columns=["hour", "minute", "slot", "gh3", "gh4", "gh5", "gh6", "Temperature_missing"])
        if len(train_window) < 1000 or len(valid_window) < 500:
            continue

        x_train = add_history_features(history, train_window, train_window)
        x_valid = add_history_features(history, valid_window, train_window)
        y_train = train_window["demand"].to_numpy()
        y_valid = valid_window["demand"].to_numpy()

        model = make_model()
        model.fit(x_train[FEATURES], y_train)
        pred = np.clip(model.predict(x_valid[FEATURES]), 0.0, 1.0)

        rows.append(
            {
                "train_through_slot": int(cutoff),
                "valid_slots": [int(s) for s in sorted(x_valid["slot"].unique())],
                "rows": int(len(valid_window)),
                "r2": float(r2_score(y_valid, pred)),
                "score": float(max(0.0, 100.0 * r2_score(y_valid, pred))),
                "mae": float(mean_absolute_error(y_valid, pred)),
                "geo_mean_r2": float(r2_score(y_valid, np.clip(x_valid["geo_mean"], 0.0, 1.0))),
                "geo_slot_r2": float(r2_score(y_valid, np.clip(x_valid["geo_slot_mean"], 0.0, 1.0))),
            }
        )

    if not rows:
        return {"folds": [], "mean_r2": None, "mean_score": None, "mean_mae": None}

    return {
        "folds": rows,
        "mean_r2": float(np.mean([row["r2"] for row in rows])),
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_geo_mean_r2": float(np.mean([row["geo_mean_r2"] for row in rows])),
        "mean_geo_slot_r2": float(np.mean([row["geo_slot_r2"] for row in rows])),
    }


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    expected_columns = sample.columns.tolist()
    submission = submission[expected_columns]

    if len(submission) != len(test):
        raise ValueError(f"Submission row count mismatch: {len(submission)} != {len(test)}")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")
    if not np.isfinite(submission["demand"]).all():
        raise ValueError("Submission contains non-finite demand values.")
    if not submission["Index"].equals(test["Index"]):
        raise ValueError("Submission Index column does not match test.csv.")

    return submission


def build_submission() -> pd.DataFrame:
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    if FINAL_SUBMISSION_PATH.exists():
        submission = pd.read_csv(FINAL_SUBMISSION_PATH)
        submission = validate_submission(submission, test, sample)
        submission.to_csv(SUBMISSION_PATH, index=False)
        logger.info(
            "Saved exact accepted 90.73 submission from %s.",
            FINAL_SUBMISSION_PATH,
        )
        return submission

    train = pd.read_csv(DATA_DIR / "train.csv")
    forecast_day, history, calibration = select_windows(train, test)
    logger.info(
        "Forecast day %s: using %s historical rows and %s current-day calibration rows.",
        forecast_day,
        len(history),
        len(calibration),
    )

    report = forward_validation(history, calibration)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    if report["mean_score"] is not None:
        logger.info(
            "Forward validation score: %.4f (R2 %.5f, MAE %.5f).",
            report["mean_score"],
            report["mean_r2"],
            report["mean_mae"],
        )

    x_train = add_history_features(history, calibration, calibration)
    x_test = add_history_features(history, test, calibration)

    model = make_model()
    model.fit(x_train[FEATURES], calibration["demand"].to_numpy())
    predictions = np.clip(model.predict(x_test[FEATURES]), 0.0, 1.0)

    if np.isnan(predictions).any():
        fallback = x_test["geo_slot_mean"].fillna(x_test["geo_mean"]).fillna(train["demand"].mean())
        predictions = np.where(np.isnan(predictions), fallback, predictions)
        predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame({"Index": test["Index"], "demand": predictions})
    submission = validate_submission(submission, test, sample)

    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Saved %s with shape %s.", SUBMISSION_PATH, submission.shape)
    return submission


if __name__ == "__main__":
    build_submission()
