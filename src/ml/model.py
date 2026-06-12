import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.features import SIGNAL_COLS, build_feature_matrix, compute_anomaly_scores

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/models"))
METRICS_PATH = MODEL_DIR / "metrics.json"
SIGNAL_STATS_PATH = MODEL_DIR / "signal_stats.pkl"


def train(con) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_matrix(con)
    df = compute_anomaly_scores(df)

    # Compute and save per-ticker signal statistics
    stats = {}
    for ticker, grp in df.groupby("ticker"):
        stats[ticker] = {
            col: {"mean": grp[col].mean(), "std": grp[col].std()}
            for col in SIGNAL_COLS
        }

    with open(SIGNAL_STATS_PATH, "wb") as f:
        pickle.dump(stats, f)

    # Summary metrics
    stressed = df[df["is_stressed"] == 1]
    metrics = {
        "total_rows": len(df),
        "stressed_days": int(stressed["is_stressed"].sum()),
        "stressed_pct": round(stressed["is_stressed"].sum() / len(df) * 100, 2),
        "avg_stress_score": round(df["stress_score"].mean(), 4),
        "max_stress_score": round(df["stress_score"].max(), 4),
        "tickers": df["ticker"].nunique(),
        "signals": SIGNAL_COLS,
        "top_stressed_tickers": (
            stressed.groupby("ticker").size()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        ),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model trained → {metrics['stressed_days']} stressed days detected "
          f"({metrics['stressed_pct']}% of all days)")
    return metrics


def predict(ticker: str, features: dict) -> dict:
    """
    Score a single observation against historical baseline.
    Returns z-scores per signal and overall stress score.
    """
    if not SIGNAL_STATS_PATH.exists():
        raise FileNotFoundError("Signal stats not found. Run train() first.")

    with open(SIGNAL_STATS_PATH, "rb") as f:
        stats = pickle.load(f)

    if ticker not in stats:
        raise ValueError(f"No baseline stats for ticker {ticker}.")

    ticker_stats = stats[ticker]
    z_scores = {}
    for col in SIGNAL_COLS:
        val = features.get(col)
        if val is None or np.isnan(val):
            z_scores[col] = 0.0
            continue
        mean = ticker_stats[col]["mean"]
        std = ticker_stats[col]["std"]
        z_scores[col] = round((val - mean) / std if std > 0 else 0.0, 4)

    stress_score = round(np.mean(np.abs(list(z_scores.values()))), 4)
    is_stressed = stress_score > 1.5

    return {
        "ticker": ticker,
        "stress_score": stress_score,
        "is_stressed": is_stressed,
        "risk_level": "HIGH" if stress_score > 2.5 else "ELEVATED" if stress_score > 1.5 else "NORMAL",
        "z_scores": z_scores,
    }


def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)