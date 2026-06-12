import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Signals we monitor for anomalies
SIGNAL_COLS = [
    "return_1d",
    "return_20d",
    "volatility_20d",
    "rsi_14",
    "price_to_sma20",
    "volume_ratio_20d",
    "vix",
    "treasury_10yr",
    "cpi_medical",
]

def build_feature_matrix(con):
    df = con.execute(f"""
        SELECT date, ticker, {', '.join(SIGNAL_COLS)}
        FROM market_health_joined
        WHERE return_1d IS NOT NULL
        ORDER BY date
    """).df()

    # Forward fill health indicators
    health_cols = ["cpi_medical", "treasury_10yr", "vix"]
    df[health_cols] = df[health_cols].ffill()
    df = df.dropna(subset=SIGNAL_COLS)

    print(f"Feature matrix: {len(df)} rows x {len(SIGNAL_COLS)} signals")
    return df


def compute_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal, compute how many standard deviations
    today's value is from the ticker's historical mean.
    A z-score > 2 means unusually high. > 3 means extreme.
    """
    out = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.copy()
        for col in SIGNAL_COLS:
            mean = grp[col].mean()
            std = grp[col].std()
            grp[f"z_{col}"] = (grp[col] - mean) / std

        # Overall stress score = average of absolute z-scores
        z_cols = [f"z_{c}" for c in SIGNAL_COLS]
        grp["stress_score"] = grp[z_cols].abs().mean(axis=1)

        # Flag as stressed if stress score > 1.5
        grp["is_stressed"] = (grp["stress_score"] > 1.5).astype(int)
        out.append(grp)

    return pd.concat(out, ignore_index=True)