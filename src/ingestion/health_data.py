import logging
import os

import duckdb
import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

HEALTH_SERIES = {
    "CPIMEDSL":      "cpi_medical",
    "UNRATE":        "unemployment_rate",
    "VIXCLS":        "vix",
    "DGS10":         "treasury_10yr",
    "PCE":           "personal_consumption",
}

def _fetch_series(series_id: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "file_type": "json",
    }
    if FRED_API_KEY:
        params["api_key"] = FRED_API_KEY

    resp = requests.get(FRED_BASE, params=params, timeout=15)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    if not obs:
        return pd.DataFrame()

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df

def fetch_health_indicators(start_date: str = "2020-01-01") -> pd.DataFrame:
    print(f"Fetching {len(HEALTH_SERIES)} FRED series from {start_date}")
    frames = {}
    
    for series_id, col_name in HEALTH_SERIES.items():
        try:
            df = _fetch_series(series_id, start_date)
            if not df.empty:
                frames[col_name] = df.set_index("date")["value"]
                print(f"  ✓ {col_name}: {len(df)} observations")
        except Exception as exc:
            print(f"  ✗ {col_name}: {exc}")

    if not frames:
        raise RuntimeError("No health data retrieved from FRED.")

    wide = pd.DataFrame(frames)
    wide.index.name = "date"
    wide = wide.reset_index()
    wide = wide.sort_values("date")
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.set_index("date").resample("D").ffill().reset_index()
    wide["date"] = wide["date"].dt.date
    return wide

def ingest_health_data(con, start_date: str = "2020-01-01") -> int:
    df = fetch_health_indicators(start_date)

    # Build column definitions dynamically from whatever columns came back
    cols = ", ".join(f"{c} DOUBLE" for c in df.columns if c != "date")
    
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS health_indicators (
            date DATE PRIMARY KEY,
            {cols}
        )
    """)

    con.execute("DELETE FROM health_indicators")
    con.register("health_df", df)
    con.execute("INSERT INTO health_indicators SELECT * FROM health_df")

    count = con.execute("SELECT COUNT(*) FROM health_indicators").fetchone()[0]
    print(f"health_indicators table: {count} rows")

    return count