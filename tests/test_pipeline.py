import numpy as np
import pandas as pd
import pytest
import duckdb

from src.ingestion.market_data import add_technical_features
from src.ml.features import SIGNAL_COLS


@pytest.fixture
def in_memory_con():
    return duckdb.connect(":memory:")


def make_fake_ohlcv(ticker="UNH", n=100):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 500 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "date": dates.date,
        "ticker": ticker,
        "open": close - np.abs(np.random.randn(n)),
        "high": close + np.abs(np.random.randn(n)),
        "low": close - np.abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


class TestTechnicalFeatures:
    def test_returns_computed(self):
        df = make_fake_ohlcv(n=60)
        result = add_technical_features(df)
        assert "return_1d" in result.columns
        assert "volatility_20d" in result.columns

    def test_rsi_bounds(self):
        df = make_fake_ohlcv(n=60)
        result = add_technical_features(df)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_target_column_present(self):
        df = make_fake_ohlcv(n=60)
        result = add_technical_features(df)
        assert "target_volatility_5d" in result.columns

    def test_multiple_tickers(self):
        df = pd.concat([make_fake_ohlcv("UNH", 60), make_fake_ohlcv("PFE", 60)])
        result = add_technical_features(df)
        assert set(result["ticker"].unique()) == {"UNH", "PFE"}


class TestDatabase:
    def test_connection(self, in_memory_con):
        result = in_memory_con.execute("SELECT 42 AS answer").fetchone()
        assert result[0] == 42

    def test_table_creation(self, in_memory_con):
        in_memory_con.execute("""
            CREATE TABLE market_data (
                date DATE, ticker VARCHAR, close DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        tables = [t[0] for t in in_memory_con.execute("SHOW TABLES").fetchall()]
        assert "market_data" in tables

    def test_upsert_deduplication(self, in_memory_con):
        in_memory_con.execute("""
            CREATE TABLE market_data (
                date DATE, ticker VARCHAR, close DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)
        in_memory_con.execute("INSERT INTO market_data VALUES ('2024-01-01', 'UNH', 500.0)")
        in_memory_con.execute("INSERT OR REPLACE INTO market_data VALUES ('2024-01-01', 'UNH', 505.0)")
        count = in_memory_con.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        assert count == 1


class TestSignals:
    def test_signal_cols_defined(self):
        assert len(SIGNAL_COLS) > 5
        assert "volatility_20d" in SIGNAL_COLS
        assert "vix" in SIGNAL_COLS
        assert "rsi_14" in SIGNAL_COLS