import logging
import os
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/healthmarket.duckdb")

DEFAULT_TICKERS = [
    ("UNH", "UnitedHealth Group"),
    ("JNJ", "Johnson & Johnson"),
    ("PFE", "Pfizer"),
    ("ABBV", "AbbVie"),
    ("MRK", "Merck"),
    ("TMO", "Thermo Fisher"),
    ("ABT", "Abbott Labs"),
    ("CVS", "CVS Health"),
    ("ELV", "Elevance Health"),
    ("HUM", "Humana"),
]

def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=4")
    return con


def initialize_database(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            date DATE,
            ticker VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            return_1d DOUBLE,
            return_5d DOUBLE,
            return_20d DOUBLE,
            volatility_20d DOUBLE,
            volatility_5d DOUBLE,
            rsi_14 DOUBLE,
            sma_20 DOUBLE,
            sma_50 DOUBLE,
            price_to_sma20 DOUBLE,
            volume_ratio_20d DOUBLE,
            target_volatility_5d DOUBLE,
            PRIMARY KEY (date, ticker)
        )
    """)

    # Seed watchlist with defaults if empty
    count = con.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
    if count == 0:
        con.executemany(
            "INSERT INTO watchlist VALUES (?, ?)", 
            DEFAULT_TICKERS
        )
        logger.info("Watchlist seeded with %d default tickers", len(DEFAULT_TICKERS))

        
def validate_and_add_ticker(symbol: str, con) -> bool:
    import yfinance as yf
    
    symbol = symbol.upper().strip()
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    if "shortName" not in info:
        print(f"'{symbol}' not found on Yahoo Finance. Check the symbol and try again.")
        return False
    
    name = info["shortName"]
    con.execute(
        "INSERT OR IGNORE INTO watchlist VALUES (?, ?)",
        [symbol, name]
    )
    print(f"✓ Added {symbol} - {name} to watchlist")
    return True


def remove_ticker(symbol: str, con) -> None:
    symbol = symbol.upper().strip()
    con.execute("DELETE FROM watchlist WHERE symbol = ?", [symbol])
    print(f"✓ Removed {symbol} from watchlist")


def get_tickers(con) -> list[str]:
    rows = con.execute("SELECT symbol FROM watchlist ORDER BY symbol").fetchall()
    return [row[0] for row in rows]

def build_joined_view(con) -> None:
    con.execute("""
        CREATE OR REPLACE VIEW market_health_joined AS
        SELECT
            m.date,
            m.ticker,
            m.open,
            m.high,
            m.low,
            m.close,
            m.volume,
            m.return_1d,
            m.return_5d,
            m.return_20d,
            m.volatility_20d,
            m.volatility_5d,
            m.rsi_14,
            m.sma_20,
            m.sma_50,
            m.price_to_sma20,
            m.volume_ratio_20d,
            m.target_volatility_5d,
            h.cpi_medical,
            h.unemployment_rate,
            h.vix,
            h.treasury_10yr,
            h.personal_consumption
        FROM market_data m
        LEFT JOIN health_indicators h ON m.date = h.date
    """)
    print("✓ market_health_joined view created")