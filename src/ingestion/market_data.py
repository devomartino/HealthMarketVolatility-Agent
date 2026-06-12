import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

## Dynamic Tickers: 

def get_tickers(con) -> list[str]:
    return [row[0] for row in con.execute("SELECT symbol FROM watchlist").fetchall()]

def fetch_ohlcv(con, period_days=730):
    tickers = get_tickers(con)
    end = datetime.today()
    start = end - timedelta(days=period_days)
    print(f"Fetching data for {len(tickers)} tickers from {start.date()} to {end.date()}")
    
    frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            continue
        df = df.reset_index()
        if isinstance(df.columns[0], tuple):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        frames.append(df)
        print(f"  ✓ {ticker}: {len(df)} rows")
    
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    return combined

def add_technical_features(df):
    out = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        
        # Returns
        grp["return_1d"] = grp["close"].pct_change(1)
        grp["return_5d"] = grp["close"].pct_change(5)
        grp["return_20d"] = grp["close"].pct_change(20)
        
        # Volatility
        grp["volatility_20d"] = grp["return_1d"].rolling(20).std()
        grp["volatility_5d"] = grp["return_1d"].rolling(5).std()
        
        # RSI
        delta = grp["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        grp["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Moving averages
        grp["sma_20"] = grp["close"].rolling(20).mean()
        grp["sma_50"] = grp["close"].rolling(50).mean()
        grp["price_to_sma20"] = grp["close"] / grp["sma_20"]
        
        # Volume
        grp["volume_ratio_20d"] = grp["volume"] / grp["volume"].rolling(20).mean()
        
        # ML target — what we're trying to predict
        grp["target_volatility_5d"] = grp["volatility_5d"].shift(-5)
        
        out.append(grp)
    
    return pd.concat(out, ignore_index=True)

def ingest_market_data(con, period_days=730) -> int:
    raw = fetch_ohlcv(con, period_days=period_days)
    featured = add_technical_features(raw)

    # Be explicit about columns to avoid type mismatch
    featured = featured[[
        "date", "ticker", "open", "high", "low", "close", "volume",
        "return_1d", "return_5d", "return_20d",
        "volatility_20d", "volatility_5d", "rsi_14",
        "sma_20", "sma_50", "price_to_sma20",
        "volume_ratio_20d", "target_volatility_5d"
    ]]

    con.execute("DELETE FROM market_data")
    con.register("featured", featured)
    con.execute("INSERT INTO market_data SELECT * FROM featured")

    count = con.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
    print(f"market_data table: {count} rows")
    return count