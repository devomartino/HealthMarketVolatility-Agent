import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.database import get_connection, initialize_database, build_joined_view
from src.ml.model import predict, get_metrics, train
from src.rag.embeddings import index_all_documents
from src.agent.chat import chat as agent_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        con = get_connection()
        initialize_database(con)
        build_joined_view(con)
        logger.info("Database ready on startup")
    except Exception as exc:
        logger.warning("DB not ready on startup: %s", exc)
    yield


app = FastAPI(
    title="HealthMarket Intelligence Agent",
    description="AI-powered healthcare stock and public health analytics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    api_key: str
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    response: str
    history: list[dict]

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        response_text, updated_history = agent_chat(
            req.message,
            api_key=req.api_key,
            history=req.history,
        )
        return ChatResponse(response=response_text, history=updated_history)
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/{ticker}")
def predict_stress(ticker: str):
    ticker = ticker.upper()
    try:
        con = get_connection()
        build_joined_view(con)
        row = con.execute(f"""
            SELECT return_1d, return_20d, volatility_20d, rsi_14,
                   price_to_sma20, volume_ratio_20d, vix,
                   treasury_10yr, cpi_medical
            FROM market_health_joined
            WHERE ticker = '{ticker}'
            ORDER BY date DESC
            LIMIT 1
        """).df()
        if row.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        features = row.iloc[0].to_dict()
        return predict(ticker, features)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tickers")
def list_tickers():
    try:
        con = get_connection()
        df = con.execute("""
            SELECT ticker, date, close, return_1d, return_5d, 
                   volatility_20d, rsi_14
            FROM market_data
            WHERE (ticker, date) IN (
                SELECT ticker, MAX(date) FROM market_data GROUP BY ticker
            )
            ORDER BY ticker
        """).df()
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/timeseries/{ticker}")
def get_timeseries(ticker: str, days: int = 90):
    """Return daily OHLCV + indicators for a ticker over the last N days."""
    ticker = ticker.upper()
    try:
        # Try joined view first (includes health indicators)
        try:
            df = db_query(f"""
                SELECT date, open, high, low, close, volume,
                       return_1d, volatility_20d, rsi_14, vix
                FROM market_health_joined
                WHERE ticker = '{ticker}'
                AND date >= CURRENT_DATE - INTERVAL {days} DAY
                ORDER BY date
            """, DB_PATH)
        except Exception:
            # Fall back to market_data only if health_indicators not yet loaded
            df = db_query(f"""
                SELECT date, open, high, low, close, volume,
                       return_1d, volatility_20d, rsi_14,
                       NULL as vix
                FROM market_data
                WHERE ticker = '{ticker}'
                AND date >= CURRENT_DATE - INTERVAL {days} DAY
                ORDER BY date
            """, DB_PATH)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/metrics")
def model_metrics():
    metrics = get_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="Model not trained yet.")
    return metrics


@app.post("/pipeline/run")
def trigger_pipeline(background_tasks: BackgroundTasks):
    def _run():
        try:
            from src.ingestion.market_data import ingest_market_data
            from src.ingestion.health_data import ingest_health_data
            con = get_connection()
            initialize_database(con)
            ingest_market_data(con)
            ingest_health_data(con)
            build_joined_view(con)
            index_all_documents(con)
            train(con)
            logger.info("Pipeline complete")
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)

    background_tasks.add_task(_run)
    return {"status": "pipeline started in background"}