import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION_NAME = "healthmarket_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

def get_collection() -> chromadb.Collection:
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

def upsert_documents(documents: list[dict]) -> int:
    if not documents:
        logger.warning("No documents to upsert.")
        return 0

    collection = get_collection()
    ids = [d["id"] for d in documents]
    texts = [d["text"] for d in documents]
    metadatas = [{"source": d.get("source", ""), "date": d.get("date", "")} for d in documents]

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    count = collection.count()
    print(f"✓ ChromaDB collection '{COLLECTION_NAME}': {count} documents")
    return count

def build_market_narratives(con) -> list[dict]:
    df = con.execute("""
        SELECT ticker, date, close, return_1d, return_5d, 
               volatility_20d, rsi_14, vix
        FROM market_health_joined
        WHERE date >= CURRENT_DATE - INTERVAL 60 DAY
        AND return_1d IS NOT NULL
        ORDER BY ticker, date
    """).df()

    docs = []
    for ticker, grp in df.groupby("ticker"):
        latest = grp.iloc[-1]
        avg_vol = grp["volatility_20d"].mean()
        best_day = grp.loc[grp["return_1d"].idxmax()]
        worst_day = grp.loc[grp["return_1d"].idxmin()]

        text = (
            f"{ticker} as of {latest['date']}: "
            f"closing at ${latest['close']:.2f}, "
            f"1-day return {latest['return_1d']*100:.2f}%, "
            f"5-day return {latest['return_5d']*100:.2f}%, "
            f"20-day volatility {latest['volatility_20d']*100:.2f}%, "
            f"RSI-14 at {latest['rsi_14']:.1f}, "
            f"VIX at {latest['vix']:.1f}. "
            f"Over the past 60 days: average volatility {avg_vol*100:.2f}%, "
            f"best day was {best_day['date']} (+{best_day['return_1d']*100:.2f}%), "
            f"worst day was {worst_day['date']} ({worst_day['return_1d']*100:.2f}%)."
        )
        docs.append({
            "id": f"market_{ticker}_{latest['date']}",
            "text": text,
            "source": "Yahoo Finance",
            "date": str(latest["date"]),
        })
    return docs


def build_health_narratives(con) -> list[dict]:
    df = con.execute("""
        SELECT date, cpi_medical, unemployment_rate, 
               vix, treasury_10yr
        FROM health_indicators
        WHERE cpi_medical IS NOT NULL
        ORDER BY date DESC
        LIMIT 6
    """).df()

    docs = []
    for _, row in df.iterrows():
        text = (
            f"Health indicators as of {row['date']}: "
            f"medical CPI at {row['cpi_medical']:.1f}, "
            f"unemployment rate {row['unemployment_rate']:.1f}%, "
            f"VIX at {row['vix']:.1f}, "
            f"10-year Treasury yield {row['treasury_10yr']:.2f}%."
        )
        docs.append({
            "id": f"health_{row['date']}",
            "text": text,
            "source": "FRED",
            "date": str(row["date"]),
        })
    return docs


def index_all_documents(con) -> int:
    market_docs = build_market_narratives(con)
    health_docs = build_health_narratives(con)
    all_docs = market_docs + health_docs
    print(f"Indexing {len(all_docs)} documents "
          f"({len(market_docs)} market, {len(health_docs)} health)")
    return upsert_documents(all_docs)