import logging
import os

import anthropic

from src.pipeline.database import get_connection, build_joined_view
from src.ml.model import predict, get_metrics
from src.rag.retriever import retrieve, format_context

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are HealthMarket Intelligence Agent — an AI analyst 
specializing in the intersection of public health indicators and healthcare 
financial markets.

You have access to real data through three tools:
1. query_database — run SQL against live market and health data
2. retrieve_context — search the knowledge base for relevant summaries
3. get_stress_prediction — run the anomaly detection model for a ticker

Always cite your data sources. Be precise with numbers. If you don't have 
enough data to answer confidently, say so clearly. Format numbers cleanly 
with % signs and $ where appropriate.
"""

TOOLS = [
    {
        "name": "query_database",
        "description": (
            "Execute a SQL query against the DuckDB database containing "
            "healthcare stock data and health indicators. Use for specific "
            "data lookups, aggregations, and time-series queries. "
            "Available tables: market_data, health_indicators. "
            "Available view: market_health_joined."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid DuckDB SQL query. Always use LIMIT to avoid large result sets.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "retrieve_context",
        "description": (
            "Semantic search over the knowledge base of market and health "
            "summaries. Use this to get recent trends, narratives, or "
            "context about specific tickers or health indicators."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_stress_prediction",
        "description": (
            "Run the anomaly detection model to get a stress score and "
            "risk level for a specific ticker. Returns z-scores for each "
            "signal and overall risk level: NORMAL, ELEVATED, or HIGH."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol e.g. UNH, PFE, JNJ",
                }
            },
            "required": ["ticker"],
        },
    },
]

def _execute_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "query_database":
            con = get_connection()
            build_joined_view(con)
            df = con.execute(tool_input["sql"]).df()
            if df.empty:
                return "Query returned no results."
            return df.to_string(index=False, max_rows=20)

        elif tool_name == "retrieve_context":
            docs = retrieve(tool_input["query"], k=4)
            return format_context(docs)

        elif tool_name == "get_stress_prediction":
            ticker = tool_input["ticker"].upper()
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
                return f"No data found for ticker {ticker}."

            features = row.iloc[0].to_dict()
            result = predict(ticker, features)
            return (
                f"Stress prediction for {ticker}:\n"
                f"  Risk level:   {result['risk_level']}\n"
                f"  Stress score: {result['stress_score']}\n"
                f"  Is stressed:  {result['is_stressed']}\n"
                f"  Z-scores:\n" +
                "\n".join(
                    f"    {k}: {v}" for k, v in result['z_scores'].items()
                )
            )

        return f"Unknown tool: {tool_name}"

    except Exception as exc:
        logger.error("Tool '%s' error: %s", tool_name, exc)
        return f"Error executing {tool_name}: {exc}"
    
def chat(user_message: str, api_key: str, history: list[dict] = None) -> tuple[str, list[dict]]:
    """
    Multi-turn chat with the agent.

    Args:
        user_message: The user's question
        api_key: User's Anthropic API key
        history: Prior conversation messages

    Returns:
        (assistant_response_text, updated_history)
    """
    client = anthropic.Anthropic(api_key=api_key)
    history = history or []

    messages = history + [{"role": "user", "content": user_message}]

    # Agentic loop — keeps running until Claude stops calling tools
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content 
                 if hasattr(block, "text")),
                ""
            )
            return final_text, messages

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.info("Tool call: %s(%s)", block.name, block.input)
                    result = _execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return "Agent encountered an unexpected state.", messages 