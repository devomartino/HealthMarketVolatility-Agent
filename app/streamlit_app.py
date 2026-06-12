import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="HealthMarket Intelligence Agent",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 HealthMarket Intelligence Agent")
st.caption("AI-powered analysis at the intersection of public health and financial markets")

# Sidebar — API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your Anthropic API key. Get one at console.anthropic.com"
    )
    if api_key:
        st.success("API key set ✓")
    else:
        st.warning("Enter your API key to use the chat agent")

    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- Yahoo Finance (stock data)")
    st.markdown("- FRED API (health indicators)")
    st.markdown("- Anthropic Claude (AI agent)")


# Helper functions
def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# Tabs
tab_chat, tab_market, tab_ml, tab_health = st.tabs([
    "💬 AI Agent",
    "📈 Market Dashboard", 
    "🤖 ML Predictions",
    "🏥 Health Indicators"
])

with tab_chat:
    st.subheader("Ask the AI Agent")
    st.markdown(
        "Try asking:\n"
        "- *What is UNH's current stress level?*\n"
        "- *How has medical CPI trended over the past 6 months?*\n"
        "- *Compare the risk of JNJ and HUM right now.*"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_history" not in st.session_state:
        st.session_state.api_history = []

    # Render existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about healthcare stocks or health indicators..."):
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent thinking..."):
                    result = api_post("/chat", {
                        "message": prompt,
                        "api_key": api_key,
                        "history": st.session_state.api_history,
                    })
                if result:
                    response_text = result["response"]
                    st.session_state.api_history = result["history"]
                    st.markdown(response_text)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text
                    })

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.api_history = []
        st.rerun()

with tab_market:
    st.subheader("Healthcare Stock Overview")
    tickers_data = api_get("/tickers")

    if tickers_data:
        df_tickers = pd.DataFrame(tickers_data)
        df_tickers["return_1d_pct"] = (df_tickers["return_1d"] * 100).round(2)
        df_tickers["return_5d_pct"] = (df_tickers["return_5d"] * 100).round(2)
        df_tickers["volatility_pct"] = (df_tickers["volatility_20d"] * 100).round(3)

        # KPI cards
        cols = st.columns(5)
        for i, (_, row) in enumerate(df_tickers.iterrows()):
            col = cols[i % 5]
            col.metric(
                label=row["ticker"],
                value=f"${row['close']:.2f}",
                delta=f"{row['return_1d_pct']:+.2f}% today",
            )

        st.divider()

        # Time series chart
        st.subheader("Price History")
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.selectbox("Ticker", df_tickers["ticker"].tolist())
        with col2:
            days = st.slider("Days", 30, 365, 90)

        ts_data = api_get(f"/timeseries/{selected}?days={days}")
        if ts_data:
            df_ts = pd.DataFrame(ts_data)
            df_ts["date"] = pd.to_datetime(df_ts["date"])

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_ts["date"],
                open=df_ts["open"],
                high=df_ts["high"],
                low=df_ts["low"],
                close=df_ts["close"],
                name=selected,
            ))
            fig.update_layout(
                title=f"{selected} — {days}-day Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Volatility comparison bar chart
        st.subheader("Volatility Comparison")
        fig2 = px.bar(
            df_tickers.sort_values("volatility_pct"),
            x="ticker",
            y="volatility_pct",
            color="volatility_pct",
            color_continuous_scale="RdYlGn_r",
            title="20-Day Volatility (%) by Ticker",
            labels={"volatility_pct": "Volatility (%)", "ticker": "Ticker"},
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab_ml:
    st.subheader("Stress Regime Detector")
    st.markdown(
        "The anomaly detection model scores each stock against its own "
        "historical baseline. A stress score above 1.5 indicates elevated risk."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        tickers_data = api_get("/tickers")
        ticker_list = [t["ticker"] for t in tickers_data] if tickers_data else []
        pred_ticker = st.selectbox("Select ticker", ticker_list, key="pred_ticker")

        if st.button("Run Stress Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                pred = api_get(f"/predict/{pred_ticker}")
            if pred:
                risk = pred["risk_level"]
                color = "🔴" if risk == "HIGH" else "🟡" if risk == "ELEVATED" else "🟢"
                st.markdown(f"### {color} {risk}")
                st.metric("Stress Score", pred["stress_score"],
                          help="Above 1.5 = elevated, above 2.5 = high risk")
                st.metric("Is Stressed", "Yes" if pred["is_stressed"] else "No")

                st.subheader("Z-Scores")
                z_df = pd.DataFrame(
                    pred["z_scores"].items(),
                    columns=["Signal", "Z-Score"]
                ).sort_values("Z-Score", key=abs, ascending=False)

                fig = px.bar(
                    z_df,
                    x="Z-Score",
                    y="Signal",
                    orientation="h",
                    color="Z-Score",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    title=f"{pred_ticker} Signal Z-Scores",
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Info")
        metrics = api_get("/metrics")
        if metrics:
            st.metric("Total Days Analyzed", metrics.get("total_rows"))
            st.metric("Stressed Days Detected",
                      f"{metrics.get('stressed_days')} ({metrics.get('stressed_pct')}%)")
            st.metric("Tickers Covered", metrics.get("tickers"))

            st.subheader("Most Stressed Tickers (Historical)")
            top = metrics.get("top_stressed_tickers", {})
            if top:
                top_df = pd.DataFrame(top.items(), columns=["Ticker", "Stressed Days"])
                fig2 = px.bar(top_df, x="Ticker", y="Stressed Days",
                              color="Stressed Days",
                              color_continuous_scale="Reds")
                st.plotly_chart(fig2, use_container_width=True)

with tab_health:
    st.subheader("Public Health & Macro Indicators")
    st.caption("Data sourced from FRED (Federal Reserve Economic Data)")

    st.markdown("""
    | Indicator | Description | Why It Matters |
    |-----------|-------------|----------------|
    | Medical CPI | Healthcare cost inflation index | Rising costs pressure insurer margins |
    | Unemployment Rate | % of labor force unemployed | Affects healthcare coverage demand |
    | VIX | Market fear gauge | High VIX = market stress |
    | Treasury 10yr | 10-year bond yield | Higher rates pressure growth stocks |
    """)

    st.divider()

    if st.button("Generate AI Health Summary"):
        if not api_key:
            st.error("Enter your API key in the sidebar first.")
        else:
            with st.spinner("Agent analyzing indicators..."):
                result = api_post("/chat", {
                    "message": (
                        "Summarize the current state of health macro indicators. "
                        "Cover medical CPI trend, unemployment, VIX level, and "
                        "treasury yields. What does this mean for healthcare stocks?"
                    ),
                    "api_key": api_key,
                    "history": [],
                })
            if result:
                st.info(result["response"])

    st.divider()
    st.subheader("How to Read the Indicators Guide")
    with st.expander("Click to expand full guide"):
        try:
            with open("INDICATORS_GUIDE.md", "r") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.markdown("INDICATORS_GUIDE.md not found in project root.")

