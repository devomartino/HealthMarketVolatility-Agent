from dotenv import load_dotenv
load_dotenv()

from src.pipeline.database import get_connection, initialize_database, build_joined_view
from src.ml.model import train, predict, get_metrics

con = get_connection()
initialize_database(con)
build_joined_view(con)

print("Training anomaly detection model...")
metrics = train(con)

print("\nModel Summary:")
print(f"  Total days analyzed:  {metrics['total_rows']}")
print(f"  Stressed days found:  {metrics['stressed_days']} ({metrics['stressed_pct']}%)")
print(f"  Avg stress score:     {metrics['avg_stress_score']}")
print(f"  Max stress score:     {metrics['max_stress_score']}")
print(f"  Tickers covered:      {metrics['tickers']}")
print(f"\nMost stressed tickers:")
for ticker, count in metrics['top_stressed_tickers'].items():
    print(f"  {ticker}: {count} stressed days")

print("\nLive prediction for UNH:")
row = con.execute("""
    SELECT return_1d, return_20d, volatility_20d, rsi_14,
           price_to_sma20, volume_ratio_20d, vix, treasury_10yr, cpi_medical
    FROM market_health_joined
    WHERE ticker = 'UNH'
    ORDER BY date DESC
    LIMIT 1
""").df().iloc[0].to_dict()

result = predict("UNH", row)
print(f"  Risk level:    {result['risk_level']}")
print(f"  Stress score:  {result['stress_score']}")
print(f"  Is stressed:   {result['is_stressed']}")
print(f"\n  Z-scores (how unusual each signal is):")
for signal, z in result['z_scores'].items():
    bar = "█" * int(abs(z))
    print(f"  {signal:<20} {z:>6.2f}  {bar}")