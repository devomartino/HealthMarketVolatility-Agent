from dotenv import load_dotenv
load_dotenv()

from src.pipeline.database import get_connection, initialize_database, build_joined_view

con = get_connection()
initialize_database(con)
build_joined_view(con)

result = con.execute("""
    SELECT date, ticker, close, return_1d, volatility_20d, 
           cpi_medical, unemployment_rate, vix
    FROM market_health_joined
    WHERE cpi_medical IS NOT NULL
    ORDER BY date DESC
    LIMIT 8
""").df()

print("Joined market + health data:")
print(result.to_string(index=False))