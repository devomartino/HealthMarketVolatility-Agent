from dotenv import load_dotenv
import os
load_dotenv()
print("FRED key:", os.getenv("FRED_API_KEY"))
print("Anthropic key:", os.getenv("ANTHROPIC_API_KEY"))
