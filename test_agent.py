from dotenv import load_dotenv
load_dotenv()

import os
from src.agent.chat import chat

api_key = os.getenv("ANTHROPIC_API_KEY")

print("Testing HealthMarket Intelligence Agent")
print("=" * 50)

# Test 1: Simple market question
print("\nQuestion 1: What is UNH's current stress level?")
response, history = chat(
    "What is UNH's current stress level and what are the key signals driving it?",
    api_key=api_key,
)
print(response)

# Test 2: Health indicator question
print("\n" + "=" * 50)
print("\nQuestion 2: Medical CPI trend")
response, history = chat(
    "How has medical CPI trended over the past few months?",
    api_key=api_key,
    history=[],
)
print(response)

# Test 3: Comparison question
print("\n" + "=" * 50)
print("\nQuestion 3: Compare two stocks")
response, history = chat(
    "Compare the volatility of JNJ and HUM. Which is riskier right now?",
    api_key=api_key,
    history=[],
)
print(response)