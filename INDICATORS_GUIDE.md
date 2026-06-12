# HealthMarket Intelligence Agent — Indicators Guide

## How to Read the Stress Score

The stress score measures how unusual a stock's current behavior is
compared to its own history. Think of it like a weather alert system.

| Stress Score | Risk Level | What It Means |
|---|---|---|
| 0.0 – 1.5 | 🟢 NORMAL | Everything looks typical. No action needed. |
| 1.5 – 2.5 | 🟡 ELEVATED | Some signals are unusual. Worth watching. |
| 2.5+ | 🔴 HIGH | Multiple signals spiking. Potential stress event. |

---

## How to Read Z-Scores

A z-score tells you how far a signal is from its normal value.
Positive = higher than usual. Negative = lower than usual.

| Z-Score | What It Means |
|---|---|
| 0 | Perfectly normal |
| 1 or -1 | Slightly unusual |
| 2 or -2 | Very unusual (only happens ~5% of the time) |
| 3 or -3 | Extreme — red flag |

---

## What Each Signal Means

### Market Signals

**return_1d** — How much the stock moved today (%)
- Positive = stock went up
- Negative = stock went down
- High z-score = unusually big move in either direction

**return_20d** — How much the stock has moved over the past month
- Negative z-score = stock has been falling for a while
- Positive z-score = stock has been on a strong run

**volatility_20d** — How wildly the stock has been swinging day to day
- High z-score = much more chaotic than usual
- Low z-score = unusually calm

**rsi_14** — Relative Strength Index (0 to 100 scale)
- Below 30 = oversold (stock may have dropped too far, too fast)
- Above 70 = overbought (stock may have risen too far, too fast)
- Negative z-score = weaker than this stock normally is

**price_to_sma20** — Where the stock is trading vs its 20-day average
- Above 1.0 = trading above its recent average (momentum)
- Below 1.0 = trading below its recent average (weakness)
- High negative z-score = stock is well below where it normally trades

**volume_ratio_20d** — How much trading activity vs normal
- 1.0 = normal volume
- 2.0 = twice the normal volume
- High z-score = unusual trading activity (could mean big news)

### Health & Macro Signals

**vix** — The market's "fear gauge" (0 to 100 scale)
- Below 15 = calm markets
- 15–25 = normal uncertainty
- Above 25 = elevated fear
- Above 35 = panic territory
- High z-score = market is much more fearful than usual

**treasury_10yr** — 10-year US Treasury interest rate (%)
- Higher rates = more pressure on healthcare stocks
- High z-score = rates are elevated vs historical norm

**cpi_medical** — Medical care inflation index
- Rising = healthcare costs going up
- High z-score = costs rising faster than usual

---

## Example: How to Read a Full Result

Ticker: HUM
Risk Level:    🔴 HIGH
Stress Score:  2.84
Z-Scores:
return_1d            -2.1  ██   ← dropped hard today
return_20d           -1.8  █    ← been falling for weeks
volatility_20d        2.3  ██   ← much more chaotic than normal
rsi_14               -2.5  ██   ← deeply oversold
price_to_sma20       -1.9  █    ← well below its average
volume_ratio_20d      3.1  ███  ← 3x normal trading volume
vix                   2.2  ██   ← whole market is fearful
treasury_10yr         0.8      ← rates slightly elevated, not alarming
cpi_medical           0.1      ← medical costs normal
