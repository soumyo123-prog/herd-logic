"""
Ticker universes for the Technical agent.

DEFAULT_UNIVERSE is a hand-picked mix of 15 NSE large-caps spanning the
major sectors — enough variety to expose rotation signals without
slowing dev loops to a crawl. Swap-in point for a live NSE constituent
fetch later; the agent signature stays the same.
"""

DEFAULT_UNIVERSE: list[str] = [
    # Banking / financials
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "BAJFINANCE",
    # IT
    "INFY", "TCS",
    # Energy
    "RELIANCE",
    # FMCG
    "HINDUNILVR", "ITC",
    # Pharma
    "SUNPHARMA",
    # Auto
    "MARUTI",
    # Metals
    "TATASTEEL",
    # Infra
    "LT", "ADANIPORTS",
]
