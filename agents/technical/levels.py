"""
Deterministic entry / stop-loss / target computation.

The LLM emits qualitative setup proposals (direction, confidence, key
signals). This module turns each proposal into concrete numeric levels
using the ticker's last close and its daily ATR. Because the math is
deterministic, two back-to-back runs produce byte-identical setups —
which makes the system auditable in a way pure-LLM output never is.

The four constants below are the *only* tuning knobs. Changing them
rebuilds every setup with new risk parameters; no other code touches
numeric levels.
"""

from __future__ import annotations

from agents.technical.schemas import LLMSetupProposal, SwingSetup

# ── Tunable constants ────────────────────────────────────────────────

# Stop-loss distance, expressed as a multiple of the daily ATR.
ATR_MULTIPLIER_STOP = 1.5

# Target risk-reward ratios.
RR_TARGET_1 = 2.0
RR_TARGET_2 = 3.0

# Entry zone — ±ENTRY_ZONE_PCT of the last close.
ENTRY_ZONE_PCT = 0.005  # ±0.5%


def compute_levels(proposal: LLMSetupProposal, snapshot: dict) -> SwingSetup:
    """Turn an LLM proposal into a fully-specified SwingSetup."""
    last = float(snapshot["last_close"])
    atr = float(snapshot["daily"]["atr14"])

    entry_low = round(last * (1 - ENTRY_ZONE_PCT), 2)
    entry_high = round(last * (1 + ENTRY_ZONE_PCT), 2)

    if proposal.direction == "long":
        stop_loss = round(last - ATR_MULTIPLIER_STOP * atr, 2)
        risk = last - stop_loss
        target_1 = round(last + RR_TARGET_1 * risk, 2)
        target_2 = round(last + RR_TARGET_2 * risk, 2)
    else:
        stop_loss = round(last + ATR_MULTIPLIER_STOP * atr, 2)
        risk = stop_loss - last
        target_1 = round(last - RR_TARGET_1 * risk, 2)
        target_2 = round(last - RR_TARGET_2 * risk, 2)

    return SwingSetup(
        ticker=proposal.ticker,
        direction=proposal.direction,
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        risk_reward_ratio=RR_TARGET_1,
        confidence=proposal.confidence,
        holding_period_days=proposal.holding_period_days,
        key_signals=proposal.key_signals,
    )
