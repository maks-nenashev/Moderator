import json
from pathlib import Path

# =========================
# Load thresholds
# =========================
THRESHOLDS_PATH = Path("artifacts/policy_thresholds.json")

with open(THRESHOLDS_PATH, "r") as f:
    THRESHOLDS = json.load(f)


def apply_policy(scores: dict) -> dict:
    """
    Apply moderation policy based on calibrated thresholds.

    Returns:
        {
          decision: "allow" | "review" | "block",
          reasons: [labels],
          scores: original scores
        }
    """

    decision = "allow"
    reasons = []

    for label, score in scores.items():
        cfg = THRESHOLDS.get(label)

        if not cfg or cfg["review"] is None:
            continue

        review_t = cfg["review"]
        block_t = cfg["block"]

        if score >= block_t:
            decision = "block"
            reasons.append(label)

        elif score >= review_t and decision != "block":
            decision = "review"
            reasons.append(label)

    return {
        "decision": decision,
        "reasons": reasons,
        "scores": scores
    }

