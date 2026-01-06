import json
from pathlib import Path
from typing import Dict, Optional

# =========================
# Load v1 thresholds
# =========================
THRESHOLDS_V1_PATH = Path("artifacts/policy_thresholds.json")
with open(THRESHOLDS_V1_PATH, "r") as f:
    THRESHOLDS_V1 = json.load(f)

# =========================
# Load v2 thresholds
# =========================
THRESHOLDS_V2_PATH = Path("artifacts/policy_thresholds_v2.json")
with open(THRESHOLDS_V2_PATH, "r") as f:
    THRESHOLDS_V2 = json.load(f)

V2_REVIEW = THRESHOLDS_V2["toxicity"]["review"]
V2_BLOCK = THRESHOLDS_V2["toxicity"]["block"]


# =========================
# V1 policy (unchanged)
# =========================
def apply_policy_v1(scores: Dict[str, float]) -> Dict:
    decision = "allow"
    reasons = []

    for label, score in scores.items():
        cfg = THRESHOLDS_V1.get(label)
        if not cfg or cfg["review"] is None:
            continue

        if score >= cfg["block"]:
            decision = "block"
            reasons.append(label)

        elif score >= cfg["review"] and decision != "block":
            decision = "review"
            reasons.append(label)

    return {
        "decision": decision,
        "reasons": reasons,
        "scores": scores
    }


# =========================
# Combined v1 + v2 policy
# =========================
def apply_policy(
    scores_v1: Dict[str, float],
    score_v2: Optional[float] = None
) -> Dict:
    """
    Orchestrated moderation policy.

    Priority:
      1) v1 block -> block
      2) v2 strong toxicity -> review
      3) v1 review -> review
      4) v2 moderate toxicity -> review
      5) else -> allow
    """

    # ---- Step 1: apply v1 ----
    v1_result = apply_policy_v1(scores_v1)

   # ---- Step 2: conditional block by v1 (confirmed by v2) ----
    if v1_result["decision"] == "block":
      # v1 block разрешён ТОЛЬКО если v2 подтверждает токсичность
      if score_v2 is not None and score_v2 >= V2_REVIEW:
        return v1_result
      else:
        # иначе понижаем до review (false positive protection)
        return {
            "decision": "review",
            "reasons": v1_result["reasons"],
            "scores": scores_v1,
            "v2_score": score_v2
        }

    # ---- Step 4: v1 review ----
    if v1_result["decision"] == "review":
        return v1_result

    # ---- Step 5: v2 soft review ----
    if score_v2 is not None and score_v2 >= V2_REVIEW:
        return {
            "decision": "review",
            "reasons": ["multilingual_toxicity"],
            "scores": scores_v1,
            "v2_score": score_v2
        }

    # ---- Step 6: allow ----
    return {
        "decision": "allow",
        "reasons": [],
        "scores": scores_v1,
        "v2_score": score_v2
    }

