import json
from pathlib import Path
from typing import Dict, Optional

# =========================
# Load v1 thresholds (review-only)
# =========================
THRESHOLDS_V1_PATH = Path("artifacts/policy_thresholds.json")
with open(THRESHOLDS_V1_PATH, "r") as f:
    THRESHOLDS_V1 = json.load(f)

# =========================
# Load v2 thresholds (ONLY source of block)
# =========================
THRESHOLDS_V2_PATH = Path("artifacts/policy_thresholds_v2.json")
with open(THRESHOLDS_V2_PATH, "r") as f:
    THRESHOLDS_V2 = json.load(f)

V2_REVIEW = THRESHOLDS_V2["toxicity"]["review"]
V2_BLOCK  = THRESHOLDS_V2["toxicity"]["block"]

# =========================
# Load v3 thresholds (sexual intent signal)
# =========================
THRESHOLDS_V3_PATH = Path("artifacts/v3/thresholds.json")
with open(THRESHOLDS_V3_PATH, "r") as f:
    THRESHOLDS_V3 = json.load(f)

V3_REVIEW = THRESHOLDS_V3["review_threshold"]

# =========================
# V1 policy (REVIEW ONLY)
# =========================
def apply_policy_v1(scores: Dict[str, float]) -> Dict:
    """
    v1 is a content classifier.
    It NEVER blocks. It can only raise review signals.
    """
    reasons = []

    for label, score in scores.items():
        cfg = THRESHOLDS_V1.get(label)
        if not cfg or cfg.get("review") is None:
            continue

        if score >= cfg["review"]:
            reasons.append(label)

    decision = "review" if reasons else "allow"

    return {
        "decision": decision,
        "reasons": reasons,
        "scores": scores
    }

# =========================
# Full policy: v1 + v2 + v3
# =========================
def apply_policy(
    scores_v1: Dict[str, float],
    score_v2: Optional[float] = None,
    v3: Optional[Dict] = None
) -> Dict:
    """
    FINAL moderation policy.

    Rules:
    - v1: weak content signal, never blocks, never alone
    - v2: ONLY source of block
    - v3: sexual intent signal → review
    """

    # ---- Step 1: v2 TERMINAL BLOCK ----
    if score_v2 is not None and score_v2 >= V2_BLOCK:
        return {
            "decision": "block",
            "reasons": ["confirmed_aggression"],
            "scores": scores_v1,
            "v2_score": score_v2
        }

    # ---- Step 2: v3 sexual intent signal ----
    if v3 is not None:
        v3_score = v3.get("score")
        if v3_score is not None and v3_score >= V3_REVIEW:
            return {
                "decision": "review",
                "reasons": ["sexual_intent_signal"],
                "scores": scores_v1,
                "v2_score": score_v2,
                "v3_score": v3_score
            }

    # ---- Step 3: v2 soft toxicity review ----
    if score_v2 is not None and score_v2 >= V2_REVIEW:
        return {
            "decision": "review",
            "reasons": ["multilingual_toxicity"],
            "scores": scores_v1,
            "v2_score": score_v2
        }

    # ---- Step 4: v1 content review (ONLY with confirmation) ----
    v1_result = apply_policy_v1(scores_v1)
    if v1_result["decision"] == "review":
        # v1 is allowed to escalate ONLY if something else already smells
        if score_v2 is not None and score_v2 >= V2_REVIEW:
            return {
                "decision": "review",
                "reasons": v1_result["reasons"],
                "scores": scores_v1,
                "v2_score": score_v2
            }

    # ---- Step 5: allow ----
    return {
        "decision": "allow",
        "reasons": [],
        "scores": scores_v1,
        "v2_score": score_v2
    }
