import json
from pathlib import Path
from typing import Dict, Optional, Any, List


def load_json(path: Path, required: bool = True, default: Optional[Dict] = None) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Threshold file not found: {path}")
        return default or {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# Load thresholds
# =========================
ARTIFACTS_DIR = Path("artifacts")
THRESHOLDS_V1_PATH = ARTIFACTS_DIR / "policy_thresholds.json"
THRESHOLDS_V2_PATH = ARTIFACTS_DIR / "policy_thresholds_v2.json"
THRESHOLDS_V3_PATH = ARTIFACTS_DIR / "v3" / "thresholds.json"
THRESHOLDS_V3_1_PATH = ARTIFACTS_DIR / "v3.1" / "thresholds.json"
THRESHOLDS_V4_PATH = ARTIFACTS_DIR / "v4" / "thresholds.json"
THRESHOLDS_V4_1_PATH = ARTIFACTS_DIR / "v4.1" / "thresholds.json" # <-- NEW

THRESHOLDS_V1 = load_json(THRESHOLDS_V1_PATH, required=False, default={})
THRESHOLDS_V2 = load_json(THRESHOLDS_V2_PATH, required=False, default={})
THRESHOLDS_V3 = load_json(THRESHOLDS_V3_PATH, required=False, default={})
THRESHOLDS_V3_1 = load_json(THRESHOLDS_V3_1_PATH, required=False, default={})
THRESHOLDS_V4 = load_json(THRESHOLDS_V4_PATH, required=False, default={})
THRESHOLDS_V4_1 = load_json(THRESHOLDS_V4_1_PATH, required=False, default={}) # <-- NEW

# Настройка порогов (v2 Toxicity)
toxicity_cfg = THRESHOLDS_V2.get("toxicity", {}) or {}
V2_REVIEW = toxicity_cfg.get("review", 0.5)
V2_BLOCK = toxicity_cfg.get("block", 0.9)

# Настройка порогов (Sexual Intent)
V3_REVIEW = THRESHOLDS_V3.get("review_threshold", 0.70)
V3_1_REVIEW = THRESHOLDS_V3_1.get("review_threshold", 0.70)
V4_REVIEW = THRESHOLDS_V4.get("review_threshold", 0.70)
V4_1_REVIEW = THRESHOLDS_V4_1.get("review_threshold", 0.70) # <-- NEW (Сленг)

# =========================
# V1 policy (REVIEW ONLY)
# =========================
def apply_policy_v1(scores: Dict[str, float]) -> Dict[str, Any]:
    reasons = []
    for label, score in scores.items():
        cfg = THRESHOLDS_V1.get(label)
        if not cfg: continue
        review_thr = cfg.get("review")
        if review_thr is not None and score >= review_thr:
            reasons.append(label)
    
    return {"decision": "review" if reasons else "allow", "reasons": reasons}

# =========================
# Full policy: v1 + v2 + v3 + v3.1 + v4 + v4.1
# =========================
def apply_policy(
    scores_v1: Dict[str, float],
    score_v2: Optional[float] = None,
    v3: Optional[Dict[str, Any]] = None,
    v3_1: Optional[Dict[str, Any]] = None,
    v4: Optional[Dict[str, Any]] = None,
    v4_1: Optional[Dict[str, Any]] = None  # <-- NEW
) -> Dict[str, Any]:
    """
    FINAL moderation policy.
    - v2: BLOCK signal (Toxicity)
    - v3/v3.1/v4/v4.1: REVIEW signal (Sexual Intent)
    """
    reasons = []
    v3_score = v3.get("score") if v3 else None
    v3_1_score = v3_1.get("score") if v3_1 else None
    v4_score = v4.get("score") if v4 else None
    v4_1_score = v4_1.get("score") if v4_1 else None # <-- NEW

    # ---- Step 1: v2 TERMINAL BLOCK ----
    if score_v2 is not None and score_v2 >= V2_BLOCK:
        return {
            "decision": "block",
            "reasons": ["confirmed_aggression"],
            "v2_score": score_v2
        }

    # ---- Step 2: v4.1 HARDCORE SLANG (High Priority) ----
    # Если сработал сленг (cycki и т.д.), это мгновенный REVIEW
    if v3_1_score is not None and v3_1_score >= V3_1_REVIEW:
        reasons.append("sexual_slang_west")

    if v4_1_score is not None and v4_1_score >= V4_1_REVIEW:
        reasons.append("sexual_slang_cee")

    # ---- Step 3: v3 & v4 Sexual Intent signals ----
    if v3_score is not None and v3_score >= V3_REVIEW:
        reasons.append("sexual_intent_west")
    
    if v4_score is not None and v4_score >= V4_REVIEW:
        reasons.append("sexual_intent_cee")

    # ---- Step 4: v2 Toxicity review ----
    if score_v2 is not None and score_v2 >= V2_REVIEW:
        reasons.append("multilingual_toxicity")

    # ---- Step 5: v1 content review (Escalation) ----
    v1_res = apply_policy_v1(scores_v1)
    if v1_res["decision"] == "review" and (score_v2 is not None or v3_score is not None or v4_score is not None or v4_1_score is not None):
        reasons.extend(v1_res["reasons"])

    # Final Decision construction
    decision = "review" if reasons else "allow"
    
    return {
        "decision": decision,
        "reasons": list(set(reasons)),
        "scores_v1": scores_v1,
        "v2_score": score_v2,
        "v3_score": v3_score,
        "v3_1_score": v3_1_score,
        "v4_score": v4_score,
        "v4_1_score": v4_1_score # <-- NEW
    }

if __name__ == "__main__":
    # Тест: v4.1 поймал польский мат
    print(apply_policy({"insult": 0.1}, score_v2=0.1, v4_1={"score": 0.95}))