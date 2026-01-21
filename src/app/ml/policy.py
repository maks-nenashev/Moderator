import json
from pathlib import Path
from typing import Dict, Optional, Any, List

# ==========================================
# Infrastructure: Artifacts & Thresholds
# ==========================================
ARTIFACTS_DIR = Path("artifacts")

def get_threshold(version: str, default: float = 0.45) -> float:
    """Загружает порог из артефактов. Если файла нет, берет безопасный дефолт."""
    path = ARTIFACTS_DIR / version / "thresholds.json"
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f).get("review_threshold", default)
    except Exception:
        return default

# Динамическая загрузка порогов для всех 8 движков
THRESHOLDS = {
    v: get_threshold(v) for v in ["v3", "v3.1", "v4", "v4.1", "v5", "v5.1", "v6", "v6.1"]
}

# Mapping регионов для формирования понятных причин (reasons)
REGION_MAP = {
    "v3": "west",
    "v4": "cee",
    "v5": "baltic",
    "v6": "cis"
}

# ==========================================
# Core Logic: Policy Engine
# ==========================================
def apply_policy(engines_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Унифицированная политика для 8 векторов.
    engines_data: словарь вида {'v3': {'score': 0.5}, 'v3.1': {...}, ...}
    """
    reasons = []
    final_scores = {}

    for version, data in engines_data.items():
        if not data:
            continue
            
        score = data.get("score", 0.0)
        final_scores[f"{version}_score"] = score
        
        # Проверка порога
        limit = THRESHOLDS.get(version, 0.5)
        if score >= limit:
            # Определяем тип (Base или Slang) и регион
            is_slang = ".1" in version
            region = REGION_MAP.get(version.replace(".1", ""), "unknown")
            category = "sexual_slang" if is_slang else "sexual_intent"
            reasons.append(f"{category}_{region}")

    # Финальное решение: если есть хотя бы одна причина — на модерацию
    decision = "review" if reasons else "allow"
    
    # Risk Control: CIS Hardcore Block (Опционально)
    # Если v6 (Base CIS) выдает экстремальный скор (>0.9), можно сразу BLOCK
    if final_scores.get("v6_score", 0) > 0.92:
        decision = "block"
        reasons.append("confirmed_toxicity_cis")

    return {
        "decision": decision,
        "reasons": list(set(reasons)),
        **final_scores
    }

if __name__ == "__main__":
    # Тест: Имитация срабатывания v6 (хлеб 0.404 при пороге 0.45)
    test_data = {
        "v6": {"score": 0.404},
        "v6.1": {"score": 0.120}
    }
    print("--- Test: CIS Safe Case ---")
    print(apply_policy(test_data))
    
    # Тест: Имитация срабатывания v4.1 (Сленг CEE)
    test_data_toxic = {
        "v4.1": {"score": 0.850}
    }
    print("\n--- Test: CEE Slang Case ---")
    print(apply_policy(test_data_toxic))