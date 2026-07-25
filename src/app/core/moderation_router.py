import joblib, json
from pathlib import Path

class ModerationRouter:
    def __init__(self, artifacts_path="artifacts"):
        self.engines = {}
        # Регистрируем наши живые движки
        for version in ["v1", "v3", "v3.1", "v3.2", "v4", "v4.1", "v4.2", "v5", "v5.1", "v5.2", "v6", "v6.1", "v6.2"]:
            path = Path(artifacts_path) / version
            model_path = path / "model.joblib"
            vec_path = path / "vectorizer.joblib"
            thr_path = path / "thresholds.json"

            if model_path.exists() and vec_path.exists():
                # Загружаем порог из файла, если нет — дефолт 0.7
                threshold = 0.7
                if thr_path.exists():
                    with open(thr_path, "r") as f:
                        threshold = json.load(f).get("review_threshold", 0.7)

                self.engines[version] = {
                    "model": joblib.load(model_path),
                    "vec": joblib.load(vec_path),
                    "threshold": threshold
                }
        print(f"✅ Loaded engines: {list(self.engines.keys())}")

    def predict(self, text):
        raw_scores = {}
        flags = []

        # 1. Снимаем показания со всех датчиков
        for name, eng in self.engines.items():
            score = eng["model"].predict_proba(eng["vec"].transform([text]))[0, 1]
            raw_scores[name] = round(score, 4)
            
            # Если скор выше порога этой конкретной модели — ставим флаг
            if score >= eng["threshold"]:
                flags.append(name)

        # 2. Логика принятия решения (Safety First)
        # Если хотя бы один движок сработал — отправляем на проверку
        if flags:
            status = "🔴 REVIEW"
            # Для удобства понимания пишем, кто именно нашел нарушение
            reason = f"Detected by: {', '.join(flags)}"
        else:
            status = "🟢 ALLOW"
            reason = "Clean"

        return {
            "text": text,
            "decision": status,
            "reason": reason,
            "raw_scores": raw_scores
        }

# --- ТЕСТ ДРАЙВ ---
router = ModerationRouter()
test_cases = [
    "The farm has a very loud cock", 
    "s-e-n-d n-u-d-e-s",             
    "Pokaż mi cycki",
    "Ho visto un uccello nel parco"
]

print(f"\n{'TEXT':<30} | {'DECISION':<12} | {'REASON'}")
print("-" * 75)
for t in test_cases:
    res = router.predict(t)
    print(f"{res['text'][:30]:<30} | {res['decision']:<12} | {res['reason']}")