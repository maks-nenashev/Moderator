import joblib
import json
import sys
from pathlib import Path

def load_engine(version):
    path = Path(f"artifacts/{version}")
    model = joblib.load(path / "model.joblib")
    vectorizer = joblib.load(path / "vectorizer.joblib")
    with open(path / "thresholds.json", "r") as f:
        thresholds = json.load(f)
    return model, vectorizer, thresholds

def predict(text):
    engines = ["v5", "v5.1"]
    print(f"\n🔍 Testing: '{text}'")
    print("-" * 50)

    for eng in engines:
        model, vec, th = load_engine(eng)
        
        # Получаем вероятность
        X = vec.transform([text])
        prob = model.predict_proba(X)[0][1]
        
        # Сверяем с порогом
        status = "❌ TOXIC" if prob >= th["review_threshold"] else "✅ CLEAN"
        
        print(f"[{eng}] Prob: {prob:.4f} | Threshold: {th['review_threshold']} | Result: {status}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'text to check'")
    else:
        predict(sys.argv[1])
