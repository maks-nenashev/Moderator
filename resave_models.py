import joblib
from pathlib import Path

engines = ["v3", "v3.1", "v4", "v4.1", "v5", "v5.1", "v6", "v6.1"]

for eng in engines:
    path = Path(f"artifacts/{eng}")
    if not path.exists():
        continue
    
    print(f"Processing {eng}...")
    model = joblib.load(path / "model.joblib")
    vec = joblib.load(path / "vectorizer.joblib")
    
    # Перезаписываем их текущей версией (1.7.2)
    joblib.dump(model, path / "model.joblib")
    joblib.dump(vec, path / "vectorizer.joblib")
    print(f"Fixed {eng}")
