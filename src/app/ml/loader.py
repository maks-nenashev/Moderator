import joblib
import json
import os
from pathlib import Path

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.thresholds = {}
        self.active_versions = ["v3", "v3.1", "v4", "v4.1", "v5", "v5.1", "v6", "v6.1"] 

    def load(self):
        for ver in self.active_versions:
            path = Path(f"artifacts/{ver}")
            if (path / "model.joblib").exists():
                self.models[ver] = joblib.load(path / "model.joblib")
                self.vectorizers[ver] = joblib.load(path / "vectorizer.joblib")
                
                with open(path / "thresholds.json", "r") as f:
                    self.thresholds[ver] = json.load(f)
                print(f"[LOADER] Engine {ver} loaded successfully.")

    def predict(self, text, version="v5"):
        if version not in self.models:
            return 0.0
        
        vec = self.vectorizers[version]
        model = self.models[version]
        
        X = vec.transform([text])
        return float(model.predict_proba(X)[:, 1][0])