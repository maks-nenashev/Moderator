import joblib
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelEngine:
    def __init__(self, version: str, artifacts_dir: Path):
        self.version = version
        self.dir = artifacts_dir / version
        self.model = joblib.load(self.dir / "model.joblib")
        self.vectorizer = joblib.load(self.dir / "vectorizer.joblib")
        
        with open(self.dir / "thresholds.json") as f:
            data = json.load(f)
            self.review_threshold = data.get("review_threshold", 0.5)

    def predict_score(self, text: str) -> float:
        X = self.vectorizer.transform([text])
        return float(self.model.predict_proba(X)[0, 1])

class ModelLoader:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.base_path = Path(artifacts_dir)
        self.engines = {}
        self.load_all()

    def load_all(self):
        if not self.base_path.exists():
            logger.error(f"Artifacts path {self.base_path} does not exist.")
            return

        for version_dir in sorted(self.base_path.iterdir()):
            if version_dir.is_dir() and (version_dir / "model.joblib").exists():
                version = version_dir.name
                try:
                    self.engines[version] = ModelEngine(version, self.base_path)
                except Exception as e:
                    logger.error(f"Failed to load engine {version}: {e}")

        print(f"✅ Loaded engines: {list(self.engines.keys())}")