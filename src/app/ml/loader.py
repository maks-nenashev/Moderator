import sys
from pathlib import Path

# Гарантируем, что корень проекта (/home/maks/Moderator) находится в sys.path
BASE_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_PROJECT_DIR))

import joblib
import json
import logging
from train.s3_sync import download_version_artifacts

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
    # Перечисляем все активные версии, которые должны быть инициализированы сервисом
    TARGET_VERSIONS = [
        "v1", 
        "v3", "v3.1", "v3.2", 
        "v4", "v4.1", "v4.2", 
        "v5", "v5.1", "v5.2", 
        "v6", "v6.1", "v6.2"
    ]

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.base_path = Path(artifacts_dir)
        self.engines = {}
        self.load_all()

    def load_all(self):
        # Гарантируем наличие базовой директории artifacts/
        self.base_path.mkdir(parents=True, exist_ok=True)

        for version in self.TARGET_VERSIONS:
            version_dir = self.base_path / version
            model_file = version_dir / "model.joblib"

            # 1. Проверяем наличие артефактов локально. Если их нет — тянем из S3
            if not model_file.exists():
                logger.warning(f"⚠️ Local artifacts for [{version}] not found. Pulling from S3...")
                success = download_version_artifacts(version)
                if not success:
                    logger.error(f"❌ Failed to download artifacts for [{version}] from S3.")
                    continue

            # 2. Инициализируем ModelEngine
            try:
                self.engines[version] = ModelEngine(version, self.base_path)
            except Exception as e:
                logger.error(f"❌ Failed to load engine [{version}]: {e}")

        print(f"✅ Successfully loaded {len(self.engines)} engines: {list(self.engines.keys())}")