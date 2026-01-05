import os

class Settings:
    ENV: str = os.getenv("ENV", "dev")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "artifacts/model.keras")
    TOKENIZER_PATH: str = os.getenv("TOKENIZER_PATH", "artifacts/tokenizer.json")


settings = Settings()
