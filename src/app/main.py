from fastapi import FastAPI
from app.api.routes import router as api_router
from app.ml.state import model_loader

app = FastAPI(
    title="Moderator",
    version="0.1.0",
    description="ML text moderation service"
)

app.include_router(api_router)


@app.on_event("startup")
def startup_event():
    # Загружает ВСЕ доступные движки (v3 ... v6.2) из artifacts/
    model_loader.load_all()


@app.get("/")
def root():
    return {"status": "ok", "service": "moderator"}