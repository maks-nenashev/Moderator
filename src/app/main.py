from fastapi import FastAPI
from app.api.routes import router as api_router
from app.ml.state import model_loader
from app.ml.state_v2 import model_loader_v2 # load second model state
from app.ml.state_v3 import model_loader_v3     # load third model state

app = FastAPI(
    title="Moderator",
    version="0.1.0",
    description="ML text moderation service"
)

app.include_router(api_router)


@app.on_event("startup")
def startup_event():
    model_loader.load()
    model_loader_v2.load() # load second model on startup
    #model_loader_v3.load()  # load third model on startup

@app.get("/")
def root():
    return {"status": "ok", "service": "moderator"}

