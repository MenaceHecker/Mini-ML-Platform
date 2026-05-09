import logging

from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.train import router as train_router
from app.api.predict import router as predict_router
from app.api.model import router as model_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(title="Mini ML Platform")
app.include_router(predict_router)
app.include_router(ingest_router)
app.include_router(train_router)
app.include_router(model_router)


@app.get("/health")
def health():
    return {"status": "ok"}
