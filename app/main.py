from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.train import router as train_router


app = FastAPI(title="Mini ML Platform")
app.include_router(ingest_router)
app.include_router(train_router)

@app.get("/health")
def health():
    return {"status": "ok"}
