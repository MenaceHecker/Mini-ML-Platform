from datetime import datetime, timezone
from pathlib import Path
import threading
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from app.model_cache import model_cache

router = APIRouter()

# One lock to serialise CSV appends and prevent file corruption under
# concurrent requests.
_log_lock = threading.Lock()
MONITORING_DIR = Path("data/monitoring")
PREDICTIONS_LOG = MONITORING_DIR / "predictions.csv"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

class PredictionRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    RoomsPerHousehold: float
    BedroomsPerRoom: float


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest] = Field(min_length=1)


def _get_model_or_503():
    """Return the cached model or raise HTTP 503 if unavailable."""
    try:
        return model_cache.get()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Production model not available yet: {str(e)}"
        )


def _append_prediction_logs(rows: list[dict]):
    with _log_lock:
        df = pd.DataFrame(rows)
        write_header = not PREDICTIONS_LOG.exists()
        df.to_csv(PREDICTIONS_LOG, mode="a", header=write_header, index=False)


@router.post("/predict")
def predict(data: PredictionRequest):
    model = _get_model_or_503()

    input_data = data.model_dump()
    df = pd.DataFrame([input_data])
    pred = model.predict(df)
    request_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    prediction = float(pred[0])

    _append_prediction_logs([{
        "request_id": request_id,
        "timestamp_utc": timestamp_utc,
        **input_data,
        "prediction": prediction
    }])

    return {"request_id": request_id, "prediction": prediction}


@router.post("/predict/batch")
def predict_batch(payload: BatchPredictionRequest):
    model = _get_model_or_503()
    rows = [record.model_dump() for record in payload.records]
    df = pd.DataFrame(rows)
    preds = model.predict(df)

    batch_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    log_rows = []
    for row, pred in zip(rows, preds):
        log_rows.append({
            "request_id": batch_id,
            "timestamp_utc": timestamp_utc,
            **row,
            "prediction": float(pred)
        })

    _append_prediction_logs(log_rows)

    return {
        "batch_id": batch_id,
        "count": len(rows),
        "predictions": [float(p) for p in preds]
    }


@router.get("/predict/stats")
def prediction_stats():
    if not PREDICTIONS_LOG.exists():
        return {
            "total_predictions": 0,
            "message": "No predictions logged yet."
        }

    df = pd.read_csv(PREDICTIONS_LOG)
    if df.empty:
        return {
            "total_predictions": 0,
            "message": "No predictions logged yet."
        }

    return {
        "total_predictions": int(len(df)),
        "average_prediction": float(df["prediction"].mean()),
        "min_prediction": float(df["prediction"].min()),
        "max_prediction": float(df["prediction"].max()),
        "latest_prediction": float(df.iloc[-1]["prediction"]),
        "latest_timestamp_utc": str(df.iloc[-1]["timestamp_utc"])
    }
