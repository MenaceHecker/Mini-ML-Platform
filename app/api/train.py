import logging

from fastapi import APIRouter

from app.model_cache import model_cache
from pipelines.training import train_model

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train")
def train():
    metrics = train_model("data/processed/california_housing.csv")

    # Immediately swap the cache to the freshly registered model so subsequent
    # /predict calls don't need a manual reload.
    try:
        cache_info = model_cache.reload()
        logger.info("Model cache refreshed after training: %s", cache_info)
    except Exception as exc:
        logger.warning("Training succeeded but cache reload failed: %s", exc)
        cache_info = model_cache.info()

    return {
        "message": "Model trained and cache refreshed successfully",
        "metrics": metrics,
        "model_cache": cache_info,
    }
