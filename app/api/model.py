from fastapi import APIRouter, HTTPException

from app.model_cache import model_cache

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/reload")
def reload_model():
    """
    Force the server to reload the latest Production model from MLflow.
    Call this after promoting or training a new model version.
    """
    try:
        info = model_cache.reload()
        return {"message": "Model reloaded successfully", **info}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to reload model: {str(e)}"
        )


@router.get("/info")
def model_info():
    """Return metadata about the currently cached model."""
    return model_cache.info()
