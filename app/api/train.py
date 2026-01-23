from fastapi import APIRouter
from pipelines.training import train_model

router = APIRouter()
@router.post("/train")
def train():
    metrics = train_model("data/processed/california_housing.csv")
    return {
        "message": "Model trained successfully",
        "metrics": metrics
    }

