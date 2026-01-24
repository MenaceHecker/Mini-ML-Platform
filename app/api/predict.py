from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from pipelines.inference import load_production_model

router = APIRouter()
model = load_production_model()

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