from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from pipelines.features import build_features

router = APIRouter()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/ingest")
async def ingest_dataset(file: UploadFile = File(...)):
    raw_path = RAW_DIR / file.filename
    processed_path = PROCESSED_DIR / file.filename

    # Saving the raw file
    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Running the feature pipeline
    build_features(
        input_path=str(raw_path),
        output_path=str(processed_path)
    )

    return {
        "message": "Dataset ingested and features generated",
        "raw_path": str(raw_path),
        "processed_path": str(processed_path)
    }