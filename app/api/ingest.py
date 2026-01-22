from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from pipelines.features import build_features

router = APIRouter()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

