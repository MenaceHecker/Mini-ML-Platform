# Mini ML Platform

This is a small end-to-end ML project built to feel closer to a real product than a notebook demo.

You can:
- ingest a dataset,
- run feature engineering,
- train and register a model with MLflow,
- serve predictions through FastAPI,
- and log predictions for lightweight monitoring.

The focus is system design and workflow, not chasing the highest model accuracy.

## Tech stack
- FastAPI for the API layer
- scikit-learn for model training
- MLflow for experiment tracking and model registry
- pandas for data handling

## Project layout
- `app/api` - API routes (`/ingest`, `/train`, `/predict`, etc.)
- `pipelines` - feature, training, and inference logic
- `data/raw` - uploaded raw datasets
- `data/processed` - processed datasets used for training
- `data/monitoring` - logged predictions for monitoring
- `experiments` - local MLflow tracking and model registry data

## API endpoints

### Health
- `GET /health`  
Returns a simple status check.

### Ingest data
- `POST /ingest`  
Upload a CSV dataset. The API saves the raw file, runs feature engineering, and writes the processed file.

### Train model
- `POST /train`  
Trains a `RandomForestRegressor`, logs metrics to MLflow, and registers the model as `CaliforniaHousingModel`.

### Single prediction
- `POST /predict`  
Runs inference on one record and returns:
- `request_id`
- `prediction`

Each prediction is logged to `data/monitoring/predictions.csv` with timestamp and input features.

### Batch prediction
- `POST /predict/batch`  
Accepts a list of records and returns:
- `batch_id`
- `count`
- `predictions`

Batch predictions are also logged to `data/monitoring/predictions.csv`.

### Prediction stats
- `GET /predict/stats`  
Returns simple monitoring stats from logged predictions:
- total predictions
- average/min/max prediction
- latest prediction and timestamp

## Notes
- Inference first tries the `Production` model stage in MLflow.
- If no production model is set yet, it falls back to the latest registered model version.

## Run locally
Start the API from the project root:

```bash
uvicorn app.main:app --reload
```

Then open:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

