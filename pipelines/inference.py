import mlflow
import mlflow.pyfunc

MODEL_NAME = "CaliforniaHousingModel"


def _load_latest_registered_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{MODEL_NAME}'. Train a model first.")

    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    return mlflow.pyfunc.load_model(model_uri)


def get_production_model():
    mlflow.set_tracking_uri("file:./experiments")
    mlflow.set_registry_uri("file:./experiments")

    try:
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    except Exception:
        return _load_latest_registered_model()


def load_production_model():
    return get_production_model()
