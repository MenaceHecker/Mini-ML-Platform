import mlflow.pyfunc

MODEL_NAME = "CaliforniaHousingModel"

def load_production_model():
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
