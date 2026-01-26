import mlflow

mlflow.set_tracking_uri("file:./experiments")
client = mlflow.tracking.MlflowClient()

print([m.name for m in client.search_registered_models()])
