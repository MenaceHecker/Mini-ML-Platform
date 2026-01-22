from fastapi import FastAPI

app = FastAPI(title="Mini ML Platform")

@app.get("/health")
def health():
    return {"status": "ok"}
