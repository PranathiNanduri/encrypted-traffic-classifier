from fastapi import FastAPI

app = FastAPI(title="Encrypted Traffic Classifier")

@app.get("/")
def home():
    return {"message": "API running locally"}