from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from disease_model import load_disease_model, predict_disease
from suggest import get_disease_details

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    load_disease_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    return predict_disease(contents)


class DiseaseDetailsRequest(BaseModel):
    disease: str
    context: str


@app.post("/disease-details")
async def get_details(request: DiseaseDetailsRequest):
    return get_disease_details(request.disease, request.context)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)