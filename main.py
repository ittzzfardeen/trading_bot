from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.responses import JSONResponse
from data_ingestion.ingestion_pipeline import Data_Ingestion  # you already have this
from agent.workflow import GraphBuilder  # this should be your graph stream handler
from data_models.models import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")

async def upload_files(files: List[UploadFile] = File(...)):
    try:
        ingestion = DataIngestion()
        ingestion.run_pipeline(files)
        return {"message": "Files successfully processed and stored."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})