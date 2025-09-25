from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile

from rag import (
    load_resume,
    rag
)

app = FastAPI()

@app.post("/predict")
async def predict_best_job(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    resume = load_resume(tmp_path)
    result = rag(resume)

    return JSONResponse(content={"result": result})