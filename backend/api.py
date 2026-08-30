import importlib.util
import os
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

AI_DETECTOR_PATH = os.path.join(ROOT, "ai-detector", "predict.py")
if os.path.join(ROOT, "ai-detector") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "ai-detector"))

spec = importlib.util.spec_from_file_location("voice_predict_api", AI_DETECTOR_PATH)
voice_predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voice_predict)
analyze_audio = voice_predict.analyze_audio

from prevention.prevention_engine import build_prevention_plan

app = FastAPI(title="Voice Clone Guard API", version="1.0.0")


@app.get("/")
def root():
    return {"service": "Voice Clone Guard API", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_name = file.filename or "sample.wav"
    suffix = os.path.splitext(file_name)[1] or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            content = await file.read()
            if not content:
                raise ValueError("Uploaded file is empty.")
            handle.write(content)
            temp_path = handle.name

        result = analyze_audio(temp_path)
        risk = result.get("risk", {})
        prevention = build_prevention_plan(risk)
        return JSONResponse(
            content={
                "status": "success",
                "analysis": result,
                "prevention": prevention,
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc)},
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
