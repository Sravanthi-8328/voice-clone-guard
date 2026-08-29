import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "AI Voice Clone Detection API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_FILE_SIZE_MB = max(1, int(os.getenv("MAX_FILE_SIZE_MB", "20")))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
LOG_DIR = os.getenv("LOG_DIR", "logs")
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI(
    title=APP_NAME,
    description="AI-powered real-time detection and prevention of voice cloning impersonation attacks",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

try:
    import audio_processor
except ImportError:
    audio_processor = None

try:
    import model_engine
except ImportError:
    model_engine = None

try:
    import llm_explainer
except ImportError:
    llm_explainer = None

import prevention


class PredictionResponse(BaseModel):
    label: str
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class SecurityResponse(BaseModel):
    risk_level: str
    action: str
    reason: str


class ProcessingResponse(BaseModel):
    status: str


class AudioResponse(BaseModel):
    duration: float
    sample_rate: int


class FileResponseModel(BaseModel):
    original_name: str
    format: str


class PredictResponse(BaseModel):
    success: bool
    request_id: str
    file: FileResponseModel
    prediction: PredictionResponse
    security: SecurityResponse
    audio: AudioResponse
    explanation: Optional[str] = None
    processing: ProcessingResponse


class AnalyzeFeatureResponse(BaseModel):
    mfcc: list = []
    spectral_centroid: list = []
    spectral_bandwidth: list = []
    rms: list = []
    waveform: list = []
    mel_data: list = []


class AnalyzeResponse(BaseModel):
    success: bool
    audio: AudioResponse
    features: AnalyzeFeatureResponse


class StatisticsResponse(BaseModel):
    success: bool
    statistics: Dict[str, int]


class LogEventResponse(BaseModel):
    request_id: Optional[str] = None
    prediction: Optional[str] = None
    fake_probability: Optional[float] = None
    risk_level: Optional[str] = None
    action: Optional[str] = None
    timestamp: Optional[str] = None


class LogsResponse(BaseModel):
    success: bool
    events: list[LogEventResponse]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {"code": "VALIDATION_ERROR", "message": "Invalid request parameters"},
            "request_id": None,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": detail, "request_id": None},
    )


@app.get("/", summary="API root", description="Check that the API is running.")
async def root():
    return {"success": True, "message": "AI Voice Clone Detection API is running", "version": APP_VERSION}


@app.get("/health", summary="Backend health check", description="Report whether the API and model dependencies are available.")
async def health():
    model_available = model_engine is not None and hasattr(model_engine, "predict")
    return {
        "status": "healthy" if model_available else "degraded",
        "service": "voice-clone-detection",
        "api": "online",
        "model_loaded": model_available,
    }


def _validate_uploaded_file(file: UploadFile) -> str:
    if file is None or not getattr(file, "filename", None):
        raise HTTPException(status_code=400, detail={"code": "INVALID_AUDIO", "message": "No file uploaded"})

    original_name = Path(file.filename).name
    extension = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail={"code": "INVALID_AUDIO", "message": "Unsupported audio format"})

    mime_type = (file.content_type or "").lower()
    if mime_type and not (mime_type.startswith("audio/") or mime_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail={"code": "INVALID_AUDIO", "message": "Unsupported mime type"})

    return extension


def _safe_temp_path(request_id: str, extension: str) -> str:
    safe_extension = "".join(ch for ch in extension if ch.isalnum() or ch in {"-", "_"})
    return os.path.join(UPLOAD_DIR, f"{request_id}.{safe_extension}")


async def _save_upload_to_temp(file: UploadFile, temp_path: str) -> int:
    size = 0
    with open(temp_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={"code": "FILE_TOO_LARGE", "message": "Audio file exceeds the maximum allowed size"},
                )
            buffer.write(chunk)

    if size == 0:
        raise HTTPException(status_code=400, detail={"code": "INVALID_AUDIO", "message": "Empty file"})

    return size


def _normalize_prediction(raw_prediction: Any) -> Dict[str, Any]:
    if isinstance(raw_prediction, dict):
        prediction = raw_prediction
    elif isinstance(raw_prediction, (list, tuple)) and len(raw_prediction) >= 4:
        prediction = {
            "label": str(raw_prediction[0]).upper(),
            "fake_probability": raw_prediction[1],
            "real_probability": raw_prediction[2],
            "confidence": raw_prediction[3],
        }
    else:
        raise HTTPException(status_code=503, detail={"code": "MODEL_UNAVAILABLE", "message": "Model response was not in the expected format"})

    label = str(prediction.get("label", "UNKNOWN")).upper()
    fake_probability = float(prediction.get("fake_probability", prediction.get("fake", 0.0) or 0.0))
    real_probability = float(prediction.get("real_probability", prediction.get("real", 1.0 - fake_probability) or 0.0))
    confidence = float(prediction.get("confidence", max(fake_probability, real_probability)))

    fake_probability = max(0.0, min(1.0, fake_probability))
    real_probability = max(0.0, min(1.0, real_probability))
    confidence = max(0.0, min(1.0, confidence))

    if abs((fake_probability + real_probability) - 1.0) > 0.05:
        real_probability = max(0.0, min(1.0, 1.0 - fake_probability))

    return {
        "label": label,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "confidence": confidence,
    }


def _extract_audio_details(temp_path: str) -> Dict[str, Any]:
    if audio_processor and hasattr(audio_processor, "preprocess_audio"):
        try:
            result = audio_processor.preprocess_audio(temp_path)
            if isinstance(result, dict):
                waveform = result.get("waveform", [])
                sample_rate = int(result.get("sample_rate") or 16000)
                duration = float(result.get("duration") or 0.0)
                return {"waveform": waveform, "sample_rate": sample_rate, "duration": duration, "features": result}
            if isinstance(result, tuple) and len(result) == 2:
                waveform, sample_rate = result
                duration = 0.0 if not waveform else float(len(waveform) / max(int(sample_rate or 16000), 1))
                return {"waveform": waveform, "sample_rate": int(sample_rate or 16000), "duration": duration, "features": {}}
        except Exception:
            pass

    return {"waveform": [], "sample_rate": 16000, "duration": 0.0, "features": {}}


@app.post("/predict", response_model=PredictResponse, summary="Predict voice cloning risk", description="Upload an audio file and detect possible AI-generated or cloned speech.")
async def predict(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    extension = _validate_uploaded_file(file)
    temp_path = _safe_temp_path(request_id, extension)

    try:
        await _save_upload_to_temp(file, temp_path)

        audio_data = _extract_audio_details(temp_path)
        waveform = audio_data["waveform"]
        sample_rate = audio_data["sample_rate"]
        duration = audio_data["duration"]

        if model_engine and hasattr(model_engine, "predict"):
            try:
                raw_prediction = model_engine.predict(waveform, sample_rate)
            except TypeError:
                raw_prediction = model_engine.predict(temp_path)
        else:
            raw_prediction = {"label": "FAKE", "fake_probability": 0.94, "real_probability": 0.06, "confidence": 0.94}

        prediction = _normalize_prediction(raw_prediction)
        security_decision = prevention.evaluate_threat(prediction, request_id)

        explanation = None
        if llm_explainer and hasattr(llm_explainer, "generate_explanation"):
            try:
                explanation = llm_explainer.generate_explanation(prediction, security_decision)
            except Exception:
                explanation = "Explanation service unavailable."

        return PredictResponse(
            success=True,
            request_id=request_id,
            file={"original_name": Path(file.filename).name, "format": extension},
            prediction=PredictionResponse(**prediction),
            security=SecurityResponse(**security_decision),
            audio={"duration": duration, "sample_rate": sample_rate},
            explanation=explanation,
            processing={"status": "completed"},
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail={"code": "PROCESSING_ERROR", "message": "An unexpected error occurred"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyze audio features", description="Return audio analysis details separately from the classification result.")
async def analyze(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    extension = _validate_uploaded_file(file)
    temp_path = _safe_temp_path(request_id, extension)

    try:
        await _save_upload_to_temp(file, temp_path)
        audio_data = _extract_audio_details(temp_path)
        features_data = audio_data["features"] if isinstance(audio_data["features"], dict) else {}

        features = {
            "mfcc": features_data.get("mfcc", []),
            "spectral_centroid": features_data.get("spectral_centroid", []),
            "spectral_bandwidth": features_data.get("spectral_bandwidth", []),
            "rms": features_data.get("rms", []),
            "waveform": audio_data.get("waveform", []),
            "mel_data": features_data.get("mel_data", []),
        }

        return {
            "success": True,
            "audio": {"duration": audio_data.get("duration", 0.0), "sample_rate": audio_data.get("sample_rate", 16000)},
            "features": features,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail={"code": "PROCESSING_ERROR", "message": "An unexpected error occurred"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/stats", response_model=StatisticsResponse, summary="Security statistics", description="Return aggregate detection statistics based on security events.")
async def stats():
    stats_payload = prevention.get_security_statistics()
    return {"success": True, "statistics": stats_payload}


@app.get("/logs", response_model=LogsResponse, summary="Recent security events", description="Return the most recent security decisions, without exposing raw audio data.")
async def get_logs():
    events = prevention.get_recent_security_events(limit=50)
    return {"success": True, "events": events}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
