# Backend API

The backend exposes a lightweight FastAPI service for audio analysis and risk evaluation.

## Responsibilities
- Accept uploaded recordings through the /analyze endpoint
- Run the voice detector on the submitted file
- Combine the model output with the risk engine
- Return both the analysis payload and the prevention plan

## Run locally

```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints
- GET / -> service metadata
- GET /health -> health probe
- POST /analyze -> analyzes an uploaded waveform and returns JSON output

## Example response

```json
{
  "status": "success",
  "analysis": {
    "voice_classification": "FAKE / AI-GENERATED VOICE",
    "ai_probability": 0.92,
    "risk": {
      "risk_level": "HIGH",
      "risk_score": 92.0
    }
  },
  "prevention": {
    "status": "BLOCK",
    "message": "High risk: ..."
  }
}
```

