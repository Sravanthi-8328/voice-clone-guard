# AI Voice Clone Detection API

This is the backend API for the **AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks** project (SIH26104).

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

3. **Run the server:**
   ```bash
   uvicorn app:app --reload
   ```

4. **View documentation:**
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Module Integration

- `app.py`: Main FastAPI application, routing, and file validation.
- `prevention.py`: Threat prevention engine which maps model predictions to risk levels and actions (MEMBER 1).
- **MEMBER 2 (Model Engine)**: Should implement `predict(waveform, sample_rate)` in `model_engine.py` returning a dictionary with keys `label`, `fake_probability`, `real_probability`, `confidence`.
- **MEMBER 3 (Audio Pipeline)**: Should implement `preprocess_audio(file_path)` in `audio_processor.py` returning `waveform` and `sample_rate`.
- **MEMBER 6 (LLM Explainer)**: Should implement `generate_explanation(prediction, security_decision)` in `llm_explainer.py`.
