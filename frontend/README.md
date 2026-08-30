# Frontend Dashboard

The dashboard is a Streamlit interface that lets operators upload a voice sample and view the deepfake assessment in a user-friendly panel.

## Features
- Upload an audio sample in common voice formats
- Display AI probability and risk score
- Highlight the final action recommendation
- Show the prevention plan with clear operational instructions

## Run locally

```bash
cd frontend
streamlit run dashboard.py
```

## Typical workflow
1. Upload a voice clip.
2. The model estimates whether it is real or synthetic.
3. The app surfaces the risk score, risk level, and prevention action.
4. Operators can act based on the displayed recommendation.
