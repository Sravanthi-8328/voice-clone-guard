import importlib.util
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

AI_DETECTOR_PATH = os.path.join(ROOT, "ai-detector", "predict.py")
if os.path.join(ROOT, "ai-detector") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "ai-detector"))

spec = importlib.util.spec_from_file_location("voice_predict_dashboard", AI_DETECTOR_PATH)
voice_predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voice_predict)
analyze_audio = voice_predict.analyze_audio

from prevention.prevention_engine import build_prevention_plan

st.set_page_config(page_title="Voice Clone Guard", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

# ==================== ULTRA MODERN STYLING ====================
st.markdown(
    """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #0f172a 50%, #1a1f35 75%, #0f172a 100%);
            background-attachment: fixed;
            color: #f1f5f9;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        /* HEADER STYLING */
        .header-container {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3), 0 0 40px rgba(59, 130, 246, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            position: relative;
            overflow: hidden;
        }
        
        .header-container::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: float 20s linear infinite;
        }
        
        @keyframes float {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }
        
        .header-title {
            font-size: 3.5rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
            background: linear-gradient(120deg, #ffffff 0%, #e0e7ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            z-index: 1;
            position: relative;
        }
        
        .header-subtitle {
            font-size: 1.15rem;
            color: rgba(226, 232, 240, 0.9);
            font-weight: 300;
            letter-spacing: 0.5px;
            z-index: 1;
            position: relative;
        }
        
        /* UPLOAD CARD */
        .upload-container {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(71, 85, 105, 0.2) 100%);
            border: 2px dashed rgba(59, 130, 246, 0.5);
            border-radius: 20px;
            padding: 3rem 2rem;
            margin-bottom: 2.5rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .upload-container:hover {
            border-color: rgba(59, 130, 246, 0.8);
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(71, 85, 105, 0.3) 100%);
            box-shadow: 0 12px 48px rgba(59, 130, 246, 0.2);
            transform: translateY(-2px);
        }
        
        .upload-label {
            font-size: 1.3rem;
            font-weight: 700;
            background: linear-gradient(120deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        /* METRIC CARDS */
        .metric-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.95) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(59, 130, 246, 0.6);
            box-shadow: 0 12px 36px rgba(59, 130, 246, 0.2);
        }
        
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: rgba(226, 232, 240, 0.6);
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 900;
            background: linear-gradient(120deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* STATUS CARD */
        .status-card {
            border-radius: 20px;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
            border: 2px solid;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .status-card.block {
            border-color: rgba(220, 38, 38, 0.5);
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.3) 0%, rgba(220, 38, 38, 0.1) 100%);
        }
        
        .status-card.verify {
            border-color: rgba(245, 158, 11, 0.5);
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.3) 0%, rgba(245, 158, 11, 0.1) 100%);
        }
        
        .status-card.allowed {
            border-color: rgba(34, 197, 94, 0.5);
            background: linear-gradient(135deg, rgba(20, 83, 45, 0.3) 0%, rgba(34, 197, 94, 0.1) 100%);
        }
        
        .status-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: rgba(226, 232, 240, 0.7);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .status-value {
            font-size: 2.8rem;
            font-weight: 900;
            margin: 0.5rem 0;
        }
        
        .status-description {
            font-size: 0.95rem;
            color: rgba(226, 232, 240, 0.8);
            margin-top: 1rem;
            line-height: 1.6;
        }
        
        /* SECTION HEADERS */
        .section-header {
            font-size: 1.5rem;
            font-weight: 800;
            margin-top: 2.5rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(120deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        /* ACTION ITEMS */
        .action-item {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.95) 100%);
            border-left: 4px solid;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 0.75rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        
        .action-item.block {
            border-left-color: #dc2626;
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.4) 0%, rgba(220, 38, 38, 0.15) 100%);
        }
        
        .action-item.verify {
            border-left-color: #f59e0b;
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.4) 0%, rgba(245, 158, 11, 0.15) 100%);
        }
        
        .action-item.info {
            border-left-color: #3b82f6;
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(59, 130, 246, 0.15) 100%);
        }
        
        /* PROGRESS BAR */
        .progress-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: rgba(226, 232, 240, 0.8);
            margin-bottom: 0.5rem;
        }
        
        /* EXPANDER */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(30, 41, 59, 0.95) 100%) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(59, 130, 246, 0.3) !important;
        }
        
        /* SMOOTH TRANSITIONS */
        [data-testid="column"] {
            transition: all 0.3s ease;
        }
        
        /* SCROLLBAR */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(30, 41, 59, 0.3);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #3b82f6, #1e40af);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #60a5fa, #3b82f6);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== HEADER ====================
st.markdown(
    """
    <div class="header-container">
        <div class="header-title">🛡️ Voice Clone Guard</div>
        <div class="header-subtitle">Real-time AI-powered detection and prevention of voice cloning impersonation attacks</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== UPLOAD SECTION ====================
st.markdown('<div class="section-header">📤 Upload Your Voice Sample</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
        <div class="upload-container">
            <div class="upload-label">🎤 Drop your audio file here</div>
            <p style="color: rgba(226, 232, 240, 0.7); margin-top: 0.5rem;">Supported: WAV, MP3, FLAC, M4A, OGG (Max 200MB)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "m4a", "ogg"], label_visibility="collapsed")


def _read_waveform(file_path):
    try:
        audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if audio is None or np.size(audio) == 0:
            return np.array([]), 0
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), sr
    except Exception:
        return np.array([]), 0


if uploaded_file is not None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1] or ".wav") as handle:
            handle.write(uploaded_file.read())
            temp_path = handle.name

        # PROGRESS INDICATOR
        with st.spinner("🔍 Analyzing voice sample..."):
            result = analyze_audio(temp_path)
        
        risk = result["risk"]
        prevention = build_prevention_plan(risk)
        waveform, sampling_rate = _read_waveform(temp_path)

        # ==================== DETECTION SUMMARY ====================
        st.markdown('<div class="section-header">📊 Detection Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">🤖 AI Probability</div>
                    <div class="metric-value">{result['ai_probability'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">⚠️ Risk Score</div>
                    <div class="metric-value">{risk['risk_score']}/100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">📈 Risk Level</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{risk['risk_level']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">✅ Confidence</div>
                    <div class="metric-value">{result['model_confidence']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ==================== RISK PROGRESS ====================
        risk_percentage = min(max(risk["risk_score"] / 100.0, 0.0), 1.0)
        st.markdown('<p class="progress-label">🎯 Overall Risk Level</p>', unsafe_allow_html=True)
        st.progress(risk_percentage)

        # ==================== MAIN RESULTS ====================
        left, right = st.columns([1.2, 1])
        
        with left:
            st.markdown('<div class="section-header">📈 Waveform Analysis</div>', unsafe_allow_html=True)
            if waveform.size > 0:
                if waveform.size > 2000:
                    waveform = waveform[:: max(1, len(waveform) // 2000)]
                st.line_chart(waveform, width='stretch')
            else:
                st.info("⚠️ Waveform preview unavailable for this file.")

        with right:
            st.markdown('<div class="section-header">🎯 Risk Decision</div>', unsafe_allow_html=True)
            status_key = prevention["status"]
            
            if status_key == "BLOCK":
                status_label = "⛔ BLOCKED"
                status_color = "#dc2626"
                card_class = "block"
            elif status_key == "VERIFY":
                status_label = "⚠️ VERIFY"
                status_color = "#f59e0b"
                card_class = "verify"
            else:
                status_label = "✅ ALLOWED"
                status_color = "#16a34a"
                card_class = "allowed"

            st.markdown(
                f"""
                <div class="status-card {card_class}">
                    <div class="status-label">Decision Status</div>
                    <div class="status-value" style="color: {status_color};">{status_label}</div>
                    <div class="status-description">
                        {risk['risk_level']} risk detected with {result['ai_probability'] * 100:.1f}% AI probability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.95) 100%);
                            border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1.2rem;
                            margin-top: 1rem; backdrop-filter: blur(10px);">
                    <p style="color: rgba(226, 232, 240, 0.9); line-height: 1.6; margin: 0;">
                        💡 <strong>Recommendation:</strong><br/>{risk['recommendation']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # ==================== PROSODY ANALYSIS ====================
        st.markdown('<div class="section-header">🎵 Prosody Analysis</div>', unsafe_allow_html=True)
        prosody = result["prosody"]
        prosody_items = {
            "Pitch variation": prosody.get("pitch_variation", 0.0),
            "Energy variation": prosody.get("energy_variation", 0.0),
            "Pause ratio": prosody.get("pause_ratio", 0.0),
            "Speaking rate": prosody.get("speaking_rate", 0.0),
        }
        st.bar_chart(prosody_items, width='stretch')

        # ==================== RECOMMENDED ACTIONS ====================
        st.markdown('<div class="section-header">✨ Recommended Actions</div>', unsafe_allow_html=True)
        
        for action in prevention["actions"]:
            action_type = action["type"]
            label = action['label']
            reason = action['reason']
            
            if action_type == "block":
                st.markdown(
                    f'<div class="action-item block"><strong>⛔ {label}</strong><br/><span style="color: rgba(226, 232, 240, 0.8); font-size: 0.9rem;">{reason}</span></div>',
                    unsafe_allow_html=True
                )
            elif action_type == "verify":
                st.markdown(
                    f'<div class="action-item verify"><strong>⚠️ {label}</strong><br/><span style="color: rgba(226, 232, 240, 0.8); font-size: 0.9rem;">{reason}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="action-item info"><strong>ℹ️ {label}</strong><br/><span style="color: rgba(226, 232, 240, 0.8); font-size: 0.9rem;">{reason}</span></div>',
                    unsafe_allow_html=True
                )

        # ==================== DETAILED ANALYSIS ====================
        with st.expander("🔍 View Detailed Analysis (JSON)"):
            st.json({
                "voice_classification": result["voice_classification"],
                "real_probability": f"{result['real_probability']*100:.2f}%",
                "ai_probability": f"{result['ai_probability']*100:.2f}%",
                "model_confidence": f"{result['model_confidence']:.2f}%",
                "prosody": result["prosody"],
                "risk": risk,
                "prevention": prevention,
            })
        
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
else:
    # LANDING PAGE
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">🎤</div>
                <h2 style="color: #60a5fa; font-size: 1.8rem; margin-bottom: 1rem;">Ready to Analyze?</h2>
                <p style="color: rgba(226, 232, 240, 0.7); font-size: 1.1rem; line-height: 1.8;">
                    Upload an audio file to instantly detect if it's a real human voice or an AI-generated clone.
                    <br/><br/>
                    <strong style="color: #60a5fa;">Advanced AI Detection • Real-time Analysis • High Accuracy</strong>
                </p>
                <div style="margin-top: 2rem; display: flex; gap: 1rem; justify-content: center;">
                    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(30, 41, 59, 0.95) 100%); 
                                border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1rem; 
                                flex: 1; backdrop-filter: blur(10px);">
                        <div style="font-size: 1.5rem;">📊</div>
                        <p style="color: #60a5fa; font-weight: 600;">86.46%</p>
                        <p style="color: rgba(226, 232, 240, 0.7); font-size: 0.9rem;">Model Accuracy</p>
                    </div>
                    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(30, 41, 59, 0.95) 100%); 
                                border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1rem; 
                                flex: 1; backdrop-filter: blur(10px);">
                        <div style="font-size: 1.5rem;">⚡</div>
                        <p style="color: #60a5fa; font-weight: 600;">Real-time</p>
                        <p style="color: rgba(226, 232, 240, 0.7); font-size: 0.9rem;">Instant Results</p>
                    </div>
                    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(30, 41, 59, 0.95) 100%); 
                                border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1rem; 
                                flex: 1; backdrop-filter: blur(10px);">
                        <div style="font-size: 1.5rem;">🔒</div>
                        <p style="color: #60a5fa; font-weight: 600;">Secure</p>
                        <p style="color: rgba(226, 232, 240, 0.7); font-size: 0.9rem;">Privacy First</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
