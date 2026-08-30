def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def calculate_risk(
    ai_probability,
    prosody_score=0.0,
    acoustic_anomaly=0.0,
    context_risk=0.0,
):
    """Combine CNN, prosody, acoustic, and context signals into one actionable risk score."""

    ai_probability = _clamp(ai_probability)
    prosody_score = _clamp(prosody_score)
    acoustic_anomaly = _clamp(acoustic_anomaly)
    context_risk = _clamp(context_risk)

    combined_score = (
        0.55 * ai_probability
        + 0.25 * prosody_score
        + 0.20 * acoustic_anomaly
        + 0.10 * context_risk
    )
    combined_score = _clamp(combined_score)

    risk_score = round(combined_score * 100, 2)

    if risk_score >= 75:
        risk_level = "HIGH"
        recommendation = (
            "WARNING: Possible AI-generated or cloned voice detected. "
            "Do not approve sensitive transactions. Verify the caller through a trusted secondary channel."
        )
    elif risk_score >= 40:
        risk_level = "MEDIUM"
        recommendation = (
            "Suspicious voice characteristics detected. "
            "Perform additional identity verification before taking sensitive actions."
        )
    else:
        risk_level = "LOW"
        recommendation = (
            "No significant synthetic voice risk detected. "
            "Continue standard verification procedures."
        )

    return {
        "ai_probability": round(ai_probability, 4),
        "prosody_score": round(prosody_score, 4),
        "acoustic_anomaly": round(acoustic_anomaly, 4),
        "context_risk": round(context_risk, 4),
        "combined_score": round(combined_score, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
    }