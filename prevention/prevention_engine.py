def build_prevention_plan(risk_result):
    """Turn risk scores into a concrete prevention decision for the application."""
    risk_level = str(risk_result.get("risk_level", "LOW")).upper()
    risk_score = float(risk_result.get("risk_score", 0.0))
    ai_probability = float(risk_result.get("ai_probability", 0.0))

    if risk_level == "HIGH" or risk_score >= 75 or ai_probability >= 0.8:
        return {
            "status": "BLOCK",
            "message": (
                "High risk: a likely AI-generated or cloned voice was detected. "
                "Block the transaction and verify the caller through a trusted secondary channel."
            ),
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "actions": [
                {"type": "block", "label": "Block sensitive action", "reason": "High synthetic voice probability."},
                {"type": "verify", "label": "Request secondary verification", "reason": "Use a trusted callback or out-of-band identity check."},
                {"type": "alert", "label": "Escalate to security team", "reason": "Possible impersonation or deepfake attack."},
                {"type": "log", "label": "Persist security event", "reason": "Keep an auditable record for investigation."},
            ],
        }

    if risk_level == "MEDIUM" or risk_score >= 40 or ai_probability >= 0.45:
        return {
            "status": "VERIFY",
            "message": (
                "Medium risk: additional verification is required before approving sensitive actions. "
                "Continue with a human review or a secondary authentication step."
            ),
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "actions": [
                {"type": "verify", "label": "Perform extra identity checks", "reason": "Suspicious voice characteristics were observed."},
                {"type": "review", "label": "Escalate to a human agent", "reason": "Manual review is recommended before proceeding."},
                {"type": "monitor", "label": "Continue monitoring the call", "reason": "Collect more evidence while the interaction is active."},
            ],
        }

    return {
        "status": "ALLOW",
        "message": (
            "Low risk: continue the standard workflow and monitor for unusual behavior. "
            "No immediate synthetic voice risk was detected."
        ),
        "risk_level": risk_level,
        "risk_score": round(risk_score, 2),
        "actions": [
            {"type": "monitor", "label": "Continue normal workflow", "reason": "The risk remains within acceptable levels."},
            {"type": "log", "label": "Keep standard audit trail", "reason": "Record the interaction for routine compliance checks."},
        ],
    }
