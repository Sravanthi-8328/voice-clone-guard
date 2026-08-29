import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
SECURITY_LOG_FILE = os.path.join(LOG_DIR, "security_prevention.log")


def _default_thresholds() -> Dict[str, float]:
    return {
        "high": float(os.getenv("HIGH_THRESHOLD", "0.85")),
        "medium": float(os.getenv("MEDIUM_THRESHOLD", "0.60")),
    }


def _coerce_probability(value: Any, default: float = 0.0) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = default
    return max(0.0, min(1.0, probability))


def _rule_fake_probability(fake_probability: float, thresholds: Dict[str, float]) -> Dict[str, str]:
    if fake_probability >= thresholds["high"]:
        return {"risk_level": "HIGH", "action": "BLOCKED", "reason": "High probability of synthetic voice detected"}
    if fake_probability >= thresholds["medium"]:
        return {"risk_level": "MEDIUM", "action": "WARNING", "reason": "Suspicious audio characteristics detected"}
    return {"risk_level": "LOW", "action": "ALLOWED", "reason": "Audio appears to be genuine"}


def _rule_model_confidence(prediction: Dict[str, Any]) -> Dict[str, str]:
    confidence = _coerce_probability(prediction.get("confidence", 0.0))
    if confidence < 0.5:
        return {"risk_level": "LOW", "action": "ALLOWED", "reason": "Model confidence is low and the sample appears inconclusive"}
    return {}


def _rule_invalid_input(prediction: Dict[str, Any]) -> Dict[str, str]:
    if not prediction or not isinstance(prediction, dict):
        return {"risk_level": "HIGH", "action": "BLOCKED", "reason": "Invalid or unsafe input received"}
    return {}


def _log_security_event(request_id: str, prediction_label: str, fake_prob: float, risk: str, action: str):
    """Write structured security metadata without storing raw audio."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"{timestamp} | "
        f"request_id={request_id} | "
        f"prediction={prediction_label} | "
        f"fake_probability={fake_prob:.4f} | "
        f"risk={risk} | "
        f"action={action}\n"
    )
    try:
        with open(SECURITY_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
    except Exception:
        pass


def evaluate_threat(prediction: Dict[str, Any], request_id: str = None) -> Dict[str, str]:
    """Evaluate the threat level from a model prediction using prototype thresholds."""
    if not request_id:
        request_id = str(uuid.uuid4())

    if not isinstance(prediction, dict):
        prediction = {"label": "UNKNOWN", "fake_probability": 0.0, "real_probability": 0.0}

    thresholds = _default_thresholds()
    fake_probability = _coerce_probability(prediction.get("fake_probability", 0.0))
    label = str(prediction.get("label", "UNKNOWN")).upper()

    invalid_rule = _rule_invalid_input(prediction)
    if invalid_rule:
        risk_level = invalid_rule["risk_level"]
        action = invalid_rule["action"]
        reason = invalid_rule["reason"]
    else:
        rule_result = _rule_fake_probability(fake_probability, thresholds)
        risk_level = rule_result["risk_level"]
        action = rule_result["action"]
        reason = rule_result["reason"]

        confidence_rule = _rule_model_confidence(prediction)
        if confidence_rule:
            risk_level = confidence_rule["risk_level"]
            action = confidence_rule["action"]
            reason = confidence_rule["reason"]

    _log_security_event(request_id, label, fake_probability, risk_level, action)
    return {"risk_level": risk_level, "action": action, "reason": reason}


def get_recent_security_events(limit: int = 50) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not os.path.exists(SECURITY_LOG_FILE):
        return events

    with open(SECURITY_LOG_FILE, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines()[-limit:]:
            parts = [part.strip() for part in line.strip().split(" | ") if part.strip()]
            if len(parts) < 6:
                continue

            event: Dict[str, Any] = {"timestamp": parts[0]}
            for part in parts[1:]:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                if key == "fake_probability":
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0
                if key == "risk":
                    key = "risk_level"
                event[key] = value
            events.append(event)
    return events


def get_security_statistics() -> Dict[str, int]:
    events = get_recent_security_events(limit=1000)
    total = len(events)
    real_count = sum(1 for event in events if str(event.get("prediction", "")).upper() == "REAL")
    fake_count = total - real_count
    blocked = sum(1 for event in events if str(event.get("action", "")).upper() == "BLOCKED")
    warnings = sum(1 for event in events if str(event.get("action", "")).upper() == "WARNING")
    allowed = sum(1 for event in events if str(event.get("action", "")).upper() == "ALLOWED")

    return {
        "total_analyzed": total,
        "real": real_count,
        "fake": fake_count,
        "blocked": blocked,
        "warnings": warnings,
        "allowed": allowed,
    }
