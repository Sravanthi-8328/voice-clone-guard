from prevention.prevention_engine import build_prevention_plan


def test_high_risk_plan_blocks_sensitive_actions():
    plan = build_prevention_plan({
        "risk_level": "HIGH",
        "risk_score": 92,
        "ai_probability": 0.92,
    })

    assert plan["status"] == "BLOCK"
    assert "verify" in plan["message"].lower()
    assert "block" in plan["message"].lower()
    assert len(plan["actions"]) >= 3


def test_low_risk_plan_allows_standard_flow():
    plan = build_prevention_plan({
        "risk_level": "LOW",
        "risk_score": 18,
        "ai_probability": 0.08,
    })

    assert plan["status"] == "ALLOW"
    assert "continue" in plan["message"].lower()
    assert any(action["type"] == "monitor" for action in plan["actions"])
