# Risk and Prevention Engine

This module converts a raw risk score into a practical action plan for operators and systems.

## Decision states
- ALLOW: low-risk traffic can continue under the normal process
- VERIFY: additional authentication or human review is required
- BLOCK: sensitive actions are stopped and manual verification is enforced

## Example

```python
from prevention.prevention_engine import build_prevention_plan

plan = build_prevention_plan({
    "risk_level": "HIGH",
    "risk_score": 92,
    "ai_probability": 0.92,
})

print(plan["status"])  # BLOCK
```

The engine is intentionally conservative: if the AI probability or risk score passes the confidence threshold, it prevents risky actions before they complete.
