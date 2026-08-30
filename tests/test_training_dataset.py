import importlib.util
import os
import sys


def load_train_module():
    train_path = os.path.join(os.path.dirname(__file__), "..", "ai-detector", "train.py")
    module_dir = os.path.dirname(train_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("voice_train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_max_samples_uses_full_dataset(monkeypatch):
    monkeypatch.delenv("MAX_SAMPLES", raising=False)
    module = load_train_module()
    assert module.resolve_max_samples() is None
