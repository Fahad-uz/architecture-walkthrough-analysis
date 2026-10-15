from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from architecture_walkthrough.vision.providers import CubiCasaGeometryProvider


def test_unimplemented_provider_never_fetches_or_promises_to_enable_weights(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/setup_cubicasa.py"
    spec = importlib.util.spec_from_file_location("setup_cubicasa_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=lambda *a, **k: calls.append(a)))
    monkeypatch.setattr(CubiCasaGeometryProvider, "inference_implemented", False)
    monkeypatch.setattr(module, "CHECKPOINT_DIR", tmp_path / "weights")
    monkeypatch.setattr(module, "CHECKPOINT", tmp_path / "weights/model.pkl")
    assert module.main() == 1
    assert calls == []
    assert not (tmp_path / "weights").exists()
    assert "inference is not implemented" in capsys.readouterr().out
