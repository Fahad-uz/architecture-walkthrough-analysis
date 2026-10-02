from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/benchmark/run_baseline.py"
spec = importlib.util.spec_from_file_location("baseline_runner", MODULE_PATH)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def manifest(tmp_path: Path) -> Path:
    image = tmp_path / "blank.png"
    Image.new("RGB", (120, 120), "white").save(image)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "samples": [
        {"id": "sample/one", "split": "development", "image": {
            "path": image.name, "sha256": runner.checksum(image),
        }},
        # Missing held-out data must not be read by a development run.
        {"id": "sample/untouched", "split": "quarantine", "image": {
            "path": "do-not-read.png", "sha256": "unknown",
        }},
    ]}))
    return path


def test_development_selection_does_not_open_quarantined_images(tmp_path: Path) -> None:
    selected = runner.load_samples(manifest(tmp_path), "development")
    assert [entry["id"] for entry in selected] == ["sample/one"]


@pytest.mark.parametrize("split", ["quarantine", "test", "evaluation"])
def test_evaluation_data_cannot_be_used_for_tuning(tmp_path: Path, split: str) -> None:
    with pytest.raises(ValueError, match="evaluation stays untouched"):
        runner.load_samples(manifest(tmp_path), split)


def test_changed_input_is_rejected_before_analysis(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    (tmp_path / "blank.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.load_samples(path, "development")


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["samples"].append(payload["samples"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique"):
        runner.load_samples(path, "development")


def test_config_cannot_enable_remote_ai_via_environment(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("ai:\n  gemini_enabled: true\n  gemini_sanity_check_enabled: true\n")
    monkeypatch.setenv("ARCH_WALK_GEMINI_ENABLED", "true")
    monkeypatch.setenv("ARCH_WALK_AI_CHECKPOINT", "unexpected.pt")
    actual = runner.effective_config(config)
    assert actual["ai"]["gemini_enabled"] is False
    assert actual["ai"]["gemini_sanity_check_enabled"] is False
    assert actual["ai"]["segmentation_checkpoint"] is None


def test_external_checkpoint_requires_a_separate_experiment(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("ai:\n  segmentation_checkpoint: unexpected.pt\n")
    with pytest.raises(ValueError, match="without external checkpoints"):
        runner.effective_config(config)


def test_worker_that_hangs_after_writing_result_is_still_timeout(tmp_path: Path) -> None:
    (tmp_path / "worker-result.json").write_text('{"status":"completed"}')
    actual = runner.execute_worker([sys.executable, "-c", "import time; time.sleep(10)"],
                                   tmp_path, 0.1)
    assert actual["status"] == "timeout"
    assert actual["peak_memory_bytes"] is None
    assert actual["runtime_seconds"] < 5


def test_crashed_worker_cannot_publish_success(tmp_path: Path) -> None:
    (tmp_path / "worker-result.json").write_text('{"status":"completed"}')
    actual = runner.execute_worker([sys.executable, "-c", "raise SystemExit(2)"], tmp_path, 5)
    assert actual["status"] == "failed"
    assert actual["worker_exit_code"] == 2


def test_run_records_real_pipeline_failure_without_accuracy_claim(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("ocr:\n  enabled: false\n")
    output = tmp_path / "results"
    report = runner.run(path, output, config, "development", 20)
    assert report["mode"] == "automatic_unedited"
    assert report["status"] == "finished"
    assert report["summary"] == {"failed": 1}
    result = report["results"][0]
    assert "no structural walls" in result["error"]
    assert result["peak_memory_bytes"] > 0
    assert len(report["provenance"]["config_sha256"]) == 64
    assert "not accuracy measurements" in report["accuracy_claim"]
    assert json.loads((output / "report.json").read_text())["results"] == report["results"]
    with pytest.raises(FileExistsError):
        runner.run(path, output, config, "development", 20)


def test_resume_preserves_results_and_rejects_changed_experiments(tmp_path: Path, monkeypatch) -> None:
    path = manifest(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("ocr:\n  enabled: false\n")
    output = tmp_path / "results"
    calls = []

    def fake_worker(command, directory, timeout):
        calls.append(directory)
        return {"status": "completed", "runtime_seconds": 1.0}

    monkeypatch.setattr(runner, "execute_worker", fake_worker)
    original = runner.run(path, output, config, "development", 20)
    resumed = runner.run(path, output, config, "development", 20, resume=True)
    assert len(calls) == 1
    assert resumed["results"] == original["results"]
    with pytest.raises(ValueError, match="changed experiment: timeout_seconds"):
        runner.run(path, output, config, "development", 30, resume=True)
    config.write_text("ocr:\n  enabled: true\n")
    with pytest.raises(ValueError, match="changed experiment: config_sha256"):
        runner.run(path, output, config, "development", 20, resume=True)


def test_interrupted_run_resumes_only_unfinished_sample(tmp_path: Path, monkeypatch) -> None:
    path = manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["samples"].append({**payload["samples"][0], "id": "sample/two"})
    path.write_text(json.dumps(payload))
    config = tmp_path / "config.yaml"
    config.write_text("ocr:\n  enabled: false\n")
    output = tmp_path / "results"
    calls = []

    def fake_worker(command, directory, timeout):
        calls.append(directory)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return {"status": "completed", "runtime_seconds": 1.0}

    monkeypatch.setattr(runner, "execute_worker", fake_worker)
    with pytest.raises(KeyboardInterrupt):
        runner.run(path, output, config, "development", 20)
    partial = json.loads((output / "report.json").read_text())
    assert len(partial["results"]) == 1
    resumed = runner.run(path, output, config, "development", 20, resume=True)
    assert len(calls) == 3
    assert calls[-1].name.endswith("attempt-2")
    assert resumed["summary"] == {"completed": 2}
    assert len({item["id"] for item in resumed["results"]}) == 2
