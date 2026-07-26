from pathlib import Path

from architecture_walkthrough.config import load_config


def test_environment_selects_default_config_file(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "container.yaml"
    config_path.write_text("defaults:\n  wall_height_m: 2.75\n", encoding="utf-8")
    monkeypatch.setenv("ARCH_WALK_CONFIG", str(config_path))

    config = load_config()

    assert config.defaults.wall_height_m == 2.75


def test_explicit_config_path_overrides_environment(tmp_path: Path, monkeypatch) -> None:
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text("defaults:\n  wall_height_m: 2.5\n", encoding="utf-8")
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text("defaults:\n  wall_height_m: 3.25\n", encoding="utf-8")
    monkeypatch.setenv("ARCH_WALK_CONFIG", str(environment_path))

    config = load_config(explicit_path)

    assert config.defaults.wall_height_m == 3.25


def test_runtime_tool_and_gemini_environment_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARCH_WALK_BLENDER_PATH", "C:/tools/blender.exe")
    monkeypatch.setenv("ARCH_WALK_FFMPEG_PATH", "C:/tools/ffmpeg.exe")
    monkeypatch.setenv("ARCH_WALK_GEMINI_ENABLED", "false")
    monkeypatch.setenv("ARCH_WALK_GEMINI_MODEL", "gemini-test")

    config = load_config(tmp_path / "missing.yaml")

    assert config.paths.blender_executable == "C:/tools/blender.exe"
    assert config.paths.ffmpeg_executable == "C:/tools/ffmpeg.exe"
    assert config.ai.gemini_enabled is False
    assert config.ai.gemini_model == "gemini-test"


def test_analysis_capacity_and_timeout_are_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "limits.yaml"
    config_path.write_text(
        "limits:\n"
        "  max_concurrent_analyses: 2\n"
        "  processing_timeout_seconds: 37\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.limits.max_concurrent_analyses == 2
    assert config.limits.processing_timeout_seconds == 37
