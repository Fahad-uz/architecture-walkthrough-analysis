from architecture_walkthrough import main as cli
from architecture_walkthrough.config import AppConfig


def test_image_to_glb_preserves_disabled_config_without_opt_in(monkeypatch) -> None:
    config = AppConfig()
    config.ai.gemini_enabled = False
    captured: dict[str, bool] = {}
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "convert_image_to_glb",
        lambda *_args, **_kwargs: captured.update(enabled=config.ai.gemini_enabled),
    )

    result = cli.main(["image-to-glb", "--input", "plan.png", "--output", "building.glb"])

    assert result == 0
    assert captured["enabled"] is False


def test_required_gemini_enables_the_provider_over_disabled_config(monkeypatch) -> None:
    config = AppConfig()
    config.ai.gemini_enabled = False
    captured: dict[str, bool] = {}
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "convert_image_to_glb",
        lambda *_args, **_kwargs: captured.update(enabled=config.ai.gemini_enabled),
    )

    result = cli.main(
        [
            "image-to-glb",
            "--input",
            "plan.png",
            "--output",
            "building.glb",
            "--require-gemini",
        ]
    )

    assert result == 0
    assert captured["enabled"] is True


def test_image_to_glb_preserves_enabled_environment_or_yaml_config(monkeypatch) -> None:
    config = AppConfig()
    config.ai.gemini_enabled = True
    captured: dict[str, bool] = {}
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "convert_image_to_glb",
        lambda *_args, **_kwargs: captured.update(enabled=config.ai.gemini_enabled),
    )

    result = cli.main(["image-to-glb", "--input", "plan.png", "--output", "building.glb"])

    assert result == 0
    assert captured["enabled"] is True


def test_no_run_blender_wins_when_both_legacy_flags_are_present(monkeypatch) -> None:
    config = AppConfig()
    captured: dict[str, bool] = {}
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "build_glb_model",
        lambda *_args, **kwargs: captured.update(run_blender=kwargs["run_blender"]),
    )

    result = cli.main(
        [
            "build-model",
            "--floorplan",
            "floorplan.json",
            "--output",
            "building.glb",
            "--use-blender",
            "--no-run-blender",
        ]
    )

    assert result == 0
    assert captured["run_blender"] is False
