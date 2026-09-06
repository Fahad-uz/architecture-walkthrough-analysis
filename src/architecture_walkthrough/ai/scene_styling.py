from __future__ import annotations

from .interfaces import DeterministicSceneStyler, SceneStyle


def default_scene_style() -> SceneStyle:
    return DeterministicSceneStyler().propose_style()
