from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from architecture_walkthrough.vision.preprocessing import preprocess_image, resize_preserving_aspect


def test_resize_preserves_aspect() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_preserving_aspect(image, max_side=100)
    assert resized.shape[:2] == (50, 100)


def test_preprocessing_writes_debug_outputs(tmp_path: Path) -> None:
    path = tmp_path / "plan.png"
    Image.new("RGB", (80, 60), "white").save(path)
    result = preprocess_image(path, tmp_path / "debug")
    assert result.edges_path.exists()
    assert result.grayscale_path.exists()
    assert result.cleaned_path.exists()
