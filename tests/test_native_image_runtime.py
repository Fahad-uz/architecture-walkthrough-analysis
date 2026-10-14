from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


_RESIZE_WORKER = """
import os
import sys

if os.name == "posix":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import cv2
import numpy as np

source = np.load(sys.argv[1], allow_pickle=False)
width, height = map(int, sys.argv[3:])
result = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
np.save(sys.argv[2], result, allow_pickle=False)
"""


@pytest.mark.parametrize(
    ("source_shape", "target_shape"),
    [
        # RapidOCR's detector rounding triggered a native ARM resize crash here.
        ((1026, 1580), (1024, 1568)),
        ((733, 1101), (736, 1088)),
        ((37, 53), (61, 79)),
    ],
)
def test_native_rgb_resize_exits_cleanly_and_preserves_bilinear_pixels(
    tmp_path: Path,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> None:
    source = np.random.default_rng(42).integers(
        0, 256, (*source_shape, 3), dtype=np.uint8,
    )
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "output.npy"
    np.save(input_path, source, allow_pickle=False)
    height, width = target_shape
    completed = subprocess.run(
        [sys.executable, "-c", _RESIZE_WORKER, str(input_path), str(output_path),
         str(width), str(height)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, (
        f"Native resize exited with {completed.returncode}: {completed.stderr}"
    )
    actual = np.load(output_path, allow_pickle=False)
    assert actual.shape == (*target_shape, 3)
    assert actual.dtype == np.uint8

    # Independently interpolate a grid including all edges. OpenCV's uint8
    # fixed-point rounding may differ from the floating-point reference by one.
    rows = np.unique(np.linspace(0, height - 1, 65, dtype=int))
    columns = np.unique(np.linspace(0, width - 1, 65, dtype=int))
    src_height, src_width = source_shape
    y = np.clip((rows + 0.5) * src_height / height - 0.5, 0, src_height - 1)
    x = np.clip((columns + 0.5) * src_width / width - 0.5, 0, src_width - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, src_height - 1), np.minimum(x0 + 1, src_width - 1)
    wx, wy = (x - x0)[None, :, None], (y - y0)[:, None, None]
    top = source[y0[:, None], x0] * (1 - wx) + source[y0[:, None], x1] * wx
    bottom = source[y1[:, None], x0] * (1 - wx) + source[y1[:, None], x1] * wx
    expected = np.rint(top * (1 - wy) + bottom * wy).astype(np.int16)
    observed = actual[rows[:, None], columns].astype(np.int16)
    assert np.max(np.abs(observed - expected)) <= 1
