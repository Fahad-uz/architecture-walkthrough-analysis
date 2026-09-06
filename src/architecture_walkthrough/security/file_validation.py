from __future__ import annotations

import re
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from architecture_walkthrough.config import LimitSettings


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MIME_BY_EXT = {
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".webp": {"image/webp"},
}
JOB_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")


class ValidatedImage(BaseModel):
    original_suffix: str
    safe_filename: str
    width: int
    height: int
    mime_type: str
    size_bytes: int


def safe_job_id() -> str:
    return uuid.uuid4().hex


def validate_job_id(job_id: str) -> str:
    """Accept one canonical ASCII spelling for every on-disk job directory."""

    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("invalid job id")
    return job_id


def ensure_within_directory(base: Path, target: Path) -> Path:
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    if base_resolved != target_resolved and base_resolved not in target_resolved.parents:
        raise ValueError("path traversal attempt rejected")
    return target_resolved


def create_job_dir(work_root: Path, job_id: str | None = None) -> Path:
    if job_id is None:
        job_id = safe_job_id()
    job_id = validate_job_id(job_id)
    path = ensure_within_directory(work_root, work_root / job_id)
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_image_file(path: Path, limits: LimitSettings, declared_mime: str | None = None) -> ValidatedImage:
    if not path.exists() or not path.is_file():
        raise ValueError("input file does not exist")
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported image extension: {suffix}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError("empty upload rejected")
    if size_bytes > limits.max_upload_mb * 1024 * 1024:
        raise ValueError("upload exceeds configured size limit")
    if declared_mime and declared_mime not in ALLOWED_MIME_BY_EXT[suffix]:
        raise ValueError("declared MIME type does not match extension")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            fmt = (image.format or "").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image integrity validation failed") from exc
    if fmt == "webp":
        mime = "image/webp"
    elif fmt in {"jpeg", "jpg"}:
        mime = "image/jpeg"
    elif fmt == "png":
        mime = "image/png"
    else:
        raise ValueError(f"unsupported decoded image format: {fmt}")
    if mime not in ALLOWED_MIME_BY_EXT[suffix]:
        raise ValueError("decoded image format does not match extension")
    if width > limits.max_image_width or height > limits.max_image_height:
        raise ValueError("image dimensions exceed configured limits")
    return ValidatedImage(
        original_suffix=suffix,
        safe_filename=f"{uuid.uuid4().hex}{suffix}",
        width=width,
        height=height,
        mime_type=mime,
        size_bytes=size_bytes,
    )
