"""Acquire a small, reproducible, noncommercial CubiCasa5K evaluation subset.

Downloads ZIP members with verified HTTP byte ranges; never downloads the full
5.5 GB archive. Assets remain in ignored outputs/benchmark-data. Official test
examples are quarantined, not a certified independent final evaluation split.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ARCHIVE_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
RECORD_URL = "https://zenodo.org/records/2613548"
LICENSE_URL = "https://raw.githubusercontent.com/CubiCasa/CubiCasa5k/master/LICENSE"
STYLES = ("colorful", "high_quality", "high_quality_architectural")
DEFAULT_ROOT = Path("outputs/benchmark-data/cubicasa5k")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class RangeArchive(io.RawIOBase):
    """Minimal seekable ZIP reader with strict range and download size checks."""

    def __init__(self, url: str = ARCHIVE_URL) -> None:
        self.url = url
        self.position = 0
        self.size = 0
        self.transferred_bytes = 0
        self.requests = 0
        self.range(0, 1)

    def range(self, start: int, length: int) -> bytes:
        if length < 0 or length > 40 * 1024 * 1024:
            raise ValueError("Refusing an unbounded or oversized range")
        if not length:
            return b""
        expected_prefix = f"bytes {start}-{start + length - 1}/"
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{start + length - 1}",
                "User-Agent": "PlanStride-research-subset/1.0",
            },
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not content_range.startswith(expected_prefix):
                        raise ValueError("Server did not honor the requested byte range")
                    archive_size = int(content_range.rsplit("/", 1)[1])
                    if self.size and self.size != archive_size:
                        raise ValueError("Remote archive size changed during acquisition")
                    self.size = archive_size
                    data = response.read(length + 1)
                    if len(data) != length:
                        raise ValueError("Truncated or oversized range response")
                    self.transferred_bytes += len(data)
                    self.requests += 1
                    return data
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(min(60, int(error.headers.get("Retry-After", "10"))))
        raise AssertionError("Retry loop exhausted")

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (self.position if whence == 1 else self.size if whence == 2 else 0) + offset
        if self.position < 0:
            raise ValueError("Negative archive position")
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        size = self.size - self.position if size < 0 else size
        result = self.range(self.position, size)
        self.position += len(result)
        return result


def read_member(archive: RangeArchive, info: zipfile.ZipInfo, end: int) -> bytes:
    """Fetch one local record, decompress it, and verify its ZIP checksum."""
    if info.file_size > 40 * 1024 * 1024:
        raise ValueError("Oversized member")
    record = archive.range(info.header_offset, end - info.header_offset)
    header = struct.unpack("<4s5H3I2H", record[:30])
    if header[0] != b"PK\x03\x04" or header[2] & 1:
        raise ValueError("Invalid or encrypted member")
    offset = 30 + header[-2] + header[-1]
    payload = record[offset : offset + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(payload, info.file_size + 1)
        if not decoder.eof or decoder.unconsumed_tail:
            raise ValueError("Invalid or oversized compressed member")
    elif info.compress_type == zipfile.ZIP_STORED:
        data = payload
    else:
        raise ValueError("Unsupported compression")
    if len(data) != info.file_size or zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise ValueError("Member size or CRC mismatch")
    return data


def select_groups(lines: list[str], count: int, excluded: set[str]) -> list[str]:
    """Deterministic style-balanced sampling, preserving source directory groups."""
    groups: dict[str, list[str]] = {style: [] for style in STYLES}
    for line in lines:
        parts = line.strip("/ \r\n").split("/")
        if len(parts) != 2 or parts[0] not in groups or not parts[1].isdigit():
            raise ValueError(f"Unexpected source group: {line!r}")
        group = "/".join(parts)
        if parts[1] not in excluded:
            groups[parts[0]].append(group)
    for values in groups.values():
        values[:] = sorted(set(values), key=lambda value: sha256(value.encode()))
    chosen: list[str] = []
    seen = set(excluded)
    while len(chosen) < count:
        previous = len(chosen)
        for style in STYLES:
            while groups[style]:
                group = groups[style].pop(0)
                group_id = group.rsplit("/", 1)[-1]
                if group_id not in seen:
                    chosen.append(group)
                    seen.add(group_id)
                    break
            if len(chosen) == count:
                break
        if len(chosen) == previous:
            raise ValueError("Insufficient distinct source groups")
    return chosen


def image_fingerprints(path: Path) -> dict[str, Any]:
    """Duplicate screening only; no semantic examination of quarantined images."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        pixels = image.tobytes()
        pixel_digest = sha256(f"{width}x{height}:".encode() + pixels)
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(small.get_flattened_data())
        bits = 0
        for row in range(8):
            for column in range(8):
                index = row * 9 + column
                bits = (bits << 1) | (values[index] > values[index + 1])
    return {"width": width, "height": height, "pixel_sha256": pixel_digest, "dhash": f"{bits:016x}"}


def duplicate_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    exact: list[list[str]] = []
    near: list[dict[str, Any]] = []
    for index, first in enumerate(samples):
        for second in samples[index + 1 :]:
            if first["image"]["pixel_sha256"] == second["image"]["pixel_sha256"]:
                exact.append([first["id"], second["id"]])
            distance = (
                int(first["image"]["dhash"], 16) ^ int(second["image"]["dhash"], 16)
            ).bit_count()
            if distance <= 6:
                near.append({"ids": [first["id"], second["id"]], "dhash_distance": distance})
    return {
        "exact_pixel_duplicates": exact,
        "near_duplicate_candidates": near,
        "near_duplicate_method": "64-bit difference hash, Hamming distance <= 6",
        "near_duplicate_review_complete": False,
        "limitation": "Hash screening does not establish building or source-family independence.",
    }


def acquire(root: Path, development_count: int, quarantine_count: int) -> dict[str, Any]:
    if development_count < 1 or quarantine_count < 1:
        raise ValueError("Both sample counts must be positive")
    root.mkdir(parents=True, exist_ok=True)
    archive = RangeArchive()
    with zipfile.ZipFile(archive) as source_zip:
        infos = sorted(source_zip.infolist(), key=lambda info: info.header_offset)
        by_name = {info.filename: info for info in infos}
        ends = {
            info.filename: infos[index + 1].header_offset
            if index + 1 < len(infos)
            else source_zip.start_dir
            for index, info in enumerate(infos)
        }
        split_data = {}
        inventory: list[dict[str, Any]] = []
        for split in ("train", "test"):
            member = f"cubicasa5k/{split}.txt"
            data = read_member(archive, by_name[member], ends[member])
            path = root / f"official-{split}.txt"
            path.write_bytes(data)
            split_data[split] = data.decode().splitlines()
            inventory.append({"path": path.name, "archive_member": member,
                              "sha256": sha256(data), "bytes": len(data)})
        development = select_groups(split_data["train"], development_count, set())
        excluded = {group.rsplit("/", 1)[-1] for group in development}
        quarantine = select_groups(split_data["test"], quarantine_count, excluded)
        samples: list[dict[str, Any]] = []
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "name": "CubiCasa5K", "version": "1.0", "record_url": RECORD_URL,
                "archive_url": ARCHIVE_URL, "archive_bytes": archive.size,
                "published_archive_md5": "0ce0b203d1e3c125b51087b219bd23b9",
                "whole_archive_checksum_verified": False,
                "license": "CC-BY-NC-4.0", "license_url": LICENSE_URL,
                "attribution": "Kalervo, Ylioinas, Haikio, Karhu and Kannala, CubiCasa5K (2019)",
                "use": "Local noncommercial evaluation only; not bundled or redistributed",
            },
            "split_status": "provisional_group_independence_unverified",
            "grouping": {
                "directory_groups_disjoint": True,
                "building_independence_verified": False,
                "source_family_independence_verified": False,
                "policy": "Official train -> development; official test -> quarantine. "
                          "One F1 image per distinct numeric source-directory ID. Style folders "
                          "are sampling strata, not independent source families.",
            },
            "quarantine_policy": "Do not run reconstruction, view predictions, or tune using "
                                 "quarantined examples. Final evaluation requires independent "
                                 "group and annotation review first.",
            "samples": samples,
            "downloads": inventory,
            "acquisition_complete": False,
        }
        for split, groups in (("development", development), ("quarantine", quarantine)):
            for group in groups:
                sample: dict[str, Any] = {
                    "id": f"cubicasa5k/{group}", "split": split,
                    "official_split": "train" if split == "development" else "test",
                    "source_group": group, "style": group.split("/")[0],
                    "source_family": None, "building_id": None,
                    "license": "CC-BY-NC-4.0", "ground_truth_reviewed": False,
                    "annotation_status": "upstream SVG; not converted or independently reviewed",
                }
                for kind, filename in (("image", "F1_scaled.png"), ("annotation", "model.svg")):
                    member = f"cubicasa5k/{group}/{filename}"
                    info = by_name[member]
                    relative = Path("samples") / group / filename
                    path = root / relative
                    # Cache integrity is checked against the upstream ZIP CRC and size.
                    data = path.read_bytes() if path.exists() else b""
                    if len(data) != info.file_size or zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
                        data = read_member(archive, info, ends[member])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    record = {"path": relative.as_posix(), "archive_member": member,
                              "sha256": sha256(data), "bytes": len(data),
                              "zip_crc32": f"{info.CRC:08x}"}
                    sample[kind] = dict(record)
                    inventory.append(record)
                sample["image"].update(image_fingerprints(root / sample["image"]["path"]))
                samples.append(sample)
                atomic_json(root / "manifest.json", manifest)
                print(f"Saved {len(samples)}/{development_count + quarantine_count}: {sample['id']}",
                      flush=True)
                time.sleep(0.6)
    with urllib.request.urlopen(LICENSE_URL, timeout=30) as response:
        license_data = response.read(64 * 1024)
    if b"Attribution-NonCommercial 4.0" not in license_data:
        raise ValueError("Upstream license changed; review before using downloaded data")
    (root / "LICENSE-CubiCasa5K.txt").write_bytes(license_data)
    inventory.append({"path": "LICENSE-CubiCasa5K.txt", "url": LICENSE_URL,
                      "sha256": sha256(license_data), "bytes": len(license_data)})
    manifest["duplicate_screening"] = duplicate_report(samples)
    manifest["counts"] = dict(Counter(sample["split"] for sample in samples))
    manifest["downloaded_file_bytes"] = sum(item["bytes"] for item in inventory)
    manifest["archive_range_bytes_transferred_this_run"] = archive.transferred_bytes
    manifest["archive_requests_this_run"] = archive.requests
    manifest["acquisition_complete"] = True
    atomic_json(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--development", type=int, default=70)
    parser.add_argument("--quarantine", type=int, default=30)
    args = parser.parse_args()
    manifest = acquire(args.root, args.development, args.quarantine)
    print(json.dumps({key: manifest[key] for key in (
        "counts", "downloaded_file_bytes", "archive_range_bytes_transferred_this_run",
        "duplicate_screening", "split_status",
    )}, indent=2))


if __name__ == "__main__":
    main()
