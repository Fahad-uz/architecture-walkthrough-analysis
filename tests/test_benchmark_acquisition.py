"""Protect dataset provenance and leakage checks without network downloads."""

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "benchmark" / "acquire_cubicasa.py"
SPEC = importlib.util.spec_from_file_location("acquire_cubicasa", SCRIPT)
assert SPEC and SPEC.loader
acquisition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquisition)


def test_selection_balances_styles_and_keeps_numeric_source_groups_together():
    lines = [f"/{style}/{number}/" for style in acquisition.STYLES for number in range(20)]
    first = acquisition.select_groups(lines, 9, {"0", "1"})
    second = acquisition.select_groups(list(reversed(lines)), 9, {"0", "1"})
    assert first == second
    assert len({group.split("/")[1] for group in first}) == 9
    assert all(group.split("/")[1] not in {"0", "1"} for group in first)
    assert {style: sum(group.startswith(style + "/") for group in first)
            for style in acquisition.STYLES} == dict.fromkeys(acquisition.STYLES, 3)


def test_selection_does_not_silently_reuse_groups_or_accept_paths():
    with pytest.raises(ValueError, match="Insufficient"):
        acquisition.select_groups(["high_quality/1", "colorful/1"], 2, set())
    with pytest.raises(ValueError, match="Unexpected"):
        acquisition.select_groups(["../../escape"], 1, set())


def test_zip_member_checksum_rejects_corruption():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("sample.txt", b"original sample")
    raw = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.getinfo("sample.txt")
        end = archive.start_dir

    class ByteRanges:
        data = raw

        def range(self, start, length):
            return self.data[start:start + length]

    remote = ByteRanges()
    assert acquisition.read_member(remote, info, end) == b"original sample"
    remote.data = raw.replace(b"original", b"modified")
    with pytest.raises(ValueError, match="CRC mismatch"):
        acquisition.read_member(remote, info, end)


def test_duplicate_screening_reports_cross_split_matches_without_certifying_independence():
    samples = [
        {"id": "a", "split": "development",
         "image": {"pixel_sha256": "same", "dhash": "0000000000000000"}},
        {"id": "b", "split": "quarantine",
         "image": {"pixel_sha256": "same", "dhash": "0000000000000000"}},
        {"id": "c", "split": "quarantine",
         "image": {"pixel_sha256": "different", "dhash": "0000000000000001"}},
    ]
    result = acquisition.duplicate_report(samples)
    assert result["exact_pixel_duplicates"] == [["a", "b"]]
    assert len(result["near_duplicate_candidates"]) == 3
    assert result["near_duplicate_review_complete"] is False


def test_range_reader_rejects_full_archive_response(monkeypatch):
    class FullResponse:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _):
            pytest.fail("Must reject a full download before reading its body")

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", lambda *a, **k: FullResponse())
    with pytest.raises(ValueError, match="did not honor"):
        acquisition.RangeArchive()
