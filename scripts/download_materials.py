"""Verify or restore the small, CC0 surface library shipped with the app."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify files without downloading")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "assets/download-manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root / "assets/textures"):
            raise ValueError(f"material destination outside assets/textures: {item['path']}")
        valid = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        if valid:
            print(f"OK {item['path']}")
            continue
        if args.check:
            failures.append(item["path"])
            continue
        if not item["url"].startswith("https://dl.polyhaven.org/file/ph-assets/Textures/"):
            raise ValueError("unsupported material source")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "ArchitectureWalkthrough/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(item["bytes"] + 1)
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"download verification failed: {item['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(path)
        print(f"Downloaded {item['path']} ({len(data):,} bytes)")
    if failures:
        raise SystemExit("Missing or modified materials: " + ", ".join(failures))


if __name__ == "__main__":
    main()
