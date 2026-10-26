"""Create independent, verified ZIP assets for the local datasets and predictions.

Run from the project root: python -m scripts.create_github_backup
Only Python's standard library is required. No project data are modified.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / ".github-backup" / "assets"
MAX_PART_BYTES = 800 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    datasets = [ROOT / name for name in ("trans_t1_a", "trans_t2_b")]
    for dataset in datasets:
        if not dataset.is_dir():
            raise SystemExit(f"Missing dataset: {dataset}")
    files = sorted(
        [path for folder in datasets for path in folder.rglob("*") if path.is_file()]
        + [path for path in (ROOT / "outputs").rglob("*")
           if path.is_file() and path.suffix.lower() in {".npy", ".npz"}],
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    groups: list[list[Path]] = []
    size = 0
    for path in files:
        length = path.stat().st_size
        if length > MAX_PART_BYTES:
            raise SystemExit(f"File exceeds archive target size: {path}")
        if not groups or size + length > MAX_PART_BYTES:
            groups.append([])
            size = 0
        groups[-1].append(path)
        size += length

    DESTINATION.mkdir(parents=True, exist_ok=True)
    if any(DESTINATION.iterdir()):
        raise SystemExit(f"Destination is not empty; preserve or move existing assets: {DESTINATION}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "independent ZIP archives; extract every part into repository root",
        "scope": "all files in trans_t1_a and trans_t2_b; all outputs .npy/.npz files",
        "excluded": ["Python environments", "model cache", "bytecode", "credentials"],
        "file_count": len(files),
        "original_bytes": sum(path.stat().st_size for path in files),
        "archives": [],
    }
    for index, group in enumerate(groups, 1):
        name = f"bn5211-data-part-{index:02d}-of-{len(groups):02d}.zip"
        archive = DESTINATION / name
        entries = []
        with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=1) as output:
            for path in group:
                relative = path.relative_to(ROOT).as_posix()
                before = path.stat()
                checksum = sha256(path)
                output.write(path, relative)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed during backup: {relative}")
                entries.append({"path": relative, "bytes": after.st_size, "sha256": checksum})
        with ZipFile(archive) as check:
            if check.namelist() != [entry["path"] for entry in entries]:
                raise RuntimeError(f"Archive file list mismatch: {archive}")
            for entry in entries:
                digest = hashlib.sha256()
                with check.open(entry["path"]) as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"Archive content mismatch: {entry['path']}")
        if archive.stat().st_size >= 2 * 1024**3:
            raise RuntimeError(f"Release asset exceeds GitHub size limit: {archive}")
        manifest["archives"].append({
            "name": name, "bytes": archive.stat().st_size,
            "sha256": sha256(archive), "files": entries,
        })
        print(f"Verified {name}: {len(entries)} files, {archive.stat().st_size / 1024**2:.1f} MiB", flush=True)

    (DESTINATION / "backup-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (DESTINATION / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['name']}\n" for item in manifest["archives"]),
        encoding="utf-8",
    )
    print(f"Complete: {len(files)} files in {len(groups)} verified archives", flush=True)


if __name__ == "__main__":
    main()
