"""Disposable playback copies, keyed by the immutable original file identity."""
import hashlib
import json
from pathlib import Path


def preview_path(root: Path, source: Path) -> Path:
    stat = source.stat()
    identity = [str(source.resolve()), stat.st_size, stat.st_mtime_ns, "960px-25fps-v1"]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return root / "runs" / "live_previews" / f"{key}.mp4"
