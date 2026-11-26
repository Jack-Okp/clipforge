"""Get a video into the project folder, from a link (yt-dlp) or an uploaded file."""
import json
import subprocess
from pathlib import Path
from typing import Callable


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, ValueError, KeyError, TypeError) as e:
        raise RuntimeError("FFmpeg could not read this file. Is it a valid video?") from e


def download_url(url: str, project_dir: Path, on_progress: Callable[[float], None]) -> tuple[Path, str]:
    """Download a video with yt-dlp. Returns (file path, title)."""
    import yt_dlp

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    opts = {
        "outtmpl": str(project_dir / "source.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():  # merged output may use .mp4 even if the prepared name differs
        path = next(project_dir.glob("source.*"))
    return path, info.get("title") or path.stem
