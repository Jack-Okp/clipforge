"""Cut clips with FFmpeg, using the NVIDIA encoder when it is available."""
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in out.stdout
    except FileNotFoundError:
        return False


def _cmd(src: Path, dst: Path, start: float, end: float, nvenc: bool) -> list[str]:
    video = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21"] if nvenc else \
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    return [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        *video, "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst),
    ]


def cut_clip(src: Path, dst: Path, start: float, end: float, use_nvenc: str = "auto") -> str:
    """Cut one clip. Returns the encoder used. Falls back to CPU if the GPU encode fails."""
    want_gpu = use_nvenc == "on" or (use_nvenc == "auto" and nvenc_available())
    if want_gpu:
        r = subprocess.run(_cmd(src, dst, start, end, True), capture_output=True)
        if r.returncode == 0:
            return "h264_nvenc"
    r = subprocess.run(_cmd(src, dst, start, end, False), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {r.stderr[-400:]}")
    return "libx264"
