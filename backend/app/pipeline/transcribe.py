"""Speech to text with word-level timestamps (faster-whisper, runs locally)."""
import json
import subprocess
from pathlib import Path
from typing import Callable


def extract_audio(video: Path, wav: Path) -> None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        err = r.stderr or ""
        if "does not contain any stream" in err or "matches no streams" in err:
            raise RuntimeError("This video has no audio track, so there is nothing to transcribe.")
        raise RuntimeError(f"FFmpeg could not extract the audio: {err.strip()[-300:] or 'unknown error'}")


def _make_model(size: str, device: str):
    from faster_whisper import WhisperModel

    if device == "cuda":
        # int8_float16 keeps 'small'/'medium' comfortably inside 4 GB of VRAM
        return WhisperModel(size, device="cuda", compute_type="int8_float16")
    return WhisperModel(size, device="cpu", compute_type="int8")


# Whisper tends to tidy "um" and "uh" away. A prompt that contains them makes it keep them,
# which is what lets the filler remover find them later.
FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."


def _run(model, wav: Path, keep_fillers: bool, on_progress: Callable[[float], None]):
    # The model is evaluated lazily: missing CUDA libraries often only show up here, on the
    # first real work, not when the model is created. So the whole run sits inside the caller's try.
    segments_iter, info = model.transcribe(
        str(wav), word_timestamps=True, vad_filter=True, beam_size=5,
        initial_prompt=FILLER_PROMPT if keep_fillers else None,
    )
    total = info.duration or 1.0
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": [
                {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word}
                for w in (seg.words or [])
            ],
        })
        on_progress(min(seg.end / total, 1.0))
    return segments, info


def transcribe(video: Path, out_json: Path, size: str, device: str,
               on_progress: Callable[[float], None], keep_fillers: bool = True) -> dict:
    wav = out_json.with_suffix(".wav")
    extract_audio(video, wav)
    gpu_error = None
    try:
        order = ["cuda", "cpu"] if device == "auto" else [device]
        for i, dev in enumerate(order):
            try:
                segments, info = _run(_make_model(size, dev), wav, keep_fillers, on_progress)
                used = dev
                break
            except Exception as e:
                if i == len(order) - 1:
                    if dev == "cuda":
                        raise RuntimeError(
                            f"GPU transcription failed ({str(e)[:200]}). Set Device to 'cpu' in Settings, "
                            "or run setup.bat again to install the CUDA libraries.") from e
                    raise
                gpu_error = str(e)[:300]   # GPU unusable: fall back to the CPU and say so
                on_progress(0.0)
    finally:
        wav.unlink(missing_ok=True)
    data = {"language": info.language, "duration": info.duration, "device": used, "segments": segments}
    if gpu_error:
        data["gpu_error"] = gpu_error
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data
