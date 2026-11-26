"""Turn a chosen moment into a finished short: reframe, clean up, caption, overlay, encode.

Frames are decoded by one FFmpeg process, reframed here with OpenCV, and piped into a second
FFmpeg process that burns in captions and overlays, mixes in the cleaned audio and encodes.
"""
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import captions, export, faces, timeline

SR = 48000


class EncoderError(Exception):
    pass


# x264 CRF / NVENC CQ / audio bitrate for the export presets
QUALITY = {"high": (19, 21, "192k"), "balanced": (23, 25, "160k"), "small": (28, 30, "128k")}


# ---------------------------------------------------------------- options

def _b(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "on", "yes")


def _f(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_opts(cfg: dict) -> dict:
    wm_path = cfg.get("watermark_path", "")
    return {
        "aspect": cfg.get("output_aspect", "9:16") if cfg.get("output_aspect") in ("9:16", "1:1", "4:5", "original") else "9:16",
        "layout": cfg.get("layout", "auto") if cfg.get("layout") in ("auto", "fill", "split", "fit") else "auto",
        "caption_style": cfg.get("caption_style", "bold"),
        "caption_position": cfg.get("caption_position", "lower"),
        "text_color": cfg.get("caption_color", "#FFFFFF"),
        "highlight_color": cfg.get("highlight_color", "#FFE600"),
        "accent_color": cfg.get("accent_color", "#3DFF8F"),
        "remove_fillers": _b(cfg.get("remove_fillers", "on")),
        "remove_silence": _b(cfg.get("remove_silence", "off")),
        "silence_gap": max(0.3, _f(cfg.get("silence_gap"), 0.8)),
        "normalize": _b(cfg.get("normalize_audio", "on")),
        "hook": _b(cfg.get("hook_title", "off")),
        "hook_seconds": _f(cfg.get("hook_seconds"), 3.0),
        "progress": _b(cfg.get("progress_bar", "off")),
        "progress_color": cfg.get("progress_color", "#FF3B5C"),
        "watermark": _b(cfg.get("watermark", "off")) and bool(wm_path) and Path(wm_path).exists(),
        "wm_path": wm_path,
        "wm_pos": cfg.get("watermark_position", "tr"),
        "wm_opacity": min(max(_f(cfg.get("watermark_opacity"), 0.85), 0.1), 1.0),
        "wm_scale": min(max(_f(cfg.get("watermark_scale"), 0.16), 0.05), 0.5),
        "use_nvenc": cfg.get("use_nvenc", "auto"),
        "quality": cfg.get("export_quality") if cfg.get("export_quality") in QUALITY else "high",
        "width": 720 if str(cfg.get("export_width")) == "720" else 1080,
    }


def needs_render(cfg: dict) -> bool:
    """False only when every Phase 2 feature is off, so a plain fast cut is enough."""
    o = parse_opts(cfg)
    return (o["aspect"] != "original" or o["caption_style"] in captions.STYLES or o["remove_fillers"]
            or o["remove_silence"] or o["normalize"] or o["hook"] or o["progress"] or o["watermark"]
            or o["quality"] != "high" or o["width"] != 1080)


# ---------------------------------------------------------------- probing

def probe(src: Path) -> dict:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(src)],
                             capture_output=True, text=True, check=True)
        streams = json.loads(out.stdout)["streams"]
    except (subprocess.CalledProcessError, ValueError, KeyError) as e:
        raise RuntimeError("FFmpeg could not read this file. Is it a valid video?") from e
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    if v is None:
        raise RuntimeError("This file has no video track, so vertical clips cannot be made from it. "
                           "Upload a video file instead.")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = rot or int(v.get("tags", {}).get("rotate", 0) or 0)
    if abs(rot) in (90, 270):  # FFmpeg auto-rotates, so decoded frames come out swapped
        w, h = h, w
    num, _, den = (v.get("avg_frame_rate") or "30/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    fps = 30 if fps <= 1 else min(30, max(15, round(fps)))
    return {"w": w, "h": h, "fps": fps, "audio": any(s["codec_type"] == "audio" for s in streams)}


def output_size(aspect: str, src_w: int, src_h: int, width: int = 1080) -> tuple[int, int]:
    ratios = {"9:16": 16 / 9, "1:1": 1.0, "4:5": 1.25}
    if aspect in ratios:
        return width, round(width * ratios[aspect] / 2) * 2
    w = min(src_w, 1920 if width >= 1080 else 1280) // 2 * 2
    return w, max(2, round(src_h * w / src_w / 2) * 2)


# ---------------------------------------------------------------- geometry

def fit_crop(sw: float, sh: float, aspect: float) -> tuple[float, float]:
    """Largest crop of the given aspect that fits inside the source."""
    return (sh * aspect, sh) if sw / sh > aspect else (sw, sw / aspect)


def crop_resize(frame: np.ndarray, cx: float, cy: float, cw: float, ch: float, ow: int, oh: int) -> np.ndarray:
    sh, sw = frame.shape[:2]
    x0 = min(max(cx - cw / 2, 0), sw - cw)
    y0 = min(max(cy - ch / 2, 0), sh - ch)
    sx, sy = ow / cw, oh / ch
    if sx >= 1.0:  # enlarging: sub-pixel warp keeps slow pans smooth
        m = np.float32([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]])
        return cv2.warpAffine(frame, m, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    xi, yi = int(round(x0)), int(round(y0))
    region = frame[yi:min(yi + int(round(ch)), sh), xi:min(xi + int(round(cw)), sw)]
    return cv2.resize(region, (ow, oh), interpolation=cv2.INTER_AREA)


def compose_fit(frame: np.ndarray, ow: int, oh: int) -> np.ndarray:
    """Whole picture in the middle over a dimmed, blurred copy of itself."""
    sh, sw = frame.shape[:2]
    cw, ch = fit_crop(sw, sh, ow / oh)
    x0, y0 = int((sw - cw) / 2), int((sh - ch) / 2)
    region = frame[y0:y0 + int(ch), x0:x0 + int(cw)]
    small = cv2.resize(region, (max(ow // 12, 2), max(oh // 12, 2)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    bg = (cv2.resize(small, (ow, oh), interpolation=cv2.INTER_LINEAR) * 0.5).astype(np.uint8)
    scale = ow / sw if sw / sh > ow / oh else oh / sh
    fw, fh = max(int(round(sw * scale)), 2), max(int(round(sh * scale)), 2)
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    xo, yo = (ow - fw) // 2, (oh - fh) // 2
    sx0, sy0 = max(-xo, 0), max(-yo, 0)
    dx0, dy0 = max(xo, 0), max(yo, 0)
    w, h = min(fw - sx0, ow - dx0), min(fh - sy0, oh - dy0)
    bg[dy0:dy0 + h, dx0:dx0 + w] = fg[sy0:sy0 + h, sx0:sx0 + w]
    return bg


# ---------------------------------------------------------------- audio

def prepare_audio(src: Path, start: float, dur: float, tl: timeline.Timeline, wav_path: Path) -> bool:
    """Cut the kept spans out of the clip's audio with tiny fades so edits never click."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                          "-vn", "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True).stdout
    if len(raw) < 4:
        return False
    audio = np.frombuffer(raw[: len(raw) // 4 * 4], np.int16).reshape(-1, 2).astype(np.float32)
    fade = int(0.008 * SR)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    chunks = []
    for s, e in tl.audio_intervals():
        seg = audio[int(round(s * SR)): int(round(e * SR))].copy()
        if len(tl.ranges) > 1 and len(seg) > 4 * fade:
            if s > 0:
                seg[:fade] *= ramp
            if e < dur - 0.01:
                seg[-fade:] *= ramp[::-1]
        chunks.append(seg)
    out = np.concatenate(chunks) if chunks else audio
    target = int(round(tl.out_duration * SR))
    out = out[:target] if len(out) >= target else np.pad(out, ((0, target - len(out)), (0, 0)))
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(out.astype(np.int16).tobytes())
    return True


# ---------------------------------------------------------------- main entry

def _overlay_xy(pos: str, margin: int) -> str:
    return {"tl": f"{margin}:{margin}", "tr": f"W-w-{margin}:{margin}",
            "bl": f"{margin}:H-h-{margin}", "br": f"W-w-{margin}:H-h-{margin}"}.get(pos, f"W-w-{margin}:{margin}")


def render_clip(src: Path, dst: Path, clip: dict, transcript: dict, cfg: dict, workdir: Path,
                on_progress: Callable[[float], None] | None = None,
                check: Callable[[], None] | None = None,
                cache_dir: Path | None = None, edits: dict | None = None) -> dict:
    """workdir is scratch space that is deleted afterwards; cache_dir (face scans) must outlive it.

    edits come from the clip editor: {"words": {key: text}, "cuts": [key], "crop": {cx, cy, zoom},
    "broll": [{path, at, dur}]}. Every part is optional."""
    edits = edits or {}
    o = parse_opts(cfg)
    prog = on_progress or (lambda f: None)
    chk = check or (lambda: None)
    info = probe(src)
    fps = info["fps"]
    start, end = float(clip["start"]), float(clip["end"])
    dur = end - start
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. cleanup plan
    all_words = timeline.clip_words(transcript, start, end)
    cut_keys = set(edits.get("cuts") or [])
    extra_cuts = [(w["start"], w["end"]) for w in all_words if w["key"] in cut_keys]
    words = timeline.apply_text_edits(all_words, edits.get("words"))
    if o["remove_fillers"] or o["remove_silence"] or extra_cuts:
        tl = timeline.compute_timeline(start, end, words, fps, o["remove_fillers"], o["remove_silence"],
                                       o["silence_gap"], extra_cuts=extra_cuts)
    else:
        tl = timeline.full_timeline(start, end, fps)

    # 2. frame size, layout and camera paths
    ow, oh = output_size(o["aspect"], info["w"], info["h"], o["width"])
    sw, sh = info["w"], info["h"]
    dec_w = min(sw, 1920) // 2 * 2
    dec_h = max(2, round(sh * dec_w / sw / 2) * 2)
    layout, paths = "original", {}
    crop = edits.get("crop") if o["aspect"] != "original" else None
    manual = False
    if crop:
        # the user placed the crop box by hand: no face scan, same camera for the whole clip
        manual = True
        layout = "fill"
        cw, ch = fit_crop(dec_w, dec_h, ow / oh)
        z = min(max(_f(crop.get("zoom"), 1.0), 1.0), 3.0)
        cw, ch = cw / z, ch / z
        n = tl.total_frames
        paths = {"cx": np.full(n, min(max(_f(crop.get("cx"), 0.5), 0.0), 1.0) * dec_w),
                 "cy": np.full(n, min(max(_f(crop.get("cy"), 0.5), 0.0), 1.0) * dec_h), "size": (cw, ch)}
    elif o["aspect"] != "original":
        if o["layout"] == "fit":
            samples, layout = [], "fit"
        else:
            cache = (cache_dir or workdir.parent / "cache") / f"faces_{start:.2f}_{end:.2f}.json"
            samples = faces.load_or_scan(cache, src, start, dur, sw, sh, check=chk,
                                         progress=lambda f: prog(f * 0.15))
            layout = faces.choose_layout(samples, o["layout"])
        ts = [s["t"] for s in samples]
        n = tl.total_frames
        k = dec_w / sw  # face coordinates are in source pixels; frames are decoded at dec size
        if layout == "fill":
            cw, ch = fit_crop(dec_w, dec_h, ow / oh)
            series = faces.speaker_series(samples)
            horizontal = cw < dec_w - 1
            if horizontal:
                p = faces.camera_path(ts, [None if s is None else s[0] * k for s in series], n, fps, 0.07 * cw, 1.5 * cw)
                paths = {"cx": p if p is not None else np.full(n, dec_w / 2), "cy": np.full(n, dec_h / 2)}
            else:
                p = faces.camera_path(ts, [None if s is None else s[1] * k + 0.1 * ch for s in series], n, fps, 0.07 * ch, 1.5 * ch)
                paths = {"cx": np.full(n, dec_w / 2), "cy": p if p is not None else np.full(n, dec_h / 2)}
            paths["size"] = (cw, ch)
        elif layout == "split":
            hh = oh // 2
            aspect = ow / hh
            ch = dec_h * 0.85
            cw = ch * aspect
            if cw > dec_w:
                cw, ch = dec_w, dec_w / aspect
            left, right = faces.split_series(samples)
            for name, ser in (("l", left), ("r", right)):
                px = faces.camera_path(ts, [None if s is None else s[0] * k for s in ser], n, fps, 0.07 * cw, 1.5 * cw)
                py = faces.camera_path(ts, [None if s is None else s[1] * k + 0.10 * ch for s in ser], n, fps, 0.07 * ch, 1.0 * ch)
                paths[name + "x"] = px if px is not None else np.full(n, dec_w / 2)
                paths[name + "y"] = py if py is not None else np.full(n, dec_h / 2)
            paths["size"] = (cw, ch)

    # 3. captions, hook title, progress bar
    out_dur = tl.out_duration
    final_words = tl.remap_words(words)
    ass_text = captions.build_ass(
        final_words, ow, oh, out_dur,
        style_name=o["caption_style"], position=o["caption_position"], text_color=o["text_color"],
        highlight_color=o["highlight_color"], accent_color=o["accent_color"],
        hook_text=clip.get("title") if o["hook"] else None, hook_seconds=o["hook_seconds"],
        progress=o["progress"], progress_color=o["progress_color"])
    has_overlay = "Dialogue:" in ass_text
    if has_overlay:
        (workdir / "sub.ass").write_text(ass_text, encoding="utf-8")

    # 3b. B-roll cut-aways: picture is replaced for a stretch, the speaker's audio carries on
    broll = _broll_plan(edits.get("broll"), fps, tl.total_frames)

    # 4. audio
    has_audio = info["audio"] and prepare_audio(src, start, dur, tl, workdir / "audio.wav")

    # 5. encode (GPU first, CPU if the GPU encoder will not start)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.stem + ".part.mp4")
    want_gpu = o["use_nvenc"] == "on" or (o["use_nvenc"] == "auto" and export.nvenc_available())
    encoder = "libx264"
    try:
        if want_gpu:
            try:
                _pipe(src, part, start, dur, info, dec_w, dec_h, ow, oh, fps, tl, layout, paths, o, has_overlay,
                      has_audio, workdir, "h264_nvenc", prog, chk, broll)
                encoder = "h264_nvenc"
            except EncoderError:
                encoder = "libx264"
        if encoder == "libx264":
            _pipe(src, part, start, dur, info, dec_w, dec_h, ow, oh, fps, tl, layout, paths, o, has_overlay,
                  has_audio, workdir, "libx264", prog, chk, broll)
        os.replace(part, dst)
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(workdir, ignore_errors=True)
    prog(1.0)
    return {"layout": "manual" if manual else layout, "encoder": encoder, "out_seconds": round(out_dur, 2),
            "removed_seconds": round(tl.removed_seconds, 2), "width": ow, "height": oh,
            "words": [{"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]} for w in final_words]}


def _broll_plan(items, fps: float, total_frames: int) -> list[dict]:
    """Cut-aways as frame ranges in clip-source frames. Missing files are skipped, never fatal."""
    plan = []
    for it in items or []:
        path = Path(str(it.get("path", "")))
        fa = int(round(_f(it.get("at"), 0.0) * fps))
        fb = min(fa + int(round(_f(it.get("dur"), 0.0) * fps)), total_frames)
        if path.is_file() and fb - fa >= 3 and fa >= 0:
            plan.append({"path": path, "fa": fa, "fb": fb})
    plan.sort(key=lambda p: p["fa"])
    for a, b in zip(plan, plan[1:]):
        a["fb"] = min(a["fb"], b["fa"])            # overlapping cut-aways: the later one wins
    return plan


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class BrollReader:
    """Gives frames of a footage file (or one still image) cropped to fill the output size."""

    def __init__(self, path: Path, ow: int, oh: int, fps: float):
        self.ow, self.oh, self.last, self.proc = ow, oh, None, None
        if path.suffix.lower() in IMAGE_EXT:
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Could not read the B-roll image {path.name}")
            h, w = img.shape[:2]
            cw, ch = fit_crop(w, h, ow / oh)
            self.last = crop_resize(img, w / 2, h / 2, cw, ch, ow, oh)
        else:
            self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
                 f"fps={fps},scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh}",
                 "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def next(self) -> np.ndarray:
        if self.proc is not None:
            raw = self.proc.stdout.read(self.ow * self.oh * 3)
            if len(raw) == self.ow * self.oh * 3:
                self.last = np.frombuffer(raw, np.uint8).reshape(self.oh, self.ow, 3)
        if self.last is None:
            raise RuntimeError("The B-roll file has no readable video.")
        return self.last

    def close(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()


def _pipe(src, part, start, dur, info, dec_w, dec_h, ow, oh, fps, tl, layout, paths, o, has_overlay,
          has_audio, workdir, encoder, prog, chk, broll=()) -> None:
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-an",
         "-vf", f"fps={fps},scale={dec_w}:{dec_h}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
           "-r", str(fps), "-i", "-"]
    idx = 1
    if has_audio:
        cmd += ["-i", "audio.wav"]
        a_idx, idx = idx, idx + 1
    if o["watermark"]:
        cmd += ["-i", str(Path(o["wm_path"]).resolve())]
        w_idx, idx = idx, idx + 1

    graph, vin = [], "[0:v]"
    if has_overlay:
        graph.append(f"{vin}ass=sub.ass[vs]")
        vin = "[vs]"
    if o["watermark"]:
        margin = round(ow * 0.04)
        graph.append(f"[{w_idx}:v]format=rgba,scale={round(ow * o['wm_scale'])}:-1,"
                     f"colorchannelmixer=aa={o['wm_opacity']}[lg]")
        graph.append(f"{vin}[lg]overlay={_overlay_xy(o['wm_pos'], margin)}[vout]")
    else:
        graph.append(f"{vin}null[vout]")
    cmd += ["-map", "[vout]"]
    if has_audio:
        graph.append(f"[{a_idx}:a]{'loudnorm=I=-14:TP=-1.5:LRA=11' if o['normalize'] else 'anull'}[aout]")
        cmd += ["-map", "[aout]"]
    cmd += ["-filter_complex", ";".join(graph)]

    crf, cq, abr = QUALITY[o["quality"]]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    cmd += ["-pix_fmt", "yuv420p", "-r", str(fps)]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", abr, "-ar", str(SR), "-shortest"]
    cmd += ["-movflags", "+faststart", str(part)]

    err_path = workdir / "encode.log"
    with open(err_path, "wb") as err:
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err, cwd=workdir)
        frame_bytes = dec_w * dec_h * 3
        written, i = 0, 0
        n_paths = tl.total_frames
        readers: dict = {}
        try:
            while True:
                raw = dec.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                if i < len(tl.mask) and tl.mask[i]:
                    frame = np.frombuffer(raw, np.uint8).reshape(dec_h, dec_w, 3)
                    j = min(i, n_paths - 1)
                    if layout == "fill":
                        cw, ch = paths["size"]
                        out = crop_resize(frame, paths["cx"][j], paths["cy"][j], cw, ch, ow, oh)
                    elif layout == "split":
                        cw, ch = paths["size"]
                        hh = oh // 2
                        out = np.vstack([crop_resize(frame, paths["lx"][j], paths["ly"][j], cw, ch, ow, hh),
                                         crop_resize(frame, paths["rx"][j], paths["ry"][j], cw, ch, ow, oh - hh)])
                    elif layout == "fit":
                        out = compose_fit(frame, ow, oh)
                    elif (dec_w, dec_h) != (ow, oh):
                        out = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
                    else:
                        out = frame
                    for b in broll:
                        if b["fa"] <= i < b["fb"]:
                            rd = readers.get(b["fa"]) or readers.setdefault(b["fa"], BrollReader(b["path"], ow, oh, fps))
                            out = rd.next()
                            break
                    try:
                        enc.stdin.write(np.ascontiguousarray(out).tobytes())
                    except (BrokenPipeError, OSError):
                        break
                    written += 1
                i += 1
                if i % 15 == 0:
                    chk()
                    prog(0.15 + 0.85 * min(i / max(tl.total_frames, 1), 1.0))
        except BaseException:  # includes cancellation raised by chk()
            enc.kill()
            raise
        finally:
            for rd in readers.values():
                rd.close()
            dec.kill()
            dec.wait()
            try:
                enc.stdin.close()
            except OSError:
                pass
            enc.wait()
    if enc.returncode != 0 or not part.exists() or written == 0:
        log = err_path.read_text(errors="replace")[-600:] if err_path.exists() else ""
        if encoder == "h264_nvenc":
            raise EncoderError(log)
        raise RuntimeError(f"FFmpeg failed while rendering the clip: {log.strip() or 'no output produced'}")
