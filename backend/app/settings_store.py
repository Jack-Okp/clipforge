"""Settings persisted in SQLite. API keys fall back to environment variables / .env."""
import os

from . import db

DEFAULTS: dict[str, str] = {
    "llm_provider": "claude",            # claude | gemini | ollama
    "anthropic_api_key": "",
    "anthropic_model": "claude-sonnet-5-5",
    "gemini_api_key": "",
    "gemini_model": "gemini-3.8-flash",   # Google limits older 2.5 models to existing users; editable in Settings
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "llama3.1:8b",
    "whisper_model": "small",            # tiny | base | small | medium  (large-v3 is too big for 4 GB VRAM)
    "whisper_device": "auto",            # auto | cuda | cpu
    "use_nvenc": "auto",                 # auto | on | off  (GPU video encoding)
    "min_clip_seconds": "20",
    "max_clip_seconds": "90",
    "max_clips": "8",
    "transcribe_fillers": "on",          # nudges Whisper to keep "um"/"uh" so they can be removed

    # ---- Phase 2: how finished clips look ----
    "output_aspect": "9:16",             # 9:16 | 1:1 | 4:5 | original
    "layout": "auto",                    # auto | fill | split | fit
    "caption_style": "bold",             # none | bold | classic | karaoke | pop | minimal
    "caption_position": "lower",         # lower | center | top
    "caption_color": "#FFFFFF",
    "highlight_color": "#FFE600",
    "accent_color": "#3DFF8F",
    "remove_fillers": "on",
    "remove_silence": "off",
    "silence_gap": "0.8",                # pauses longer than this (seconds) get shortened
    "normalize_audio": "on",
    "hook_title": "off",
    "hook_seconds": "3",
    "progress_bar": "off",
    "progress_color": "#FF3B5C",
    "watermark": "off",
    "watermark_path": "",                # set by the logo upload, never by the form
    "watermark_position": "tr",          # tl | tr | bl | br
    "watermark_opacity": "0.85",
    "watermark_scale": "0.16",

    # ---- Phase 3 ----
    "export_quality": "high",            # high | balanced | small
    "export_width": "1080",              # 1080 | 720
    "broll_folder": "",                  # folder of your own footage/images for B-roll suggestions
}

# The look of a finished clip. These can be overridden for a single re-render.
STYLE_KEYS = [
    "output_aspect", "layout", "caption_style", "caption_position", "caption_color", "highlight_color",
    "accent_color", "remove_fillers", "remove_silence", "silence_gap", "normalize_audio", "hook_title",
    "hook_seconds", "progress_bar", "progress_color", "watermark", "watermark_position",
    "watermark_opacity", "watermark_scale", "export_quality", "export_width",
]

CHOICES = {
    "output_aspect": {"9:16", "1:1", "4:5", "original"},
    "layout": {"auto", "fill", "split", "fit"},
    "caption_style": {"none", "bold", "classic", "karaoke", "pop", "minimal"},
    "caption_position": {"lower", "center", "top"},
    "watermark_position": {"tl", "tr", "bl", "br"},
    "remove_fillers": {"on", "off"}, "remove_silence": {"on", "off"}, "normalize_audio": {"on", "off"},
    "hook_title": {"on", "off"}, "progress_bar": {"on", "off"}, "watermark": {"on", "off"},
    "transcribe_fillers": {"on", "off"}, "use_nvenc": {"auto", "on", "off"},
    "whisper_model": {"tiny", "base", "small", "medium"}, "whisper_device": {"auto", "cuda", "cpu"},
    "export_quality": {"high", "balanced", "small"}, "export_width": {"1080", "720"},
}


def validate(values: dict[str, str]) -> str | None:
    """Return a message for the first bad value, or None when everything is acceptable."""
    import re
    for k, v in values.items():
        if k in CHOICES and v not in CHOICES[k]:
            return f"'{v}' is not a valid choice for {k}"
        if k.endswith("_color") and not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            return f"{k} must be a colour like #FFE600"
        if k in ("silence_gap", "hook_seconds", "watermark_opacity", "watermark_scale",
                 "min_clip_seconds", "max_clip_seconds"):
            try:
                if float(v) <= 0:
                    return f"{k} must be greater than zero"
            except ValueError:
                return f"{k} must be a number"
        if k == "broll_folder" and v.strip() and not os.path.isdir(v.strip()):
            return "That B-roll folder does not exist on this computer"
        if k == "max_clips" and not (v.isdigit() and 1 <= int(v) <= 30):
            return "max_clips must be a whole number from 1 to 30"
    lo, hi = values.get("min_clip_seconds"), values.get("max_clip_seconds")
    if lo and hi and float(lo) >= float(hi):
        return "min_clip_seconds must be smaller than max_clip_seconds"
    return None

SECRET_KEYS = {"anthropic_api_key", "gemini_api_key"}

ENV_FALLBACK = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "ollama_base_url": "OLLAMA_BASE_URL",
}


def get_all() -> dict[str, str]:
    stored = {r["key"]: r["value"] for r in db.rows("SELECT key, value FROM settings")}
    out = dict(DEFAULTS)
    out.update({k: v for k, v in stored.items() if k in DEFAULTS})
    for key, env in ENV_FALLBACK.items():
        if not out.get(key) and os.environ.get(env):
            out[key] = os.environ[env]
    return out


def set_many(values: dict[str, str]) -> None:
    for key, value in values.items():
        if key not in DEFAULTS:
            continue
        db.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


def public_view() -> dict:
    """Settings safe to send to the browser: secrets are replaced by a masked hint."""
    from pathlib import Path
    cfg = get_all()
    out: dict = {"watermark_set": bool(cfg["watermark_path"]) and Path(cfg["watermark_path"]).exists()}
    for key, value in cfg.items():
        if key == "watermark_path":
            continue  # a local file path is of no use to the browser
        if key in SECRET_KEYS:
            out[key] = ""
            out[key + "_set"] = bool(value)
            out[key + "_hint"] = ("…" + value[-4:]) if len(value) >= 8 else ""
        else:
            out[key] = value
    return out
