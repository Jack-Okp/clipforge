# ClipForge (Phase 2)

A local clipping tool: give it a long video, get back ranked, ready-to-post short clips.
Everything runs on your machine except the AI call, which you can point at Claude, Gemini,
or a local Ollama model.

## Install (Windows)

1. Install **Python 3.10+** (tick "Add to PATH") from python.org.
2. Install **FFmpeg**: open a terminal and run `winget install Gyan.FFmpeg`, then reopen the terminal.
3. Double-click **setup.bat** (first run downloads packages, a few minutes).
4. Double-click **start.bat**. The app opens at http://127.0.0.1:8000.

**Upgrading from Phase 1?** Unzip over the old folder (keep your `data` folder) and run
`setup.bat` again so the new packages (OpenCV) are installed. Old projects and settings are kept.

## What a finished clip gets (Phase 2)

| Feature | What it does |
|---------|--------------|
| **Vertical reframing** | 9:16, 4:5, 1:1 or original shape. |
| **Follows the speaker** | Finds faces, picks who is talking, and pans smoothly like a camera operator (ignores small movements, glides when the speaker really moves). |
| **Split screen** | When two people are on screen together, they are stacked top and bottom, each centred. |
| **Fit on blur** | When there are no faces (slides, screen recordings), the whole picture is shown over a blurred copy of itself so nothing is cut off. |
| **Animated captions** | Word-by-word highlighting from the Whisper timestamps. Five styles: Bold, Pop, Classic, Karaoke sweep, Minimal box. Your own text, highlight and keyword colours. Numbers, names in capitals and long words get the keyword colour. |
| **Cleanup** | Removes "um", "uh", "er", "hmm"; can shorten long pauses; evens out loudness (-14 LUFS, the level social apps use). Cuts are placed on video frames with tiny audio fades, so there are no clicks and captions stay in sync. |
| **Overlays** | Hook title at the start, a progress bar, and your logo (position, size, opacity). |

All of it is set under **Settings** (defaults for new projects, with a live preview) and can be
changed per project under **Restyle clips**, which re-renders from the original video without
re-transcribing or asking the AI again. Each clip also has its own **Restyle this clip** button.

## Switching AI models

Open **Settings** in the app.

| Provider | What you need |
|----------|---------------|
| Claude   | API key from console.anthropic.com |
| Gemini   | API key from aistudio.google.com |
| Ollama   | Install Ollama, then `ollama pull llama3.1:8b` (or any model). No key. |

Click a provider card, press **Test connection**, then **Save settings**. You can also re-pick
clips for a finished project with a different model ("Re-pick clips with"). Keys are stored in
plain text in `data/clipforge.db` on your computer and never shown back in the browser, so treat
that file like a password.

Model names change over time. The Gemini default is `gemini-3.8-flash` (Google limits its older 2.5
models to existing users). If a provider rejects the model name, type a current one into the Model box
and press **Test connection**.

ClipForge only answers on `127.0.0.1` / `localhost`, and it refuses requests that come from other
websites, so a page open in another browser tab cannot use your keys or start jobs.

## Your hardware (i7-10850H, 16 GB RAM, Quadro T1000 4 GB)

- Whisper `small` or `medium` run on the GPU. Avoid `large-v3` (needs more than 4 GB).
- Video encoding uses the GPU (NVENC) automatically and falls back to the CPU if it cannot start.
- Reframing itself runs on the CPU. Expect a clip to take roughly 1 to 2 times its own length to
  render; the first render of a clip also scans for faces, restyles after that reuse the scan.
- Ollama: stick to 3B-8B models. They are slower and less accurate than Claude or Gemini on long transcripts.

## Honest limits

- **Face detection** uses the detectors built into OpenCV, so nothing extra is downloaded. They are
  best with faces turned mostly towards the camera, and can miss a face that is small, in heavy
  shadow, or turned far away. When no face is found for a stretch, the camera holds its last position.
- **Who is talking** is judged from mouth movement, not from the audio. It is reliable for telling two
  people apart in a normal shot, but can be fooled by someone nodding or laughing. Split screen avoids the
  problem entirely when both people are visible.
- **Fillers**: Whisper tends to leave "um" out of transcripts. ClipForge nudges it to keep them
  (Settings > Keep "um"/"uh" in transcript), but some will still be missed. Only sounds that are never
  real words are removed; "like" and "you know" are left alone on purpose.
- **Emojis** in captions are not included, because the subtitle renderer cannot draw colour emojis reliably.
- **Fonts**: captions use Arial Black and Arial, which ship with Windows.
- **Source quality**: a 1080p landscape video cropped to vertical is enlarged about 1.8 times, so it will
  look softer than the original. A 4K source looks noticeably better.

## Troubleshooting

- **"Cannot connect to Ollama"**: start the Ollama app or run `ollama serve`.
- **Transcription is slow**: the transcript heading shows which device was used. If it says `cpu`, a red note under it gives the reason the GPU could not be used (usually missing CUDA libraries; re-run setup.bat). Transcription falls back to the CPU by itself, so the job still finishes.
- **A project says "ClipForge was closed while this was running"**: closing the window mid-job cuts the job off. Press **Run again**, or **Restyle clips** if the clips were already made.
- **"This file has no video track"**: only video files can be used. Audio-only files (mp3, wav) are rejected.
- **FFmpeg not found when starting**: install it with `winget install Gyan.FFmpeg`, then open a new window.
- **Link download fails**: update yt-dlp with `venv\Scripts\python.exe -m pip install -U yt-dlp`.
- **A clip shows a red "Render failed" card**: the message says why. Press **Restyle this clip** to retry; other clips are unaffected.
- **Captions look wrong or the font is different**: Arial Black is missing on your system; the renderer substitutes another font.

## Roadmap

Phase 3: clip editor (trim, edit transcript, drag the crop), batch queue polish, prompt-based clipping, brand kits, export presets and SRT export.

## Developers

Tests: `cd backend` then `python -m pytest tests -q` (needs `pip install pytest`).
