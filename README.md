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

