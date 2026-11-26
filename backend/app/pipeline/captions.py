"""Animated captions, hook title and progress bar, written as one ASS subtitle file.

ASS is the format FFmpeg's libass renderer understands, so every overlay is burned in by a
single FFmpeg pass with no extra drawing code. Times here are in edited-clip time.
"""
import re
from dataclasses import dataclass

STOPWORDS = set("""
about after again also because before being between could doing during every first from have having
here into just know like maybe might more most much never only other over really right should since
some something still such than that their them then there these they thing think this those through
today under until very want were what when where which while whole will with would your yours
""".split())


@dataclass(frozen=True)
class CaptionStyle:
    font: str
    size: float          # fraction of the layout unit
    bold: bool
    upper: bool
    max_words: int
    max_chars: int
    outline: float       # fraction of font size
    shadow: float        # fraction of font size
    mode: str            # highlight | pop | karaoke | plain
    box: bool
    active_scale: float
    keywords: bool


STYLES: dict[str, CaptionStyle] = {
    "bold":    CaptionStyle("Arial Black", 0.098, True, True, 3, 22, 0.09, 0.0, "highlight", False, 1.12, True),
    "classic": CaptionStyle("Arial", 0.072, True, False, 6, 34, 0.07, 0.04, "highlight", False, 1.0, False),
    "karaoke": CaptionStyle("Arial", 0.078, True, False, 5, 30, 0.07, 0.03, "karaoke", False, 1.0, False),
    "pop":     CaptionStyle("Arial Black", 0.105, True, True, 2, 16, 0.09, 0.0, "pop", False, 1.0, True),
    "minimal": CaptionStyle("Arial", 0.056, False, False, 7, 40, 0.0, 0.0, "plain", True, 1.0, False),
}


def hex_to_ass(color: str, alpha: int = 0) -> str:
    c = (color or "").lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", c):
        c = "FFFFFF"
    return f"&H{alpha:02X}{c[4:6]}{c[2:4]}{c[0:2]}".upper()


def ass_time(t: float) -> str:
    cs = int(round(max(t, 0.0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean(text: str) -> str:
    """ASS treats braces and backslashes as markup, so they must never reach the file."""
    return re.sub(r"[{}\\]", "", text).replace("\n", " ").strip()


def is_keyword(word: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", word)
    if re.search(r"\d|[$%€£]", word):
        return True
    if len(letters) >= 3 and letters.isupper():
        return True
    return len(letters) >= 7 and letters.lower() not in STOPWORDS


def build_pages(words: list[dict], style: CaptionStyle) -> list[list[dict]]:
    """Group words into short caption pages, breaking at sentence ends and long pauses."""
    pages: list[list[dict]] = []
    cur: list[dict] = []
    chars = 0
    for w in words:
        text = clean(w["word"])
        if not text:
            continue
        if cur:
            gap = w["start"] - cur[-1]["end"]
            ended = cur[-1]["word"].rstrip().endswith((".", "?", "!"))
            if (len(cur) >= style.max_words or chars + len(text) + 1 > style.max_chars
                    or gap > 0.8 or ended):
                pages.append(cur)
                cur, chars = [], 0
        cur.append({"start": w["start"], "end": w["end"], "word": text})
        chars += len(text) + 1
    if cur:
        pages.append(cur)
    return pages


def srt_time(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def build_srt(words: list[dict], max_words: int = 7, max_chars: int = 42) -> str:
    """Plain subtitle file (for YouTube, Premiere...). Words are the clip's final, re-timed words."""
    style = CaptionStyle("Arial", 0.07, False, False, max_words, max_chars, 0, 0, "plain", False, 1.0, False)
    pages = build_pages(words, style)
    out, prev_end = [], 0.0
    for n, page in enumerate(pages, 1):
        a = max(page[0]["start"], prev_end)
        nxt = pages[n][0]["start"] if n < len(pages) else None
        b = max(page[-1]["end"], a + 0.3)
        if nxt is not None:
            b = min(b, max(nxt, a + 0.05))               # never overlap the next line
        prev_end = b
        out.append(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{' '.join(w['word'] for w in page)}\n")
    return "\n".join(out)


def _layout_unit(w: int, h: int) -> float:
    return min(w, h * 0.75)


def build_ass(words: list[dict], width: int, height: int, duration: float, *,
              style_name: str = "bold", position: str = "lower",
              text_color: str = "#FFFFFF", highlight_color: str = "#FFE600", accent_color: str = "#3DFF8F",
              hook_text: str | None = None, hook_seconds: float = 3.0,
              progress: bool = False, progress_color: str = "#FF3B5C") -> str:
    unit = _layout_unit(width, height)
    vertical = height >= width * 1.2
    events: list[str] = []
    styles: list[str] = []

    # ---- captions ----
    if style_name in STYLES and words:
        st = STYLES[style_name]
        size = round(st.size * unit)
        outline = round(st.outline * size) if not st.box else round(0.018 * unit)
        shadow = round(st.shadow * size)
        align = {"lower": 2, "center": 5, "top": 8}.get(position, 2)
        margin_v = 0
        if align == 2:
            margin_v = round(height * (0.20 if vertical else 0.10 if width == height else 0.08))
        elif align == 8:
            margin_v = round(height * (0.25 if vertical else 0.15))
        outline_col = hex_to_ass("#000000", 0x80) if st.box else hex_to_ass("#000000")
        primary = hex_to_ass(highlight_color) if st.mode == "karaoke" else hex_to_ass(text_color)
        secondary = hex_to_ass(text_color) if st.mode == "karaoke" else hex_to_ass(highlight_color)
        styles.append(
            f"Style: Cap,{st.font},{size},{primary},{secondary},{outline_col},{hex_to_ass('#000000', 0x60)},"
            f"{-1 if st.bold else 0},0,0,0,100,100,0,0,{3 if st.box else 1},{outline},{shadow},"
            f"{align},{round(width * 0.07)},{round(width * 0.07)},{margin_v},1"
        )
        hl, ac = hex_to_ass(highlight_color), hex_to_ass(accent_color)

        def show(word: str) -> str:
            return word.upper() if st.upper else word

        for page in build_pages(words, st):
            page_end = page[-1]["end"]
            if st.mode == "karaoke":
                parts = []
                for j, w in enumerate(page):
                    nxt = page[j + 1]["start"] if j + 1 < len(page) else w["end"]
                    parts.append(f"{{\\kf{max(int(round((nxt - w['start']) * 100)), 1)}}}{show(w['word'])}")
                events.append(f"Dialogue: 1,{ass_time(page[0]['start'])},{ass_time(page_end)},Cap,,0,0,0,,{' '.join(parts)}")
            elif st.mode == "plain":
                text = " ".join(show(w["word"]) for w in page)
                events.append(f"Dialogue: 1,{ass_time(page[0]['start'])},{ass_time(page_end)},Cap,,0,0,0,,{text}")
            else:
                for j, active in enumerate(page):
                    start = active["start"]
                    end = page[j + 1]["start"] if j + 1 < len(page) else page_end + 0.12
                    if end - start < 0.04:
                        end = start + 0.04
                    parts = []
                    for k, w in enumerate(page):
                        txt = show(w["word"])
                        if k == j:
                            if st.mode == "pop":
                                tag = (f"{{\\1c{hl}\\fscx80\\fscy80\\t(0,100,\\fscx118\\fscy118)"
                                       f"\\t(100,180,\\fscx105\\fscy105)}}")
                            else:
                                s = round(st.active_scale * 100)
                                tag = f"{{\\1c{hl}\\fscx{s}\\fscy{s}}}"
                            parts.append(f"{tag}{txt}{{\\r}}")
                        elif st.keywords and is_keyword(w["word"]):
                            parts.append(f"{{\\1c{ac}}}{txt}{{\\r}}")
                        else:
                            parts.append(txt)
                    events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{' '.join(parts)}")

    # ---- hook title ----
    if hook_text and clean(hook_text):
        size = round(0.060 * unit)
        box = round(0.022 * unit)
        styles.append(
            f"Style: Hook,Arial Black,{size},{hex_to_ass('#000000')},{hex_to_ass('#000000')},{hex_to_ass('#FFFFFF')},"
            f"{hex_to_ass('#FFFFFF')},-1,0,0,0,100,100,0,0,3,{box},0,8,{round(width * 0.10)},{round(width * 0.10)},"
            f"{round(height * (0.09 if vertical else 0.06))},1"
        )
        text = clean(hook_text)[:90]
        end = min(max(hook_seconds, 0.5), max(duration, 0.5))
        events.append(f"Dialogue: 2,{ass_time(0)},{ass_time(end)},Hook,,0,0,0,,{{\\fad(250,250)}}{text}")

    # ---- progress bar ----
    if progress and duration > 0:
        bar_h = max(8, round(height * 0.006))
        styles.append(f"Style: Bar,Arial,20,{hex_to_ass(progress_color)},{hex_to_ass(progress_color)},"
                      f"{hex_to_ass(progress_color)},{hex_to_ass(progress_color)},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1")
        ms = max(int(duration * 1000), 1)
        events.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(duration)},Bar,,0,0,0,,"
            f"{{\\an7\\pos(0,{height - bar_h})\\clip(0,0,0,{height})\\t(0,{ms},\\clip(0,0,{width},{height}))\\p1}}"
            f"m 0 0 l {width} 0 l {width} {bar_h} l 0 {bar_h}{{\\p0}}"
        )

    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
        "MarginL,MarginR,MarginV,Encoding\n"
    )
    ev = "\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    return head + "\n".join(styles) + "\n" + ev + "\n".join(events) + "\n"
