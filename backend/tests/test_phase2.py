"""Phase 2 tests. Run from the backend folder:  python -m pytest tests -q"""
import numpy as np
import pytest

from app.pipeline import captions, faces, render, timeline

FPS = 25


def W(t, text, d=0.4):
    return {"start": t, "end": t + d, "word": text}


# ------------------------------------------------------------------ timeline

def test_no_cleanup_keeps_everything():
    tl = timeline.compute_timeline(0, 10, [W(1, "hello")], FPS, False, False)
    assert tl.out_frames == tl.total_frames == 250 and tl.removed_seconds == 0


def test_filler_is_cut_and_real_words_are_not():
    words = [W(1, "so"), W(2, "um,"), W(3, "like"), W(4, "Uh"), W(5, "umbrella")]
    tl = timeline.compute_timeline(0, 8, words, FPS, True, False)
    assert tl.removed_seconds == pytest.approx(0.8, abs=0.1)
    kept = [w["word"] for w in tl.remap_words(words)]
    assert kept == ["so", "like", "umbrella"]       # "like" and "umbrella" must survive


def test_long_silence_is_shortened_but_leaves_a_natural_pause():
    words = [W(0.5, "a"), W(1.0, "b"), W(5.0, "c")]
    tl = timeline.compute_timeline(0, 6, words, FPS, False, True, silence_gap=0.8, pad=0.18)
    assert tl.removed_seconds == pytest.approx(3.6 - 0.36, abs=0.1)
    assert tl.removed_seconds < 3.6                  # some pause is kept


def test_short_pauses_are_left_alone():
    words = [W(0.5, "a"), W(1.4, "b"), W(2.3, "c")]
    tl = timeline.compute_timeline(0, 2.8, words, FPS, False, True, silence_gap=0.8)
    assert tl.removed_seconds == 0


def test_dead_air_after_the_last_word_is_trimmed_but_keeps_a_short_tail():
    words = [W(0.5, "a"), W(1.4, "b"), W(2.3, "c")]          # last word ends 2.7, clip runs to 4.0
    tl = timeline.compute_timeline(0, 4, words, FPS, False, True, silence_gap=0.8, pad=0.18)
    assert tl.removed_seconds == pytest.approx(1.3 - 0.18, abs=0.05)
    assert tl.out_duration == pytest.approx(2.88, abs=0.05)


def test_map_time_shifts_words_after_a_cut():
    words = [W(1, "a"), W(2, "um"), W(3, "b")]
    tl = timeline.compute_timeline(0, 5, words, FPS, True, False)
    out = tl.remap_words(words)
    assert out[0]["start"] == pytest.approx(1.0, abs=0.05)
    assert out[1]["start"] == pytest.approx(3.0 - 0.4, abs=0.08)   # 0.4 s earlier


def test_cuts_land_on_frame_boundaries_and_audio_matches_video():
    words = [W(1.013, "a"), W(2.0071, "um", 0.37), W(3.31, "b")]
    tl = timeline.compute_timeline(0, 5, words, FPS, True, False)
    spans = tl.audio_intervals()
    assert sum(e - s for s, e in spans) == pytest.approx(tl.out_duration)
    for s, e in spans:
        assert round(s * FPS) == pytest.approx(s * FPS) and round(e * FPS) == pytest.approx(e * FPS)


def test_clip_words_clamps_to_range_and_falls_back_without_word_timestamps():
    tr = {"segments": [
        {"start": 0, "end": 10, "text": "x", "words": [W(0.5, "early"), W(4.9, "edge", 0.4), W(7, "late")]},
        {"start": 10, "end": 14, "text": "no word times here", "words": []},
    ]}
    inside = timeline.clip_words(tr, 4.0, 5.0)
    assert [w["word"] for w in inside] == ["edge"] and inside[0]["end"] <= 5.0
    spread = timeline.clip_words(tr, 10, 14)
    assert [w["word"] for w in spread] == ["no", "word", "times", "here"]
    assert spread[0]["start"] == 10 and spread[-1]["end"] == pytest.approx(14)


# ------------------------------------------------------------------ captions

def test_colour_and_time_formats():
    assert captions.hex_to_ass("#FFE600") == "&H00" + "00E6FF"
    assert captions.hex_to_ass("garbage") == "&H00FFFFFF"
    assert captions.ass_time(3725.456) == "1:02:05.46"
    assert captions.ass_time(-1) == "0:00:00.00"


def test_markup_characters_never_reach_the_file():
    ass = captions.build_ass([W(0, "{\\b1}evil"), W(0.5, "ok")], 1080, 1920, 2)
    body = ass.split("[Events]")[1]
    assert "{\\b1}" not in body and "\\b1" not in body


def test_pages_break_on_sentence_end_pause_and_word_limit():
    st = captions.STYLES["bold"]                      # max 3 words
    words = [W(0, "one"), W(0.5, "two."), W(1.0, "three"), W(1.5, "four"), W(3.0, "five"),
             W(3.5, "six"), W(4.0, "seven"), W(4.5, "eight")]
    pages = [[w["word"] for w in p] for p in captions.build_pages(words, st)]
    assert pages == [["one", "two."], ["three", "four"], ["five", "six", "seven"], ["eight"]]


def test_highlight_style_one_event_per_word_with_active_colour():
    words = [W(0, "hello"), W(0.5, "world")]
    ass = captions.build_ass(words, 1080, 1920, 2, style_name="bold", highlight_color="#FF0000")
    ev = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(ev) == 2
    assert "&H000000FF" in ev[0]                      # red (BGR) on the active word
    assert "HELLO" in ev[0] and "WORLD" in ev[0]      # whole page visible, one word lit


def test_events_are_continuous_inside_a_page():
    ass = captions.build_ass([W(0, "a"), W(0.5, "b"), W(1.0, "c")], 1080, 1920, 3, style_name="classic")
    ev = [l.split(",")[1:3] for l in ass.splitlines() if l.startswith("Dialogue")]
    assert ev[0][1] == ev[1][0] and ev[1][1] == ev[2][0]      # no flicker between words


def test_karaoke_uses_one_event_per_page_with_sweep_tags():
    ass = captions.build_ass([W(0, "a"), W(0.5, "b")], 1080, 1920, 2, style_name="karaoke")
    ev = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(ev) == 1 and ev[0].count("\\kf") == 2


def test_hook_and_progress_bar_are_added_only_when_asked():
    plain = captions.build_ass([W(0, "a")], 1080, 1920, 5, style_name="none")
    assert "Dialogue" not in plain
    full = captions.build_ass([W(0, "a")], 1080, 1920, 5, style_name="bold", hook_text="Big hook",
                              progress=True)
    assert "Big hook" in full and ",Bar," in full and ",Hook," in full


def test_keyword_detection():
    assert captions.is_keyword("$500") and captions.is_keyword("NASA") and captions.is_keyword("incredible")
    assert not captions.is_keyword("the") and not captions.is_keyword("because")


def test_square_and_landscape_sizes_scale_sensibly():
    for w, h in ((1080, 1920), (1080, 1080), (1920, 1080)):
        ass = captions.build_ass([W(0, "hi")], w, h, 2)
        size = int(ass.split("Style: Cap,")[1].split(",")[1])
        assert 50 <= size <= 120, (w, h, size)


# ------------------------------------------------------------------ camera

def test_lazy_follow_ignores_jitter_and_follows_real_movement():
    jitter = 500 + np.sin(np.arange(100)) * 8
    assert np.ptp(faces.lazy_follow(jitter, 25, dead=40, vmax=900)) == 0
    move = np.concatenate([np.full(50, 400.0), np.full(150, 900.0)])
    out = faces.lazy_follow(move, 25, dead=40, vmax=900)
    assert out[-1] > 850 and np.max(np.abs(np.diff(out))) <= 900 / 25 + 1e-6


def test_camera_path_is_smooth_and_none_without_faces():
    assert faces.camera_path([], [], 100, 25, 40, 900) is None
    ts = [i / 6 for i in range(24)]
    vals = [500.0] * 12 + [900.0] * 12
    p = faces.camera_path(ts, vals, 100, 25, 40, 900)
    assert len(p) == 100 and np.max(np.abs(np.diff(p, 2))) < 3      # no sudden jerks


def test_gaps_hold_the_last_position():
    ts = [0, 1, 2, 3]
    p = faces.camera_path(ts, [None, 300.0, None, None], 100, 25, 10, 900)
    assert p[0] == pytest.approx(p[-1], abs=2)


def face(x, y=300, w=100, mouth=0.0):
    return [x, y, w, w, mouth]


def test_speaker_switches_to_the_one_talking_but_not_on_noise():
    left_talks = [{"t": i / 6, "faces": [face(300, mouth=10), face(900, mouth=0.5)]} for i in range(12)]
    right_talks = [{"t": 2 + i / 6, "faces": [face(300, mouth=0.5), face(900, mouth=10)]} for i in range(24)]
    series = faces.speaker_series(left_talks + right_talks)
    assert series[5][0] == 300                       # follows the left speaker first
    assert series[-1][0] == 900                      # then the right one
    noisy = [{"t": i / 6, "faces": [face(300, mouth=3 + (i % 2)), face(900, mouth=3 + ((i + 1) % 2))]}
             for i in range(30)]
    assert len({round(s[0]) for s in faces.speaker_series(noisy)}) == 1     # no flicker


def test_layout_choice():
    one = [{"t": 0, "faces": [face(500)]}] * 10
    two = [{"t": 0, "faces": [face(300), face(900)]}] * 10
    dup = [{"t": 0, "faces": [face(500), face(510)]}] * 10          # same face detected twice
    none = [{"t": 0, "faces": []}] * 10
    assert faces.choose_layout(one, "auto") == "fill"
    assert faces.choose_layout(two, "auto") == "split"
    assert faces.choose_layout(dup, "auto") == "fill"
    assert faces.choose_layout(none, "auto") == "fit"
    assert faces.choose_layout(one, "split") == "fill"               # cannot split one person
    assert faces.choose_layout(two, "fit") == "fit"


def test_split_assigns_left_and_right_people():
    samples = [{"t": i, "faces": [face(900), face(300)]} for i in range(5)] + [{"t": 5, "faces": [face(880)]}]
    left, right = faces.split_series(samples)
    assert all(l[0] == 300 for l in left[:5]) and all(r[0] == 900 for r in right[:5])
    assert left[5] is None and right[5][0] == 880


def test_isolated_false_detections_are_dropped():
    samples = [{"t": i, "faces": [face(500)]} for i in range(8)]
    samples[4]["faces"].append(face(1500))           # appears in one sample only
    cleaned = faces.stabilise(samples)
    assert all(len(s["faces"]) == 1 for s in cleaned)


# ------------------------------------------------------------------ geometry and options

def test_output_sizes():
    assert render.output_size("9:16", 1920, 1080) == (1080, 1920)
    assert render.output_size("4:5", 1920, 1080) == (1080, 1350)
    assert render.output_size("original", 3840, 2160) == (1920, 1080)


def test_fit_crop_stays_inside_the_source():
    cw, ch = render.fit_crop(1920, 1080, 9 / 16)
    assert ch == 1080 and cw == pytest.approx(607.5)
    cw, ch = render.fit_crop(1080, 1920, 1.0)
    assert cw == 1080 and ch == 1080


def test_crop_and_fit_produce_exact_output_size():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    assert render.crop_resize(frame, 960, 540, 607, 1080, 1080, 1920).shape == (1920, 1080, 3)
    assert render.crop_resize(frame, 0, 0, 1215, 1080, 1080, 960).shape == (960, 1080, 3)   # clamped at the edge
    assert render.compose_fit(frame, 1080, 1920).shape == (1920, 1080, 3)
    assert render.compose_fit(frame, 1080, 1080).shape == (1080, 1080, 3)


def test_options_parse_safely_and_decide_when_to_render():
    o = render.parse_opts({"output_aspect": "banana", "silence_gap": "x", "watermark": "on"})
    assert o["aspect"] == "9:16" and o["silence_gap"] == 0.8 and o["watermark"] is False   # no logo file
    off = {"output_aspect": "original", "caption_style": "none", "remove_fillers": "off",
           "remove_silence": "off", "normalize_audio": "off", "hook_title": "off", "progress_bar": "off",
           "watermark": "off"}
    assert render.needs_render(off) is False
    assert render.needs_render({**off, "caption_style": "bold"}) is True
    assert render.needs_render({**off, "normalize_audio": "on"}) is True


# ------------------------------------------------------------------ real face

def test_detects_a_real_face():
    skimage = pytest.importorskip("skimage")
    import cv2
    from skimage import data
    gray = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY)
    found = faces.detect_in_frame(gray, 40)
    assert len(found) == 1
    x, y, w, h = found[0]
    assert 150 < x + w / 2 < 300 and 50 < y + h / 2 < 180
