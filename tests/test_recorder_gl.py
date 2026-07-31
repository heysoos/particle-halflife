"""Integration tests for the GL side of recording: what actually lands in the file.

These need a real window and OpenGL context (WSLg provides one), so they are
skipped wherever a display isn't available. They cover the requirement that
unit tests can't reach: the REC badge is visible on screen but **never** in the
recorded video, and the HUD is included only when shown.

The trick that makes these deterministic: update() is never called, so
_n_particles_to_draw stays 0 and the scene is a uniform background colour. Any
non-background pixel in a recorded frame therefore came from an overlay, and we
can say exactly which one.
"""

import os
import subprocess
import shutil

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
moderngl = pytest.importorskip("moderngl")

HAVE_FFMPEG = shutil.which("ffmpeg") is not None

# Even dimensions — libx264's yuv420p requires them. Small for speed.
W, H = 320, 240


def _have_gl() -> bool:
    try:
        pygame.init()
        pygame.display.set_mode((32, 32), pygame.OPENGL | pygame.DOUBLEBUF)
        moderngl.create_context()
        return True
    except Exception:
        return False
    finally:
        pygame.display.quit()


pytestmark = [
    pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH"),
    pytest.mark.skipif(not _have_gl(), reason="no OpenGL context available"),
]


@pytest.fixture
def renderer(tmp_path):
    """A real Renderer at 320x240 writing recordings into tmp_path."""
    from halflife.config import SimConfig
    from halflife.renderer import Renderer

    config = SimConfig(
        num_particles=64,
        num_species=2,
        window_width=W,
        window_height=H,
        recording_dir=str(tmp_path),
        recording_fps=30,
    )
    r = Renderer(config)
    try:
        yield r
    finally:
        r.close()          # also stops any recording still running
        pygame.display.quit()


def _record(renderer, n_frames: int) -> str:
    """Record n_frames of whatever the renderer currently draws; return the path.

    Asserts the badge was actually painted while recording, so no test here can
    pass vacuously by way of a badge that never drew at all. Checked before
    stop(), which clears _badge_text.
    """
    path = renderer.toggle_recording()
    assert path is not None, "recording failed to start"
    for i in range(n_frames):
        renderer.render(fps=60.0, step_count=i, n_alive=0)
    assert renderer._badge_text is not None, "badge never painted — tests would be vacuous"
    renderer.toggle_recording()
    return path


def _decode_first_frame(path: str) -> np.ndarray:
    """Decode frame 1 of an MP4 to an (H, W, 3) uint8 array."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw[: W * H * 3], dtype=np.uint8).reshape(H, W, 3).copy()


def _badge_region(renderer, frame: np.ndarray) -> np.ndarray:
    """Slice the frame where the badge actually got painted.

    Taken from the bounding box of the non-transparent pixels left on
    _badge_surface by the last paint, rather than hard-coded — the badge's size
    depends on its text and its position on window width. pygame surface coords
    and the decoded (vflipped) frame share a top-left origin, so the rect maps
    across directly.
    """
    rect = renderer._badge_surface.get_bounding_rect()
    assert rect.width > 0 and rect.height > 0, "badge surface is blank"
    return frame[rect.top:rect.bottom, rect.left:rect.right]


# ── The core requirement ──────────────────────────────────────────────────────

def test_badge_is_on_screen_but_not_in_the_video(renderer):
    """The whole point: the indicator must not be recorded.

    With the HUD hidden and no particles drawn, every recorded frame should be a
    single flat background colour. The badge is a dark-red panel with a bright
    red dot — if it leaked into the capture, this frame would not be uniform.
    """
    renderer.toggle_hud()                      # hide HUD → frame is pure background
    assert renderer._show_hud is False

    path = _record(renderer, 12)   # also asserts the badge really was painted

    frame = _decode_first_frame(path)
    spread = frame.max(axis=(0, 1)).astype(int) - frame.min(axis=(0, 1)).astype(int)
    assert spread.max() <= 8, (
        f"recorded frame is not uniform (per-channel spread {spread}) — "
        f"something was captured that should not have been"
    )
    # And specifically: nothing bright in the rect the badge occupied.
    corner = _badge_region(renderer, frame)
    assert corner[..., 0].max() <= int(frame[..., 0].min()) + 8, (
        "found bright pixels where the badge was drawn — it leaked into the video"
    )


def test_hud_is_recorded_when_shown_but_badge_still_is_not(renderer):
    """H controls the HUD in the video; the badge is excluded either way."""
    assert renderer._show_hud is True

    path = _record(renderer, 12)
    frame = _decode_first_frame(path)

    # The HUD (left-edge buttons, bottom hints) must be visible in the capture.
    spread = int(frame.max()) - int(frame.min())
    assert spread > 40, f"HUD does not appear in the recording (spread {spread})"

    # The badge's own rect must hold no trace of it. paint_rec_badge draws a
    # (150, 40, 45) border and a (235, 60, 60) dot — strongly red-dominant —
    # whereas every HUD element up there is blue-grey. So: count red-dominant
    # pixels in the rect. (Counting matters: the badge's brightest pixel is its
    # near-white *text*, so testing only the max-red pixel misses the leak.)
    corner = _badge_region(renderer, frame).reshape(-1, 3).astype(int)
    red_dominant = int(((corner[:, 0] > 100) & (corner[:, 0] > 2 * corner[:, 1])).sum())
    assert red_dominant <= 10, (
        f"{red_dominant} red-dominant pixels in the badge rect — "
        f"the REC badge leaked into the video"
    )


def test_hidden_hud_records_less_than_shown_hud(renderer):
    """Directly contrast the two: H must actually change what gets captured."""
    shown_frame = _decode_first_frame(_record(renderer, 8))
    renderer.toggle_hud()
    hidden_frame = _decode_first_frame(_record(renderer, 8))

    assert shown_frame.std() > hidden_frame.std() + 5, (
        f"hiding the HUD did not reduce recorded content "
        f"(std {shown_frame.std():.2f} shown vs {hidden_frame.std():.2f} hidden)"
    )


# ── Orientation, in the real pipeline ─────────────────────────────────────────

def test_recorded_video_is_not_upside_down(renderer):
    """The HUD's key hints sit at the BOTTOM of the window (hud.py paints them
    at window_height - …), and the top-centre strip is empty. If vflip were
    missing or doubled, the hint text would show up along the top instead."""
    frame = _decode_first_frame(_record(renderer, 8))

    strip_w = slice(W // 4, 3 * W // 4)     # centre column: avoids the left buttons
    top     = frame[0:28, strip_w].std()
    bottom  = frame[H - 28:H, strip_w].std()
    assert bottom > top + 3, (
        f"bottom-centre strip should hold the key hints, top should be empty "
        f"(std top {top:.2f}, bottom {bottom:.2f}) — video may be flipped"
    )


# ── Frame accounting through the real render loop ─────────────────────────────

def test_one_video_frame_per_rendered_frame(renderer):
    renderer.toggle_hud()
    n = 15
    path = _record(renderer, n)
    assert renderer.recorder.frame_count + renderer.recorder.dropped == n
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert int(out) == renderer.recorder.frame_count


def test_not_recording_leaves_no_file_and_no_badge(renderer, tmp_path):
    """Rendering without recording must not touch the recordings dir."""
    for i in range(5):
        renderer.render(fps=60.0, step_count=i, n_alive=0)
    assert not renderer.recorder.is_recording
    assert renderer._badge_text is None
    assert list(tmp_path.glob("*.mp4")) == []


# ── Recording controls in the render-settings panel ───────────────────────────

def test_rec_fps_slider_exists_with_the_requested_range(renderer):
    """A 15-90 fps bar, defaulting to the config value."""
    s = next(s for s in renderer._render_sliders if s.field == "recording_fps")
    assert s._linear_range == (15.0, 90.0)
    assert s.value == pytest.approx(renderer.config.recording_fps)


def test_rec_fps_slider_drag_sets_the_next_recordings_fps(renderer):
    """Dragging the bar must reach the recorder — but only on the next start."""
    s = next(s for s in renderer._render_sliders if s.field == "recording_fps")
    renderer.toggle_render_params()                 # open the panel
    track = s._track_rect
    assert renderer.handle_mousedown_slider((track.right, track.centery)) is True
    assert renderer._render_settings["recording_fps"] == pytest.approx(90.0)

    before = renderer.recorder.fps
    renderer.toggle_recording()
    assert renderer.recorder.fps == 90, f"slider value never reached the recorder (was {before})"
    renderer.toggle_recording()


def test_realtime_toggle_flips_and_reaches_the_recorder(renderer):
    assert renderer._render_settings["recording_realtime"] is False

    renderer.toggle_render_params()
    rect = renderer._rec_realtime_rect
    assert renderer.handle_mousedown_slider(rect.center) is True
    assert renderer._render_settings["recording_realtime"] is True

    renderer.toggle_recording()
    assert renderer.recorder.realtime is True
    renderer.toggle_recording()


def test_rec_controls_are_inert_while_the_panel_is_closed(renderer):
    """The toggle rect must not be clickable when its panel isn't showing."""
    assert renderer._show_render_params is False
    assert renderer.handle_mousedown_slider(renderer._rec_realtime_rect.center) is False
    assert renderer._render_settings["recording_realtime"] is False


def test_rec_controls_are_inert_while_the_hud_is_hidden(renderer):
    renderer.toggle_render_params()
    renderer.toggle_hud()
    assert renderer.handle_mousedown_slider(renderer._rec_realtime_rect.center) is False
    assert renderer._render_settings["recording_realtime"] is False


def test_panel_reset_restores_both_fps_and_realtime(renderer):
    """The panel-level reset owns the toggle too, even though it has no slider."""
    renderer.toggle_render_params()
    s = next(s for s in renderer._render_sliders if s.field == "recording_fps")
    renderer.handle_mousedown_slider((s._track_rect.right, s._track_rect.centery))
    renderer.handle_mousedown_slider(renderer._rec_realtime_rect.center)
    assert renderer._render_settings["recording_fps"] == pytest.approx(90.0)
    assert renderer._render_settings["recording_realtime"] is True

    renderer.handle_mousedown_slider(renderer._render_params_reset_rect.center)
    assert renderer._render_settings["recording_fps"] == pytest.approx(
        renderer.config.recording_fps)
    assert renderer._render_settings["recording_realtime"] is False


def test_render_panel_fits_inside_the_production_window(tmp_path):
    """Adding the fps slider + toggle row must not push the panel off-screen.

    Checked at the real 1280x720 window, not the small fixture: the HUD layout
    is absolute (button strip pitch, fixed slider rows) and only ever had to fit
    the production size. This is the constraint that kept the recording controls
    out of a 10th left-strip button — that would shift slider_start_y down 30px,
    and the physics panel is already at its minimum row pitch.
    """
    from halflife.config import SimConfig
    from halflife.renderer import Renderer

    config = SimConfig(num_particles=64, num_species=2, recording_dir=str(tmp_path))
    r = Renderer(config)
    try:
        top    = r._slider_start_y - 30
        bottom = top + r._render_slider_content_h + 36
        assert bottom <= config.window_height, (
            f"render panel runs to y={bottom}, past the {config.window_height}px window"
        )
        assert r._rec_realtime_rect.bottom <= bottom, "toggle spills out of its panel"
        # Not asserted: the PHYSICS panel already overflows ~12px at 720 in
        # edges mode (15 sliders against the _MIN_ROW_H=28 floor in
        # _rebuild_physics_sliders). That predates the recording work and is
        # unrelated to it — but it is why the fps slider went into the render
        # panel instead of behind a new button, which would have shifted
        # slider_start_y down another 30px and made it decidedly worse.
    finally:
        r.close()
        pygame.display.quit()


def test_realtime_recording_produces_a_playable_file(renderer, tmp_path):
    """End-to-end through the real GL path with realtime pacing on."""
    renderer.toggle_hud()
    renderer._render_settings["recording_realtime"] = True
    path = _record(renderer, 10)
    assert renderer.recorder.realtime is True
    frame = _decode_first_frame(path)
    assert frame.shape == (H, W, 3)
    # Realtime writes at the output rate regardless of how fast frames arrived.
    assert renderer.recorder.frame_count >= 1


def test_close_finalizes_an_active_recording(tmp_path):
    """Quitting mid-recording must leave a playable file, not a truncated one."""
    from halflife.config import SimConfig
    from halflife.renderer import Renderer

    config = SimConfig(num_particles=64, num_species=2,
                       window_width=W, window_height=H,
                       recording_dir=str(tmp_path), recording_fps=30)
    r = Renderer(config)
    path = r.toggle_recording()
    for i in range(10):
        r.render(fps=60.0, step_count=i, n_alive=0)
    r.close()                     # simulates Q / crash path via the finally block
    pygame.display.quit()

    assert not r.recorder.is_recording
    # A file with no moov atom makes ffprobe fail; decoding proves it finalized.
    frame = _decode_first_frame(path)
    assert frame.shape == (H, W, 3)
