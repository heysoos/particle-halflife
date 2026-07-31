"""Tests for halflife/render/recorder.py — the ffmpeg video pipe.

Pure CPU: no GL context, no display, no JAX. The GL side (ctx.screen.read())
cannot run headless here and is covered by the manual checklist in
docs/superpowers/specs/2026-07-29-simulation-recording-design.md.

Tests that actually encode are skipped when ffmpeg is absent so the suite still
passes on machines without it.
"""

import json
import os
import queue
import shutil
import subprocess
import time

import numpy as np
import pytest

from halflife.render.recorder import (
    QUEUE_MAXSIZE,
    RecorderUnavailable,
    VideoRecorder,
    unique_path,
)

HAVE_FFMPEG  = shutil.which("ffmpeg") is not None
HAVE_FFPROBE = shutil.which("ffprobe") is not None

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
needs_ffprobe = pytest.mark.skipif(not HAVE_FFPROBE, reason="ffprobe not on PATH")

# Small enough to encode fast, but libx264's yuv420p needs even dimensions.
W, H, FPS = 64, 48, 30


def _gradient_frame(i: int) -> bytes:
    """A distinguishable RGB frame: horizontal ramp, brightness varying with i."""
    row = np.linspace(0, 255, W, dtype=np.uint8)
    img = np.stack([np.tile(row, (H, 1))] * 3, axis=-1)
    img = (img.astype(np.uint16) * (i % 8 + 1) // 8).astype(np.uint8)
    return img.tobytes()


def _probe(path: str) -> dict:
    """ffprobe the first video stream as a dict."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_streams", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["streams"][0]


# ── unique_path ───────────────────────────────────────────────────────────────

def test_unique_path_avoids_existing_file(tmp_path):
    """A pre-existing name must not be handed out again, and must survive."""
    first = unique_path(str(tmp_path))
    with open(first, "wb") as f:
        f.write(b"do not clobber me")

    second = unique_path(str(tmp_path))
    assert second != first
    # The occupied slot is untouched.
    with open(first, "rb") as f:
        assert f.read() == b"do not clobber me"


def test_unique_path_survives_many_collisions(tmp_path):
    """Repeated collisions inside one second keep producing fresh names."""
    seen = set()
    for _ in range(5):
        p = unique_path(str(tmp_path))
        assert p not in seen
        seen.add(p)
        open(p, "wb").close()
    assert len(seen) == 5


def test_unique_path_uses_mp4_extension(tmp_path):
    assert unique_path(str(tmp_path)).endswith(".mp4")
    assert os.path.basename(unique_path(str(tmp_path))).startswith("halflife_")


# ── Round trip ────────────────────────────────────────────────────────────────

@needs_ffmpeg
@needs_ffprobe
def test_round_trip_frame_count_and_dimensions(tmp_path):
    """N submitted frames must come out as N frames at the requested size."""
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    path = rec.start()
    n = 30
    for i in range(n):
        rec.submit(_gradient_frame(i))
        # Pace like a real render loop. Submitting as fast as Python can loop
        # outruns ffmpeg's ~55 ms startup and the queue legitimately drops
        # frames — that is the designed backpressure behaviour, tested
        # separately below, not what this test is about.
        time.sleep(1 / FPS)
    assert rec.stop() == path

    assert os.path.exists(path)
    assert os.stat(path).st_size > 0
    assert rec.dropped == 0, "a paced 64x48 encode should never fall behind"

    stream = _probe(path)
    assert int(stream["nb_read_frames"]) == n
    assert (stream["width"], stream["height"]) == (W, H)
    assert stream["pix_fmt"] == "yuv420p"  # required for broad playability


@needs_ffmpeg
def test_every_submitted_frame_is_either_encoded_or_counted_dropped(tmp_path):
    """The conservation invariant: nothing vanishes silently.

    Submitting flat-out deliberately overruns the encoder, which is exactly when
    a frame could go missing without anyone noticing. frame_count + dropped must
    still account for every submission.
    """
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    rec.start()
    n = 200
    for i in range(n):
        rec.submit(_gradient_frame(i))
    rec.stop()
    assert rec.frame_count + rec.dropped == n
    assert rec.dropped > 0, "an unpaced 200-frame burst should overrun the queue"


@needs_ffmpeg
def test_output_dir_is_created(tmp_path):
    """out_dir need not exist beforehand."""
    target = tmp_path / "nested" / "recordings"
    rec = VideoRecorder(W, H, FPS, out_dir=str(target))
    path = rec.start()
    rec.submit(_gradient_frame(0))
    rec.stop()
    assert target.is_dir()
    assert os.path.exists(path)


@needs_ffmpeg
def test_consecutive_recordings_get_distinct_files(tmp_path):
    """start → stop → start must open a new file, not append to the old one."""
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    p1 = rec.start()
    for i in range(5):
        rec.submit(_gradient_frame(i))
    rec.stop()

    p2 = rec.start()
    for i in range(5):
        rec.submit(_gradient_frame(i))
    rec.stop()

    assert p1 != p2
    assert os.path.exists(p1) and os.path.exists(p2)


# ── Orientation ───────────────────────────────────────────────────────────────

@needs_ffmpeg
def test_vflip_corrects_bottom_up_rows(tmp_path):
    """The single most likely defect: a silently upside-down video.

    glReadPixels returns rows bottom-up, so a frame whose *last* rows are white
    represents an image whose *top* is white. After the vflip filter the decoded
    frame must therefore be white in its top half.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[H // 2:, :, :] = 255          # bottom half of the byte buffer …
                                      # … = top half of the image, in GL order

    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    path = rec.start()
    for _ in range(6):                # a few frames so x264 definitely emits one
        rec.submit(img.tobytes())
    rec.stop()

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    decoded = np.frombuffer(raw[: W * H * 3], dtype=np.uint8).reshape(H, W, 3)

    top    = decoded[: H // 2].mean()
    bottom = decoded[H // 2:].mean()
    assert top > 200,    f"top half should be white, got mean {top:.1f}"
    assert bottom < 55,  f"bottom half should be black, got mean {bottom:.1f}"


# ── Backpressure ──────────────────────────────────────────────────────────────

def test_full_queue_drops_instead_of_blocking(tmp_path):
    """A stalled encoder must cost frames, never block the render thread."""
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    # Stand in for a started recorder whose writer thread never drains. No
    # ffmpeg needed — submit() only touches _active and _queue.
    rec._active = True
    rec._queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    n = QUEUE_MAXSIZE + 20
    for i in range(n):
        rec.submit(_gradient_frame(i))   # must not raise, must not hang

    assert rec._queue.qsize() == QUEUE_MAXSIZE
    assert rec.dropped == n - QUEUE_MAXSIZE
    assert rec.frame_count == 0          # nothing reached the encoder


# ── State machine ─────────────────────────────────────────────────────────────

def test_submit_before_start_is_noop(tmp_path):
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    rec.submit(_gradient_frame(0))       # must not raise
    assert rec.frame_count == 0
    assert rec.dropped == 0
    assert not rec.is_recording


def test_stop_before_start_returns_none(tmp_path):
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    assert rec.stop() is None
    assert not rec.is_recording


@needs_ffmpeg
def test_double_stop_is_safe(tmp_path):
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    rec.start()
    rec.submit(_gradient_frame(0))
    assert rec.stop() is not None
    assert rec.stop() is None            # second call is a no-op
    assert not rec.is_recording


@needs_ffmpeg
def test_submit_after_stop_is_noop(tmp_path):
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    rec.start()
    rec.submit(_gradient_frame(0))
    rec.stop()
    before = rec.frame_count
    rec.submit(_gradient_frame(1))       # must not raise
    assert rec.frame_count == before


@needs_ffmpeg
def test_double_start_is_idempotent(tmp_path):
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    p1 = rec.start()
    p2 = rec.start()                     # already recording → same path
    assert p1 == p2
    rec.stop()
    assert len(list(tmp_path.glob("*.mp4"))) == 1


# ── Introspection ─────────────────────────────────────────────────────────────

def test_elapsed_is_video_time(tmp_path):
    """elapsed reports frames/fps — the duration of the finished file."""
    rec = VideoRecorder(W, H, 60, out_dir=str(tmp_path))
    assert rec.elapsed == 0.0
    rec._frame_count = 90
    assert rec.elapsed == pytest.approx(1.5)


def test_size_bytes_zero_before_start(tmp_path):
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    assert rec.size_bytes == 0


def test_speed_factor_zero_before_any_frames(tmp_path):
    """No data yet must read as 0.0, not divide by zero."""
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    assert rec.speed_factor == 0.0
    assert rec.wall_elapsed == 0.0


def test_speed_factor_reports_fixed_fps_speedup(tmp_path):
    """30 frames captured over 3 s of real time, written at 60 fps → 6x fast."""
    rec = VideoRecorder(W, H, 60, out_dir=str(tmp_path))
    rec._t_start     = 100.0
    rec._t_end       = 103.0     # 3 s wall
    rec._frame_count = 30        # → 10 fps live
    assert rec.wall_elapsed == pytest.approx(3.0)
    assert rec.speed_factor == pytest.approx(6.0)


def test_speed_factor_is_one_when_output_matches_live_rate(tmp_path):
    rec = VideoRecorder(W, H, 10, out_dir=str(tmp_path))
    rec._t_start, rec._t_end, rec._frame_count = 0.0, 3.0, 30
    assert rec.speed_factor == pytest.approx(1.0)


@needs_ffmpeg
def test_wall_elapsed_freezes_at_stop(tmp_path):
    """wall_elapsed must measure the captured session, not include teardown."""
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    rec.start()
    rec.submit(_gradient_frame(0))
    rec.stop()
    frozen = rec.wall_elapsed
    time.sleep(0.15)
    assert rec.wall_elapsed == frozen


@needs_ffmpeg
def test_size_bytes_reads_the_output_file(tmp_path):
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    rec.start()
    for i in range(20):
        rec.submit(_gradient_frame(i))
    rec.stop()
    assert rec.size_bytes == os.stat(rec.path).st_size > 0


# ── Realtime pacing ───────────────────────────────────────────────────────────
#
# These bypass submit() and push (frame, timestamp) tuples straight onto the
# queue, so the writer thread's wall-clock arithmetic is exercised against
# chosen times instead of whatever the machine happened to do. No sleeps, no
# clock patching, no flakiness.

def _drain(rec, timeout=5.0):
    """Wait for the writer thread to consume everything queued so far."""
    deadline = time.monotonic() + timeout
    while not rec._queue.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)          # let the last item finish being written


def _feed(rec, times):
    """Enqueue one frame per entry in `times` (seconds after recording start)."""
    frame = _gradient_frame(0)
    for dt in times:
        rec._queue.put((frame, rec._t_start + dt))
    _drain(rec)


@needs_ffmpeg
def test_realtime_duplicates_frames_when_app_is_slower_than_output(tmp_path):
    """10 fps live at 30 fps output → each frame written ~3x, so video time
    tracks wall time instead of running 3x fast."""
    rec = VideoRecorder(W, H, 30, out_dir=str(tmp_path), realtime=True)
    rec.start()
    _feed(rec, [i / 10.0 for i in range(10)])   # 10 frames spanning 0.9 s
    rec.stop()

    # Frame at dt fills slots through int(dt*30); last is dt=0.9 → 28 frames.
    assert rec.frame_count == 28
    assert rec.elapsed == pytest.approx(28 / 30, abs=0.02)
    assert rec.skipped == 0
    assert rec.dropped == 0


@needs_ffmpeg
def test_realtime_skips_frames_when_app_is_faster_than_output(tmp_path):
    """60 fps live at 30 fps output → half the frames have no slot."""
    rec = VideoRecorder(W, H, 30, out_dir=str(tmp_path), realtime=True)
    rec.start()
    n = 20
    _feed(rec, [i / 60.0 for i in range(n)])    # 20 frames spanning 0.317 s
    rec.stop()

    assert rec.frame_count == 10
    assert rec.skipped == n - rec.frame_count
    assert rec.dropped == 0, "skipping is deliberate resampling, not a failure"


@needs_ffmpeg
def test_realtime_first_frame_is_never_skipped(tmp_path):
    """dt=0 must still produce a frame — otherwise every video starts late."""
    rec = VideoRecorder(W, H, 30, out_dir=str(tmp_path), realtime=True)
    rec.start()
    _feed(rec, [0.0])
    rec.stop()
    assert rec.frame_count == 1
    assert rec.skipped == 0


@needs_ffmpeg
def test_realtime_burst_after_a_stall_is_capped_and_counted(tmp_path):
    """A long stall (e.g. a JIT retrace) must not emit thousands of duplicates,
    and whatever is withheld must be reported rather than silently dropped."""
    fps = 30
    rec = VideoRecorder(W, H, fps, out_dir=str(tmp_path), realtime=True)
    rec.start()
    _feed(rec, [0.0, 30.0])          # 30-second gap → wants ~900 duplicates
    rec.stop()

    cap = fps * 2
    assert rec.frame_count == 1 + cap
    assert rec.clamped > 0
    assert rec.clamped == (int(30.0 * fps) + 1) - 1 - cap


@needs_ffmpeg
def test_realtime_speed_factor_is_about_one(tmp_path):
    """The whole point of realtime: playback matches what you saw."""
    rec = VideoRecorder(W, H, 30, out_dir=str(tmp_path), realtime=True)
    rec.start()
    _feed(rec, [i / 12.0 for i in range(12)])   # ~12 fps live
    rec.stop()
    rec._t_start, rec._t_end = 0.0, 11 / 12.0   # the wall span we simulated
    assert rec.speed_factor == pytest.approx(1.0, abs=0.1)


@needs_ffmpeg
def test_fixed_mode_is_still_one_frame_in_one_frame_out(tmp_path):
    """Realtime must be opt-in — the default path is unchanged."""
    rec = VideoRecorder(W, H, 30, out_dir=str(tmp_path))
    assert rec.realtime is False
    rec.start()
    _feed(rec, [i / 10.0 for i in range(10)])   # same input as the 28-frame case
    rec.stop()
    assert rec.frame_count == 10
    assert rec.skipped == 0 and rec.clamped == 0


def test_fps_and_realtime_are_settable_before_start(tmp_path):
    """The UI sets these between takes; they must reach the ffmpeg command."""
    rec = VideoRecorder(W, H, 60, out_dir=str(tmp_path))
    rec.fps, rec.realtime = 24, True
    assert rec._ffmpeg_command("/o.mp4")[
        rec._ffmpeg_command("/o.mp4").index("-r") + 1] == "24"
    assert rec.realtime is True


# ── Failure modes ─────────────────────────────────────────────────────────────

def test_missing_ffmpeg_raises_recorder_unavailable(tmp_path, monkeypatch):
    """No encoder → a clear error, and the recorder stays idle."""
    monkeypatch.setattr("halflife.render.recorder.shutil.which", lambda _: None)
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    with pytest.raises(RecorderUnavailable, match="ffmpeg"):
        rec.start()
    assert not rec.is_recording
    assert rec.path is None


def test_unwritable_out_dir_raises_recorder_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("halflife.render.recorder.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "halflife.render.recorder.os.makedirs",
        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only file system")),
    )
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path / "nope"))
    with pytest.raises(RecorderUnavailable, match="cannot create"):
        rec.start()
    assert not rec.is_recording


@needs_ffmpeg
def test_broken_pipe_deactivates_and_records_error(tmp_path):
    """ffmpeg dying mid-recording must clear is_recording (so the badge drops)
    and leave a readable error, not an exception on the render thread."""
    rec = VideoRecorder(W, H, FPS, out_dir=str(tmp_path))
    rec.start()
    rec._proc.kill()
    rec._proc.wait()
    # Keep submitting until the writer thread notices the dead pipe. The queue
    # is bounded, so this cannot spin forever on a live recorder either.
    for _ in range(500):
        if not rec.is_recording:
            break
        rec.submit(_gradient_frame(0))
    assert not rec.is_recording
    assert rec.error is not None
    # stop() must still reap the corpse rather than early-returning on _active.
    rec.stop()
    assert rec._proc is None and rec._thread is None


def test_ffmpeg_command_shape(tmp_path):
    """Guard the flags the design depends on: rawvideo in, vflip, yuv420p out."""
    rec = VideoRecorder(1280, 720, 60, out_dir=str(tmp_path), crf=18)
    cmd = rec._ffmpeg_command("/out/x.mp4")
    assert cmd[0] == "ffmpeg"
    assert cmd[-1] == "/out/x.mp4"
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "rawvideo"
    assert cmd[cmd.index("-vf") + 1] == "vflip"
    assert cmd[cmd.index("-s") + 1] == "1280x720"
    assert cmd[cmd.index("-r") + 1] == "60"
    assert cmd[cmd.index("-crf") + 1] == "18"
    # rgb24 for the input, yuv420p for the output — both -pix_fmt flags present.
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-pix_fmt"] == ["rgb24", "yuv420p"]
