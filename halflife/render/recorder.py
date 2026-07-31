"""Video recording: raw framebuffer bytes → H.264 MP4 via an ffmpeg subprocess.

Deliberately knows nothing about OpenGL, pygame, JAX, or the simulation — it
accepts raw RGB byte blobs and writes a file. That keeps it testable without a
GL context (see tests/test_recorder.py) and keeps renderer.py from growing
another responsibility.

Design notes:

  * ffmpeg is fed rawvideo on stdin from a background daemon thread. The queue
    between submit() and that thread is BOUNDED and drops on overflow, so a
    slow encoder degrades the video but never stalls the render loop. Writing
    to a full pipe from the render thread would couple simulation framerate to
    encoder throughput.

  * OpenGL's framebuffer origin is bottom-left, so glReadPixels hands back rows
    in bottom-up order. ffmpeg's `vflip` filter corrects this — cheaper than
    flipping a 2.7 MB numpy array on the CPU every frame.

  * Every public method is safe to call in any state (submit() before start()
    is a silent no-op, stop() when idle returns None). Call sites in render()
    therefore need no state checks of their own.

  * Two pacing modes, both producing a CONSTANT-framerate file (ffmpeg is
    launched with a fixed -r, and rawvideo carries no timestamps):

      fixed (default)  one rendered frame → one video frame. Deterministic; a
                       session slower than `fps` plays back sped up.
      realtime         the writer thread repeats or skips frames against the
                       wall clock so video time tracks real time. This is how
                       screen recorders do it — duplicates encode to almost
                       nothing, so the file-size cost is far below the frame
                       multiplier.
"""

import os
import queue
import shutil
import subprocess
import threading
import time


class RecorderUnavailable(RuntimeError):
    """Raised by start() when recording cannot begin (no ffmpeg, bad out_dir).

    Callers are expected to catch this and carry on — a missing encoder should
    never take down the simulation.
    """


# Bounded hand-off between the render thread and the ffmpeg writer thread.
#
# Sized off measurements at 1280x720 (libx264 veryfast crf18, this machine):
#   * ffmpeg needs ~55 ms after Popen before it reads the first frame, so the
#     queue's real job is absorbing that startup burst — without slack the
#     opening ~3 frames of every recording would be dropped.
#   * steady-state encode is ~10.5 ms/frame, i.e. a ~95 fps ceiling. A submit
#     rate sustained above that drops frames no matter how deep the queue is,
#     so depth beyond the startup burst buys nothing but latency and RAM.
# 16 frames ≈ 265 ms of slack at 60 fps, ≈ 44 MB of raw RGB at 720p — held only
# while recording. Deeper would silently buffer hundreds of MB and hide a real
# sustained stall that the dropped-frame counter should be surfacing instead.
QUEUE_MAXSIZE = 16

# Sentinel pushed by stop() to tell the writer thread to finish and exit.
_SENTINEL = object()


def unique_path(out_dir: str, prefix: str = "halflife", ext: str = "mp4") -> str:
    """Return a not-yet-existing path `out_dir/<prefix>_<YYYYmmdd_HHMMSS>.<ext>`.

    Second-resolution timestamps can collide if two recordings start inside the
    same second, so a `_1`, `_2`, … suffix is appended until the name is free.
    Recordings are never overwritten.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(out_dir, f"{prefix}_{stamp}.{ext}")
    n = 0
    while os.path.exists(candidate):
        n += 1
        candidate = os.path.join(out_dir, f"{prefix}_{stamp}_{n}.{ext}")
    return candidate


class VideoRecorder:
    """Encodes submitted RGB frames to an MP4 through a piped ffmpeg process."""

    def __init__(self, width: int, height: int, fps: int,
                 out_dir: str = "recordings", crf: int = 18,
                 realtime: bool = False):
        self.width   = int(width)
        self.height  = int(height)
        self.out_dir = out_dir
        self.crf     = int(crf)
        # fps and realtime are plain mutable attributes, read at start(): the UI
        # sets them from its slider/toggle just before a recording begins.
        # Changing them mid-recording would desync the already-launched ffmpeg
        # (-r is fixed at spawn), so they only ever apply to the NEXT take.
        self.fps      = int(fps)
        self.realtime = bool(realtime)

        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._path: str | None = None
        self._active = False
        self._t_end: float | None = None

        self._frame_count = 0
        self._dropped     = 0
        # Realtime bookkeeping: frames skipped because the app outran the output
        # rate, and duplicate frames withheld by the burst cap after a stall.
        self._skipped   = 0
        self._clamped   = 0
        self._t_start: float | None = None
        # Set by the writer thread when ffmpeg's stdin dies mid-recording, read
        # by the render thread. A plain str assignment is atomic enough here —
        # no lock needed for a single-writer / single-reader string slot.
        self._error: str | None = None

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._active

    @property
    def path(self) -> str | None:
        return self._path

    @property
    def frame_count(self) -> int:
        """Frames handed to the writer thread (dropped frames not counted)."""
        return self._frame_count

    @property
    def dropped(self) -> int:
        """Frames lost to encoder backpressure — a defect, unlike `skipped`."""
        return self._dropped

    @property
    def skipped(self) -> int:
        """Realtime only: frames with no slot because the app outran `fps`."""
        return self._skipped

    @property
    def clamped(self) -> int:
        """Realtime only: duplicate frames withheld by the burst cap."""
        return self._clamped

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def elapsed(self) -> float:
        """Duration of the video produced so far, in seconds.

        This is *video* time (frame_count / fps), not wall-clock: the output is
        a fixed-framerate stream, so this is what the finished file will report.
        """
        return self._frame_count / self.fps if self.fps else 0.0

    @property
    def wall_elapsed(self) -> float:
        """Seconds of real time since start(). Frozen at stop()."""
        if self._t_start is None:
            return 0.0
        return (self._t_end if self._t_end is not None else time.monotonic()) - self._t_start

    @property
    def speed_factor(self) -> float:
        """How much faster the finished video plays than real time.

        In the default mode the output is fixed-framerate, so a session captured
        at 10 fps live and written at 30 fps plays 3x fast. Worth reporting: it
        is invisible in the file, and the cure is the realtime toggle (or
        lowering recording_fps). In realtime mode this should read ~1.0 by
        construction. Returns 0.0 when there isn't enough data to say.
        """
        wall = self.wall_elapsed
        if wall <= 0 or self._frame_count == 0:
            return 0.0
        return self.fps / (self._frame_count / wall)

    @property
    def size_bytes(self) -> int:
        """Current size of the output file; 0 before ffmpeg's first flush."""
        if self._path is None:
            return 0
        try:
            return os.stat(self._path).st_size
        except OSError:
            return 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _ffmpeg_command(self, path: str) -> list:
        """Build the ffmpeg argv. Split out so tests can assert on the flags."""
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            # Input: raw RGB frames on stdin, at the declared output rate.
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            # glReadPixels rows arrive bottom-up; undo that here.
            "-vf", "vflip",
            # yuv420p is required for playback in browsers/QuickTime — libx264
            # would otherwise pick yuv444p from rgb24 input and produce a file
            # most players refuse.
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            path,
        ]

    def start(self) -> str:
        """Spawn ffmpeg and the writer thread. Returns the output path.

        Raises RecorderUnavailable if ffmpeg is missing or the output directory
        cannot be created/written. Idempotent while already recording (returns
        the in-progress path).
        """
        if self._active:
            return self._path
        # A previous recording that died mid-flight (broken pipe) leaves _active
        # False but its process and thread still around. Reap it before
        # overwriting the slots, so pressing R again after a failure recovers
        # cleanly instead of leaking a zombie ffmpeg.
        self.stop()

        if shutil.which("ffmpeg") is None:
            raise RecorderUnavailable(
                "ffmpeg not found on PATH — install it to enable recording "
                "(WSL: sudo apt install ffmpeg)"
            )

        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except OSError as e:
            raise RecorderUnavailable(f"cannot create {self.out_dir!r}: {e}") from e

        path = unique_path(self.out_dir)
        try:
            proc = subprocess.Popen(
                self._ffmpeg_command(path),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
        except OSError as e:
            raise RecorderUnavailable(f"failed to launch ffmpeg: {e}") from e

        self._proc        = proc
        self._path        = path
        self._queue       = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._frame_count = 0
        self._dropped     = 0
        self._skipped     = 0
        self._clamped     = 0
        self._error       = None
        self._active      = True
        self._t_start     = time.monotonic()
        self._t_end       = None

        self._thread = threading.Thread(
            target=self._writer_loop, name="video-writer", daemon=True
        )
        self._thread.start()
        return path

    def submit(self, frame: bytes) -> None:
        """Enqueue one raw RGB frame. Never blocks; drops when the queue is full.

        No-op when not recording, so render() can call this unconditionally.
        The wall-clock stamp is taken HERE, on the render thread, so realtime
        pacing reflects when the frame was actually drawn rather than when the
        writer thread got around to it.
        """
        if not self._active or self._queue is None:
            return
        try:
            self._queue.put_nowait((frame, time.monotonic()))
        except queue.Full:
            # Encoder is behind. Dropping keeps the render loop at full speed;
            # the count surfaces in the stop() summary.
            self._dropped += 1

    def stop(self) -> str | None:
        """Drain, close ffmpeg's stdin, and wait for a finalized file.

        Returns the output path, or None if there was nothing to tear down.
        Waiting on the process matters: without it the MP4's moov atom may not
        be written and the file won't play.

        Keyed on the *presence* of a process/thread rather than on _active, so
        it also cleans up a recording the writer thread aborted after a broken
        pipe (which clears _active on its own). Safe to call repeatedly.
        """
        if self._proc is None and self._thread is None:
            return None
        self._active = False
        # Freeze wall_elapsed here, before the drain/join below spends time that
        # isn't part of the captured session.
        self._t_end = time.monotonic()

        path = self._path
        if self._queue is not None:
            # Sentinel rather than a flag so the writer thread wakes up
            # immediately instead of sitting in a blocking get().
            try:
                self._queue.put(_SENTINEL, timeout=2.0)
            except queue.Full:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)

        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        self._queue  = None
        self._thread = None
        return path

    # ── Writer thread ─────────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        """Pull frames off the queue and write them to ffmpeg's stdin.

        In realtime mode this is where wall-clock pacing happens: each frame is
        repeated (or skipped) so that video time tracks real time. Doing it here
        rather than in submit() keeps the duplication cost off the render thread
        — and duplicate frames cost x264 almost nothing, since they encode as
        near-empty P-frames.
        """
        q     = self._queue
        proc  = self._proc
        stdin = proc.stdin if proc is not None else None
        # Cap the burst a single frame may expand into, so a multi-second stall
        # (e.g. the JIT retrace on a bond-mode switch) can't emit thousands of
        # duplicates. Withheld frames are counted, never silently swallowed.
        max_repeats = max(2, self.fps * 2)
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            frame, t_submit = item
            if stdin is None:
                continue

            if self.realtime:
                # How many output frames should exist once this one is placed:
                # it occupies every slot through floor(dt*fps), hence the +1.
                # (Without it the very first frame, at dt=0, would want zero
                # slots and get skipped, starting the video a frame late.)
                #
                # The epsilon guards the truncation against float error: at
                # fps=30, dt=0.9 evaluates to 26.999999999999996, and a bare
                # int() would silently lose a frame every time a timestamp
                # lands on a slot boundary. 1 microsecond is far below any
                # meaningful video timing.
                target  = int((t_submit - self._t_start) * self.fps + 1e-6) + 1
                repeats = target - self._frame_count
                if repeats <= 0:
                    # The app is rendering faster than the output rate — this
                    # frame has no slot. Deliberate resampling, not a failure.
                    self._skipped += 1
                    continue
                if repeats > max_repeats:
                    self._clamped += repeats - max_repeats
                    repeats = max_repeats
            else:
                repeats = 1

            try:
                for _ in range(repeats):
                    stdin.write(frame)
                    self._frame_count += 1
            except (BrokenPipeError, ValueError, OSError) as e:
                # ffmpeg died (bad args, disk full, killed). Stop accepting
                # frames so the render loop sees is_recording False and drops
                # the REC badge — a visible signal that something broke.
                self._error  = f"encoder write failed: {e}"
                self._active = False
                break
        try:
            if stdin is not None and not stdin.closed:
                stdin.close()
        except OSError:
            pass
