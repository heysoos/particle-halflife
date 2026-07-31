# Simulation Recording + HUD Hiding

**Date:** 2026-07-29
**Status:** Implemented. Cost/size figures below are measured, not estimated —
the initial file-size estimate was wrong by ~14× and the playback speed-up by ~3×;
both are corrected in place and noted where they changed the design.

## Motivation

There is no way to get the simulation *out* of the app. The only capture path is
`S` → a single PNG screenshot (`main.py:342`), which cannot show the dynamics —
and the dynamics are the whole point of a simulator where everything decays. Sharing
a result today means pointing a third-party screen recorder at the window, which
captures the desktop chrome, the taskbar, and whatever else is on screen.

This spec adds first-class video recording (`R`) plus a HUD hide toggle (`H`), so a
clean, shareable MP4 of the running simulation can be produced from inside the app.

## Goals

- `R` starts / stops recording to a uniquely-named file under `recordings/`.
- `H` hides the entire HUD, so a recording can be made without buttons, sliders,
  or stats panels in the frame.
- A visible on-screen indicator confirms recording is live — and is **never**
  present in the recorded video.
- Zero cost when not recording. No new Python dependencies.

## Addendum (2026-07-30): live fps control + realtime pacing

Shipped after the measurements below showed the fixed-framerate speed-up was
much worse in practice than assumed at design time (3–8×, not ~2×). Three changes:

- **`recording_fps` default 60 → 30**, and live-tunable from a `rec fps` slider
  ([15, 90]) in the render-settings panel.
- **A `Realtime` toggle** beside it. When on, the writer thread repeats or skips
  frames against the wall clock so video time tracks real time — the standard
  screen-recorder approach, and the only one available given ffmpeg fixes `-r` at
  spawn and rawvideo carries no per-frame timestamps. This supersedes the
  "variable framerate — out of scope" non-goal below; it turned out to need no
  VFR plumbing at all.
- Both are read at `start()` only, since `-r` cannot change mid-recording. The
  REC badge shows the pacing actually in force (`30fps` vs `30fps live`).

Placement: the render-settings panel rather than a new left-strip button, because
the button strip is full — a 10th button shifts `slider_start_y` down 30px and the
physics panel already overflows ~12px at 720 in edges mode (pre-existing, from the
`_MIN_ROW_H = 28` floor).

Measured at 1280×720 / 5k particles, same live rate, burned-in loop:

| | fixed | realtime |
|---|---|---|
| playback | 2.14× fast | **1.00× real time** |
| file size | 4.8 MB | 4.5 MB |

The realtime file is *smaller* despite holding 258 frames against 90 — duplicate
frames encode as near-empty P-frames, so the frame multiplier does not translate
into a size multiplier.

Two details worth keeping:

- The slot arithmetic is `target = int(dt·fps + 1e-6) + 1`. The `+1` stops the
  first frame (dt=0) from claiming zero slots and starting the video late; the
  epsilon guards the truncation, since `0.9*30` evaluates to `26.999999999999996`
  and a bare `int()` silently loses a frame whenever a timestamp lands on a slot
  boundary.
- A single frame's duplicate burst is capped at `2·fps`, so a multi-second stall
  (a JIT retrace on a bond-mode switch) can't emit thousands of duplicates. The
  withheld count is reported at stop rather than dropped silently.

## Non-goals

- Audio. There is none.
- Recording at a resolution other than the window's (`1280x720`). The window is
  deliberately non-resizable (`renderer.py:389`) to avoid texture-resize complexity;
  offscreen supersampled rendering is a separate project.
- Pausing / resuming a single recording, in-app trimming, or GIF export.
- Asynchronous (PBO) framebuffer readback. See "Known costs".
- ~~Variable-framerate output.~~ Superseded — see the addendum above. Real-time
  playback is now supported, achieved by wall-clock frame resampling into a
  constant-framerate stream rather than by per-frame timestamps.

## Architecture

Three units, with the new one carrying almost all the logic.

### 1. `halflife/render/recorder.py` — `VideoRecorder` (new)

Owns everything about turning frames into a file. Knows nothing about OpenGL, pygame,
JAX, or the simulation — it accepts raw bytes. `renderer.py` is already 1708 lines, and
this responsibility is cleanly separable and independently testable, so it gets its own
module.

```python
class VideoRecorder:
    def __init__(self, width: int, height: int, fps: int,
                 out_dir: str = "recordings", crf: int = 18)

    def start(self) -> str            # spawn ffmpeg + writer thread; returns output path
    def submit(self, frame: bytes)    # non-blocking enqueue; no-op when not recording
    def stop(self) -> str | None      # close stdin, join thread, finalize; None if inactive

    # read-only properties
    is_recording: bool
    path: str | None
    frame_count: int          # frames successfully handed to ffmpeg
    dropped: int              # frames discarded due to encoder backpressure
    elapsed: float            # frame_count / fps — video-time, not wall-time
    size_bytes: int           # os.stat of the output file (0 if not yet flushed)
```

**Interface contract:** every method is safe to call in any state. `submit()` before
`start()` or after `stop()` is a silent no-op; `stop()` on an inactive recorder returns
`None`. This keeps the call sites in `render()` free of state checks.

#### ffmpeg invocation

```
ffmpeg -y -loglevel error
       -f rawvideo -pix_fmt rgb24 -s <W>x<H> -r <fps> -i -
       -vf vflip
       -c:v libx264 -preset veryfast -crf <crf> -pix_fmt yuv420p
       recordings/halflife_<timestamp>.mp4
```

- `-vf vflip` — `glReadPixels` returns rows bottom-up; OpenGL's origin is bottom-left.
- `-pix_fmt yuv420p` on the output — required for playback in browsers, QuickTime,
  and most players. Without it libx264 defaults to `yuv444p` from `rgb24` input.
- `-preset veryfast -crf 18` — encodes comfortably faster than realtime at 720p while
  staying visually near-lossless. ~15 MB per minute.
- `-loglevel error` — keeps ffmpeg's banner out of the app's stdout, but still surfaces
  real failures.

ffmpeg is resolved via `shutil.which("ffmpeg")`. If absent, `start()` raises
`RecorderUnavailable` with a message naming the missing binary, and `is_recording`
stays `False` — the app keeps running.

#### Filename uniqueness

`recordings/halflife_YYYYmmdd_HHMMSS.mp4`, resolved at `start()`. If that path already
exists (two recordings begun inside the same second), `_1`, `_2`, … is appended until
the name is free. Nothing is ever overwritten. `out_dir` is created with
`os.makedirs(..., exist_ok=True)`.

#### Backpressure

A `queue.Queue(maxsize=8)` feeds a daemon writer thread that does the blocking
`ffmpeg.stdin.write()`. `submit()` uses `put_nowait()`; on `queue.Full` it increments
`dropped` and discards the frame.

This is a deliberate trade: **a slow encoder degrades the video, never the simulation.**
Writing to a full pipe from the render thread would stall the main loop, coupling sim
framerate to encoder throughput. Dropped frames are reported when recording stops so a
sustained problem is visible rather than silent.

`stop()` drains the queue, closes stdin, joins the writer thread, then `wait()`s on the
ffmpeg process so the MP4's moov atom is written and the file is playable.

### 2. `halflife/renderer.py` — capture point and badge pass

New state: `self._show_hud = True`, `self.recorder = VideoRecorder(...)` (constructed
inactive at init, so no per-toggle setup cost).

The capture slots into the existing tail of `render()` (currently `renderer.py:1604-1624`),
between the HUD blit and the flip:

```
  tonemap trail layer  → ctx.screen        (existing)
  tonemap fresh layer  → ctx.screen        (existing)
  if self._show_hud:  paint + blit HUD     (existing, now gated)
★ if recorder.is_recording:                (new — capture)
★     recorder.submit(ctx.screen.read(components=3, alignment=1))
★ if recorder.is_recording:  blit REC badge   (new — after capture)
  pygame.display.flip()                    (existing)
```

Capturing *after* the HUD blit but *before* the badge blit is what satisfies both
requirements at once: the video contains whatever the HUD is currently showing (so the
stats panel can be recorded deliberately, and `H` yields a clean scene-only video), while
the badge — drawn strictly after the read — cannot reach the file.

New public methods, matching the existing `toggle_*` conventions:

```python
def toggle_hud(self) -> None            # flips _show_hud, sets _hud_dirty
def toggle_recording(self) -> str | None   # start ⇄ stop; returns path on either edge
```

#### The REC badge

A **second full-window RGBA surface/texture pair** (`_badge_surface`, `_badge_texture`)
blitted through the *existing* `hud_prog` + `_hud_quad_vao`. No new shaders, no new
geometry, no NDC arithmetic — it is the HUD compositing path, reused.

The alternative (a small positioned quad) would need either a new vertex shader or
UV-remapping arithmetic, to save 3.7 MB of VRAM on a machine that already holds two
RGBA16F trail buffers. Not worth the complexity.

Repaint is gated on the badge's *displayed* text changing — elapsed time and size are
shown at 1-second resolution, so the surface repaints and re-uploads roughly once per
second rather than every frame.

Contents, top-right corner:
- A filled red dot, alpha-pulsed at 1 Hz off `frame_count / fps` (not wall-clock, so
  the pulse rate matches the video's own time and stays deterministic).
- `REC 01:23 · 47 MB` — `elapsed` as `MM:SS`, `size_bytes` as whole MB.

Drawing the badge in its own pass means it renders even when the HUD is hidden, which
is required: `H` must not be able to hide the recording indicator.

#### Input gating while the HUD is hidden

With the HUD hidden its buttons and sliders are invisible but their hit rects are still
live, so a click intended to frame a shot could silently hit "Reset". `handle_click()`
returns `None` and `handle_mousedown_slider()` returns `False` when `_show_hud` is
`False`.

Camera pan / zoom / right-click-reset and particle picking stay active — those are how
you frame a shot, and none of them depend on a visible widget.

### 3. `halflife/main.py` — key bindings

| Key | Before | After |
|-----|--------|-------|
| `R` | reset | **toggle recording** |
| `N` | — | **reset** |
| `H` | — | **toggle HUD** |

`R` is the natural mnemonic for record and the user chose to give it up from reset; reset
moves to `N` ("new world") to keep it a single unmodified keypress, and the HUD's Reset
button is unaffected.

The main loop gains a `try/finally` so that `Q`, a window close, or an unhandled
exception all still reach `recorder.stop()`. Without it a crash mid-recording leaves a
raw, unfinalized file with no moov atom — unplayable. Finalization goes in
`Renderer.close()` alongside the other resource releases.

The module docstring's keyboard table is updated.

### `halflife/render/hud.py`

The bottom key hint (`hud.py:590`) becomes:

```
[Space] pause  [+/-] speed  [B] viz  [M] bond mode  [N] reset  [R] rec  [H] hide HUD  [Q] quit
```

### `halflife/config.py`

Three fields added to `SimConfig`, next to the existing render knobs
(`window_width`, `window_height`, `fps_target`, `background_color`):

```python
recording_dir: str = "recordings"
recording_fps: int = 60
recording_crf: int = 18
```

`SimConfig` is `static_argnums` in the JIT'd kernels, but these fields never enter a
traced function, so the emitted HLO is unchanged and the on-disk XLA cache still hits.

### `.gitignore`

Add `recordings/` and `screenshots/` — neither is currently ignored, and both hold
generated binary output.

## Data flow

```
 GPU default framebuffer (scene + HUD, LDR)
        │  ctx.screen.read()        ← synchronous glReadPixels, 2.7 MB/frame
        ▼
 bytes (rgb24, bottom-up)
        │  submit() → put_nowait
        ▼
 queue.Queue(maxsize=8)             ← full ⇒ drop + count, never block
        │  writer thread
        ▼
 ffmpeg stdin (rawvideo)
        │  vflip → libx264 → yuv420p
        ▼
 recordings/halflife_<ts>.mp4
```

## Behaviour decisions

**Fixed output framerate.** Every rendered frame becomes exactly one video frame at a
constant declared `recording_fps` (default 60). Video duration is therefore
`frames / 60`, independent of wall-clock. This is deterministic and needs no
variable-framerate plumbing; the alternative (per-frame PTS) is a later change if the
speed-up proves annoying.

Measurement showed the speed-up is larger than assumed at design time: the default
config runs at ~10–19 FPS while recording, so a 60 FPS output plays **3–6× fast**, not
the ~2× sketched above. Since this is invisible in the finished file, `stop()` prints the
measured factor and the `recording_fps` value that would give real-time playback:

```
Recording stopped: recordings/halflife_20260729_194958.mp4 (300 frames, 5.0s video)
  plays 6.0x faster than real time (10 fps live → 60 fps output).
  For real-time playback set SimConfig.recording_fps≈10.
```

`VideoRecorder` therefore also tracks wall-clock (`wall_elapsed`, frozen at `stop()`)
and exposes `speed_factor`.

**Recording continues while paused**, capturing identical frames. A pause therefore
holds on screen in the video, which is the useful behaviour for narrating a structure.

**No auto-stop and no size cap.** The badge shows elapsed time and file size so growth
is visible, but a forgotten recording keeps consuming disk — measured at **~210 MB per
minute of video** (~35 MB per minute of real time) at 1280×720 crf 18 on a dense 5k-particle
field. That is far heavier than typical screen capture because the particle field is
high-entropy and compresses poorly; the initial ~15 MB/min estimate was wrong by ~14×.
Raising `recording_crf` to ~23 roughly halves it. Adding a hard cap would mean choosing
a limit for the user; the indicator is the mitigation.

**Toggling `R` while recording finalizes the current file.** The next `R` starts a fresh
one under a new name. There is no append.

## Known costs

`ctx.screen.read()` is a synchronous `glReadPixels`: it stalls the GL pipeline until the
2.7 MB (1280×720×3) readback completes, every frame.

Measured on the real config (1280×720, 5000 particles, `angle_mode="vsepr"`):

| | recording off | recording on |
|---|---|---|
| frame time | 52.8 ms | 68.1 ms |
| FPS | 18.9 | 14.7 |

**+15.3 ms/frame, a 29% FPS loss** — the upper end of the 10–30% estimate. Combined with
the fixed-framerate output above, this makes a recording play back faster than it looked
live (and recording itself worsens the factor by lowering live FPS).

Encoder headroom is not the bottleneck: ffmpeg sustains ~10.5 ms/frame (~95 FPS) at this
resolution, comfortably above the ~15 FPS submit rate, and 300-frame runs dropped zero
frames.

Both costs are accepted for v1. The fix is a ring of pixel-buffer objects for asynchronous
readback, which `moderngl` does not expose directly and which would need a 1–2 frame
latency budget in the capture path. Out of scope.

## Error handling

| Condition | Behaviour |
|---|---|
| `ffmpeg` not on PATH | `start()` raises `RecorderUnavailable`; `main.py` catches, prints a one-line hint, keeps running. `is_recording` stays `False`. |
| ffmpeg exits mid-recording (bad args, disk full) | Writer thread catches `BrokenPipeError`, marks the recorder failed and stops accepting frames. Next `render()` sees `is_recording == False`, so the badge disappears — a visible signal. Error printed once. |
| Encoder can't keep up | Frames dropped and counted; total reported at `stop()`. |
| `out_dir` not writable | `makedirs` / `open` raises at `start()`; same handling as missing ffmpeg. |
| App exits or crashes while recording | `try/finally` → `Renderer.close()` → `recorder.stop()` finalizes a playable file. |
| `stop()` called when not recording | Returns `None`. No-op. |

## Testing

### `tests/test_recorder.py` — automated, CPU-only

No GL context, no JAX, no display. Guarded by
`pytest.mark.skipif(shutil.which("ffmpeg") is None)` so the suite still passes where
ffmpeg is absent.

1. **Filename uniqueness** — pre-create `halflife_<ts>.mp4` in a tmpdir; assert the next
   `start()` picks a different path and leaves the existing file untouched.
2. **Round trip** — `start()`, submit 30 synthetic gradient frames, `stop()`; assert the
   file exists, is non-empty, and `ffprobe -count_frames` reports 30 frames at the
   requested dimensions.
3. **vflip correctness** — submit a frame that is white in its top half only (in raw
   bottom-up byte order), decode frame 1 back via `ffmpeg -f rawvideo`, and assert the
   white band lands in the top half of the decoded image. Guards against a silently
   upside-down video, which is the single most likely defect here.
4. **Backpressure** — with a stubbed writer that never drains, submit far more than
   `maxsize` frames; assert no exception is raised and `dropped > 0`.
5. **State machine** — `submit()` and `stop()` before `start()` are no-ops;
   double `stop()` is safe; `start()` after `stop()` opens a new distinct path.
6. **Missing ffmpeg** — patch `shutil.which` to return `None`; assert
   `RecorderUnavailable` is raised and `is_recording` is `False`.

### `tests/test_recorder_gl.py` — automated integration

WSLg turns out to expose a real OpenGL context (`D3D12 / Intel UHD, Mesa 23.2`), so the
capture path is machine-verified after all rather than left to a checklist. Skipped when
no context is available.

The trick that makes these deterministic: `update()` is never called, so
`_n_particles_to_draw` stays 0 and the scene is a uniform background colour. Any
non-background pixel in a recorded frame therefore came from an overlay, and the test can
say which one.

1. **Badge excluded** — HUD hidden, record 12 frames; the decoded frame must be uniform
   (per-channel spread ≤ 8), and the rect the badge actually occupied (taken from
   `_badge_surface.get_bounding_rect()`, not hard-coded) must contain nothing bright.
2. **HUD included, badge still excluded** — HUD shown; the frame must be non-uniform, but
   the badge's rect must hold ≤ 10 red-dominant pixels. (Counting matters: the badge's
   *brightest* pixel is its near-white text, so a max-red-pixel test misses the leak.)
3. **H changes the capture** — recorded std with the HUD shown must exceed the hidden case.
4. **Orientation in the real pipeline** — the HUD's key hints are painted at the bottom of
   the window, so the bottom-centre strip must carry more variance than the top-centre one.
5. **Frame accounting** — `frame_count + dropped` equals frames rendered, and `ffprobe`
   agrees with `frame_count`.
6. **Not recording** — rendering leaves no file and never paints the badge.
7. **`close()` finalizes** — quitting mid-recording still yields a decodable file.

Every recording helper asserts the badge was painted *before* stopping, so no test here
can pass vacuously via a badge that never drew.

**Mutation-checked:** moving `_blit_rec_badge()` to before the capture makes tests 1, 2
and 4 fail. Verified, then reverted.

### Manual verification — remaining

A short list the automated tests don't cover, since they never open a visible window
under a human's eye:

1. Confirm the REC badge is legible and well-placed over a *busy* scene (the automated
   tests only ever draw it over flat background).
2. With the HUD hidden, click where the Reset button sits; confirm nothing resets and that
   drag-pan still works.
3. Press `N`; confirm the world resets (the key moved off `R`).
4. Press `R`, then `Q` while still recording; confirm the file is finalized and plays.
5. Watch a recording back and judge whether the fixed-framerate speed-up is acceptable, or
   whether `recording_fps` wants lowering to the value `stop()` suggests.
