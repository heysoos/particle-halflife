# Pan/Zoom Camera + Particle Inspector — Implementation Plan

**Goal:** Add scroll-to-zoom, click-and-drag-to-pan, and click-to-select with a live info panel showing the selected particle's stats (and its composite's stats if it's a member).

**Architecture:**
- A 2D affine camera (`view_center: vec2`, `view_scale: float`) is layered into the existing world→NDC mapping in every world-space vertex shader (particle, bond, event, plus a new highlight shader). The simulation is untouched — only what the user sees moves.
- Mouse handling in `main.py`'s pygame event loop splits clicks into pan vs select based on whether the cursor moved more than a small threshold during the press.
- The renderer caches the selected particle's data each frame and draws a pygame info panel in the top-right corner; a small ring is drawn around the selected particle in the world view.

**Tech Stack:** No new dependencies. Only `renderer.py` and `main.py` change.

**Mockup:** [notes/2026-05-14-particle-info-panel-mockup.html](../../../notes/2026-05-14-particle-info-panel-mockup.html) — open in a browser to preview the panel.

---

## Environment

Claude Code is running natively in WSL. Activate venv directly; no `wsl bash -c` wrapper.

**Run the sim** (for visual verification after each task):
```bash
source .venv/bin/activate && python -m halflife.main
```

**Git commit** with inline identity (no global git config in WSL):
```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add <files>
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "..."
```

**Comment-preservation rule (`CLAUDE.md`):** when editing existing functions, preserve every comment and docstring that does not specifically describe code being removed. Be surgical. New code follows the default minimal-comments rule (comment only the non-obvious WHY).

---

## File Structure

| File              | Responsibility                                                                                                          | Status |
|-------------------|-------------------------------------------------------------------------------------------------------------------------|--------|
| `halflife/renderer.py` | Camera state and uniforms; screen↔world conversions; particle hit-test; selection state and stats caching; highlight ring; info-panel rendering; close-button click-handling. | Modify |
| `halflife/main.py`     | pygame event loop: scroll-wheel zoom, click-drag pan with movement threshold, click-without-drag dispatch to `select_particle`. | Modify |

No other files change. JAX simulation logic is unaffected.

---

## Task 1: Camera transform uniforms in world-space shaders

After this task, the renderer has a camera but no UI to drive it. Defaults render identically to today.

**Files:**
- Modify: `halflife/renderer.py`

### Step 1: Add `u_view_center` and `u_view_scale` uniforms to the particle vertex shader

Replace `PARTICLE_VERTEX_SHADER` at the top of `renderer.py` with:

```python
PARTICLE_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec4  in_color;
in float in_size;

out vec4 v_color;

uniform vec2  u_world_size;
uniform vec2  u_view_center;   // world point at screen center
uniform float u_view_scale;    // 1.0 = fit world; >1 zooms in
uniform float u_size_mult;
uniform float u_alpha_mult;

void main() {
    // Camera: (pos - center) * scale, then center on (world_size / 2) so the
    // result still lives in [0, world_size] when scale=1 and center is the
    // world midpoint — keeps the no-camera default visually identical.
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = in_size * u_size_mult * u_view_scale;
    v_color = vec4(in_color.rgb, in_color.a * u_alpha_mult);
}
"""
```

Note: `gl_PointSize` is also multiplied by `u_view_scale` so particles grow as you zoom in (their world-size remains constant).

### Step 2: Same camera uniforms in the bond vertex shader

Replace `BOND_VERTEX_SHADER` (currently a few blocks below) with:

```python
BOND_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec4 in_color;

out vec4 v_color;

uniform vec2  u_world_size;
uniform vec2  u_view_center;
uniform float u_view_scale;

void main() {
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
"""
```

### Step 3: Same camera uniforms in the event vertex shader

Replace `EVENT_VERTEX_SHADER` with:

```python
EVENT_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec3  in_color;
in float in_age;

out vec3  v_color;
out float v_age;

uniform vec2  u_world_size;
uniform vec2  u_view_center;
uniform float u_view_scale;

void main() {
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = mix(60.0, 20.0, in_age) * u_view_scale;
    v_color = in_color;
    v_age   = in_age;
}
"""
```

### Step 4: Add camera state to `Renderer.__init__`

Near the existing `# ── Render-settings dict ──` block (just below it is a good spot), insert:

```python
        # ── Camera (pan + zoom) ──────────────────────────────────────────────
        # World point at the screen center, and how many world-units-per-pixel
        # is shrunk vs default. view_scale = 1.0 → default (whole world visible
        # at start), > 1.0 → zoomed in. The default view_center sits at the
        # world midpoint so the initial view is identical to the no-camera
        # version of the renderer.
        self._view_center = [config.world_width * 0.5, config.world_height * 0.5]
        self._view_scale  = 1.0
        self._view_scale_min = 0.25
        self._view_scale_max = 40.0
```

### Step 5: Set the new uniforms each frame in `render()`

Currently each world-space program has `u_world_size` set once in `__init__` (because window/world size doesn't change). The camera uniforms change every frame, so they must be pushed before each draw.

In `render()`, after the line `rs = self._render_settings` (or anywhere before the trail/scene draws begin), add:

```python
        # Push camera uniforms to all world-space programs. Cheap; saves
        # branching them later.
        vc = (float(self._view_center[0]), float(self._view_center[1]))
        vs = float(self._view_scale)
        self.particle_prog['u_view_center'].value = vc
        self.particle_prog['u_view_scale'].value  = vs
        self.bond_prog['u_view_center'].value     = vc
        self.bond_prog['u_view_scale'].value      = vs
        self.event_prog['u_view_center'].value    = vc
        self.event_prog['u_view_scale'].value     = vs
```

### Step 6: Smoke test

Run:
```bash
source .venv/bin/activate && timeout 4 python -m halflife.main
```

Expected: sim looks identical to before — view defaults are world center and scale 1.0. Exit code 143 from SIGTERM only.

### Step 7: Commit

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): camera uniforms (view_center + view_scale) in world shaders

Particle, bond, and event vertex shaders now take u_view_center and
u_view_scale uniforms so the world→NDC mapping is centered on a
configurable world point and scaled. Defaults are world midpoint and
1.0 → visually identical to the no-camera version. Particles and event
sprites grow with zoom so their world-size stays constant. No UI yet."
```

---

## Task 2: Scroll-wheel zoom toward cursor + right-click reset

**Files:**
- Modify: `halflife/renderer.py` (zoom_at + reset_view methods, screen↔world conversion helpers)
- Modify: `halflife/main.py` (pygame.MOUSEWHEEL + right-mouse-button event handling)

### Step 1: Add screen↔world conversion helpers to `Renderer`

Place these next to the existing `_periodic_com` helper (around line 1225):

```python
    def _screen_to_world(self, sx: int, sy: int) -> tuple:
        """Convert window-pixel coords → world coords using current camera.

        Inverse of the vertex-shader math:
            view = (pos - center) * scale + world_size * 0.5
            ndc  = view / world_size * 2 - 1
        Solving for pos:
            pos = ndc * world_size / 2 / scale + center
        """
        config = self.config
        ndc_x = (sx / config.window_width)  * 2.0 - 1.0
        ndc_y = 1.0 - (sy / config.window_height) * 2.0   # pygame Y is top-down
        wx = ndc_x * config.world_width  * 0.5 / self._view_scale + self._view_center[0]
        wy = ndc_y * config.world_height * 0.5 / self._view_scale + self._view_center[1]
        return wx, wy

    def zoom_at(self, sx: int, sy: int, factor: float) -> None:
        """Zoom by `factor` while keeping the world point under (sx, sy) fixed.

        Pin the world-coord that lives under the cursor, change scale, then
        recompute view_center so that same world-coord still maps to the same
        screen coord.
        """
        # World point under cursor BEFORE zoom
        wx_before, wy_before = self._screen_to_world(sx, sy)

        new_scale = float(np.clip(self._view_scale * factor,
                                   self._view_scale_min, self._view_scale_max))
        if abs(new_scale - self._view_scale) < 1e-6:
            return
        self._view_scale = new_scale

        # World point under cursor AFTER zoom (with unchanged view_center)
        wx_after, wy_after = self._screen_to_world(sx, sy)

        # Slide view_center so the pre-zoom point lands back under the cursor
        self._view_center[0] += wx_before - wx_after
        self._view_center[1] += wy_before - wy_after

    def reset_view(self) -> None:
        """Reset pan and zoom back to defaults (world center, scale 1.0)."""
        config = self.config
        self._view_center = [config.world_width * 0.5, config.world_height * 0.5]
        self._view_scale  = 1.0
```

### Step 2: Forward scroll events from main.py

In `halflife/main.py`, find the pygame event loop (search for `pygame.MOUSEBUTTONDOWN`). The MOUSEWHEEL event has `event.y` for vertical scroll (positive = scroll up). Add a new branch:

```python
                elif event.type == pygame.MOUSEWHEEL:
                    # Zoom toward the current mouse position. y > 0 = scroll
                    # up = zoom in.
                    if event.y != 0:
                        mx, my = pygame.mouse.get_pos()
                        factor = 1.15 ** event.y       # ~15% per notch
                        renderer.zoom_at(mx, my, factor)
```

Place this branch alongside the existing MOUSEBUTTONDOWN/MOUSEBUTTONUP/MOUSEMOTION cases.

### Step 3: Right-click resets the view to defaults

Same event loop, add a branch for the right mouse button (pygame button code 3). Press is enough — no drag tracking needed:

```python
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    # Right-click: snap back to the default framing (world
                    # midpoint, scale 1.0). Quick way to get unlost after
                    # zooming and panning around.
                    renderer.reset_view()
```

Place it alongside the other MOUSEBUTTONDOWN branches (the Task 3 left-click branch will sit next to this one).

### Step 4: Smoke test

Run the sim:
- Scroll up/down → zoom in/out around the cursor.
- World point under the cursor stays roughly fixed during zoom.
- Zoom is clamped (won't shrink below 0.25× or grow above 40×).
- Right-click anywhere → view snaps back to the default framing.

### Step 5: Commit

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py halflife/main.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): scroll-wheel zoom toward cursor + right-click reset view

MOUSEWHEEL → renderer.zoom_at(x, y, factor) — adjusts view_scale by
1.15 per notch and re-pins view_center so the world point under the
cursor stays put. Clamped to [0.25, 40]×.

Right-click (mouse button 3) → renderer.reset_view() — snaps pan and
zoom back to world midpoint and 1.0 scale. Quick recovery from
getting lost after a lot of zoom/pan."
```

---

## Task 3: Click-and-drag pan

**Files:**
- Modify: `halflife/renderer.py` (pan_by method, drag state)
- Modify: `halflife/main.py` (mousedown/motion/up handling with movement threshold)

### Step 1: Add `pan_by` to the renderer

Place alongside `zoom_at`:

```python
    def pan_by(self, dx_pixels: int, dy_pixels: int) -> None:
        """Translate the view by (dx, dy) screen pixels.

        Converts pixel deltas → world deltas via the current zoom and shifts
        view_center in the opposite direction (dragging right scrolls the
        world content right, i.e. view_center moves left).
        """
        config = self.config
        # 1 pixel = world_width / window_width / view_scale world units
        world_dx = -dx_pixels * (config.world_width  / config.window_width)  / self._view_scale
        world_dy =  dy_pixels * (config.world_height / config.window_height) / self._view_scale
        # ↑ y inverted because pygame Y goes down while world Y goes up
        self._view_center[0] += world_dx
        self._view_center[1] += world_dy
```

### Step 2: Drag-vs-click state in main.py

Above the event loop in `main.py` (look for `while running:` and the variables defined just before it), add or augment whatever state-init block is most natural:

```python
        # Mouse drag-vs-click state. A press becomes a drag only after the
        # cursor moves more than DRAG_PIXEL_THRESHOLD; a press-and-release
        # without crossing the threshold is treated as a click and dispatched
        # to renderer.select_at() (Task 4).
        DRAG_PIXEL_THRESHOLD = 4
        mouse_down_pos = None     # (x, y) at MOUSEBUTTONDOWN, or None
        is_panning = False
```

### Step 3: Wire mousedown/motion/up

Find the existing mousedown handling for HUD button + slider clicks. The new pan logic must run only when those return False (i.e. the press wasn't on a HUD button or slider handle). Wrap or augment the existing block. The structure should end up looking like:

```python
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    # 1) HUD button hits (existing dispatch returns an action string)
                    action = renderer.handle_click(pos)
                    if action is not None:
                        # ... existing action-dispatch chain (pause / toggle_bonds / …)
                        ...
                    # 2) Slider drag start (existing)
                    elif renderer.handle_mousedown_slider(pos):
                        pass
                    # 3) Otherwise, this might be the start of a pan or click-to-select.
                    else:
                        mouse_down_pos = pos
                        is_panning = False

                elif event.type == pygame.MOUSEMOTION:
                    # Existing slider-motion handling
                    renderer.handle_mousemotion(event.pos)
                    # New: pan handling. Only active while the user is
                    # mid-press on empty space (not on a HUD widget).
                    if mouse_down_pos is not None:
                        dx = event.pos[0] - mouse_down_pos[0]
                        dy = event.pos[1] - mouse_down_pos[1]
                        if not is_panning and (abs(dx) + abs(dy)) > DRAG_PIXEL_THRESHOLD:
                            is_panning = True
                        if is_panning:
                            renderer.pan_by(event.rel[0], event.rel[1])

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    renderer.handle_mouseup()
                    if mouse_down_pos is not None and not is_panning:
                        # Click without drag → particle select (Task 4)
                        renderer.select_at(*mouse_down_pos)
                    mouse_down_pos = None
                    is_panning = False
```

The `renderer.handle_click` and `renderer.handle_mousedown_slider` paths are unchanged in behavior — the new code only activates when neither claims the press.

### Step 4: Smoke test pan

- `renderer.select_at` doesn't exist yet — calling it will crash. To smoke-test pan only, temporarily replace the `renderer.select_at(*mouse_down_pos)` line with `pass`.
- Run the sim, press and drag with the left mouse button on the world: the view should scroll.
- Click without dragging: nothing should happen (yet).
- Click on a HUD button: button still works as before.
- After verifying, leave `renderer.select_at(*mouse_down_pos)` in place — Task 4 will define it.

### Step 5: Commit

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py halflife/main.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): click-and-drag pan

Renderer.pan_by(dx, dy) shifts view_center by pixel-deltas converted
to world-deltas via the current zoom. main.py treats a press-and-drag
of >4px on empty world space as a pan; <4px is reserved as click-to-
select (Task 4). HUD buttons and slider drags are unaffected — they
claim the press before pan logic sees it."
```

---

## Task 4: Click-to-select + selected-stats caching

**Files:**
- Modify: `halflife/renderer.py` (select_at, _selected_idx, stats caching in update())

### Step 1: Selection state in `__init__`

Place this with the other runtime state (near `self._n_particles_to_draw = 0`):

```python
        # ── Particle selection (click-to-inspect) ────────────────────────────
        # _selected_idx is the particle index of the current selection, or -1
        # for none. The cached snapshot is refreshed every frame from the
        # already-CPU-transferred state arrays so the panel reads stay live.
        self._selected_idx = -1
        self._selected_snapshot = None   # dict or None
        # Selection radius in WORLD UNITS at zoom 1.0 — a click further than
        # this from any particle clears the selection.
        self._select_radius_world = 2.5
```

### Step 2: `select_at` method

Place this near `pan_by`:

```python
    def select_at(self, sx: int, sy: int) -> None:
        """Pick the nearest particle within a world-radius around (sx, sy).

        Reads the cached positions stashed during the last update() rather
        than re-fetching from the GPU. If no particle falls within the radius,
        the selection is cleared.
        """
        if not hasattr(self, '_last_positions') or self._last_positions is None:
            return
        wx, wy = self._screen_to_world(sx, sy)
        pos = self._last_positions    # (N, 2) numpy
        # Linear scan — N is fixed at config.num_particles, fine for occasional
        # click events. At 5000 particles this is microseconds.
        dx = pos[:, 0] - wx
        dy = pos[:, 1] - wy
        # Periodic min-image so clicks near a wrap edge can pick particles
        # that "look" close but live on the other side of the world.
        if self.config.boundary_mode == "periodic":
            dx -= self.config.world_width  * np.round(dx / self.config.world_width)
            dy -= self.config.world_height * np.round(dy / self.config.world_height)
        d2 = dx * dx + dy * dy
        best = int(np.argmin(d2))
        # Radius is in world units, fixed regardless of zoom — so clicks have
        # the same "feel" at all zoom levels.
        if d2[best] <= self._select_radius_world ** 2:
            self._selected_idx = best
        else:
            self._selected_idx = -1
        self._hud_dirty = True
```

### Step 3: Cache positions and per-frame stats in `update()`

At the top of `update()`, just after the existing `(pos, vel, species, ...) = jax.device_get((...))` batch, stash references to the arrays the picker and panel need:

```python
        # Stash for click-to-pick and live inspector panel reads. Both run
        # after this method returns; the data is already on CPU here.
        self._last_positions = pos
        self._last_velocities = vel
        self._last_species = species
        self._last_mass = mass
        self._last_energy = _energy
        self._last_comp_id = comp_id
        self._last_comp_members = comp_members
        self._last_comp_count = comp_count
        self._last_comp_alive = comp_alive
        self._last_comp_species_hash = comp_species_hash
```

`_energy` is named with a leading underscore in the unpacking because the line currently has `_energy` to indicate it's not used elsewhere — keep the underscore and just remove its "unused" status by reading it here.

### Step 4: Build the inspector snapshot when selection changes or live values move

Add a small helper `_refresh_selected_snapshot` and call it from `update()` after the cache assignments:

```python
    def _refresh_selected_snapshot(self) -> None:
        """Rebuild the cached dict of stats shown in the inspector panel.

        Called once per frame; cheap because it touches only a few scalars.
        """
        i = self._selected_idx
        if i < 0 or self._last_positions is None:
            self._selected_snapshot = None
            return

        config = self.config
        px, py = float(self._last_positions[i, 0]), float(self._last_positions[i, 1])
        vx, vy = float(self._last_velocities[i, 0]), float(self._last_velocities[i, 1])
        speed = float(np.hypot(vx, vy))
        species_i = int(self._last_species[i])
        mass_i    = float(self._last_mass[i])
        energy_i  = float(self._last_energy[i])
        cid       = int(self._last_comp_id[i])

        # Per-species valence is computed once at __init__ — see Task 6 for
        # the cache. For now read from a lazily-built dict.
        if not hasattr(self, '_species_valence') or self._species_valence is None:
            self._build_species_valence()
        valence_i = int(self._species_valence[species_i]) if config.use_valence else None

        snap = {
            'idx':      i,
            'pos':      (px, py),
            'vel':      (vx, vy),
            'speed':    speed,
            'species':  species_i,
            'valence':  valence_i,
            'mass':     mass_i,
            'energy':   energy_i,
            'composite': None,
        }

        if cid >= 0 and bool(self._last_comp_alive[cid]):
            members = self._last_comp_members[cid]
            count   = int(self._last_comp_count[cid])
            member_species = [int(self._last_species[m])
                              for m in members[:count] if m >= 0]
            snap['composite'] = {
                'id':           cid,
                'size':         count,
                'hash':         int(self._last_comp_species_hash[cid]),
                'members':      member_species,
                # binding_energy / half_life / age / free_bonds are read from
                # the composite arrays we already pulled — see Task 5 for the
                # extra fields added to the device_get batch.
                'binding_energy': float(self._last_comp_binding_energy[cid]),
                'half_life':      float(self._last_comp_half_life[cid]),
                'age':            float(self._last_comp_age[cid]),
                'free_bonds':     (int(self._last_comp_free_bonds[cid])
                                   if config.use_valence else None),
            }
        self._selected_snapshot = snap

    def _build_species_valence(self) -> None:
        """Compute the per-species valence using the same hash as chemistry."""
        from halflife.chemistry import _hash_to_valence
        config = self.config
        self._species_valence = np.zeros(config.num_species, dtype=np.int32)
        for s in range(config.num_species):
            self._species_valence[s] = int(_hash_to_valence(s, config))
```

Add at the end of `update()`:

```python
        self._refresh_selected_snapshot()
```

### Step 5: Smoke test — `select_at` works, no panel yet

Run the sim, left-click on a particle. Nothing visual should change (the panel isn't drawn yet) but no crash. Optionally add `print(self._selected_snapshot)` inside `_refresh_selected_snapshot` temporarily to confirm a snapshot is built.

If `_last_comp_binding_energy` and friends crash with `AttributeError`, that's expected — Task 5 wires them.

### Step 6: Commit

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): particle selection + per-frame stats snapshot

Renderer.select_at(sx, sy) maps a screen click to a particle index via
the inverse camera transform plus a periodic-aware nearest-neighbor
scan, with a fixed world-space pick radius so clicks have the same
feel at any zoom. The selected particle's stats are rebuilt each frame
into _selected_snapshot from the already-CPU-transferred state arrays.

No UI yet — Task 6 draws the panel."
```

---

## Task 5: Pull additional composite fields into the per-frame state transfer

`_refresh_selected_snapshot` references `binding_energy`, `half_life`, `age`, and `free_bonds` on the composite — currently NOT included in the `jax.device_get` batch in `update()`. Add them.

**Files:**
- Modify: `halflife/renderer.py` (one tuple expansion in `update()`)

### Step 1: Augment the device_get batch

Find the current batch in `Renderer.update`:

```python
        (pos, vel, species, mass, _energy, comp_id,
         comp_members, comp_count, comp_alive, comp_species_hash,
         total_energy, step_count, sim_time) = jax.device_get((
            particles.position, particles.velocity,
            particles.species,
            particles.mass,     particles.energy,
            particles.composite_id,
            composites.members, composites.member_count, composites.alive,
            composites.species_hash,
            state.total_energy, state.step_count, state.time,
         ))
```

Expand it to:

```python
        (pos, vel, species, mass, _energy, comp_id,
         comp_members, comp_count, comp_alive, comp_species_hash,
         comp_binding_energy, comp_half_life, comp_age, comp_free_bonds,
         total_energy, step_count, sim_time) = jax.device_get((
            particles.position, particles.velocity,
            particles.species,
            particles.mass,     particles.energy,
            particles.composite_id,
            composites.members, composites.member_count, composites.alive,
            composites.species_hash,
            composites.binding_energy, composites.half_life, composites.age,
            composites.free_bonds,
            state.total_energy, state.step_count, state.time,
         ))
```

And in the cache-stash block from Task 4 Step 3, add:

```python
        self._last_comp_binding_energy = comp_binding_energy
        self._last_comp_half_life      = comp_half_life
        self._last_comp_age            = comp_age
        self._last_comp_free_bonds     = comp_free_bonds
```

### Step 2: Verify composites NamedTuple has these fields

Open `halflife/state.py` and confirm the `CompositeState` NamedTuple has `binding_energy`, `half_life`, `age`, and `free_bonds` (mentioned in the CLAUDE.md data structure section, so it should). If `free_bonds` is missing (e.g. an older revision), the snapshot dict in Task 4 will fail on `_last_comp_free_bonds` access. Guard that access on `config.use_valence`:

```python
        'free_bonds': (int(self._last_comp_free_bonds[cid])
                       if (config.use_valence and self._last_comp_free_bonds is not None) else None),
```

(The existing Task 4 code already has the `config.use_valence` guard; this step is purely a "confirm fields exist" gate.)

### Step 3: Smoke test

Run the sim, click a particle that's in a composite. Add a temporary `print` of `self._selected_snapshot['composite']` inside `_refresh_selected_snapshot` to verify all fields populate.

### Step 4: Commit

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): pull composite binding_energy/half_life/age/free_bonds to CPU

Inspector panel needs these per frame for the selected particle's
composite. Folding them into the existing batched device_get keeps
the CPU↔GPU sync cost flat (still one CUDA sync per frame)."
```

---

## Task 6: Render the inspector panel + close button

**Files:**
- Modify: `halflife/renderer.py` (`_render_inspector_panel` helper; call from `_render_hud_surface`; close-button hit-test)

### Step 1: Reserve the panel position

In `__init__`, near the buttons block, add:

```python
        # Inspector panel anchor — top-right corner below the Stats button.
        # Width fits the existing right-rail; height depends on whether the
        # selected particle is a composite member and is computed at draw time.
        self._inspector_panel_w = 235
        self._inspector_panel_x = config.window_width - self._inspector_panel_w - 8
        self._inspector_panel_y = self._stats_btn_rect.bottom + 8 + (panel_h_when_stats_open := 0)
        # Close button rect (computed at draw time using actual panel_y;
        # stashed on the renderer so handle_click can pick it up).
        self._inspector_close_rect = pygame.Rect(0, 0, 18, 18)
```

(The `panel_h_when_stats_open` walrus is just a placeholder — the inspector panel will be drawn below the Stats panel if Stats is open. Position is computed inside `_render_inspector_panel`.)

### Step 2: Implement `_render_inspector_panel`

Add this method to `Renderer`, near `_render_hud_surface`:

```python
    def _render_inspector_panel(self, surface: pygame.Surface) -> None:
        """Top-right panel showing the selected particle's stats."""
        snap = self._selected_snapshot
        if snap is None:
            return

        font  = self._font
        # Style constants — same palette as the HUD panels above.
        BG          = (15, 18, 35, 235)
        BORDER      = (70, 100, 150, 220)
        DIVIDER     = (70, 100, 150, 110)
        LABEL_FG    = (160, 185, 230)
        VALUE_FG    = (220, 230, 255)
        BODY_FG     = (190, 215, 255)
        MUTED_FG    = (120, 140, 165)
        CLOSE_BG    = (80, 30, 30, 220)
        CLOSE_BORDER= (150, 80, 80, 220)
        CLOSE_FG    = (255, 160, 160)

        comp = snap['composite']

        # Compute height: header(22) + species(20) + 7 kv rows(15 each) +
        # optional composite section(approx 8+18+5*15+24)
        base_h = 22 + 20 + 7 * 15 + 8
        comp_h = (8 + 18 + 5 * 15 + 24) if comp is not None else 0
        panel_h = base_h + comp_h + 10
        panel_w = self._inspector_panel_w

        # Anchor below Stats button; if Stats panel is open, slide down further.
        panel_x = self._inspector_panel_x
        panel_y = self._stats_btn_rect.bottom + 6
        if self._show_stats:
            # Skip past the stats panel (whose total height matches the
            # constant computed in the stats block of _render_hud_surface).
            stats_panel_h = (4 * 16 + 6 * 33 + 4 + 18 + 64 + 20 + 10)
            panel_y += stats_panel_h + 4

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(surface, BG, panel_rect, border_radius=6)
        pygame.draw.rect(surface, BORDER, panel_rect, 1, border_radius=6)

        x_text = panel_x + 10
        y      = panel_y + 6

        # Header
        title = font.render(f"Particle #{snap['idx']}", True, VALUE_FG)
        surface.blit(title, (x_text, y))

        # Close button — stash rect for click-handling
        close_size = 18
        self._inspector_close_rect = pygame.Rect(
            panel_x + panel_w - close_size - 8, y - 1, close_size, close_size
        )
        pygame.draw.rect(surface, CLOSE_BG, self._inspector_close_rect, border_radius=3)
        pygame.draw.rect(surface, CLOSE_BORDER, self._inspector_close_rect, 1, border_radius=3)
        close_lbl = font.render("×", True, CLOSE_FG)
        surface.blit(close_lbl,
                     (self._inspector_close_rect.centerx - close_lbl.get_width() // 2,
                      self._inspector_close_rect.centery - close_lbl.get_height() // 2 - 1))
        y += 18
        pygame.draw.line(surface, DIVIDER, (panel_x + 6, y),
                         (panel_x + panel_w - 6, y), 1)
        y += 4

        # Species line with color swatch
        sp = snap['species']
        col_lin = self.species_colors[sp]
        # Linear sRGB → display sRGB (~ x^(1/2.2)). HUD draws in sRGB space.
        col_disp = np.clip(col_lin, 0.0, 1.0) ** (1.0 / 2.2)
        col_rgb = tuple(int(round(c * 255)) for c in col_disp)
        sw_rect = pygame.Rect(x_text, y + 3, 14, 14)
        pygame.draw.rect(surface, col_rgb, sw_rect, border_radius=2)
        pygame.draw.rect(surface, (255, 255, 255, 50), sw_rect, 1, border_radius=2)
        lbl = font.render(f"Species {sp}", True, VALUE_FG)
        surface.blit(lbl, (x_text + 18, y))
        if snap['valence'] is not None:
            v_txt = font.render(f"valence {snap['valence']}", True, MUTED_FG)
            surface.blit(v_txt, (panel_x + panel_w - v_txt.get_width() - 10, y))
        y += 20

        # KV rows (label/value pairs)
        def kv(label: str, value: str, color=VALUE_FG):
            nonlocal y
            l = font.render(label, True, LABEL_FG)
            v = font.render(value, True, color)
            surface.blit(l, (x_text, y))
            surface.blit(v, (panel_x + panel_w - v.get_width() - 10, y))
            y += 15

        px, py = snap['pos']
        vx, vy = snap['vel']
        kv("Position", f"{px:.1f}, {py:.1f}")
        kv("Velocity", f"{vx:.2f}, {vy:.2f}")
        kv("Speed",    f"{snap['speed']:.2f}")
        kv("Mass",     f"{snap['mass']:.2f}")
        kv("Energy",   f"{snap['energy']:.2f}")
        kv("Age",      f"—")    # ParticleState.age is dropped from the cache;
                                 # add to Task 5 batch if/when you want this live
        kv("Composite", "free" if comp is None else f"#{comp['id']}",
           color=MUTED_FG if comp is None else VALUE_FG)

        if comp is not None:
            y += 4
            pygame.draw.line(surface, DIVIDER, (panel_x + 6, y),
                             (panel_x + panel_w - 6, y), 1)
            y += 4
            hdr = font.render(f"Composite #{comp['id']} — {comp['size']} members",
                              True, VALUE_FG)
            surface.blit(hdr, (x_text, y))
            y += 18
            kv("Hash",       f"{comp['hash']:08x}"[:8])
            kv("Binding E",  f"{comp['binding_energy']:.2f}")
            kv("Age",        f"{comp['age']:.1f} s")
            kv("Half-life",  f"{comp['half_life']:.1f} s")
            if comp['free_bonds'] is not None:
                # Total valence Σv_s of members for the denominator
                total_v = sum(int(self._species_valence[s]) for s in comp['members'])
                kv("Free bonds", f"{comp['free_bonds']} / {total_v}")
            # Members chips
            chip_y = y + 2
            chip_x = x_text
            chip_h = 16
            chip_pad = 4
            members_lbl = font.render("Members", True, LABEL_FG)
            surface.blit(members_lbl, (x_text, chip_y))
            chip_y += 16
            chip_x = x_text
            for s in comp['members']:
                col_lin = self.species_colors[s]
                col_disp = np.clip(col_lin, 0.0, 1.0) ** (1.0 / 2.2)
                col_rgb = tuple(int(round(c * 255)) for c in col_disp)
                txt = font.render(str(int(s)), True, VALUE_FG)
                cw = txt.get_width() + 12
                if chip_x + cw > panel_x + panel_w - 10:
                    chip_x = x_text
                    chip_y += chip_h + 2
                chip_rect = pygame.Rect(chip_x, chip_y, cw, chip_h)
                pygame.draw.rect(surface, (70, 100, 150, 46), chip_rect, border_radius=3)
                pygame.draw.circle(surface, col_rgb,
                                   (chip_rect.left + 6, chip_rect.centery), 4)
                surface.blit(txt, (chip_rect.left + 12, chip_rect.centery - txt.get_height() // 2))
                chip_x += cw + chip_pad
```

Notes:
- The "Age" particle-age row reads "—" because `ParticleState.age` isn't currently in the cache batch. Add it to Task 5's `device_get` call and stash if you want the live age value.
- The colour swatches do a quick `^(1/2.2)` gamma encode since the species palette is stored in linear sRGB but the pygame HUD surface is in sRGB display space.

### Step 3: Call the panel renderer at the end of `_render_hud_surface`

Just before the bottom-key-hint rendering at the very end of `_render_hud_surface`:

```python
        # ── Inspector panel ──────────────────────────────────────────────────
        self._render_inspector_panel(surface)
```

### Step 4: Click-handling for the close button

Modify `handle_click`:

```python
    def handle_click(self, pos) -> str | None:
        """Return action string if a button was clicked, else None."""
        if self._stats_btn_rect.collidepoint(pos):
            return "toggle_stats"
        # Close button on the inspector panel (only meaningful when something
        # is selected — the rect is reset to (0,0,18,18) otherwise).
        if self._selected_idx >= 0 and self._inspector_close_rect.collidepoint(pos):
            self._selected_idx = -1
            self._selected_snapshot = None
            self._hud_dirty = True
            return "clear_selection"
        # Gear nub overlays the right edge of the Trails button. …
        if self._trails_gear_rect.collidepoint(pos):
            return "toggle_render_params"
        for _label, rect, action in self._buttons:
            if rect.collidepoint(pos):
                return action
        return None
```

And in `main.py`'s click dispatch, the `clear_selection` action needs no handler (the renderer already cleared the selection internally) — but to keep the dispatch chain clean:

```python
                    elif action == 'clear_selection':
                        pass
```

(Optional — the chain doesn't crash if you skip this; the renderer already cleared state.)

### Step 5: Force HUD redraw while a selection is active

The inspector panel updates every frame (live position/velocity/energy). The HUD-dirty mechanism currently only refreshes when something changes; force a refresh while a selection is active. Find the start of `render()`'s HUD section (look for `if self._show_stats: self._hud_dirty = True`) and add:

```python
        if self._show_stats or self._selected_idx >= 0:
            self._hud_dirty = True
```

### Step 6: Smoke test

Run the sim:
- Left-click a particle → inspector appears top-right with stats.
- Drag a particle by click-and-drag → pans (no selection).
- Click empty space → selection clears (no particle within radius).
- Click another particle → selection swaps.
- Click × button on panel → selection clears.
- Click a composite member → composite section appears with member chips.
- Stats panel + inspector both open → no overlap (inspector slides below).
- Pause / resume / reroll work as before.

### Step 7: Commit

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py halflife/main.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): inspector panel for the selected particle

Top-right panel below the Stats button shows live stats for the
clicked particle: species + valence, position/velocity/speed, mass,
energy, and its composite (if any) with hash, binding energy, age,
half-life, free bonds, and species-tagged member chips. Close button
(×) clears the selection; clicking another particle swaps; clicking
empty space within the pick radius does nothing (out of radius
clears)."
```

---

## Task 7: Highlight ring around the selected particle

A subtle ring in the world view so you can see which particle the panel describes.

**Files:**
- Modify: `halflife/renderer.py` (new highlight shader + draw call)

### Step 1: Add the highlight shader

Near the EVENT shader source, add:

```python
HIGHLIGHT_VERTEX_SHADER = """
#version 330

in vec2 in_position;

uniform vec2  u_world_size;
uniform vec2  u_view_center;
uniform float u_view_scale;
uniform float u_size_px;     // ring outer radius in screen pixels

void main() {
    vec2 view = (in_position - u_view_center) * u_view_scale + (u_world_size * 0.5);
    vec2 ndc  = (view / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_size_px;
}
"""

HIGHLIGHT_FRAGMENT_SHADER = """
#version 330

out vec4 fragColor;

void main() {
    float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
    // Ring at r ≈ 0.85, ~0.08 thick, with soft falloff.
    float ring = 1.0 - clamp(abs(r - 0.85) / 0.08, 0.0, 1.0);
    if (ring < 0.02) discard;
    fragColor = vec4(1.0, 1.0, 1.0, ring);   // pure-white ring
}
"""
```

### Step 2: Build the program and a one-vertex VBO in `__init__`

```python
        # Selection highlight ring
        self.highlight_prog = self.ctx.program(
            vertex_shader=HIGHLIGHT_VERTEX_SHADER,
            fragment_shader=HIGHLIGHT_FRAGMENT_SHADER,
        )
        self.highlight_prog['u_world_size'].value = (config.world_width, config.world_height)
        self._highlight_vbo = self.ctx.buffer(reserve=2 * 4)   # one vec2 float32
        self._highlight_vao = self.ctx.vertex_array(
            self.highlight_prog,
            [(self._highlight_vbo, '2f', 'in_position')],
        )
```

Release in `close()`:
```python
        self._highlight_vbo.release()
        self._highlight_vao.release()
        self.highlight_prog.release()
```

### Step 3: Draw the highlight in the fresh-overlay pass

Highlights belong on top of bonds/events so they're always visible — the fresh FBO already composites above the trail. After the event-sprite draw inside the fresh-overlay block in `render()`, add:

```python
        # Selection highlight ring
        if self._selected_idx >= 0 and hasattr(self, '_last_positions') and self._last_positions is not None:
            sel_pos = self._last_positions[self._selected_idx].astype(np.float32)
            self._highlight_vbo.write(sel_pos.tobytes())
            self.highlight_prog['u_view_center'].value = (
                float(self._view_center[0]), float(self._view_center[1])
            )
            self.highlight_prog['u_view_scale'].value  = float(self._view_scale)
            # Ring outer radius in screen pixels — fixed regardless of zoom
            # so the highlight is always visible even on tiny zoomed-out
            # particles. ~24 px works at the default window size.
            self.highlight_prog['u_size_px'].value = 26.0
            self._highlight_vao.render(moderngl.POINTS, vertices=1)
```

### Step 4: Smoke test

Click a particle → a white ring appears around it. Pan or zoom → the ring follows that particle (because we re-read its world position each frame). Clear selection → ring disappears.

### Step 5: Commit

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(render): white highlight ring around the selected particle

Single-vertex point-sprite shader stamps a soft white ring at the
selected particle's world position into the fresh-overlay FBO so it
sits above bonds and events. Ring size is constant in screen pixels
so the selection stays visible at all zoom levels."
```

---

## Task 8: Final visual smoke pass

- [ ] Run the sim. Walk through:
  - Scroll up/down: zooms toward cursor; clamped at extremes.
  - Right-click: snaps view back to default framing.
  - Click + drag empty space: pans smoothly.
  - Click without drag on a particle: inspector appears, ring drawn.
  - Click another particle: snaps to it.
  - Click empty space: out of pick radius clears selection.
  - × button: clears selection.
  - Stats panel + inspector: no overlap.
  - Params / Trails panels still open via their buttons.
  - All Reroll buttons work without GL errors.
  - Inspector composite section shows when clicking a bonded particle.
  - Ring follows the selected particle as it moves.

- [ ] Optional: hit `R` to reset sim — selection should clear when particles re-init (composite-id changes; the snapshot is rebuilt next frame). If selection sticks awkwardly, clear `_selected_idx` from the reset code path in `main.py`.

- [ ] No further commit if nothing else needs fixing.

---

## Self-review notes

**Spec coverage:**
- Scroll zoom toward cursor → Task 2.
- Click-drag pan → Task 3.
- Click-to-select particle + info panel → Tasks 4, 5, 6.
- Highlight ring → Task 7.

**Placeholder scan:** No "TBD" / "TODO" / "implement later". Every step has code or a concrete command.

**Type consistency:**
- `_view_center` always a `list[float]` length 2 (mutated by `pan_by` / `zoom_at`).
- `_view_scale` always `float`.
- `_selected_idx` always `int`, `-1` for none.
- `_selected_snapshot` always `dict | None`.
- Composite-fields in snapshot match the augmented `device_get` batch from Task 5.

**Risks:**
- ParticleState.age is in the user's CLAUDE.md but isn't currently in the renderer's `device_get` batch. The panel shows "—" for age; if you want live age, append `particles.age` to the batch and stash like the other fields.
- At very high zoom, periodic boundaries may make particles "vanish" near the edges because their wrapped copy isn't drawn. V1 doesn't tile the view; can be added later if it bothers you.
- Inspector close-button rect is updated every frame inside `_render_inspector_panel`. Until the panel has been drawn once after a new selection, the rect holds the previous panel's position. Not a real bug because `_selected_idx >= 0` only gates clicks once the panel has been drawn at least once (`_refresh_selected_snapshot` runs in `update()` which runs before `render()` which calls `_render_hud_surface`). If a click arrives between `update()` and `render()` (only possible on the very first frame after a selection) the rect might be stale by 1 frame. Harmless.
