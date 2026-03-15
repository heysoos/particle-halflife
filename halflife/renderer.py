"""
Real-time ModernGL + pygame renderer for the Half-Life simulator.

Renders particles as point sprites (circular blobs), colored by species
and sized by mass. Composites can be shown in two modes (toggle with B):
  - Bonds mode:  GL_LINES connecting composite member particles
  - Merged mode: single large point at the composite center of mass

Clickable HUD overlay (rendered as a pygame surface → OpenGL texture):
  Buttons along the left edge; stats panel (toggle) on the right; event
  sprites (expanding rings) at fusion/fission/spawn/decay sites.

Data flow per frame:
  1. np.asarray() — transfer JAX GPU arrays to CPU numpy
  2. Pack vertex data: (x, y, r, g, b, a, size) per particle
  3. vbo.write() — upload to GPU via OpenGL
  4. Draw particles (GL_POINTS), bonds (GL_LINES), events (GL_POINTS)
  5. Render HUD: pygame surface → moderngl texture → fullscreen quad
  6. pygame.display.flip()
"""

import collections
import numpy as np
import pygame
import moderngl

from halflife.config import SimConfig
from halflife.state import WorldState, get_species_colors


# ── GLSL Shaders ──────────────────────────────────────────────────────────────

PARTICLE_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec4  in_color;
in float in_size;

out vec4 v_color;

uniform vec2 u_world_size;

void main() {
    vec2 ndc = (in_position / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = in_size;
    v_color = in_color;
}
"""

PARTICLE_FRAGMENT_SHADER = """
#version 330

in  vec4 v_color;
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r = length(coord) * 2.0;
    float alpha = 1.0 - smoothstep(0.6, 1.0, r);
    float brightness = 1.0 - smoothstep(0.0, 0.5, r) * 0.3;
    fragColor = vec4(v_color.rgb * brightness, v_color.a * alpha);
}
"""

BOND_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec4 in_color;

out vec4 v_color;

uniform vec2 u_world_size;

void main() {
    vec2 ndc = (in_position / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
"""

BOND_FRAGMENT_SHADER = """
#version 330

in  vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"""

# HUD: fullscreen quad with pygame surface as texture
HUD_VERTEX_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
    // No Y-flip needed: pygame surface is pre-flipped on upload
}
"""

HUD_FRAGMENT_SHADER = """
#version 330

in  vec2 v_uv;
uniform sampler2D hud_tex;
out vec4 fragColor;

void main() {
    fragColor = texture(hud_tex, v_uv);
}
"""

# Event sprites: expanding rings at fusion/fission/spawn/decay sites
EVENT_VERTEX_SHADER = """
#version 330

in vec2  in_position;
in vec3  in_color;
in float in_age;

out vec3  v_color;
out float v_age;

uniform vec2 u_world_size;

void main() {
    vec2 ndc = (in_position / u_world_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = mix(60.0, 20.0, in_age);
    v_color = in_color;
    v_age   = in_age;
}
"""

EVENT_FRAGMENT_SHADER = """
#version 330

in  vec3  v_color;
in  float v_age;
out vec4  fragColor;

void main() {
    float r    = length(gl_PointCoord - vec2(0.5)) * 2.0;
    float ring = 1.0 - abs(r - v_age) / 0.12;
    ring = clamp(ring, 0.0, 1.0) * (1.0 - v_age);
    fragColor = vec4(v_color, ring);
}
"""


# ── Slider ────────────────────────────────────────────────────────────────────

class Slider:
    """Horizontal slider widget — log-scale multiplier centered at 1× (the default)."""

    EXPO_MIN, EXPO_MAX = -1.0, 2.0   # 0.1× to 100× the default value

    def __init__(self, label: str, field: str, default_value: float,
                 track_rect: pygame.Rect, fmt: str = "{:.3f}"):
        self._label = label
        self._field = field
        self._default = float(default_value)
        self._exponent = 0.0  # 1× = default
        self._track_rect = track_rect
        self._fmt = fmt

    @property
    def field(self) -> str:
        return self._field

    @property
    def value(self) -> float:
        return self._default * (10.0 ** self._exponent)

    def reset(self) -> None:
        self._exponent = 0.0

    def _handle_x(self) -> int:
        t = (self._exponent - self.EXPO_MIN) / (self.EXPO_MAX - self.EXPO_MIN)
        return int(self._track_rect.left + t * self._track_rect.width)

    def draw(self, surface: pygame.Surface, font) -> None:
        r = self._track_rect
        mult = 10.0 ** self._exponent
        label_str = f"{self._label}: {self._fmt.format(self.value)} ({mult:.2f}\u00d7)"
        txt = font.render(label_str, True, (190, 215, 255))
        surface.blit(txt, (r.left, r.top - 14))
        # Track
        pygame.draw.rect(surface, (50, 60, 80, 220), r, border_radius=3)
        # Fill
        hx = self._handle_x()
        fill_r = pygame.Rect(r.left, r.top, hx - r.left, r.height)
        if fill_r.width > 0:
            pygame.draw.rect(surface, (60, 130, 200, 220), fill_r, border_radius=3)
        # Handle
        pygame.draw.circle(surface, (180, 210, 255), (hx, r.centery), 6)

    def hit_handle(self, pos) -> bool:
        hx = self._handle_x()
        return abs(pos[0] - hx) <= 8 and abs(pos[1] - self._track_rect.centery) <= 10

    def handle_drag(self, pos) -> float:
        r = self._track_rect
        t = (pos[0] - r.left) / max(r.width, 1)
        self._exponent = self.EXPO_MIN + float(max(0.0, min(1.0, t))) * (self.EXPO_MAX - self.EXPO_MIN)
        return self.value


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    ModernGL + pygame renderer. One instance per simulation run.

    After initialization, call update(state) + render(fps, step_count, n_alive)
    each frame.
    """

    MODE_BONDS  = "bonds"
    MODE_MERGED = "merged"

    def __init__(self, config: SimConfig):
        self.config = config
        self.composite_mode = self.MODE_BONDS

        # ── pygame + OpenGL context ──────────────────────────────────────────
        pygame.init()
        pygame.display.set_caption("Half-Life Particle Simulator")
        # No RESIZABLE: avoids texture/surface resize complexity
        pygame.display.set_mode(
            (config.window_width, config.window_height),
            pygame.OPENGL | pygame.DOUBLEBUF
        )
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # ── Particle shader ──────────────────────────────────────────────────
        self.particle_prog = self.ctx.program(
            vertex_shader=PARTICLE_VERTEX_SHADER,
            fragment_shader=PARTICLE_FRAGMENT_SHADER,
        )
        self.particle_prog['u_world_size'].value = (config.world_width, config.world_height)

        self._particle_vertex_size = 7  # floats: x,y,r,g,b,a,size
        self._particle_buf_size = config.max_particles * self._particle_vertex_size * 4

        self.particle_vbo = self.ctx.buffer(reserve=self._particle_buf_size)
        self.particle_vao = self.ctx.vertex_array(
            self.particle_prog,
            [(self.particle_vbo, '2f 4f 1f', 'in_position', 'in_color', 'in_size')],
        )

        # ── Bond shader ──────────────────────────────────────────────────────
        self.bond_prog = self.ctx.program(
            vertex_shader=BOND_VERTEX_SHADER,
            fragment_shader=BOND_FRAGMENT_SHADER,
        )
        self.bond_prog['u_world_size'].value = (config.world_width, config.world_height)

        max_bonds = config.max_composites * config.max_composite_size
        self._bond_vertex_size = 6  # floats: x,y,r,g,b,a
        self._bond_buf_size = max_bonds * 2 * self._bond_vertex_size * 4
        self.bond_vbo = self.ctx.buffer(reserve=self._bond_buf_size)
        self.bond_vao = self.ctx.vertex_array(
            self.bond_prog,
            [(self.bond_vbo, '2f 4f', 'in_position', 'in_color')],
        )

        # ── HUD shader (pygame surface → fullscreen quad) ────────────────────
        self.hud_prog = self.ctx.program(
            vertex_shader=HUD_VERTEX_SHADER,
            fragment_shader=HUD_FRAGMENT_SHADER,
        )
        # Fullscreen quad: 2 triangles (6 vertices)
        quad_verts = np.array([
            -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,  1.0, -1.0,  1.0,
        ], dtype=np.float32)
        self._hud_quad_vbo = self.ctx.buffer(quad_verts.tobytes())
        self._hud_quad_vao = self.ctx.vertex_array(
            self.hud_prog,
            [(self._hud_quad_vbo, '2f', 'in_pos')],
        )
        self._hud_surface = pygame.Surface(
            (config.window_width, config.window_height), pygame.SRCALPHA
        )
        self._hud_texture = self.ctx.texture(
            (config.window_width, config.window_height), 4
        )
        self._hud_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # ── Event sprite shader ──────────────────────────────────────────────
        self.event_prog = self.ctx.program(
            vertex_shader=EVENT_VERTEX_SHADER,
            fragment_shader=EVENT_FRAGMENT_SHADER,
        )
        self.event_prog['u_world_size'].value = (config.world_width, config.world_height)

        self._event_max = 200
        self._event_vbo = self.ctx.buffer(reserve=self._event_max * 6 * 4)
        self._event_vao = self.ctx.vertex_array(
            self.event_prog,
            [(self._event_vbo, '2f 3f 1f', 'in_position', 'in_color', 'in_age')],
        )
        self._n_event_vertices = 0
        # Each event: (x, y, r, g, b, birth_sim_time, lifetime)
        self._events = []

        # ── Color palette ─────────────────────────────────────────────────────
        self.species_colors = get_species_colors(config)

        # ── HUD buttons: (label, pygame.Rect, action_str) ────────────────────
        btn_w, btn_h, gap = 108, 26, 4
        btn_x = 8
        self._buttons = []
        for k, (label, action) in enumerate([
            ("Pause",   "pause"),
            ("Bonds",   "toggle_bonds"),
            ("Stats",   "toggle_stats"),
            ("Events",  "toggle_events"),
            ("Reset",   "reset"),
            ("Params",  "toggle_params"),
        ]):
            rect = pygame.Rect(btn_x, 8 + k * (btn_h + gap), btn_w, btn_h)
            self._buttons.append((label, rect, action))

        # ── Physics sliders ───────────────────────────────────────────────────
        self._show_params = False
        self._dragging_slider = None
        self._physics_updates = {}
        panel_x = btn_x
        slider_track_w = 200
        # First slider starts just below the "Params" button (index 5)
        slider_start_y = 8 + 6 * (btn_h + gap) + 8  # 8px below bottom of Params btn
        slider_row_h = 38
        slider_specs = [
            ("damping",                  "damping",     0.995, "{:.4f}"),
            ("repulsion_strength",       "repulsion",   2.0,   "{:.2f}"),
            ("fusion_threshold",         "fuse thresh", 0.2,   "{:.3f}"),
            ("polarity_fusion_scale",    "pol fuse",    0.3,   "{:.3f}"),
            ("polarity_stability_scale", "pol stab",    0.5,   "{:.3f}"),
            ("binding_energy_scale",     "bind energy", 1.0,   "{:.3f}"),
            ("repulsion_radius",         "repulse r",   0.8,   "{:.2f}"),
            ("r_cutoff_scale",           "attract r\u00d7",  1.0,   "{:.2f}"),
            ("spring_k",                 "spring k",    50.0,  "{:.1f}"),
            ("attraction_scale",         "attract \u00d7",   1.0,   "{:.2f}"),
        ]
        self._sliders = []
        for k, (field, label, default, fmt) in enumerate(slider_specs):
            row_y = slider_start_y + k * slider_row_h
            track = pygame.Rect(panel_x + 4, row_y + 18, slider_track_w, 8)
            self._sliders.append(Slider(label, field, default, track, fmt))
        self._params_reset_rect = pygame.Rect(panel_x + 4, slider_start_y - 26, 100, 20)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._n_particles_to_draw = 0
        self._n_bond_vertices = 0
        self._font = pygame.font.SysFont('monospace', 13)
        self._clock = pygame.time.Clock()

        self._show_stats  = False
        self._show_events = True
        self._paused      = False   # mirror of main loop paused state

        self._prev_alive      = None
        self._prev_comp_alive = None

        self._stats_alive    = 0
        self._stats_free     = 0
        self._stats_n_comp   = 0
        self._stats_energy   = 0.0
        self._stats_step     = 0
        self._stats_sim_time = 0.0
        self._stats_hist     = np.zeros(config.max_composite_size, dtype=np.int32)

        # Event counters
        self._fusion_total   = 0
        self._decay_total    = 0   # composite decays (fissions)
        self._fusion_rate    = 0.0
        self._decay_rate     = 0.0
        self._event_history  = collections.deque(maxlen=300)

        # Sparkline buffers
        SPARK_LEN = 150
        self._spark_free   = collections.deque(maxlen=SPARK_LEN)
        self._spark_comp   = collections.deque(maxlen=SPARK_LEN)
        self._spark_energy = collections.deque(maxlen=SPARK_LEN)
        self._spark_fusion = collections.deque(maxlen=SPARK_LEN)
        self._spark_decay  = collections.deque(maxlen=SPARK_LEN)

    # ── Public interface ──────────────────────────────────────────────────────

    def toggle_composite_mode(self):
        """Cycle between bonds and merged visualization modes."""
        if self.composite_mode == self.MODE_BONDS:
            self.composite_mode = self.MODE_MERGED
        else:
            self.composite_mode = self.MODE_BONDS

    def toggle_stats(self):
        self._show_stats = not self._show_stats

    def toggle_events(self):
        self._show_events = not self._show_events

    def set_paused(self, paused: bool):
        """Keep the renderer in sync with the main loop's pause state."""
        self._paused = paused

    def handle_click(self, pos) -> str | None:
        """Return action string if a button was clicked, else None."""
        for _label, rect, action in self._buttons:
            if rect.collidepoint(pos):
                return action
        return None

    def toggle_params(self) -> None:
        self._show_params = not self._show_params

    def handle_mousedown_slider(self, pos) -> bool:
        """Start dragging a slider if pos hits a handle, or reset all if reset button hit."""
        if not self._show_params:
            return False
        if self._params_reset_rect.collidepoint(pos):
            for s in self._sliders:
                s.reset()
            self._physics_updates.update({s.field: s.value for s in self._sliders})
            return True
        for slider in self._sliders:
            if slider.hit_handle(pos):
                self._dragging_slider = slider
                slider.handle_drag(pos)
                self._physics_updates[slider.field] = slider.value
                return True
        return False

    def handle_mousemotion(self, pos) -> None:
        if self._dragging_slider is not None:
            self._dragging_slider.handle_drag(pos)
            self._physics_updates[self._dragging_slider.field] = self._dragging_slider.value

    def handle_mouseup(self) -> None:
        self._dragging_slider = None

    def get_physics_updates(self) -> dict:
        """Return accumulated {field: float} updates and clear the buffer."""
        updates = dict(self._physics_updates)
        self._physics_updates.clear()
        return updates

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, state: WorldState):
        """
        Transfer simulation state to GPU buffers for rendering.
        Called once per frame before render().
        """
        config = self.config
        particles  = state.particles
        composites = state.composites

        # Transfer to CPU
        pos        = np.asarray(particles.position)
        vel        = np.asarray(particles.velocity)
        species    = np.asarray(particles.species)
        alive      = np.asarray(particles.alive)
        mass       = np.asarray(particles.mass)
        energy     = np.asarray(particles.energy)
        comp_id    = np.asarray(particles.composite_id)

        comp_members = np.asarray(composites.members)
        comp_count   = np.asarray(composites.member_count)
        comp_alive   = np.asarray(composites.alive)

        # ── Particle vertex data ──────────────────────────────────────────────
        alive_idx = np.where(alive)[0]
        n_alive   = len(alive_idx)

        if n_alive > 0:
            p_pos    = pos[alive_idx]
            p_sp     = species[alive_idx]
            p_mass   = mass[alive_idx]
            p_comp   = comp_id[alive_idx]

            colors = self.species_colors[p_sp]
            speed  = np.linalg.norm(vel[alive_idx], axis=-1)
            brightness = np.clip(0.5 + speed / config.init_speed, 0.5, 1.5)
            colors = np.clip(colors * brightness[:, None], 0.0, 1.0)

            if self.composite_mode == self.MODE_MERGED:
                alpha = np.where(p_comp >= 0, 0.15, 1.0).astype(np.float32)
            else:
                alpha = np.ones(n_alive, dtype=np.float32)

            size = np.clip(
                config.point_size_min + np.log1p(p_mass) * 3.0,
                config.point_size_min, config.point_size_max
            ).astype(np.float32)

            vertex_data = np.concatenate([
                p_pos.astype(np.float32),
                colors.astype(np.float32),
                alpha[:, None],
                size[:, None],
            ], axis=1).astype(np.float32)

            self.particle_vbo.write(vertex_data.flatten().tobytes())
            self._n_particles_to_draw = n_alive
        else:
            self._n_particles_to_draw = 0

        # ── Bond / merged vertex data ─────────────────────────────────────────
        self._n_bond_vertices = 0
        alive_comp_idx = np.where(comp_alive)[0]

        if self.composite_mode == self.MODE_BONDS and len(alive_comp_idx) > 0:
            bond_chunks = []
            for c in alive_comp_idx:
                n = comp_count[c]
                members = comp_members[c, :n]
                if n < 2:
                    continue
                # Vectorized pair generation via triu_indices
                ii, jj = np.triu_indices(n, k=1)
                mem_a, mem_b = members[ii], members[jj]
                valid = (mem_a >= 0) & (mem_b >= 0)
                if not np.any(valid):
                    continue

                mem_a_v = mem_a[valid]
                mem_b_v = mem_b[valid]

                comp_color = np.mean(self.species_colors[species[members]], axis=0)
                bond_rgba  = np.array([*comp_color, 0.5], dtype=np.float32)

                pos_a = pos[mem_a_v].copy()
                dx    = pos[mem_b_v] - pos_a
                if config.boundary_mode == "periodic":
                    dx[:, 0] -= config.world_width  * np.round(dx[:, 0] / config.world_width)
                    dx[:, 1] -= config.world_height * np.round(dx[:, 1] / config.world_height)
                pos_b = pos_a + dx

                n_pairs = len(mem_a_v)
                # Interleave: [a0, b0, a1, b1, ...]
                pairs_pos = np.empty((n_pairs * 2, 2), dtype=np.float32)
                pairs_pos[0::2] = pos_a
                pairs_pos[1::2] = pos_b
                rgba_tiled = np.tile(bond_rgba, (n_pairs * 2, 1))
                bond_chunks.append(np.concatenate([pairs_pos, rgba_tiled], axis=1))

            if bond_chunks:
                bond_data = np.concatenate(bond_chunks, axis=0).astype(np.float32)
                n_bytes   = min(bond_data.nbytes, self._bond_buf_size)
                self.bond_vbo.write(bond_data.flatten().tobytes()[:n_bytes])
                self._n_bond_vertices = len(bond_data)

        elif self.composite_mode == self.MODE_MERGED and len(alive_comp_idx) > 0:
            merged_verts = []
            for c in alive_comp_idx:
                n = comp_count[c]
                members = comp_members[c, :n]
                if n < 1:
                    continue
                valid_m = members[members >= 0]
                if len(valid_m) == 0:
                    continue
                com        = np.mean(pos[valid_m], axis=0)
                total_mass = np.sum(mass[valid_m])
                avg_color  = np.mean(self.species_colors[species[valid_m]], axis=0)
                sz = np.clip(
                    config.point_size_min + np.log1p(total_mass) * 4.0,
                    config.point_size_min, config.point_size_max * 1.5
                )
                merged_verts.append(
                    np.array([*com, *avg_color, 1.0, sz], dtype=np.float32)
                )

            if merged_verts:
                merged_data = np.stack(merged_verts, axis=0).astype(np.float32)
                offset      = self._n_particles_to_draw * self._particle_vertex_size * 4
                extra_bytes = merged_data.flatten().tobytes()
                avail       = self._particle_buf_size - offset
                self.particle_vbo.write(extra_bytes[:min(len(extra_bytes), avail)], offset=offset)
                self._n_particles_to_draw += len(merged_verts)

        # ── Stats ─────────────────────────────────────────────────────────────
        self._stats_alive    = int(np.sum(alive))
        self._stats_free     = int(np.sum(alive & (comp_id < 0)))
        self._stats_n_comp   = int(np.sum(comp_alive))
        self._stats_energy   = float(np.asarray(state.total_energy))
        self._stats_step     = int(np.asarray(state.step_count))
        self._stats_sim_time = float(np.asarray(state.time))

        if self._stats_n_comp > 0:
            alive_counts = comp_count[comp_alive]
            self._stats_hist, _ = np.histogram(
                alive_counts,
                bins=np.arange(1, config.max_composite_size + 2)
            )
        else:
            self._stats_hist = np.zeros(config.max_composite_size, dtype=np.int32)

        # ── Event detection ───────────────────────────────────────────────────
        current_sim_time  = self._stats_sim_time
        comp_lifetime_secs = 50.0 * config.dt
        part_lifetime_secs = 20.0 * config.dt

        # Expire old events
        self._events = [
            ev for ev in self._events
            if current_sim_time - ev[5] < ev[6]
        ]

        if self._show_events and self._prev_alive is not None:
            new_comps  = ~self._prev_comp_alive & comp_alive
            dead_comps = self._prev_comp_alive  & ~comp_alive

            budget = max(0, self._event_max - len(self._events))

            # Fusion events (gold)
            for c in np.where(new_comps)[0][:min(40, budget // 3)]:
                n       = comp_count[c]
                mems    = comp_members[c, :n]
                valid_m = mems[mems >= 0]
                if len(valid_m) > 0:
                    ex = float(np.mean(pos[valid_m, 0]))
                    ey = float(np.mean(pos[valid_m, 1]))
                    self._events.append(
                        (ex, ey, 1.0, 0.85, 0.0, current_sim_time, comp_lifetime_secs)
                    )

            # Fission events (cyan)
            for c in np.where(dead_comps)[0][:min(40, budget // 3)]:
                n       = comp_count[c]
                mems    = comp_members[c, :n]
                valid_m = mems[mems >= 0]
                if len(valid_m) > 0:
                    ex = float(np.mean(pos[valid_m, 0]))
                    ey = float(np.mean(pos[valid_m, 1]))
                    self._events.append(
                        (ex, ey, 0.0, 1.0, 1.0, current_sim_time, comp_lifetime_secs)
                    )

            # Hard cap
            self._events = self._events[-self._event_max:]

        # Accumulate event counts (composite-level only; free particles don't decay)
        if self._prev_alive is not None:
            n_fusion = int(np.sum(~self._prev_comp_alive & comp_alive))
            n_decay  = int(np.sum(self._prev_comp_alive  & ~comp_alive))
            self._fusion_total += n_fusion
            self._decay_total  += n_decay
            self._event_history.append(
                (self._stats_sim_time, n_fusion, n_decay)
            )
            recent = [e for e in self._event_history
                      if e[0] >= self._stats_sim_time - 5.0]
            if len(recent) >= 2:
                dt = max(0.01, recent[-1][0] - recent[0][0])
                self._fusion_rate = sum(e[1] for e in recent) / dt
                self._decay_rate  = sum(e[2] for e in recent) / dt

        # Sparklines
        self._spark_free.append(self._stats_free)
        self._spark_comp.append(self._stats_n_comp)
        self._spark_energy.append(self._stats_energy)
        self._spark_fusion.append(self._fusion_rate)
        self._spark_decay.append(self._decay_rate)

        self._prev_alive      = alive.copy()
        self._prev_comp_alive = comp_alive.copy()

        # ── Build event VBO ───────────────────────────────────────────────────
        self._n_event_vertices = 0
        if self._show_events and self._events:
            ev_data = []
            for (x, y, r, g, b, bt, lt) in self._events:
                age = float(np.clip((current_sim_time - bt) / max(lt, 1e-8), 0.0, 1.0))
                ev_data.append([x, y, r, g, b, age])
            ev_arr = np.array(ev_data, dtype=np.float32)
            n_bytes = min(ev_arr.nbytes, self._event_max * 6 * 4)
            self._event_vbo.write(ev_arr.flatten().tobytes()[:n_bytes])
            self._n_event_vertices = len(ev_data)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, fps: float, step_count: int, n_alive: int):
        """Clear screen, draw scene, draw HUD, flip buffers."""
        bg = self.config.background_color
        self.ctx.clear(*bg)

        # Particles
        if self._n_particles_to_draw > 0:
            self.particle_vao.render(moderngl.POINTS, vertices=self._n_particles_to_draw)

        # Bonds
        if self.composite_mode == self.MODE_BONDS and self._n_bond_vertices > 0:
            self.ctx.line_width = 1.0
            self.bond_vao.render(moderngl.LINES, vertices=self._n_bond_vertices)

        # Event sprites
        if self._show_events and self._n_event_vertices > 0:
            self.event_prog['u_world_size'].value = (
                self.config.world_width, self.config.world_height
            )
            self._event_vao.render(moderngl.POINTS, vertices=self._n_event_vertices)

        # HUD overlay
        self._render_hud_surface(fps)
        surf_data = pygame.image.tostring(self._hud_surface, 'RGBA', True)
        self._hud_texture.write(surf_data)
        self._hud_texture.use(location=0)
        self.hud_prog['hud_tex'].value = 0
        self._hud_quad_vao.render(moderngl.TRIANGLES, vertices=6)

        pygame.display.flip()

    # ── HUD surface drawing ───────────────────────────────────────────────────

    def _draw_sparkline(self, surface, data, x, y, w, h, color):
        if len(data) < 2:
            return
        arr = np.array(data, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            hi = lo + 1
        pts = [
            (x + int(k * w / (len(arr) - 1)),
             y + h - int((v - lo) / (hi - lo) * h))
            for k, v in enumerate(arr)
        ]
        pygame.draw.lines(surface, color, False, pts, 1)

    def _render_hud_surface(self, fps: float):
        """Draw buttons and optional stats panel onto the transparent HUD surface."""
        surface = self._hud_surface
        surface.fill((0, 0, 0, 0))  # clear to transparent
        font    = self._font

        # ── Buttons ──────────────────────────────────────────────────────────
        for label, rect, action in self._buttons:
            # Dynamic label
            display_label = label
            if action == 'pause':
                display_label = "Resume" if self._paused else "Pause"
            elif action == 'toggle_bonds':
                display_label = "Merged" if self.composite_mode == self.MODE_BONDS else "Bonds"
            elif action == 'toggle_stats':
                display_label = "Stats ON" if self._show_stats else "Stats"
            elif action == 'toggle_events':
                display_label = "Events ON" if self._show_events else "Events"
            elif action == 'toggle_params':
                display_label = "Params ON" if self._show_params else "Params"

            # Background
            if action in ('reset',):
                bg_col = (90, 40, 40, 200)
            elif action == 'pause':
                bg_col = (40, 80, 50, 200) if not self._paused else (80, 60, 30, 200)
            else:
                bg_col = (40, 55, 80, 200)
            pygame.draw.rect(surface, bg_col, rect, border_radius=4)
            pygame.draw.rect(surface, (100, 140, 200, 180), rect, 1, border_radius=4)

            # Text
            txt = font.render(display_label, True, (210, 230, 255))
            surface.blit(txt, (rect.centerx - txt.get_width() // 2,
                                rect.centery - txt.get_height() // 2))

        # ── Stats panel ───────────────────────────────────────────────────────
        if self._show_stats:
            config = self.config
            # panel_h: 4 static rows (16px) + 5 spark-stat rows (15+18px) +
            #          gap + header (18px) + chart (64px) + ticks (20px) + bottom (10px)
            panel_h = (4 * 16 + 5 * 33 + 4 + 18 + 64 + 20 + 10)
            panel_w = 215
            panel_x = config.window_width - panel_w - 8
            panel_y = 8

            panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
            pygame.draw.rect(surface, (15, 18, 35, 185), panel_rect, border_radius=6)
            pygame.draw.rect(surface, (70, 100, 150, 180), panel_rect, 1, border_radius=6)

            y_off = panel_y + 6

            # Static lines (no sparkline)
            for line in [
                f"FPS:        {fps:.1f}",
                f"Step:       {self._stats_step:,}",
                f"Sim time:   {self._stats_sim_time:.1f}",
                f"Particles:  {self._stats_alive:,}",
            ]:
                txt = font.render(line, True, (190, 215, 255))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 16

            # Spark-stat rows: label on one line, sparkline below it
            spark_w = panel_w - 16
            spark_h = 14
            spark_x = panel_x + 8
            for label, spark_data, color in [
                (f"Free:       {self._stats_free:,}",   self._spark_free,   (80, 180, 255)),
                (f"Composites: {self._stats_n_comp:,}", self._spark_comp,   (120, 220, 120)),
                (f"Energy:     {self._stats_energy:.0f}", self._spark_energy, (220, 180, 80)),
            ]:
                txt = font.render(label, True, (190, 215, 255))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 15
                self._draw_sparkline(surface, spark_data, spark_x, y_off, spark_w, spark_h, color)
                y_off += 18

            # Spark-stat rows for fusions and decays
            y_off += 4
            for label, spark_data, color in [
                (f"Fusions: {self._fusion_total:,} ({self._fusion_rate:.1f}/s)", self._spark_fusion, (220, 170, 60)),
                (f"Decays:  {self._decay_total:,}  ({self._decay_rate:.1f}/s)",  self._spark_decay,  (80, 200, 220)),
            ]:
                txt = font.render(label, True, (190, 215, 255))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 15
                self._draw_sparkline(surface, spark_data, spark_x, y_off, spark_w, spark_h, color)
                y_off += 18

            # Histogram — vertical bar chart
            y_off += 4
            txt = font.render("Composite sizes:", True, (160, 185, 230))
            surface.blit(txt, (panel_x + 6, y_off))
            y_off += 18

            chart_w = panel_w - 16
            chart_h = 64
            chart_x = panel_x + 8
            chart_y = y_off
            n_bins   = min(len(self._stats_hist) - 1, 40)
            bar_w    = max(1, (chart_w - n_bins) // max(1, n_bins))
            max_count = max(1, int(np.max(self._stats_hist[1:n_bins + 2]))
                            if len(self._stats_hist) > 1 else 1)

            for sz_idx in range(1, n_bins + 1):
                count = int(self._stats_hist[sz_idx]) if sz_idx < len(self._stats_hist) else 0
                if count == 0:
                    continue
                bh = max(1, int(chart_h * count / max_count))
                bx = chart_x + (sz_idx - 1) * (bar_w + 1)
                pygame.draw.rect(surface, (60, 140, 220, 200),
                                 pygame.Rect(bx, chart_y + chart_h - bh, bar_w, bh))

            pygame.draw.line(surface, (80, 110, 160),
                             (chart_x, chart_y + chart_h),
                             (chart_x + chart_w, chart_y + chart_h), 1)
            raw_ticks = np.linspace(2, n_bins, 5).astype(int)
            for tick in sorted(set(raw_ticks)):
                tick_idx = tick - 1
                if 0 <= tick_idx < n_bins:
                    tx = chart_x + tick_idx * (bar_w + 1)
                    lbl = font.render(str(tick), True, (120, 150, 190))
                    surface.blit(lbl, (tx - lbl.get_width() // 2, chart_y + chart_h + 2))
            y_off = chart_y + chart_h + 20

        # ── Params panel ──────────────────────────────────────────────────────
        if self._show_params:
            btn_w, btn_h, gap = 108, 26, 4
            panel_x = 8
            slider_start_y = 8 + 6 * (btn_h + gap) + 8
            panel_w = 220
            panel_h = len(self._sliders) * 38 + 10 + 26
            panel_rect = pygame.Rect(panel_x - 4, slider_start_y - 30, panel_w, panel_h)
            pygame.draw.rect(surface, (15, 18, 35, 185), panel_rect, border_radius=6)
            pygame.draw.rect(surface, (70, 100, 150, 180), panel_rect, 1, border_radius=6)
            # Reset Params button
            reset_rect = self._params_reset_rect
            pygame.draw.rect(surface, (80, 30, 30, 200), reset_rect, border_radius=4)
            pygame.draw.rect(surface, (150, 80, 80, 180), reset_rect, 1, border_radius=4)
            reset_txt = font.render("Reset Params", True, (255, 160, 160))
            surface.blit(reset_txt, (reset_rect.centerx - reset_txt.get_width() // 2,
                                     reset_rect.centery - reset_txt.get_height() // 2))
            for slider in self._sliders:
                slider.draw(surface, font)

        # ── Bottom key hint ───────────────────────────────────────────────────
        hint = "[Space] pause  [+/-] speed  [B] viz  [R] reset  [Q] quit"
        hint_surf = font.render(hint, True, (120, 140, 160))
        surface.blit(hint_surf,
                     (self.config.window_width // 2 - hint_surf.get_width() // 2,
                      self.config.window_height - hint_surf.get_height() - 4))

    # ── Legacy HUD (title bar only) ───────────────────────────────────────────

    def _update_title(self, fps: float, step_count: int, n_alive: int):
        pygame.display.set_caption(
            f"Half-Life | FPS:{fps:.0f} | Step:{step_count:,} | Alive:{n_alive:,}"
        )

    def close(self):
        """Release all GPU resources."""
        self.particle_vbo.release()
        self.particle_vao.release()
        self.bond_vbo.release()
        self.bond_vao.release()
        self.particle_prog.release()
        self.bond_prog.release()
        self._hud_quad_vbo.release()
        self._hud_quad_vao.release()
        self._hud_texture.release()
        self.hud_prog.release()
        self._event_vbo.release()
        self._event_vao.release()
        self.event_prog.release()
        self.ctx.release()
        pygame.quit()
