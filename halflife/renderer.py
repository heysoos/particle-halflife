"""
Real-time ModernGL + pygame renderer for the Half-Life simulator.

Renders particles as point sprites (circular blobs), colored by species
and sized by mass. Composites can be shown in two modes (toggle with B):
  - Bonds mode:  GL_LINES connecting composite member particles
  - Merged mode: single large point at the composite center of mass

Data flow per frame:
  1. np.asarray() — transfer JAX GPU arrays to CPU numpy (~0.5ms for 50K particles)
  2. Pack vertex data: (x, y, r, g, b, a, size) per particle
  3. vbo.write() — upload to GPU via OpenGL
  4. glDrawArrays(GL_POINTS, ...) + optional GL_LINES for bonds
  5. pygame.display.flip()
"""

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
    // Map world coordinates [0, world_size] to NDC [-1, 1]
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
    // Circular point sprite with smooth edge and glow falloff
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r = length(coord) * 2.0;

    // Glow: bright center, soft edge
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


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    ModernGL + pygame renderer. One instance per simulation run.

    After initialization, call update(state, config) + render() each frame.
    """

    # Composite visualization modes
    MODE_BONDS  = "bonds"   # draw lines between bonded particles
    MODE_MERGED = "merged"  # show composite as single large point

    def __init__(self, config: SimConfig):
        self.config = config
        self.composite_mode = self.MODE_BONDS

        # ── pygame + OpenGL context ──────────────────────────────────────────
        pygame.init()
        pygame.display.set_caption("Half-Life Particle Simulator")
        pygame.display.set_mode(
            (config.window_width, config.window_height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # ── Shaders ──────────────────────────────────────────────────────────
        self.particle_prog = self.ctx.program(
            vertex_shader=PARTICLE_VERTEX_SHADER,
            fragment_shader=PARTICLE_FRAGMENT_SHADER,
        )
        self.bond_prog = self.ctx.program(
            vertex_shader=BOND_VERTEX_SHADER,
            fragment_shader=BOND_FRAGMENT_SHADER,
        )

        # Upload world/window size uniforms
        self.particle_prog['u_world_size'].value = (config.world_width, config.world_height)
        self.bond_prog['u_world_size'].value = (config.world_width, config.world_height)

        # ── Buffers: Particles ────────────────────────────────────────────────
        # Vertex format: x, y, r, g, b, a, size  (7 float32 per particle)
        self._particle_vertex_size = 7  # floats
        self._particle_buf_size = config.max_particles * self._particle_vertex_size * 4  # bytes

        self.particle_vbo = self.ctx.buffer(reserve=self._particle_buf_size)
        self.particle_vao = self.ctx.vertex_array(
            self.particle_prog,
            [(self.particle_vbo, '2f 4f 1f', 'in_position', 'in_color', 'in_size')],
        )

        # ── Buffers: Bonds ────────────────────────────────────────────────────
        # Each bond = 2 vertices, each vertex = x, y, r, g, b, a (6 floats)
        max_bonds = config.max_composites * config.max_composite_size
        self._bond_vertex_size = 6  # floats
        self._bond_buf_size = max_bonds * 2 * self._bond_vertex_size * 4  # bytes
        self.bond_vbo = self.ctx.buffer(reserve=self._bond_buf_size)
        self.bond_vao = self.ctx.vertex_array(
            self.bond_prog,
            [(self.bond_vbo, '2f 4f', 'in_position', 'in_color')],
        )

        # ── Color palette ─────────────────────────────────────────────────────
        # (num_species, 3) float32 RGB
        self.species_colors = get_species_colors(config)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._n_particles_to_draw = 0
        self._n_bond_vertices = 0
        self._font = pygame.font.SysFont('monospace', 14)
        self._clock = pygame.time.Clock()

    def toggle_composite_mode(self):
        """Cycle between bonds and merged visualization modes."""
        if self.composite_mode == self.MODE_BONDS:
            self.composite_mode = self.MODE_MERGED
        else:
            self.composite_mode = self.MODE_BONDS

    def update(self, state: WorldState):
        """
        Transfer simulation state to GPU buffers for rendering.

        Called once per frame before render().
        The np.asarray() calls here are the JAX→CPU data transfer.
        """
        config = self.config
        particles = state.particles
        composites = state.composites

        # Transfer to CPU (this is the GPU→CPU sync point)
        pos     = np.asarray(particles.position)   # (N, 2)
        vel     = np.asarray(particles.velocity)   # (N, 2)
        species = np.asarray(particles.species)    # (N,)
        alive   = np.asarray(particles.alive)      # (N,) bool
        mass    = np.asarray(particles.mass)       # (N,)
        energy  = np.asarray(particles.energy)     # (N,)
        comp_id = np.asarray(particles.composite_id)  # (N,)

        comp_members = np.asarray(composites.members)       # (C, M)
        comp_count   = np.asarray(composites.member_count)  # (C,)
        comp_alive   = np.asarray(composites.alive)         # (C,)

        # ── Build particle vertex data ────────────────────────────────────────
        alive_idx = np.where(alive)[0]
        n_alive = len(alive_idx)

        if n_alive > 0:
            p_pos     = pos[alive_idx]       # (n_alive, 2)
            p_species = species[alive_idx]   # (n_alive,)
            p_mass    = mass[alive_idx]      # (n_alive,)
            p_energy  = energy[alive_idx]    # (n_alive,)
            p_comp    = comp_id[alive_idx]   # (n_alive,)

            # Colors from species palette, brightness modulated by energy
            colors = self.species_colors[p_species]  # (n_alive, 3)
            speed = np.linalg.norm(vel[alive_idx], axis=-1)  # (n_alive,)
            brightness = np.clip(0.5 + speed / config.init_speed, 0.5, 1.5)
            colors = np.clip(colors * brightness[:, None], 0.0, 1.0)

            # Alpha: full for free particles, slightly dimmed for composite members
            alpha = np.where(p_comp >= 0, 0.7, 1.0).astype(np.float32)

            # Size: scales with log(mass), clamped to [min, max]
            size = np.clip(
                config.point_size_min + np.log1p(p_mass) * 3.0,
                config.point_size_min,
                config.point_size_max
            ).astype(np.float32)

            # In merged mode, hide individual particles that are in composites
            if self.composite_mode == self.MODE_MERGED:
                alpha = np.where(p_comp >= 0, 0.0, alpha)

            # Pack: x, y, r, g, b, a, size  — (n_alive, 7)
            vertex_data = np.concatenate([
                p_pos.astype(np.float32),
                colors.astype(np.float32),
                alpha[:, None],
                size[:, None],
            ], axis=1).astype(np.float32)

            flat = vertex_data.flatten()
            self.particle_vbo.write(flat.tobytes())
            self._n_particles_to_draw = n_alive
        else:
            self._n_particles_to_draw = 0

        # ── Build bond / merged vertex data ───────────────────────────────────
        self._n_bond_vertices = 0
        alive_comp_idx = np.where(comp_alive)[0]

        if self.composite_mode == self.MODE_BONDS and len(alive_comp_idx) > 0:
            bond_verts = []
            for c in alive_comp_idx:
                n = comp_count[c]
                members = comp_members[c, :n]
                if n < 2:
                    continue
                # Average color of member species
                comp_color = np.mean(self.species_colors[species[members]], axis=0)
                bond_rgba = np.array([*comp_color, 0.5], dtype=np.float32)

                for m_a in range(n):
                    for m_b in range(m_a + 1, n):
                        i_a = members[m_a]
                        i_b = members[m_b]
                        if i_a < 0 or i_b < 0:
                            continue
                        # Two vertices per bond line: pos_a and pos_b
                        bond_verts.append(np.concatenate([pos[i_a], bond_rgba]))
                        bond_verts.append(np.concatenate([pos[i_b], bond_rgba]))

            if bond_verts:
                bond_data = np.stack(bond_verts, axis=0).astype(np.float32)
                n_bytes = min(bond_data.nbytes, self._bond_buf_size)
                self.bond_vbo.write(bond_data.flatten().tobytes()[:n_bytes])
                self._n_bond_vertices = len(bond_verts)

        elif self.composite_mode == self.MODE_MERGED and len(alive_comp_idx) > 0:
            merged_verts = []
            for c in alive_comp_idx:
                n = comp_count[c]
                members = comp_members[c, :n]
                if n < 1:
                    continue
                valid_members = members[members >= 0]
                if len(valid_members) == 0:
                    continue
                com = np.mean(pos[valid_members], axis=0)
                total_mass = np.sum(mass[valid_members])
                avg_color = np.mean(self.species_colors[species[valid_members]], axis=0)
                size_merged = np.clip(
                    config.point_size_min + np.log1p(total_mass) * 4.0,
                    config.point_size_min,
                    config.point_size_max * 1.5
                )
                alpha_m = 1.0
                merged_verts.append(
                    np.array([*com, *avg_color, alpha_m, size_merged], dtype=np.float32)
                )

            if merged_verts:
                merged_data = np.stack(merged_verts, axis=0).astype(np.float32)
                # Write into particle buffer after the alive particles
                offset = self._n_particles_to_draw * self._particle_vertex_size * 4
                extra_bytes = merged_data.flatten().tobytes()
                avail = self._particle_buf_size - offset
                write_bytes = extra_bytes[:min(len(extra_bytes), avail)]
                self.particle_vbo.write(write_bytes, offset=offset)
                self._n_particles_to_draw += len(merged_verts)

    def render(self, fps: float, step_count: int, n_alive: int):
        """
        Clear screen, draw particles and bonds, flip buffers.

        Args:
            fps:        current frames per second
            step_count: simulation step counter
            n_alive:    number of alive particles
        """
        bg = self.config.background_color
        self.ctx.clear(*bg)

        # Draw particles (point sprites)
        if self._n_particles_to_draw > 0:
            self.particle_vao.render(moderngl.POINTS, vertices=self._n_particles_to_draw)

        # Draw bonds
        if self.composite_mode == self.MODE_BONDS and self._n_bond_vertices > 0:
            self.ctx.line_width = 1.0
            self.bond_vao.render(moderngl.LINES, vertices=self._n_bond_vertices)

        # HUD overlay via pygame surface
        self._draw_hud(fps, step_count, n_alive)

        pygame.display.flip()

    def _draw_hud(self, fps: float, step_count: int, n_alive: int):
        """Blit a small text overlay onto the pygame surface."""
        mode_str = "BONDS" if self.composite_mode == self.MODE_BONDS else "MERGED"
        lines = [
            f"FPS: {fps:.1f}",
            f"Step: {step_count:,}",
            f"Alive: {n_alive:,} / {self.config.max_particles:,}",
            f"Viz: {mode_str}  [B] toggle",
            f"[Space] pause  [+/-] speed  [R] reset  [Q] quit",
        ]
        # Render text to a pygame surface then blit to screen
        # (pygame and moderngl share the same window; blit to an offscreen surface,
        #  then render as a texture — or use a simpler title bar update)
        pygame.display.set_caption(
            f"Half-Life | FPS:{fps:.0f} | Step:{step_count:,} | Alive:{n_alive:,}"
        )

    def close(self):
        """Release resources."""
        self.particle_vbo.release()
        self.particle_vao.release()
        self.bond_vbo.release()
        self.bond_vao.release()
        self.particle_prog.release()
        self.bond_prog.release()
        self.ctx.release()
        pygame.quit()
