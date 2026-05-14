"""HUD overlay painter — buttons, stats panel, params panels, inspector.

Owns the pygame-surface drawing that gets uploaded as the HUD texture every
frame. Pulled out of renderer.py so HUD/UI changes don't churn the GL
pipeline file. All state still lives on the Renderer (back-reference via
self.r); this class is a *painter*, not a state container — by design, so
mutations like `_selected_idx` and the panel-toggle bools stay on the same
object that handles input.

The Renderer instantiates one HUDPainter in __init__ as `self.hud` and:
  - calls `self.hud.refresh_selected_snapshot()` once per frame from update()
  - calls `self.hud.paint(fps)` once per frame from render() when HUD is dirty
"""

import jax.numpy as jnp
import numpy as np
import pygame


class HUDPainter:
    """Paints the HUD surface from Renderer state. Holds a back-ref `self.r`.

    All field reads (`self.r._show_stats`, etc.) and mutations
    (`self.r._inspector_close_rect = ...`) go through that reference. This
    keeps the move from renderer.py mechanical: every former `self.X` in the
    moved methods becomes `self.r.X` here.
    """

    def __init__(self, renderer):
        self.r = renderer

    # ── Per-frame snapshot ────────────────────────────────────────────────────

    def _build_species_valence(self) -> None:
        """Compute per-species valence using the same hash chemistry uses."""
        from halflife.chemistry import _hash_to_valence
        config = self.r.config
        self.r._species_valence = np.zeros(config.num_species, dtype=np.int32)
        # _hash_to_valence expects a JAX-style scalar (it calls .astype on its
        # input), so wrap the Python int. Called once, cached after.
        for s in range(config.num_species):
            self.r._species_valence[s] = int(_hash_to_valence(jnp.int32(s), config))

    def refresh_selected_snapshot(self) -> None:
        """Rebuild the dict of stats shown in the inspector panel.

        Called once per frame from Renderer.update(); cheap (touches a few
        scalars). Stashes None when nothing is selected so the panel renderer
        can early-out.
        """
        r = self.r
        i = r._selected_idx
        cs = r._cpu_state
        if i < 0 or cs is None:
            r._selected_snapshot = None
            return

        config = r.config
        px, py = float(cs.positions[i, 0]),  float(cs.positions[i, 1])
        vx, vy = float(cs.velocities[i, 0]), float(cs.velocities[i, 1])
        speed  = float(np.hypot(vx, vy))
        species_i = int(cs.species[i])
        mass_i    = float(cs.mass[i])
        energy_i  = float(cs.energy[i])
        age_i     = float(cs.age[i])
        cid       = int(cs.comp_id[i])

        if config.use_valence:
            if r._species_valence is None:
                self._build_species_valence()
            valence_i = int(r._species_valence[species_i])
        else:
            valence_i = None

        snap = {
            'idx':       i,
            'pos':       (px, py),
            'vel':       (vx, vy),
            'speed':     speed,
            'species':   species_i,
            'valence':   valence_i,
            'mass':      mass_i,
            'energy':    energy_i,
            'age':       age_i,
            'composite': None,
        }

        if cid >= 0 and bool(cs.comp_alive[cid]):
            count = int(cs.comp_count[cid])
            member_species = [int(cs.species[m])
                              for m in cs.comp_members[cid][:count] if m >= 0]
            snap['composite'] = {
                'id':             cid,
                'size':           count,
                'hash':           int(cs.comp_species_hash[cid]),
                'members':        member_species,
                'binding_energy': float(cs.comp_binding_energy[cid]),
                'half_life':      float(cs.comp_half_life[cid]),
                'age':            float(cs.comp_age[cid]),
                'free_bonds':     (int(cs.comp_free_bonds[cid])
                                   if config.use_valence else None),
            }
        r._selected_snapshot = snap

    # ── Small drawing helpers ─────────────────────────────────────────────────

    def _species_display_rgb(self, sp: int) -> tuple:
        """Linear-sRGB species color → 8-bit display-sRGB tuple for pygame draws.

        pygame surfaces are sRGB-encoded, so the linear colors used in the
        GL pipeline need a 1/2.2 gamma encoding before being handed to the
        software rasterizer. Used by the inspector panel's species swatch
        and member chips.
        """
        col_disp = np.clip(self.r.species_colors[sp], 0.0, 1.0) ** (1.0 / 2.2)
        return tuple(int(round(c * 255)) for c in col_disp)

    def _button_bg_color(self, action: str) -> tuple:
        """Look up the background tint for a button by its action string.

        Pause shows an alternate tint while the sim is paused. Reroll
        variants share one purple tint via prefix match. Everything else
        falls back to the default blue tint.
        """
        bg = self.r.BUTTON_BG
        if action == 'pause':
            return bg['pause_active'] if self.r._paused else bg['pause']
        if action.startswith('reroll_'):
            return bg['reroll']
        return bg.get(action, bg['default'])

    def _draw_slider_panel(self, surface: pygame.Surface, font,
                           sliders: list, content_h: int,
                           reset_rect: pygame.Rect, reset_label: str) -> None:
        """Draw a single slider-panel block.

        Used for both the physics-params panel and the render-params (trails)
        panel — they share the same X column, geometry, and styling, so they
        ride on the same helper. Caller supplies the slider list, the
        content height, the reset-button rect, and the reset-button label.
        """
        btn_w, btn_h, gap = 108, 26, 4
        panel_x = 8 + btn_w + 8   # right of the button strip
        slider_start_y = 8 + self.r._n_buttons * (btn_h + gap) + 8
        panel_w = 244   # track(200) + gap(4) + reset-btn(14) + margins
        panel_h = content_h + 10 + 26
        panel_rect = pygame.Rect(panel_x - 4, slider_start_y - 30, panel_w, panel_h)
        pygame.draw.rect(surface, (15, 18, 35, 185), panel_rect, border_radius=6)
        pygame.draw.rect(surface, (70, 100, 150, 180), panel_rect, 1, border_radius=6)
        # Reset button (panel-level "reset all sliders in this panel")
        pygame.draw.rect(surface, (80, 30, 30, 200), reset_rect, border_radius=4)
        pygame.draw.rect(surface, (150, 80, 80, 180), reset_rect, 1, border_radius=4)
        reset_txt = font.render(reset_label, True, (255, 160, 160))
        surface.blit(reset_txt, (reset_rect.centerx - reset_txt.get_width() // 2,
                                 reset_rect.centery - reset_txt.get_height() // 2))
        for slider in sliders:
            slider.draw(surface, font)

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

    # ── Phase methods called from paint() ─────────────────────────────────────

    def _paint_buttons(self, surface: pygame.Surface, font) -> None:
        """Paint the left-edge button strip + the top-right Stats button."""
        r = self.r
        for label, rect, action in r._buttons:
            # Dynamic label
            display_label = label
            if action == 'pause':
                display_label = "Resume" if r._paused else "Pause"
            elif action == 'toggle_bonds':
                # Label shows the CURRENT composite-view mode (not the next
                # one the click will switch to — that "next-click" wording
                # made the button read as opposite to what was on screen).
                current_label = {
                    r.MODE_BONDS:  "Bonds",
                    r.MODE_MERGED: "Merged",
                    r.MODE_NONE:   "None",
                }
                display_label = current_label.get(r.composite_mode, "Bonds")
            elif action == 'toggle_events':
                display_label = "Events ON" if r._show_events else "Events"
            elif action == 'toggle_trails':
                display_label = "Trails ON" if r._render_settings['trails_on'] else "Trails"
            elif action == 'toggle_params':
                display_label = "Params ON" if r._show_params else "Params"

            # Background
            bg_col = self._button_bg_color(action)
            pygame.draw.rect(surface, bg_col, rect, border_radius=4)
            pygame.draw.rect(surface, (100, 140, 200, 180), rect, 1, border_radius=4)

            # Text — for the Trails button, the right ~18px belongs to the
            # gear nub (separate click region opening the trails panel), so
            # center the label in the remaining left portion.
            txt = font.render(display_label, True, (210, 230, 255))
            if action == 'toggle_trails':
                text_cx = rect.left + (rect.width - 18) // 2
                surface.blit(txt, (text_cx - txt.get_width() // 2,
                                    rect.centery - txt.get_height() // 2))
            else:
                surface.blit(txt, (rect.centerx - txt.get_width() // 2,
                                    rect.centery - txt.get_height() // 2))

            # Gear nub on the Trails button — opens the trails settings panel.
            # Active state (panel currently open) uses a brighter background.
            if action == 'toggle_trails':
                gear_rect = r._trails_gear_rect
                nub_col = (60, 80, 110, 220) if r._show_render_params else (30, 40, 60, 220)
                pygame.draw.rect(surface, nub_col, gear_rect, border_radius=3)
                pygame.draw.rect(surface, (100, 140, 200, 180), gear_rect, 1, border_radius=3)
                gear_glyph = font.render("⚙", True, (200, 220, 240))
                surface.blit(gear_glyph,
                             (gear_rect.centerx - gear_glyph.get_width() // 2,
                              gear_rect.centery - gear_glyph.get_height() // 2))

        # Stats button (top-right corner)
        stats_rect = r._stats_btn_rect
        stats_bg = (40, 55, 80, 200)
        pygame.draw.rect(surface, stats_bg, stats_rect, border_radius=4)
        pygame.draw.rect(surface, (100, 140, 200, 180), stats_rect, 1, border_radius=4)
        stats_lbl = "Stats ON" if r._show_stats else "Stats"
        txt = font.render(stats_lbl, True, (210, 230, 255))
        surface.blit(txt, (stats_rect.centerx - txt.get_width() // 2,
                            stats_rect.centery - txt.get_height() // 2))

    def _paint_stats_panel(self, surface: pygame.Surface, font, fps: float) -> None:
        """Stats panel (top-right under the Stats button): FPS, counts, sparklines, histogram."""
        r = self.r
        config = r.config
        # panel_h: see Renderer.STATS_PANEL_H class constant for the row breakdown.
        panel_h = r.STATS_PANEL_H
        panel_w = 215
        panel_x = config.window_width - panel_w - 8
        panel_y = r._stats_btn_rect.bottom + 4

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(surface, (15, 18, 35, 185), panel_rect, border_radius=6)
        pygame.draw.rect(surface, (70, 100, 150, 180), panel_rect, 1, border_radius=6)

        y_off = panel_y + 6

        # Static lines (no sparkline)
        for line in [
            f"FPS:        {fps:.1f}",
            f"Step:       {r._stats_step:,}",
            f"Sim time:   {r._stats_sim_time:.1f}",
            f"Particles:  {r._stats_alive:,}",
        ]:
            txt = font.render(line, True, (190, 215, 255))
            surface.blit(txt, (panel_x + 6, y_off))
            y_off += 16

        # Spark-stat rows: label on one line, sparkline below it
        spark_w = panel_w - 16
        spark_h = 14
        spark_x = panel_x + 8
        for label, spark_data, color in [
            (f"Free:       {r._stats_free:,}",     r._spark_free,   (80, 180, 255)),
            (f"Composites: {r._stats_n_comp:,}",   r._spark_comp,   (120, 220, 120)),
            (f"Unique:     {r._stats_n_unique:,}", r._spark_unique, (200, 140, 220)),
            (f"Energy:     {r._stats_energy:.0f}", r._spark_energy, (220, 180, 80)),
        ]:
            txt = font.render(label, True, (190, 215, 255))
            surface.blit(txt, (panel_x + 6, y_off))
            y_off += 15
            self._draw_sparkline(surface, spark_data, spark_x, y_off, spark_w, spark_h, color)
            y_off += 18

        # Spark-stat rows for fusions and decays
        y_off += 4
        for label, spark_data, color in [
            (f"Fusions: {r._fusion_total:,} ({r._fusion_rate:.1f}/s)", r._spark_fusion, (220, 170, 60)),
            (f"Decays:  {r._decay_total:,}  ({r._decay_rate:.1f}/s)",  r._spark_decay,  (80, 200, 220)),
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
        # X-axis auto-zooms to the largest live composite so the chart
        # always uses the full chart_w (rather than reserving space for a
        # max-size composite that may never form again after a reroll).
        # Bounded below by a 2-size floor (so empty/early states aren't
        # a single bar) and above by config.max_composite_size (the
        # absolute physical cap). When size_max is large, bins are
        # *widened* (bin_width > 1) so we never exceed MAX_BINS_HIST
        # bars — 1px bars + 1px gaps always fit inside chart_w.
        MAX_BINS_HIST = 100
        # hist[i] = count of composites with member_count == i+1, so the
        # largest live size = (highest non-zero index) + 1.
        nz = np.flatnonzero(r._stats_hist) if len(r._stats_hist) else np.array([], dtype=np.int32)
        largest_live = int(nz.max()) + 1 if len(nz) > 0 else 2
        target_size  = max(2, min(largest_live, config.max_composite_size))
        # Cached x-axis upper bound: grow immediately if a larger composite
        # appears (otherwise its bar would clip off-chart), but only
        # *shrink* once every HIST_AXIS_SHRINK_FRAMES frames so the axis
        # doesn't twitch every time fusion/fission rearranges the tail.
        HIST_AXIS_SHRINK_FRAMES = 100
        r._hist_axis_age += 1
        if target_size > r._hist_size_max_cached:
            r._hist_size_max_cached = target_size
            r._hist_axis_age = 0
        elif (target_size < r._hist_size_max_cached
              and r._hist_axis_age >= HIST_AXIS_SHRINK_FRAMES):
            r._hist_size_max_cached = target_size
            r._hist_axis_age = 0
        size_max  = r._hist_size_max_cached
        bin_width = max(1, -(-size_max // MAX_BINS_HIST))   # ceil(size_max / MAX_BINS_HIST)
        n_bins    = -(-size_max // bin_width)               # ceil(size_max / bin_width)
        bar_w     = max(1, (chart_w - n_bins) // max(1, n_bins))

        # Aggregate hist[i] (= count of composites with member_count == i+1)
        # into n_bins of bin_width consecutive sizes. Right-pad with zeros
        # so the reshape divides evenly without shifting counts.
        padded  = np.zeros(n_bins * bin_width, dtype=np.int64)
        src_len = min(len(r._stats_hist), padded.size)
        padded[:src_len] = r._stats_hist[:src_len]
        binned    = padded.reshape(n_bins, bin_width).sum(axis=1)
        max_count = max(1, int(binned.max()))

        for b in range(n_bins):
            count = int(binned[b])
            if count == 0:
                continue
            bh = max(1, int(chart_h * count / max_count))
            bx = chart_x + b * (bar_w + 1)
            pygame.draw.rect(surface, (60, 140, 220, 200),
                             pygame.Rect(bx, chart_y + chart_h - bh, bar_w, bh))

        pygame.draw.line(surface, (80, 110, 160),
                         (chart_x, chart_y + chart_h),
                         (chart_x + chart_w, chart_y + chart_h), 1)
        # Ticks label the upper-edge size of selected bins. Bin b spans
        # sizes [b*bin_width + 1, (b+1)*bin_width]; ticks are evenly
        # spaced along the size axis so the rightmost tick reads size_max.
        raw_ticks = np.linspace(bin_width, size_max, 5).astype(int)
        seen = set()
        for tick in raw_ticks:
            tick = int(tick)
            if tick in seen:
                continue
            seen.add(tick)
            b = (tick - 1) // bin_width
            if 0 <= b < n_bins:
                tx = chart_x + b * (bar_w + 1)
                lbl = font.render(str(tick), True, (120, 150, 190))
                surface.blit(lbl, (tx - lbl.get_width() // 2, chart_y + chart_h + 2))
        y_off = chart_y + chart_h + 20

        # Composite size metrics (if profiling enabled)
        if r.metrics is not None:
            max_comp_size = r.metrics.max_composite_size_observed
            num_samples = len(r.metrics.composite_size_samples)

            y_off += 4
            txt = font.render(f"Max composite: {max_comp_size} members", True, (160, 185, 230))
            surface.blit(txt, (panel_x + 6, y_off))
            y_off += 16

            if num_samples > 0:
                recent_samples = r.metrics.composite_size_samples[-10:]  # Last 10 samples
                recent_max_sizes = [s[1] for s in recent_samples]
                avg_recent = sum(recent_max_sizes) / len(recent_max_sizes)
                txt = font.render(f"Recent avg max: {avg_recent:.1f}", True, (160, 185, 230))
                surface.blit(txt, (panel_x + 6, y_off))
                y_off += 16

    def _paint_inspector(self, surface: pygame.Surface) -> None:
        """Top-right panel showing the selected particle's stats.

        Returns early if nothing is selected. Layout follows the mockup at
        notes/2026-05-14-particle-info-panel-mockup.html.
        """
        r = self.r
        snap = r._selected_snapshot
        if snap is None:
            return

        font = r._font

        BG           = (15, 18, 35, 235)
        BORDER       = (70, 100, 150, 220)
        DIVIDER      = (70, 100, 150, 110)
        LABEL_FG     = (160, 185, 230)
        VALUE_FG     = (220, 230, 255)
        MUTED_FG     = (120, 140, 165)
        CLOSE_BG     = (80, 30, 30, 220)
        CLOSE_BORDER = (150, 80, 80, 220)
        CLOSE_FG     = (255, 160, 160)

        comp = snap['composite']

        # Height: header(22) + species(20) + 8 kv rows(15 each) + bottom pad,
        # plus composite section if applicable.
        base_h = 22 + 20 + 8 * 15 + 8
        comp_h = 0
        if comp is not None:
            # divider(8) + section title(18) + 5 kv rows(15) + members
            # label(15) + chip rows(~16 each, estimate 1-2 rows) + pad
            est_chip_rows = max(1, (len(comp['members']) + 7) // 8)
            comp_h = 8 + 18 + 5 * 15 + 15 + est_chip_rows * 18 + 8
        panel_w = 235
        panel_h = base_h + comp_h + 6

        panel_x = r.config.window_width - panel_w - 8
        panel_y = r._stats_btn_rect.bottom + 6
        if r._show_stats:
            # Slide below the open Stats panel.
            panel_y += r.STATS_PANEL_H + 4

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
        r._inspector_close_rect = pygame.Rect(
            panel_x + panel_w - close_size - 8, y - 1, close_size, close_size
        )
        pygame.draw.rect(surface, CLOSE_BG, r._inspector_close_rect, border_radius=3)
        pygame.draw.rect(surface, CLOSE_BORDER, r._inspector_close_rect, 1, border_radius=3)
        close_lbl = font.render("×", True, CLOSE_FG)
        surface.blit(close_lbl,
                     (r._inspector_close_rect.centerx - close_lbl.get_width() // 2,
                      r._inspector_close_rect.centery - close_lbl.get_height() // 2 - 1))
        y += 18
        pygame.draw.line(surface, DIVIDER, (panel_x + 6, y),
                         (panel_x + panel_w - 6, y), 1)
        y += 4

        # Species line with color swatch
        sp = snap['species']
        col_rgb = self._species_display_rgb(sp)
        sw_rect = pygame.Rect(x_text, y + 3, 14, 14)
        pygame.draw.rect(surface, col_rgb, sw_rect, border_radius=2)
        pygame.draw.rect(surface, (255, 255, 255, 50), sw_rect, 1, border_radius=2)
        lbl = font.render(f"Species {sp}", True, VALUE_FG)
        surface.blit(lbl, (x_text + 18, y))
        if snap['valence'] is not None:
            v_txt = font.render(f"valence {snap['valence']}", True, MUTED_FG)
            surface.blit(v_txt, (panel_x + panel_w - v_txt.get_width() - 10, y))
        y += 20

        def kv(label: str, value: str, value_color=VALUE_FG):
            nonlocal y
            l = font.render(label, True, LABEL_FG)
            v = font.render(value, True, value_color)
            surface.blit(l, (x_text, y))
            surface.blit(v, (panel_x + panel_w - v.get_width() - 10, y))
            y += 15

        px, py = snap['pos']
        vx, vy = snap['vel']
        kv("Position",  f"{px:.1f}, {py:.1f}")
        kv("Velocity",  f"{vx:.2f}, {vy:.2f}")
        kv("Speed",     f"{snap['speed']:.2f}")
        kv("Mass",      f"{snap['mass']:.2f}")
        kv("Energy",    f"{snap['energy']:.2f}")
        kv("Age",       f"{snap['age']:.1f} s")
        kv("Composite", "free" if comp is None else f"#{comp['id']}",
           value_color=MUTED_FG if comp is None else VALUE_FG)

        if comp is not None:
            y += 4
            pygame.draw.line(surface, DIVIDER, (panel_x + 6, y),
                             (panel_x + panel_w - 6, y), 1)
            y += 4
            hdr = font.render(
                f"Composite #{comp['id']} — {comp['size']} members",
                True, VALUE_FG
            )
            surface.blit(hdr, (x_text, y))
            y += 18

            kv("Hash",      f"{comp['hash']:08x}"[:8])
            kv("Binding E", f"{comp['binding_energy']:.2f}")
            kv("Age",       f"{comp['age']:.1f} s")
            kv("Half-life", f"{comp['half_life']:.1f} s")
            if comp['free_bonds'] is not None:
                total_v = sum(int(r._species_valence[s]) for s in comp['members'])
                kv("Free bonds", f"{comp['free_bonds']} / {total_v}")

            # Members chips — wrap onto multiple rows if needed.
            ml = font.render("Members", True, LABEL_FG)
            surface.blit(ml, (x_text, y))
            y += 16
            chip_x = x_text
            chip_h = 16
            for s in comp['members']:
                crgb = self._species_display_rgb(s)
                txt = font.render(str(int(s)), True, VALUE_FG)
                cw = txt.get_width() + 16
                if chip_x + cw > panel_x + panel_w - 10:
                    chip_x = x_text
                    y += chip_h + 2
                chip_rect = pygame.Rect(chip_x, y, cw, chip_h)
                pygame.draw.rect(surface, (70, 100, 150, 46), chip_rect, border_radius=3)
                pygame.draw.circle(surface, crgb,
                                   (chip_rect.left + 6, chip_rect.centery), 4)
                surface.blit(txt, (chip_rect.left + 12,
                                   chip_rect.centery - txt.get_height() // 2))
                chip_x += cw + 4

    # ── Entry point ───────────────────────────────────────────────────────────

    def paint(self, fps: float) -> None:
        """Draw the whole HUD onto self.r._hud_surface in phase order."""
        r = self.r
        surface = r._hud_surface
        surface.fill((0, 0, 0, 0))  # clear to transparent
        font    = r._font

        self._paint_buttons(surface, font)

        if r._show_stats:
            self._paint_stats_panel(surface, font, fps)

        # ── Params panels ─────────────────────────────────────────────────────
        # Physics params and render-params (trails) share the same panel
        # geometry — same X column, same width, same reset-button styling —
        # so they go through one helper. Mutual exclusion (only one open at
        # a time) is enforced by the toggle handlers; this just draws.
        if r._show_params:
            self._draw_slider_panel(surface, font, r._sliders,
                                    r._slider_content_h,
                                    r._params_reset_rect, "Reset Params")
        if r._show_render_params:
            self._draw_slider_panel(surface, font, r._render_sliders,
                                    r._render_slider_content_h,
                                    r._render_params_reset_rect, "Reset Trails")

        # Inspector panel (live stats for the selected particle)
        self._paint_inspector(surface)

        # Bottom key hint
        hint = "[Space] pause  [+/-] speed  [B] viz  [M] bond mode  [R] reset  [Q] quit"
        hint_surf = font.render(hint, True, (120, 140, 160))
        surface.blit(hint_surf,
                     (r.config.window_width // 2 - hint_surf.get_width() // 2,
                      r.config.window_height - hint_surf.get_height() - 4))

        # Bond-mode status — small badge above the hint so the user can see
        # which kernel is live (edges / star_spring / off) at a glance.
        mode_label = {
            "edges":       "bond: edges (covalent)",
            "star_spring": "bond: star spring (legacy)",
            "off":         "bond: off",
        }.get(r.config.bond_mode, f"bond: {r.config.bond_mode}")
        mode_surf = font.render(mode_label, True, (140, 170, 200))
        surface.blit(mode_surf,
                     (r.config.window_width // 2 - mode_surf.get_width() // 2,
                      r.config.window_height - hint_surf.get_height() - mode_surf.get_height() - 6))
