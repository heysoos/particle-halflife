"""2D pan/zoom camera for the renderer.

Owns view_center (world point at screen center) and view_scale (1.0 = full
world visible at the default world/window ratio; > 1 zooms in). Knows how
to:
  - convert a screen-pixel position into a world-coord (screen_to_world),
  - apply a zoom factor around a screen point while pinning that point's
    underlying world-coord (zoom_at),
  - pan by a screen-pixel delta (pan_by),
  - reset to the default view (reset),
  - push its current state onto a list of moderngl programs that read the
    u_view_center / u_view_scale uniforms (push_uniforms).

Lives in its own module so the renderer can grow other UI / world-space
features (gizmos, selection brushes) without bloating renderer.py.
"""

import numpy as np

from halflife.config import SimConfig


class Camera:
    """Pan + zoom state, shared by every world-space draw call.

    Coordinate convention matches the world-space vertex shaders:
        view = (pos - center) * scale + world_size * 0.5
        ndc  = view / world_size * 2 - 1
    so view_scale = 1 and view_center = world midpoint reproduces the
    no-camera identity transform exactly.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        # World point currently at the screen center.
        self.view_center = [config.world_width * 0.5, config.world_height * 0.5]
        self.view_scale  = 1.0
        # Clamp range — 0.25× lets the user pull back to see > 1 world worth
        # of tiling on each axis; 40× is roughly the point where a single
        # particle fills the screen at default sprite sizes.
        self.view_scale_min = 0.25
        self.view_scale_max = 40.0

    def reset(self) -> None:
        """Reset pan and zoom to defaults (world midpoint, scale 1.0)."""
        config = self.config
        self.view_center = [config.world_width * 0.5, config.world_height * 0.5]
        self.view_scale  = 1.0

    def screen_to_world(self, sx: int, sy: int) -> tuple:
        """Convert window-pixel coords → world coords using the current camera.

        Inverse of the vertex-shader math:
            view = (pos - center) * scale + world_size * 0.5
            ndc  = view / world_size * 2 - 1
        Solving for pos:
            pos = ndc * world_size / 2 / scale + center
        """
        config = self.config
        ndc_x = (sx / config.window_width)  * 2.0 - 1.0
        ndc_y = 1.0 - (sy / config.window_height) * 2.0   # pygame Y is top-down
        wx = ndc_x * config.world_width  * 0.5 / self.view_scale + self.view_center[0]
        wy = ndc_y * config.world_height * 0.5 / self.view_scale + self.view_center[1]
        return wx, wy

    def zoom_at(self, sx: int, sy: int, factor: float) -> None:
        """Zoom by `factor` while keeping the world point under (sx, sy) fixed.

        Pin the world-coord that lives under the cursor, change scale, then
        recompute view_center so that same world-coord still maps to the same
        screen coord. Result: cursor sits on the same molecule before and
        after the zoom step.
        """
        wx_before, wy_before = self.screen_to_world(sx, sy)
        new_scale = float(np.clip(self.view_scale * factor,
                                   self.view_scale_min, self.view_scale_max))
        if abs(new_scale - self.view_scale) < 1e-6:
            return
        self.view_scale = new_scale
        wx_after, wy_after = self.screen_to_world(sx, sy)
        self.view_center[0] += wx_before - wx_after
        self.view_center[1] += wy_before - wy_after

    def pan_by(self, dx_pixels: int, dy_pixels: int) -> None:
        """Translate the view by (dx, dy) screen pixels.

        Converts pixel deltas → world deltas via the current zoom and shifts
        view_center in the opposite direction — dragging right scrolls world
        content right, so view_center must move left.
        """
        config = self.config
        world_dx = -dx_pixels * (config.world_width  / config.window_width)  / self.view_scale
        world_dy =  dy_pixels * (config.world_height / config.window_height) / self.view_scale
        # y inverted because pygame Y goes down while world Y goes up
        self.view_center[0] += world_dx
        self.view_center[1] += world_dy

    def push_uniforms(self, programs) -> None:
        """Push current view_center / view_scale onto every program in `programs`.

        Cheap; saves branching on whether the camera moved this frame. The
        caller (Renderer) keeps the program list built once in __init__.
        """
        vc = (float(self.view_center[0]), float(self.view_center[1]))
        vs = float(self.view_scale)
        for prog in programs:
            prog['u_view_center'].value = vc
            prog['u_view_scale'].value  = vs
