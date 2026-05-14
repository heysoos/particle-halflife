"""Reusable HUD widgets — currently just the horizontal Slider used by the
physics-params and render-settings panels. Lives outside renderer.py so the
widget code can grow (and be unit-tested if we ever want to) without bloating
the GL pipeline file.
"""

import numpy as np
import pygame


class Slider:
    """Horizontal slider widget — log-scale multiplier (0.1×–10×) or optional linear range."""

    EXPO_MIN, EXPO_MAX = -1.0, 1.0   # 0.1× to 10× the default value

    def __init__(self, label: str, field: str, default_value: float,
                 track_rect: pygame.Rect, fmt: str = "{:.3f}",
                 linear_range=None, target_dict: dict = None):
        self._label = label
        self._field = field
        self._default = float(default_value)
        self._track_rect = track_rect
        self._fmt = fmt
        self._linear_range = linear_range
        # When set, this slider writes into target_dict[field] instead of the
        # renderer's generic _physics_updates dict. Used to keep render-only
        # sliders out of the PhysicsParams update pipeline.
        self._target_dict = target_dict
        self._reset_rect = pygame.Rect(track_rect.right + 4, track_rect.centery - 7, 14, 14)
        # Default exponent: the exponent value that makes self.value == self._default.
        # Cached so reset() and the "is at default" check in draw() can be one-liners.
        self._default_exponent = self._exponent_for_default()
        self._exponent = self._default_exponent

    def _exponent_for_default(self) -> float:
        """Compute the exponent that would produce self._default as the slider value."""
        if self._linear_range is not None:
            lo, hi = self._linear_range
            t = (self._default - lo) / max(hi - lo, 1e-8)
            return self.EXPO_MIN + float(np.clip(t, 0.0, 1.0)) * (self.EXPO_MAX - self.EXPO_MIN)
        return 0.0  # 1× = default on the log scale

    @property
    def field(self) -> str:
        return self._field

    @property
    def target_dict(self) -> dict:
        return self._target_dict

    @property
    def value(self) -> float:
        if self._linear_range is not None:
            lo, hi = self._linear_range
            t = (self._exponent - self.EXPO_MIN) / (self.EXPO_MAX - self.EXPO_MIN)
            return lo + float(np.clip(t, 0.0, 1.0)) * (hi - lo)
        return self._default * (10.0 ** self._exponent)

    def reset(self) -> None:
        self._exponent = self._default_exponent

    def commit(self, fallback_dict: dict) -> None:
        """Push current value into target_dict (if set) or fallback_dict.

        Centralizes the "render-setting vs physics-update" routing that used
        to be duplicated at every callsite in renderer.py.
        """
        target = self._target_dict if self._target_dict is not None else fallback_dict
        target[self._field] = self.value

    def hit_reset(self, pos) -> bool:
        return self._reset_rect.collidepoint(pos)

    def _handle_x(self) -> int:
        t = (self._exponent - self.EXPO_MIN) / (self.EXPO_MAX - self.EXPO_MIN)
        return int(self._track_rect.left + t * self._track_rect.width)

    def draw(self, surface: pygame.Surface, font) -> None:
        r = self._track_rect
        if self._linear_range is not None:
            label_str = f"{self._label}: {self._fmt.format(self.value)}"
        else:
            mult = 10.0 ** self._exponent
            label_str = f"{self._label}: {self._fmt.format(self.value)} ({mult:.2f}×)"
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
        # Per-slider reset button (↺)
        is_default = abs(self._exponent - self._default_exponent) < 0.01
        bg_col = (40, 25, 25, 200) if not is_default else (25, 30, 40, 200)
        pygame.draw.rect(surface, bg_col, self._reset_rect, border_radius=3)
        pygame.draw.rect(surface, (120, 80, 80, 180), self._reset_rect, 1, border_radius=3)
        lbl = font.render("↺", True, (200, 140, 140) if not is_default else (80, 90, 110))
        surface.blit(lbl, (self._reset_rect.centerx - lbl.get_width() // 2,
                            self._reset_rect.centery - lbl.get_height() // 2))

    def hit_handle(self, pos) -> bool:
        # Whole track is draggable, not just the knob — clicking anywhere on
        # the bar grabs the slider and snaps the knob to that x.
        r = self._track_rect
        return (r.left <= pos[0] <= r.right and
                abs(pos[1] - r.centery) <= 10)

    def handle_drag(self, pos) -> float:
        r = self._track_rect
        t = (pos[0] - r.left) / max(r.width, 1)
        self._exponent = self.EXPO_MIN + float(max(0.0, min(1.0, t))) * (self.EXPO_MAX - self.EXPO_MIN)
        return self.value
