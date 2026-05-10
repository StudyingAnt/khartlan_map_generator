"""Khartlan world simulator — refactored from the original notebook.

Physics is unchanged from the original:

    1. Plates grow on a triple-grid sphere via random-walk seeding from
       per-plate centers. Stalled cells are mode-filled afterwards.
    2. Initial terrain forms from many transient volcanoes whose ejecta
       diffuses outward each timestep.
    3. Plates then drift along per-plate direction sequences. Overlap
       compresses, gaps are mode-filled, hotspot volcanoes and weathering
       act each step.

The "triple grid" is a cubic projection of a sphere onto three rectangular
panels: an equatorial band (``mid``) plus two square polar caps (``north``
and ``south``). Each panel carries a 1-cell ghost ring whose values are
mirrored from the matching cells of neighboring panels by
:meth:`World.sync_borders`.

Dimension convention: ``n_eq = 4 * n_pol`` and ``n_pol = 2 * k`` so that
the four edges of each polar cap map cleanly onto one quadrant of mid's
top or bottom edge.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# Direction primitives
# ---------------------------------------------------------------------------

# Ordered so that consecutive entries are 45 degrees CCW apart. The plate
# random walk in `select_directions` relies on this ordering.
DIRECTIONS_8: list[tuple[int, int]] = [
    (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
    ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
]


def _flip_v(d):     return (-d[0],  d[1])
def _rot_ccw_90(d): return (-d[1],  d[0])
def _rot_cw_90(d):  return ( d[1], -d[0])


# ---------------------------------------------------------------------------
# Border maps: dict[(i, j) on source panel] -> (i, j) on target panel
# ---------------------------------------------------------------------------

def _map_mid2north(k: int) -> dict[tuple[int, int], tuple[int, int]]:
    n_pol = 2 * k
    n_eq = 4 * n_pol
    src, dst = [], []

    src.append((0, 0));         dst.append((1, k + 1))
    for j in range(1, k + 1):
        src.append((0, j));     dst.append((1, k + 1 - j))
    for j in range(k + 1, k + 1 + n_pol):
        src.append((0, j));     dst.append((j - k + 1, 1))
    for j in range(k + 1 + n_pol, k + 1 + 2 * n_pol):
        src.append((0, j));     dst.append((n_pol + 1, j - k - n_pol + 1))
    for j in range(k + 1 + 2 * n_pol, k + 1 + 3 * n_pol):
        src.append((0, j));     dst.append(((k + 1) + 3 * n_pol - j, n_pol + 1))
    for j in range(k + 1 + 3 * n_pol, n_eq + 1):
        src.append((0, j));     dst.append((1, (k + 1) + 4 * n_pol - j))
    src.append((0, n_eq + 1));  dst.append((1, k))
    return dict(zip(src, dst))


def _map_north2mid(k: int) -> dict[tuple[int, int], tuple[int, int]]:
    n_pol = 2 * k
    src, dst = [], []

    src.append((0, 0));               dst.append((1, k + 1))
    for j in range(1, k + 1):
        src.append((0, j));           dst.append((1, k + 1 - j))
    for j in range(k + 1, n_pol + 1):
        src.append((0, j));           dst.append((1, (k + 1) + 4 * n_pol - j))
    src.append((0, n_pol + 2));       dst.append((1, (k + 1) + 3 * n_pol))
    for i in range(1, n_pol + 1):
        src.append((i, n_pol + 2));   dst.append((1, (k + 1) + 3 * n_pol - i))
    src.append((n_pol + 2, n_pol + 2)); dst.append((1, (k + 1) + 2 * n_pol))
    for j in range(2, n_pol + 2):
        src.append((n_pol + 2, j));   dst.append((1, (k + 1) + n_pol + j - 2))
    src.append((n_pol + 2, 0));       dst.append((1, (k + 1) + n_pol))
    for i in range(2, n_pol + 2):
        src.append((i, 0));           dst.append((1, k + 1 + i - 2))
    return dict(zip(src, dst))


def _map_mid2south(k: int) -> dict[tuple[int, int], tuple[int, int]]:
    n_pol = 2 * k
    n_eq = 4 * n_pol
    src, dst = [], []

    src.append((n_eq + 1, 0));         dst.append((n_pol + 1, k + 1))
    for j in range(1, k + 1):
        src.append((n_eq + 1, j));     dst.append((n_pol + 1, k + 1 - j))
    for j in range(k + 1, k + 1 + n_pol):
        src.append((n_eq + 1, j));     dst.append(((k + 1) + n_pol - j, 1))
    for j in range(k + 1 + n_pol, k + 1 + 2 * n_pol):
        src.append((n_eq + 1, j));     dst.append((1, j - k - n_pol + 1))
    for j in range(k + 1 + 2 * n_pol, k + 1 + 3 * n_pol):
        src.append((n_eq + 1, j));     dst.append((j - k - 2 * n_pol + 1, n_pol + 1))
    for j in range(k + 1 + 3 * n_pol, n_eq + 1):
        src.append((n_eq + 1, j));     dst.append((n_pol + 1, (k + 1) + 4 * n_pol - j))
    src.append((n_eq + 1, n_eq + 1));  dst.append((n_pol + 1, k))
    return dict(zip(src, dst))


def _map_south2mid(k: int) -> dict[tuple[int, int], tuple[int, int]]:
    n_pol = 2 * k
    n_eq = 4 * n_pol
    src, dst = [], []

    src.append((0, 0));                 dst.append((n_eq, (k + 1) + n_pol))
    for j in range(2, n_pol + 2):
        src.append((0, j));             dst.append((n_eq, (k + 1) + n_pol + j - 2))
    src.append((0, n_pol + 2));         dst.append((n_eq, (k + 1) + 2 * n_pol))
    for i in range(2, n_pol + 2):
        src.append((i, n_pol + 2));     dst.append((n_eq, (k + 1) + 2 * n_pol + i - 2))
    src.append((n_pol + 2, n_pol + 2)); dst.append((n_eq, 1))
    for j in range(k + 1, n_pol + 1):
        src.append((n_pol + 2, j));     dst.append((n_eq, (k + 1) + 4 * n_pol - j))
    for j in range(1, k + 1):
        src.append((n_pol + 2, j));     dst.append((n_eq, k + 1 - j))
    src.append((n_pol + 2, 0));         dst.append((n_eq, k + 1))
    for i in range(1, n_pol + 1):
        src.append((i, 0));             dst.append((n_eq, (k + 1) + n_pol - i))
    return dict(zip(src, dst))


def _dict_to_arrays(d: dict[tuple[int, int], tuple[int, int]]):
    keys = np.array(list(d.keys()), dtype=int)
    vals = np.array(list(d.values()), dtype=int)
    return keys, vals


def _padded_array(shape: tuple[int, int], *, fill: int, inner: int) -> np.ndarray:
    arr = np.full(shape, fill, dtype=int)
    arr[1:-1, 1:-1] = inner
    return arr


# ---------------------------------------------------------------------------
# World topology
# ---------------------------------------------------------------------------

class World:
    """Cubic projection of a sphere onto three rectangular panels.

    Attributes
    ----------
    k : int
        Half-width of a polar cap edge in the original parameterization.
    n_pol, n_eq : int
        Polar cap edge length (``2 k``) and equatorial circumference
        (``4 n_pol``).
    shape_mid, shape_pol : tuple[int, int]
        Allocated array shapes including the 1-cell ghost ring.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.n_pol = 2 * k
        self.n_eq = 4 * self.n_pol

        self.mid2north = _map_mid2north(k)
        self.north2mid = _map_north2mid(k)
        self.mid2south = _map_mid2south(k)
        self.south2mid = _map_south2mid(k)

        # Vectorized index views for fast border syncing.
        self._mn_keys, self._mn_vals = _dict_to_arrays(self.mid2north)
        self._nm_keys, self._nm_vals = _dict_to_arrays(self.north2mid)
        self._ms_keys, self._ms_vals = _dict_to_arrays(self.mid2south)
        self._sm_keys, self._sm_vals = _dict_to_arrays(self.south2mid)

    # ---- shapes / factories ------------------------------------------------

    @property
    def shape_mid(self) -> tuple[int, int]:
        return (self.n_eq + 2, self.n_eq + 2)

    @property
    def shape_pol(self) -> tuple[int, int]:
        return (self.n_pol + 3, self.n_pol + 3)

    def empty_plates(self):
        """Three plate-label arrays: -1 in the ghost ring, 0 in the interior."""
        return (
            _padded_array(self.shape_mid, fill=-1, inner=0),
            _padded_array(self.shape_pol, fill=-1, inner=0),
            _padded_array(self.shape_pol, fill=-1, inner=0),
        )

    def random_terrain(self, rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng()
        return (
            rng.random(self.shape_mid),
            rng.random(self.shape_pol),
            rng.random(self.shape_pol),
        )

    def zeros_terrain(self):
        return (
            np.zeros(self.shape_mid),
            np.zeros(self.shape_pol),
            np.zeros(self.shape_pol),
        )

    # ---- panel iteration ---------------------------------------------------

    def panels(self, mid: np.ndarray, north: np.ndarray, south: np.ndarray):
        """Yield ``(name, array, inner_slice)`` triples for the three panels."""
        inner_eq  = (slice(1, self.n_eq + 1),  slice(1, self.n_eq + 1))
        inner_pol = (slice(1, self.n_pol + 2), slice(1, self.n_pol + 2))
        return [
            ("mid",   mid,   inner_eq),
            ("north", north, inner_pol),
            ("south", south, inner_pol),
        ]

    # ---- border syncing ----------------------------------------------------

    def sync_borders(self, mid: np.ndarray, north: np.ndarray, south: np.ndarray) -> None:
        """Copy neighbor cells into each panel's ghost ring (in-place)."""
        n_eq = self.n_eq

        # Equatorial wrap-around (left <-> right).
        mid[:, 0]        = mid[:, n_eq]
        mid[:, n_eq + 1] = mid[:, 1]

        mid[self._mn_keys[:, 0], self._mn_keys[:, 1]]   = north[self._mn_vals[:, 0], self._mn_vals[:, 1]]
        north[self._nm_keys[:, 0], self._nm_keys[:, 1]] = mid[self._nm_vals[:, 0], self._nm_vals[:, 1]]
        mid[self._ms_keys[:, 0], self._ms_keys[:, 1]]   = south[self._ms_vals[:, 0], self._ms_vals[:, 1]]
        south[self._sm_keys[:, 0], self._sm_keys[:, 1]] = mid[self._sm_vals[:, 0], self._sm_vals[:, 1]]

        self._fix_pole_corners(north, south)

    def _fix_pole_corners(self, north: np.ndarray, south: np.ndarray) -> None:
        n = self.n_pol
        north[0,     2  ] = (north[1,     1  ] + north[1,     2  ]) / 2
        north[0,     n+1] = (north[1,     n+1] + north[2,     n+1]) / 2
        north[n + 1, n+2] = (north[n,     n+1] + north[n + 1, n+1]) / 2
        north[n + 2, 1  ] = (north[n + 1, 1  ] + north[n + 1, 2  ]) / 2

        south[0,     1  ] = (south[1,     1  ] + south[2,     1  ]) / 2
        south[1,     n+2] = (south[1,     n  ] + south[2,     n+1]) / 2
        south[n + 2, n+1] = (south[n + 1, n  ] + south[n + 1, n+1]) / 2
        south[n + 1, 0  ] = (south[n,     1  ] + south[n + 1, 1  ]) / 2

    # ---- diffusion ---------------------------------------------------------

    def diffuse(self, mid: np.ndarray, north: np.ndarray, south: np.ndarray, coef: float) -> None:
        """3x3 Laplacian diffusion over the inner region of each panel.

        Equivalent to ``new = val + coef * sum_{neighbors}(neighbor - val)``
        applied with a 3x3 box stencil. Borders are re-synced after.
        """
        for _, arr, inner in self.panels(mid, north, south):
            _diffuse_panel(arr, inner, coef)
        self.sync_borders(mid, north, south)


def _diffuse_panel(arr: np.ndarray, inner: tuple[slice, slice], coef: float) -> None:
    kernel = np.ones((3, 3), dtype=arr.dtype)
    box_sum = convolve(arr, kernel, mode="constant", cval=0.0)
    arr[inner] = arr[inner] + coef * (box_sum[inner] - 9.0 * arr[inner])


# ---------------------------------------------------------------------------
# Volcanos
# ---------------------------------------------------------------------------

@dataclass
class Volcano:
    """A single volcano with a fixed location and time-varying strength."""
    vid: int
    loc: int                  # 0 = mid, 1 = north, 2 = south
    coord: tuple[int, int]
    duration: np.ndarray      # shape (t_max,), 0/1 mask
    strength: np.ndarray      # shape (t_max,), float

    def is_active(self, t: int) -> bool:
        return bool(self.duration[t])


def _random_t_span(t_max: int, min_diff: int, max_diff: int,
                    rng: np.random.Generator) -> np.ndarray:
    """0/1 mask of length t_max with a single contiguous active window."""
    t1 = int(rng.integers(0, max_diff))
    if t1 + min_diff < t_max:
        t2 = int(rng.integers(t1 + min_diff, min(t_max, t1 + min_diff + 1)))
    else:
        t1 = t_max - min_diff - 1
        t2 = t_max - 1
    span = np.zeros(t_max, dtype=int)
    span[t1:t2 + 1] = 1
    return span


def _volcano_strength_curve(duration: np.ndarray, m_mean: float, m_std: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Linear ramp inside the active window, zero outside."""
    indices = np.where(duration == 1)[0]
    if len(indices) == 0:
        raise ValueError("duration mask must contain at least one active step")

    M = float(np.clip(rng.normal(loc=m_mean, scale=m_std), 0.01, None))
    t1, t2 = indices[0], indices[-1]
    if t1 == t2:
        strength = np.zeros_like(duration, dtype=float)
        strength[t1] = M
        return strength

    a = 0.5 * M / (t1 - t2)
    b = M - 0.5 * M * t1 / (t1 - t2)
    y = a * np.arange(t1, t2 + 1) + b

    strength = np.zeros_like(duration, dtype=float)
    strength[t1:t2 + 1] = y
    return strength


def random_volcanos(
    world: World,
    n: int,
    t_max: int,
    *,
    m_mean: float,
    m_std: float,
    min_t_diff: int,
    max_t_diff: int,
    rng: np.random.Generator | None = None,
) -> list[Volcano]:
    """Sample ``n`` volcanos uniformly distributed by panel area."""
    rng = rng if rng is not None else np.random.default_rng()
    weights = np.array([world.n_eq ** 2,
                        (world.n_pol + 1) ** 2,
                        (world.n_pol + 1) ** 2], dtype=float)
    weights /= weights.sum()

    volcanos: list[Volcano] = []
    for vid in range(n):
        loc = int(rng.choice([0, 1, 2], p=weights))
        if loc == 0:
            coord = tuple(int(x) for x in rng.integers(1, world.n_eq, size=2))
        else:
            coord = tuple(int(x) for x in rng.integers(1, world.n_pol + 1, size=2))
        duration = _random_t_span(t_max, min_t_diff, max_t_diff, rng)
        strength = _volcano_strength_curve(duration, m_mean, m_std, rng)
        volcanos.append(Volcano(vid=vid, loc=loc, coord=coord,
                                duration=duration, strength=strength))
    return volcanos


def apply_volcanos(volcanos: Iterable[Volcano], t: int,
                   mid: np.ndarray, north: np.ndarray, south: np.ndarray) -> None:
    """Add each active volcano's strength at its coordinate (in-place)."""
    targets = (mid, north, south)
    for v in volcanos:
        if v.is_active(t):
            targets[v.loc][v.coord] += v.strength[t]


# ---------------------------------------------------------------------------
# Plate dynamics
# ---------------------------------------------------------------------------

def select_directions(n: int, rng: np.random.Generator | None = None) -> list[tuple[int, int]]:
    """Random walk over DIRECTIONS_8 with momentum: 0.7 stay, 0.15/0.15 +/-45 deg."""
    rng = rng if rng is not None else np.random.default_rng()
    idx = int(rng.integers(0, len(DIRECTIONS_8)))
    out = [DIRECTIONS_8[idx]]
    probs = [0.7, 0.15, 0.15]
    for _ in range(n - 1):
        candidates = [idx, (idx - 1) % len(DIRECTIONS_8), (idx + 1) % len(DIRECTIONS_8)]
        idx = int(rng.choice(candidates, p=probs))
        out.append(DIRECTIONS_8[idx])
    return out


# --- direction fix-up inside polar caps ------------------------------------
#
# When a plate sitting on a polar cap tries to move in a direction that
# would leave the cap "wrong", the direction is rotated to the orientation
# of the quadrant the plate currently occupies most.

def _divide_polar_quadrants(n: int):
    """Return the four triangular quadrants of an (n x n) polar interior."""
    middle = n // 2
    upper, lower, left, right = [], [], [], []
    for i in range(middle):
        for j in range(i, n - i):
            upper.append((i + 1, j + 1))
    for i in range(middle + 1, n):
        for j in range(n - i - 1, i + 1):
            lower.append((i + 1, j + 1))
    for j in range(middle):
        for i in range(j, n - j):
            left.append((i + 1, j + 1))
    for j in range(middle + 1, n):
        for i in range(n - j - 1, j + 1):
            right.append((i + 1, j + 1))
    return upper, lower, left, right


def _argmax_with_random_tiebreak(arr: np.ndarray, rng: np.random.Generator) -> int:
    max_val = np.max(arr)
    candidates = np.where(arr == max_val)[0]
    return int(rng.choice(candidates))


def _redirect_in_polar(plate_idx: int, direction: tuple[int, int],
                        plates: np.ndarray, *, is_north: bool,
                        rng: np.random.Generator) -> tuple[int, int]:
    n_pol = plates.shape[0] - 3
    upper, lower, left, right = _divide_polar_quadrants(n_pol + 1)
    counts = []
    for tri in (upper, lower, left, right):
        rows, cols = zip(*tri)
        counts.append(int(np.count_nonzero(plates[rows, cols] == plate_idx + 1)))
    counts = np.asarray(counts)
    chosen = _argmax_with_random_tiebreak(counts, rng)

    if is_north:
        # north: upper flips, lower stays, left CW, right CCW
        if chosen == 0: return _flip_v(direction)
        if chosen == 1: return direction
        if chosen == 2: return _rot_cw_90(direction)
        return _rot_ccw_90(direction)
    else:
        # south: upper stays, lower flips, left CCW, right CW
        if chosen == 0: return direction
        if chosen == 1: return _flip_v(direction)
        if chosen == 2: return _rot_ccw_90(direction)
        return _rot_cw_90(direction)


# --- per-plate movement -----------------------------------------------------

def _step_one_cell(world: World, x: int, y: int, dx: int, dy: int, loc: int,
                    plates_for_redirect: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                    plate_idx: int | None,
                    rng: np.random.Generator) -> tuple[int, int, int]:
    """Move one cell ``(x, y)`` on panel ``loc`` by ``(dx, dy)``.

    Returns ``(new_x, new_y, new_loc)``. Cross-panel transitions are handled
    via the world's border maps. Plates that would leave a polar cap from
    a corner stay put (matching original behavior). When a plate moves
    *within* a polar cap the direction may be re-fixed via
    :func:`_redirect_in_polar` once.
    """
    n_eq = world.n_eq
    n_pol = world.n_pol
    new_x, new_y = x + dx, y + dy

    if loc == 0:
        if 1 <= new_x <= n_eq and new_y == 0:
            return new_x, n_eq, 0
        if 1 <= new_x <= n_eq and new_y == n_eq + 1:
            return new_x, 1, 0
        if new_x == 0:
            tx, ty = world.mid2north[(new_x, new_y)]
            return tx, ty, 1
        if new_x == n_eq + 1:
            tx, ty = world.mid2south[(new_x, new_y)]
            return tx, ty, 2
        return new_x, new_y, 0

    # Polar caps. is_north decides which redirect / corner conventions apply.
    is_north = (loc == 1)
    if is_north:
        corners = {(1, 0), (0, n_pol + 1), (n_pol + 1, n_pol + 2), (n_pol + 2, 1)}
        bridge = world.north2mid
    else:
        corners = {(0, 1), (1, n_pol + 2), (n_pol + 2, n_pol + 1), (n_pol + 1, 0)}
        bridge = world.south2mid

    if (new_x, new_y) in corners:
        return x, y, loc
    if new_x in (0, n_pol + 2) or new_y in (0, n_pol + 2):
        tx, ty = bridge[(new_x, new_y)]
        return tx, ty, 0

    # In-bounds inside the polar cap: redirect once and try again.
    if plates_for_redirect is None or plate_idx is None:
        return new_x, new_y, loc
    plates_panel = plates_for_redirect[loc]
    new_dx, new_dy = _redirect_in_polar(plate_idx, (dx, dy), plates_panel,
                                         is_north=is_north, rng=rng)
    new_x, new_y = x + new_dx, y + new_dy
    if (new_x, new_y) in corners:
        return x, y, loc
    if new_x in (0, n_pol + 2) or new_y in (0, n_pol + 2):
        tx, ty = bridge[(new_x, new_y)]
        return tx, ty, 0
    return new_x, new_y, loc


def move_plate_coordinates(
    world: World,
    plate_idx: int,
    direction: tuple[int, int],
    loc: int,
    plates_panel: np.ndarray,
    plates_all: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]]]:
    """Compute origin / destination / target-panel for every cell of one plate."""
    coords = np.argwhere(plates_panel == plate_idx + 1)
    original = [tuple(c) for c in coords]
    new_coords: list[tuple[int, int]] = []
    locs: list[int] = []
    dx, dy = direction
    for x, y in original:
        nx, ny, nloc = _step_one_cell(
            world, x, y, dx, dy, loc,
            plates_for_redirect=plates_all, plate_idx=plate_idx, rng=rng,
        )
        new_coords.append((nx, ny))
        locs.append(nloc)
    return original, locs, new_coords


# --- plate growth -----------------------------------------------------------

def _has_unfilled(plates_trio, world: World) -> bool:
    mid, north, south = plates_trio
    return (np.any(mid[1:world.n_eq + 1, 1:world.n_eq + 1] == 0)
            or np.any(north[1:world.n_pol + 2, 1:world.n_pol + 2] == 0)
            or np.any(south[1:world.n_pol + 2, 1:world.n_pol + 2] == 0))


def grow_plates(
    world: World,
    n_plates: int,
    *,
    delay_max: int = 10,
    rng: np.random.Generator | None = None,
    on_step: Callable[[int, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-walk plate growth from random mid-panel seed centers.

    Each plate grows by examining all 8-neighbors of its current cells and
    claiming a random empty one. Per-plate ``delays`` stagger the start of
    growth so plates end up at different sizes.
    """
    rng = rng if rng is not None else np.random.default_rng()
    mid_p, north_p, south_p = world.empty_plates()
    n_eq = world.n_eq

    # Seed centers in the mid panel.
    centers = rng.integers(1, n_eq, size=(n_plates, 2))
    for i, c in enumerate(centers):
        mid_p[tuple(c)] = i + 1
    delays = rng.integers(1, delay_max, size=(n_plates,))

    t = 0
    while _has_unfilled((mid_p, north_p, south_p), world):
        for panel_loc, panel in ((0, mid_p), (1, north_p), (2, south_p)):
            for i, j in np.argwhere(panel > 0):
                plate_idx_1 = int(panel[i, j])
                if t < delays[plate_idx_1 - 1]:
                    continue
                possible: list[tuple[int, int, int]] = []
                for dx, dy in DIRECTIONS_8:
                    nx, ny, nloc = _step_one_cell(
                        world, i, j, dx, dy, panel_loc,
                        plates_for_redirect=None, plate_idx=None, rng=rng,
                    )
                    target = (mid_p, north_p, south_p)[nloc]
                    if target[nx, ny] == 0:
                        possible.append((nloc, nx, ny))
                if not possible:
                    continue
                pick = possible[rng.integers(0, len(possible))]
                target = (mid_p, north_p, south_p)[pick[0]]
                target[pick[1], pick[2]] = plate_idx_1
        t += 1
        if on_step is not None:
            on_step(t, mid_p, north_p, south_p)

    return mid_p, north_p, south_p


# --- plate label cleanup ----------------------------------------------------

def _smooth_panel_mode(plates: np.ndarray, lo: int, hi: int, *,
                       fill_zeros_only: bool,
                       terrain: np.ndarray | None = None) -> bool:
    """One pass of 3x3 mode-fill. Returns True if any change occurred.

    If ``terrain`` is provided, when a 0-cell takes on a new label its terrain
    is set to half the average terrain over the 3x3 cells already carrying
    that label (matching the original behavior).
    """
    changed = False
    for i in range(lo, hi):
        for j in range(lo, hi):
            current = plates[i, j]
            if fill_zeros_only and current != 0:
                continue
            window = plates[i - 1:i + 2, j - 1:j + 2].ravel()
            positive = window[window > 0]
            if positive.size == 0:
                continue
            new_val = int(stats.mode(positive, keepdims=False)[0])
            if new_val == current:
                continue
            if terrain is not None:
                mask = plates[i - 1:i + 2, j - 1:j + 2] == new_val
                if mask.any():
                    avg = terrain[i - 1:i + 2, j - 1:j + 2][mask].mean()
                    terrain[i, j] = 0.5 * avg
            plates[i, j] = new_val
            changed = True
    return changed


def smooth_plates_until_stable(world: World,
                                mid_p: np.ndarray, north_p: np.ndarray, south_p: np.ndarray,
                                *, fill_zeros_only: bool,
                                terrain_trio: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                                ) -> None:
    """Repeat 3x3 mode-fill on each panel until no labels change."""
    panels = [
        (mid_p,   1, world.n_eq + 1,  None if terrain_trio is None else terrain_trio[0]),
        (north_p, 1, world.n_pol + 2, None if terrain_trio is None else terrain_trio[1]),
        (south_p, 1, world.n_pol + 2, None if terrain_trio is None else terrain_trio[2]),
    ]
    for plates, lo, hi, terrain in panels:
        while _smooth_panel_mode(plates, lo, hi,
                                 fill_zeros_only=fill_zeros_only,
                                 terrain=terrain):
            pass


# --- one timestep of plate movement -----------------------------------------

def step_plate_movement(
    world: World,
    plates_trio: tuple[np.ndarray, np.ndarray, np.ndarray],
    terrain_trio: tuple[np.ndarray, np.ndarray, np.ndarray],
    plate_movements: Sequence[Sequence[tuple[int, int]]],
    plate_order: Sequence[int],
    t: int,
    *,
    comp: float,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray],
           tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Advance plates and the terrain riding on them by one timestep.

    Returns fresh ``(plates_trio_new, terrain_trio_new)``. Densest plates
    move first (per ``plate_order``); later plates' labels overwrite earlier
    ones in the new arrays. Where new terrain stacks to >= 1.5x the original
    cell value, it is compressed by ``comp``.
    """
    new_plates = world.empty_plates()
    new_terrain = world.zeros_terrain()
    panels_old = list(plates_trio)
    terrain_old = list(terrain_trio)

    for plate_idx in plate_order:
        direction = plate_movements[plate_idx][t]
        for src_loc in (0, 1, 2):
            orig, locs, new_coords = move_plate_coordinates(
                world, plate_idx, direction, src_loc,
                panels_old[src_loc], plates_trio, rng,
            )
            for (ox, oy), (nx, ny), nloc in zip(orig, new_coords, locs):
                new_plates[nloc][nx, ny] = plate_idx + 1
                new_terrain[nloc][nx, ny] += terrain_old[src_loc][ox, oy]

    # Compress where new terrain stacks above 1.5x the corresponding old value.
    for new, old in zip(new_terrain, terrain_old):
        mask = new >= 1.5 * old
        new[mask] *= comp

    smooth_plates_until_stable(world, *new_plates,
                                fill_zeros_only=True,
                                terrain_trio=new_terrain)

    return tuple(new_plates), tuple(new_terrain)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_three_panels(
    mid: np.ndarray, north: np.ndarray, south: np.ndarray,
    *, figsize: tuple[float, float] = (10, 6),
    cmap: str = "terrain",
    vmin: float | None = None, vmax: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Build a figure with mid (large, left) and the two polar caps (right)."""
    if vmin is None:
        vmin = float(min(mid.min(), north.min(), south.min()))
    if vmax is None:
        vmax = float(max(mid.max(), north.max(), south.max()))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[2, 1], figure=fig)

    ax_mid = fig.add_subplot(gs[:, 0])
    ax_mid.imshow(mid, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_mid.axis("off")

    ax_n = fig.add_subplot(gs[0, 1])
    ax_n.imshow(north, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_n.axis("off")

    ax_s = fig.add_subplot(gs[1, 1])
    ax_s.imshow(south, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_s.axis("off")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def render_contour_panels(
    mid: np.ndarray, north: np.ndarray, south: np.ndarray,
    *, levels: int = 20, cmap: str = "terrain",
    figsize: tuple[float, float] = (6, 5),
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Three separate contour-filled figures, one per panel, sharing vmin/vmax."""
    vmin = float(min(mid.min(), north.min(), south.min()))
    vmax = float(max(mid.max(), north.max(), south.max()))
    figs = []
    for arr, name in ((mid, "Mid"), (north, "North"), (south, "South")):
        fig, ax = plt.subplots(figsize=figsize)
        c = ax.contourf(arr, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(c, ax=ax, label="Height")
        ax.set_title(f"{name} Terrain")
        figs.append(fig)
    return tuple(figs)


# Topographic palette inspired by Arcanographia's "Islamabad" map.
# Stops 0.0..0.499 map to ocean colors, 0.5..1.0 map to land colors.
# The hard 0.499 / 0.500 split makes the coastline a sharp boundary.
_ISLAMABAD_STOPS: list[tuple[float, str]] = [
    (0.000, "#B9E1FF"),  # deep ocean
    (0.250, "#DCF0FF"),  # shallow ocean
    (0.499, "#F5FAFA"),  # shoreline foam
    (0.500, "#AAC378"),  # coastal lowland (just above sea level)
    (0.600, "#E6CD9B"),  # foothills
    (0.700, "#C89B69"),  # hills
    (0.800, "#C3733C"),  # mountains
    (0.900, "#AA4B00"),  # high peaks
    (1.000, "#A0321E"),  # summit
]


def _islamabad_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("islamabad", _ISLAMABAD_STOPS)


def _resolve_cmap(name):
    """Look up a colormap by name. Adds custom Khartlan palettes."""
    if name == "islamabad":
        return _islamabad_cmap()
    return plt.get_cmap(name)


def save_mid_terrain_image(
    mid_terrain: np.ndarray,
    output_path: str | Path,
    *,
    size: tuple[int, int] | int | None = None,
    cmap: str = "islamabad",
    vmin: float | None = None,
    vmax: float | None = None,
    strip_ghost: bool = True,
    scale_power: float = 1.0,
    interp_method: str = "linear",
    ocean_ratio: float | None = 0.65,
) -> Path:
    """Save the mid panel as a pure image — no axes, no colorbar, no padding.

    The output is exactly target pixels: one pixel per interpolated cell, with
    a colormap applied via PIL so there are no matplotlib decorations to fight.

    Parameters
    ----------
    mid_terrain
        2D array. By default the 1-cell ghost ring is stripped.
    output_path
        Where to save (PNG recommended). Parent dirs are created if missing.
    size
        Target size. ``None`` keeps the input shape. An ``int`` makes a square.
        A ``(height, width)`` tuple lets you choose a specific aspect ratio.
    cmap
        Colormap name. ``"islamabad"`` is the built-in topographic atlas
        palette (default); any matplotlib colormap name works too.
    vmin, vmax
        Manual normalization range. Ignored when ``ocean_ratio`` is set.
    scale_power
        Apply ``arr ** scale_power`` before rendering. The original notebook
        used 1.35 to exaggerate elevation contrast.
    interp_method
        Passed to scipy's ``RegularGridInterpolator`` ("linear" or "nearest").
    ocean_ratio
        If set (0 < ratio < 1), defines the fraction of the final map that
        should be ocean. The corresponding elevation quantile is treated as
        sea level: pixels at that quantile become 0, anything below becomes
        ocean (negative), anything above becomes land (positive). The
        colormap is then anchored so its midpoint sits at sea level — pair
        this with the ``"islamabad"`` cmap for a topographic-atlas look.
        Pass ``None`` to disable and use a plain min/max stretch.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm

    colormap = _resolve_cmap(cmap)

    arr = mid_terrain[1:-1, 1:-1] if strip_ghost else mid_terrain
    arr = np.asarray(arr, dtype=float)
    if scale_power != 1.0:
        arr = np.maximum(arr, 0.0) ** scale_power

    if size is not None:
        target_h, target_w = (size, size) if isinstance(size, int) else size
        h, w = arr.shape
        f = RegularGridInterpolator(
            (np.arange(h), np.arange(w)), arr, method=interp_method,
        )
        ys = np.linspace(0, h - 1, target_h)
        xs = np.linspace(0, w - 1, target_w)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        arr = f((Y, X))

    if ocean_ratio is not None:
        if not 0.0 < ocean_ratio < 1.0:
            raise ValueError("ocean_ratio must be strictly between 0 and 1")
        sea_level = float(np.quantile(arr, ocean_ratio))
        shifted = arr - sea_level
        v_lo = float(shifted.min())
        v_hi = float(shifted.max())
        if v_lo >= 0.0:
            v_lo = -max(abs(v_hi), 1e-6)
        if v_hi <= 0.0:
            v_hi = max(abs(v_lo), 1e-6)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=v_lo, vmax=v_hi)
        rgba = colormap(norm(shifted))
    else:
        if vmin is None:
            vmin = float(arr.min())
        if vmax is None:
            vmax = float(arr.max())
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = colormap(norm(arr))

    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)
    return output_path


class FrameRecorder:
    """Capture matplotlib figures as PIL frames and emit a GIF.

    Use ``capture(fig)`` after each rendered step, then ``save(path)``.
    """

    def __init__(self) -> None:
        self.frames: list[Image.Image] = []

    def capture(self, fig: plt.Figure, *, dpi: int = 80) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        self.frames.append(Image.open(buf).copy())
        buf.close()

    def save(self, path: str | Path, *, duration: int = 300, loop: int = 0) -> Path:
        if not self.frames:
            raise RuntimeError("No frames captured; nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration,
            loop=loop,
        )
        return path


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Knobs for :func:`run_simulation`."""
    k: int = 15
    n_plates: int = 10
    plate_delay_max: int = 10

    initial_t_max: int = 500
    initial_n_volcanos: int = 1000
    initial_volcano_mean: float = 200.0
    initial_volcano_std: float = 180.0
    initial_diffusion_coef: float = 0.01

    plate_t_moves: int = 15
    movement_n_volcanos: int = 15
    movement_volcano_mean: float = 200.0
    movement_volcano_std: float = 150.0
    weathering: float = 0.01
    comp: float = 0.65

    seed: int | None = None
    sample_every: int = 1            # render every Nth frame to save GIF size


def run_simulation(
    config: SimConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    record_initial_terrain: bool = True,
    record_plate_movement: bool = True,
    verbose: bool = True,
) -> dict:
    """End-to-end pipeline: plates -> initial terrain -> plate motion.

    Returns a dict with the final arrays, the World, and any saved file paths.
    """
    config = config if config is not None else SimConfig()
    rng = np.random.default_rng(config.seed)
    world = World(config.k)
    out_dir = Path(output_dir) if output_dir is not None else None

    # ----- 1) Plate generation -------------------------------------------------
    if verbose: print("[1/3] growing plates...")
    mid_p, north_p, south_p = grow_plates(
        world, config.n_plates,
        delay_max=config.plate_delay_max, rng=rng,
    )
    smooth_plates_until_stable(world, mid_p, north_p, south_p,
                                fill_zeros_only=False)

    # ----- 2) Initial terrain via volcanoes + diffusion -----------------------
    if verbose: print("[2/3] forming initial terrain...")
    mid_t, north_t, south_t = world.random_terrain(rng)
    world.sync_borders(mid_t, north_t, south_t)

    initial_volcanos = random_volcanos(
        world, config.initial_n_volcanos, config.initial_t_max,
        m_mean=config.initial_volcano_mean,
        m_std=config.initial_volcano_std,
        min_t_diff=50, max_t_diff=config.initial_t_max, rng=rng,
    )

    initial_recorder = FrameRecorder() if record_initial_terrain else None
    for t in range(config.initial_t_max):
        apply_volcanos(initial_volcanos, t, mid_t, north_t, south_t)
        coef = (config.initial_t_max - t) / config.initial_t_max * config.initial_diffusion_coef
        world.diffuse(mid_t, north_t, south_t, coef)
        if initial_recorder is not None and t % config.sample_every == 0:
            fig = render_three_panels(mid_t, north_t, south_t,
                                      title=f"Initial terrain  t={t}")
            initial_recorder.capture(fig)
            plt.close(fig)

    # Snapshot for replay if needed.
    snap_plates = (mid_p.copy(), north_p.copy(), south_p.copy())
    snap_terrain = (mid_t.copy(), north_t.copy(), south_t.copy())

    # ----- 3) Plate movement ---------------------------------------------------
    if verbose: print("[3/3] moving plates...")
    plate_movements = [select_directions(config.plate_t_moves, rng=rng)
                       for _ in range(config.n_plates)]
    plate_density = rng.integers(1, 100, size=config.n_plates)
    plate_order = list(np.argsort(plate_density)[::-1])

    movement_volcanos = random_volcanos(
        world, config.movement_n_volcanos, config.initial_t_max,
        m_mean=config.movement_volcano_mean,
        m_std=config.movement_volcano_std,
        min_t_diff=3, max_t_diff=15, rng=rng,
    )

    plates_trio = (mid_p, north_p, south_p)
    terrain_trio = (mid_t, north_t, south_t)

    move_recorder = FrameRecorder() if record_plate_movement else None
    for t in range(config.plate_t_moves):
        plates_trio, terrain_trio = step_plate_movement(
            world, plates_trio, terrain_trio,
            plate_movements, plate_order, t,
            comp=config.comp, rng=rng,
        )
        world.sync_borders(*terrain_trio)
        apply_volcanos(movement_volcanos, t, *terrain_trio)
        world.diffuse(*terrain_trio, config.weathering)

        if move_recorder is not None and t % config.sample_every == 0:
            mid_t_v, north_t_v, south_t_v = terrain_trio
            fig = render_three_panels(
                mid_t_v[1:world.n_eq, 1:world.n_eq],
                north_t_v[1:world.n_pol + 1, 1:world.n_pol + 1],
                south_t_v[1:world.n_pol + 1, 1:world.n_pol + 1],
                title=f"Plate motion  t={t}",
            )
            move_recorder.capture(fig)
            plt.close(fig)

    # ----- save outputs --------------------------------------------------------
    saved_paths: dict[str, Path] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if initial_recorder is not None:
            saved_paths["initial_terrain_gif"] = initial_recorder.save(
                out_dir / "initial_terrain.gif", duration=80,
            )
        if move_recorder is not None:
            saved_paths["plate_movement_gif"] = move_recorder.save(
                out_dir / "plate_movement.gif", duration=400,
            )
        # Save final still frames (matching the original notebook's contour outputs).
        mid_t, north_t, south_t = terrain_trio
        figs = render_contour_panels(
            mid_t[1:world.n_eq, 1:world.n_eq],
            north_t[1:world.n_pol + 1, 1:world.n_pol + 1],
            south_t[1:world.n_pol + 1, 1:world.n_pol + 1],
        )
        for fig, name in zip(figs, ("mid_terrain", "north_terrain", "south_terrain")):
            p = out_dir / f"{name}.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            saved_paths[name] = p
            plt.close(fig)

    return {
        "world": world,
        "plates": plates_trio,
        "terrain": terrain_trio,
        "snapshot_plates": snap_plates,
        "snapshot_terrain": snap_terrain,
        "saved_paths": saved_paths,
    }


__all__ = [
    "DIRECTIONS_8",
    "World",
    "Volcano",
    "FrameRecorder",
    "SimConfig",
    "apply_volcanos",
    "grow_plates",
    "move_plate_coordinates",
    "random_volcanos",
    "render_contour_panels",
    "render_three_panels",
    "run_simulation",
    "save_mid_terrain_image",
    "select_directions",
    "smooth_plates_until_stable",
    "step_plate_movement",
]
