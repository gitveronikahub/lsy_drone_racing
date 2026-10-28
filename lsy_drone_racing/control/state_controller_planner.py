"""Local paths and obstacle halfspaces for the state-controller MPC.

The oriented collision boxes match ``envs/assets/gate.xml``. Poles are conservatively
treated as vertical cylinders; clearances include the drone and tracking error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

_BOX_CENTRES = np.array([[0, 0, 0.28], [0, 0, -0.28], [0, -0.28, 0], [0, 0.28, 0], [0, 0, -0.86]])
_BOX_SIZES = np.array(
    [
        [0.01, 0.36, 0.08],
        [0.01, 0.36, 0.08],
        [0.01, 0.08, 0.36],
        [0.01, 0.08, 0.36],
        [0.05, 0.05, 0.5],
    ]
)


class LocalTrackPlanner:
    """Build collision-checked local paths through oriented gates."""

    def __init__(self, lower: NDArray, upper: NDArray):
        """Store configured flight bounds."""
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    def update(self, obs: dict[str, NDArray]) -> None:
        """Copy observable geometry, without reading hidden simulator state."""
        self.gates = np.array(obs["gates_pos"], dtype=float, copy=True).reshape(-1, 3)
        self.quats = np.array(obs["gates_quat"], copy=True).reshape(-1, 4)
        self.rotations = Rotation.from_quat(self.quats).as_matrix().reshape(-1, 3, 3)
        self.obstacles = np.array(obs["obstacles_pos"], dtype=float, copy=True).reshape(-1, 3)
        self.gates_visited = np.array(obs["gates_visited"], copy=True)
        self.obstacles_visited = np.array(obs["obstacles_visited"], copy=True)
        self.pole_radii = np.where(self.obstacles_visited, 0.19, 0.25)
        self.box_centres = (
            self.gates[:, None, :] + np.einsum("gij,bj->gbi", self.rotations, _BOX_CENTRES)
        ).reshape(-1, 3)
        self.box_rotations = np.repeat(self.rotations, len(_BOX_CENTRES), axis=0)
        self.box_sizes = np.tile(_BOX_SIZES, (len(self.gates), 1))

    def changed(self, obs: dict[str, NDArray]) -> bool:
        """Detect discovery flags, position changes, and orientation-only changes."""
        alignment = np.abs(np.sum(self.quats * obs["gates_quat"], axis=-1))
        return bool(
            not np.array_equal(obs["gates_visited"], self.gates_visited)
            or not np.array_equal(obs["obstacles_visited"], self.obstacles_visited)
            or np.any(np.abs(obs["gates_pos"] - self.gates) > 0.005)
            or np.any(np.abs(obs["obstacles_pos"] - self.obstacles) > 0.005)
            or np.any(alignment < np.cos(0.005 / 2))
        )

    def clear_segments(self, starts: NDArray, ends: NDArray) -> NDArray[np.bool_]:
        """Check complete segments against inflated geometry and flight bounds."""
        starts, ends = np.atleast_2d(starts), np.atleast_2d(ends)
        delta = ends - starts
        clear = np.all((starts >= self.lower) & (starts <= self.upper), axis=-1)
        clear &= np.all((ends >= self.lower) & (ends <= self.upper), axis=-1)
        if len(self.obstacles):
            relative = self.obstacles[None, :, :2] - starts[:, None, :2]
            denominator = np.maximum(np.sum(delta[:, :2] ** 2, axis=-1), 1e-10)
            fraction = np.clip(
                np.sum(relative * delta[:, None, :2], axis=-1) / denominator[:, None], 0, 1
            )
            distance = np.linalg.norm(relative - fraction[..., None] * delta[:, None, :2], axis=-1)
            clear &= np.all(distance > self.pole_radii[None, :], axis=-1)
        local_start = np.einsum(
            "gij,egi->egj", self.rotations, starts[:, None, :] - self.gates[None, :, :]
        )
        local_delta = np.einsum("gij,ei->egj", self.rotations, delta)
        parallel = np.abs(local_delta) < 1e-9
        divisor = np.where(parallel, 1, local_delta)
        for centre, size in zip(_BOX_CENTRES, _BOX_SIZES, strict=True):
            half = size + 0.12
            relative = local_start - centre
            bound1, bound2 = (-half - relative) / divisor, (half - relative) / divisor
            entry = np.where(parallel, -np.inf, np.minimum(bound1, bound2)).max(axis=-1)
            exit_ = np.where(parallel, np.inf, np.maximum(bound1, bound2)).min(axis=-1)
            separated = np.any(parallel & (np.abs(relative) > half), axis=-1)
            clear &= ~np.any(~separated & (np.maximum(entry, 0) <= np.minimum(exit_, 1)), axis=-1)
        return clear

    def _connector(self, start: NDArray, end: NDArray) -> NDArray | None:
        """Find a visibility connector, reporting failure instead of returning an unsafe edge."""
        if self.clear_segments(start, end)[0]:
            return np.array([start, end])
        height = np.clip((start[2] + end[2]) / 2, 0.35, self.upper[2] - 0.1)
        nodes = [start, end]
        for pole, radius in zip(self.obstacles, self.pole_radii, strict=True):
            for angle in np.arange(8) * np.pi / 4:
                xy = pole[:2] + (radius + 0.05) * np.array([np.cos(angle), np.sin(angle)])
                nodes.append(np.r_[xy, height])
        for gate, rotation in zip(self.gates, self.rotations, strict=True):
            for x in (-0.25, 0.25):
                for y, z in ((-0.52, 0), (0.52, 0), (0, 0.52), (0, -0.52)):
                    nodes.append(gate + rotation @ np.array([x, y, z]))
            # Tight exits may need a sideways turn just beyond the frame to avoid a pole.
            for x in (-0.15, 0.15):
                for y in (-0.15, 0.15):
                    nodes.append(gate + rotation @ np.array([x, y, 0.0]))
        nodes = np.asarray(nodes)
        valid = self.clear_segments(nodes, nodes)
        if not valid[0] or not valid[1]:
            return None
        nodes = nodes[np.r_[True, True, valid[2:]]]
        first, second = np.triu_indices(len(nodes), 1)
        valid = self.clear_segments(nodes[first], nodes[second])
        graph = np.full((len(nodes), len(nodes)), np.inf)
        graph[first[valid], second[valid]] = np.linalg.norm(
            nodes[first[valid]] - nodes[second[valid]], axis=-1
        )
        graph = np.minimum(graph, graph.T)
        distance, previous = shortest_path(
            graph, directed=False, indices=0, return_predecessors=True
        )
        if not np.isfinite(distance[1]):
            return None
        route = [1]
        while route[-1] != 0:
            route.append(int(previous[route[-1]]))
        return nodes[route[::-1]]

    def gate_path(
        self, start: NDArray, gate_id: int, direction: int
    ) -> tuple[NDArray | None, float]:
        """Connect to a gate approach, then cross its opening in the requested direction."""
        gate = self.gates[gate_id]
        normal = self.rotations[gate_id, :, 0] * direction
        endpoints = []
        for sign in (-1, 1):
            point = None
            for distance in (0.35, 0.28, 0.22, 0.16, 0.12, 0.08, 0.05):
                candidate = gate + normal * sign * distance
                if self.clear_segments(gate, candidate)[0]:
                    point = candidate
                    break
            if point is None:
                return None, 0.0
            endpoints.append(point)
        entry, exit_ = endpoints
        connector = self._connector(np.asarray(start), entry)
        if connector is None or not self.clear_segments(entry, exit_)[0]:
            return None, 0.0
        points = np.vstack((connector, exit_))
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=-1) > 0.01]
        return points[keep], float(np.dot(exit_ - gate, normal))

    def separating_planes(self, reference: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return local halfspaces ``normal @ position >= offset`` for each prediction stage."""
        stages, normals, offsets = [], [], []
        if len(self.obstacles):
            delta = reference[:, None, :2] - self.obstacles[None, :, :2]
            distance = np.linalg.norm(delta, axis=-1)
            stage, pole = np.nonzero(distance < 1.0)
            normal = np.column_stack(
                (
                    delta[stage, pole] / np.maximum(distance[stage, pole, None], 1e-6),
                    np.zeros(len(stage)),
                )
            )
            stages.extend(stage)
            normals.extend(normal)
            offsets.extend(np.sum(normal * self.obstacles[pole], axis=-1) + self.pole_radii[pole])
        relative = np.einsum(
            "bij,kbi->kbj", self.box_rotations, reference[:, None, :] - self.box_centres[None, :, :]
        )
        half = self.box_sizes + 0.115
        excess = np.abs(relative) - half[None, :, :]
        stage, box = np.nonzero(excess.max(axis=-1) < 0.4)
        axis = excess[stage, box].argmax(axis=-1)
        sign = np.where(relative[stage, box, axis] >= 0, 1, -1)
        normal = self.box_rotations[box, :, axis] * sign[:, None]
        stages.extend(stage)
        normals.extend(normal)
        offsets.extend(np.sum(normal * self.box_centres[box], axis=-1) + half[box, axis])
        return (
            np.asarray(stages, dtype=int),
            np.asarray(normals).reshape(-1, 3),
            np.asarray(offsets),
        )
