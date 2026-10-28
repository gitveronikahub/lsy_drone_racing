"""Receding-horizon translational control through the Crazyflie full-state interface.

A visibility path selects gate approaches and exits. A warm-started quadratic program
predicts motion from measured state, limits acceleration and its rate of change, and
follows the local path with a variable speed profile. Inner attitude control is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.state_controller_planner import LocalTrackPlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class StateController(Controller):
    """Plan and control local gate manoeuvres using a linear acceleration MPC."""

    def __init__(self, obs: dict[str, NDArray], info: dict, config: ConfigDict):
        """Build the prediction matrices and the first observable local route."""
        super().__init__(obs, info, config)
        self._config = config
        self._needs_reset = False
        self._dt = 1 / config.env.freq
        self._horizon_steps = 12
        self._prediction_dt = 0.06
        self._max_speed = 6.5  # Per-axis velocity bound, m/s.
        self._max_acceleration = 7.0  # Per-axis acceleration bound, m/s².
        self._max_jerk = 60.0  # Per-axis command slew limit, m/s³.
        self._prediction_jerk = 18.0  # Smoother future inputs over the coarser prediction grid.
        self._gate_speed = 2.3
        self._gate_sequence = np.array(obs["gate_sequence"], dtype=int, copy=True)
        self._gate_directions = np.array(obs["gate_sequence_direction"], dtype=int, copy=True)
        self.current_gate_pass_idx = int(obs["n_gates_passed"])
        self._target_pass = min(self.current_gate_pass_idx, len(self._gate_sequence) - 1)
        self._previous_acceleration = np.zeros(3)
        self._previous_command = np.zeros(3)
        self._previous_velocity = np.array(obs["vel"], dtype=float, copy=True)
        self._estimated_acceleration = np.zeros(3)
        self._response_time = np.array([0.11, 0.11, 0.06])
        self._disturbance = np.zeros(3)
        self._tick = 0
        self.solver_failures = 0
        self.last_solver_status = "initializing"
        self.predicted_positions = np.asarray(obs["pos"])[None, :].copy()
        limits = config.env.track.safety_limits
        self._planner = LocalTrackPlanner(limits.pos_limit_low, limits.pos_limit_high)
        self._planner.update(obs)
        self._build_prediction_matrices()
        self._replan(obs)

    def _build_prediction_matrices(self) -> None:
        """Condense position/velocity dynamics with first-order inner-loop response."""
        n, dt = self._horizon_steps, self._prediction_dt
        tau = self._response_time
        decay = np.exp(-dt / tau)
        velocity_response = tau * (1 - decay)
        position_response = tau * dt - tau**2 * (1 - decay)
        dynamics = np.eye(9)
        dynamics[:3, 3:6] = dt * np.eye(3)
        dynamics[:3, 6:9] = np.diag(position_response)
        dynamics[3:6, 6:9] = np.diag(velocity_response)
        dynamics[6:9, 6:9] = np.diag(decay)
        inputs = np.vstack(
            (
                np.diag(0.5 * dt**2 - position_response),
                np.diag(dt - velocity_response),
                np.diag(1 - decay),
            )
        )
        self._position_matrix = np.zeros((3 * n, 3 * n))
        self._velocity_matrix = np.zeros((3 * n, 3 * n))
        self._free_position_matrix = np.zeros((3 * n, 9))
        self._free_velocity_matrix = np.zeros((3 * n, 9))
        response, free = np.zeros((9, 3 * n)), np.eye(9)
        for stage in range(n):
            response = dynamics @ response
            response[:, 3 * stage : 3 * stage + 3] += inputs
            free = dynamics @ free
            self._position_matrix[3 * stage : 3 * stage + 3] = response[:3]
            self._velocity_matrix[3 * stage : 3 * stage + 3] = response[3:6]
            self._free_position_matrix[3 * stage : 3 * stage + 3] = free[:3]
            self._free_velocity_matrix[3 * stage : 3 * stage + 3] = free[3:6]
        self._difference_matrix = np.eye(3 * n) - np.eye(3 * n, k=-3)
        self._position_weights = np.tile([100.0, 100.0, 140.0], n)
        p, v, d = self._position_matrix, self._velocity_matrix, self._difference_matrix
        self._hessian = 2 * (
            p.T @ (self._position_weights[:, None] * p)
            + 4 * v.T @ v
            + 0.08 * np.eye(3 * n)
            + 0.2 * d.T @ d
        )
        self._jerk_steps = np.r_[
            np.full(3, self._max_jerk * self._dt), np.full(3 * (n - 1), self._prediction_jerk * dt)
        ]
        # Geometry/velocity slacks allow recovery after sudden measurement updates.
        self._slack_matrix = np.repeat(np.eye(n), 3, axis=0)
        # Reserve one additional halfspace per stage to commit to the current crossing.
        self._planes_per_stage = len(self._planner.obstacles) + len(self._planner.box_centres) + 1
        self._qp_rows = 15 * n + n * self._planes_per_stage
        self._qp_hessian = np.zeros((4 * n, 4 * n))
        self._qp_hessian[: 3 * n, : 3 * n] = self._hessian
        self._qp_hessian[3 * n :, 3 * n :] = 2e4 * np.eye(n)
        # Expose the causal block sparsity rather than factoring structural zeros at every step.
        pattern = np.zeros((self._qp_rows, 4 * n), dtype=bool)
        pattern[: 3 * n, : 3 * n] = self._difference_matrix != 0
        pattern[3 * n : 15 * n, : 3 * n] = np.tile(np.abs(self._velocity_matrix) > 0, (4, 1))
        pattern[3 * n : 15 * n, 3 * n :] = np.tile(self._slack_matrix != 0, (4, 1))
        for stage in range(n):
            rows = slice(
                15 * n + stage * self._planes_per_stage,
                15 * n + (stage + 1) * self._planes_per_stage,
            )
            pattern[rows, : 3 * (stage + 1)] = True
            pattern[rows, 3 * n + stage] = True
        row, column = np.nonzero(pattern)
        constraint_sparsity = ca.Sparsity.triplet(
            self._qp_rows, 4 * n, row.tolist(), column.tolist()
        )
        row, column = np.nonzero(self._qp_hessian)
        hessian_sparsity = ca.Sparsity.triplet(4 * n, 4 * n, row.tolist(), column.tolist())
        self._qp_solver = ca.conic(
            "state_mpc",
            "osqp",
            {"h": hessian_sparsity, "a": constraint_sparsity},
            {
                "error_on_fail": False,
                "osqp": {"verbose": False, "eps_abs": 1e-4, "eps_rel": 1e-4, "max_iter": 1000},
            },
        )
        self._qp_warm_start = np.zeros(4 * n)
        self._qp_dual = np.zeros(self._qp_rows)
        self._qp_bound_dual = np.zeros(4 * n)
        self._qp_lower = np.r_[np.full(3 * n, -self._max_acceleration), np.zeros(n)]
        self._qp_upper = np.r_[np.full(3 * n, self._max_acceleration), np.full(n, np.inf)]

    def _replan(self, obs: dict[str, NDArray]) -> None:
        """Plan through upcoming gates so intermediate exits are not stopping targets."""
        self._path = np.asarray(obs["pos"], dtype=float)[None, :]
        self._gate_windows = []
        self._exit_distance = 0.0
        length = 0.0
        # Two further passes keep the endpoint beyond the short MPC horizon, including
        # when consecutive gates are close together. Never append an unchecked connector.
        for pass_idx in range(
            self._target_pass, min(self._target_pass + 3, len(self._gate_sequence))
        ):
            gate_id = self._gate_sequence[pass_idx]
            route, exit_distance = self._planner.gate_path(
                self._path[-1], gate_id, self._gate_directions[pass_idx]
            )
            if route is None:
                break
            if pass_idx == self._target_pass:
                self._exit_distance = exit_distance
                if int(obs["n_gates_passed"]) > pass_idx:
                    # A discovery between the crossing and exit clearance must not
                    # send the drone back to the entry of an already passed gate.
                    route = self._planner._connector(self._path[-1], route[-1])
                    if route is None:
                        break
            extension = np.linalg.norm(np.diff(route, axis=0), axis=-1).sum()
            gate_start = length + extension - exit_distance
            length += extension
            self._gate_windows.append((max(0.0, gate_start - 0.35), length))
            self._path = np.vstack((self._path, route[1:]))
        self._route_blocked = not self._gate_windows
        keep = np.r_[True, np.linalg.norm(np.diff(self._path, axis=0), axis=-1) > 1e-6]
        self._path = self._path[keep]
        self._segments = np.diff(self._path, axis=0)
        self._lengths = np.linalg.norm(self._segments, axis=-1)
        self._cumulative = np.r_[0, np.cumsum(self._lengths)]
        self._exit_vertex = (
            int(np.argmin(np.abs(self._cumulative - self._gate_windows[0][1])))
            if self._gate_windows
            else 0
        )
        # Approximate the speed sustainable through a rounded polyline corner, using
        # available turning distance and acceleration rather than a fixed 0.4/0.8 m/s.
        tangents = self._segments / self._lengths[:, None]
        cosine = np.clip(np.sum(tangents[:-1] * tangents[1:], axis=-1), -1, 1)
        blend = np.minimum(0.25, 0.45 * np.minimum(self._lengths[:-1], self._lengths[1:]))
        radius = blend / np.maximum(np.tan(0.5 * np.arccos(cosine)), 1e-6)
        self._corner_speeds = np.minimum(
            self._max_speed, np.sqrt(0.6 * self._max_acceleration * radius)
        )
        self._commit_exit = (
            0 < self._exit_vertex < len(self._segments)
            and np.dot(tangents[self._exit_vertex - 1], tangents[self._exit_vertex]) < 0
        )
        self._route_progress = 0.0
        self._replan_tick = self._tick

    def _references(self, obs: dict[str, NDArray]) -> tuple[NDArray, NDArray]:
        """Follow measured progress with continuous velocity through intermediate exits."""
        n = self._horizon_steps
        if not len(self._segments):
            return np.tile(self._path[0], (n, 1)), np.zeros((n, 3))
        fraction = np.clip(
            np.sum((obs["pos"] - self._path[:-1]) * self._segments, axis=-1) / self._lengths**2,
            0,
            1,
        )
        projected = self._path[:-1] + fraction[:, None] * self._segments
        distance = np.linalg.norm(projected - obs["pos"], axis=-1)
        distance[self._cumulative[1:] < self._route_progress - 0.05] = np.inf
        # Nearby/repeated future gates must not attract measured progress to a later
        # route segment. Only the environment's passage count advances the target.
        distance[self._cumulative[:-1] >= self._gate_windows[0][1] - 1e-6] = np.inf
        segment = int(np.argmin(distance))
        self._route_progress = max(
            self._route_progress,
            self._cumulative[segment] + fraction[segment] * self._lengths[segment],
        )
        position, velocity = [], []
        self._reference_progress = []
        progress = self._route_progress
        previous_speed = float(np.linalg.norm(obs["vel"]))
        for _ in range(n):
            segment = min(
                np.searchsorted(self._cumulative, progress, side="right") - 1,
                len(self._segments) - 1,
            )
            speed = min(
                self._max_speed,
                np.sqrt(2 * self._max_acceleration * max(0, self._cumulative[-1] - progress)),
            )
            for entry, exit_ in self._gate_windows:
                if progress <= exit_:
                    speed = min(speed, np.sqrt(self._gate_speed**2 + 4 * max(0, entry - progress)))
            for corner in range(segment + 1, len(self._segments)):
                speed = min(
                    speed,
                    np.sqrt(
                        self._corner_speeds[corner - 1] ** 2
                        + 4 * max(0, self._cumulative[corner] - progress)
                    ),
                )
            # A turn's speed limit must release gradually. An instantaneous jump on
            # the outgoing segment can pull the optimizer back before it crosses.
            speed = min(speed, previous_speed + 0.6 * self._max_acceleration * self._prediction_dt)
            previous_speed = speed
            progress = min(progress + speed * self._prediction_dt, self._cumulative[-1])
            self._reference_progress.append(progress)
            segment = min(
                np.searchsorted(self._cumulative, progress, side="right") - 1,
                len(self._segments) - 1,
            )
            tangent = self._segments[segment] / self._lengths[segment]
            position.append(self._path[segment] + tangent * (progress - self._cumulative[segment]))
            velocity.append(tangent * speed if progress < self._cumulative[-1] else np.zeros(3))
            if self._commit_exit and progress >= self._gate_windows[0][1]:
                # A reversal is only released after measured exit clearance. Keep a
                # positive crossing velocity rather than letting the future return leg
                # cancel the gate approach inside the finite prediction horizon.
                position[-1] = self._path[self._exit_vertex]
                exit_tangent = (
                    self._segments[self._exit_vertex - 1] / self._lengths[self._exit_vertex - 1]
                )
                velocity[-1] = exit_tangent * max(0.4, self._corner_speeds[self._exit_vertex - 1])
        return np.asarray(position), np.asarray(velocity)

    def _solve(self, obs: dict[str, NDArray], position: NDArray, velocity: NDArray) -> NDArray:
        """Solve the constrained QP; return bounded braking on a failed or nonfinite solve."""
        n = self._horizon_steps
        state = np.r_[obs["pos"], obs["vel"], self._estimated_acceleration]
        free_position = self._free_position_matrix @ state
        free_velocity = self._free_velocity_matrix @ state
        previous = np.r_[self._previous_acceleration, np.zeros(3 * (n - 1))]
        gradient = 2 * (
            self._position_matrix.T
            @ (self._position_weights * (free_position - position.reshape(-1)))
            + 4 * self._velocity_matrix.T @ (free_velocity - velocity.reshape(-1))
            - 0.2 * self._difference_matrix.T @ previous
        )
        previous_input = np.r_[self._previous_command + self._disturbance, np.zeros(3 * (n - 1))]
        matrix = np.zeros((self._qp_rows, 4 * n))
        lower = np.full(self._qp_rows, -np.inf)
        upper = np.full(self._qp_rows, np.inf)
        matrix[: 3 * n, : 3 * n] = self._difference_matrix
        lower[: 3 * n] = previous_input - self._jerk_steps
        upper[: 3 * n] = previous_input + self._jerk_steps
        matrix[3 * n : 15 * n, : 3 * n] = np.vstack(
            (
                self._velocity_matrix,
                -self._velocity_matrix,
                self._position_matrix,
                -self._position_matrix,
            )
        )
        matrix[3 * n : 15 * n, 3 * n :] = -np.tile(self._slack_matrix, (4, 1))
        upper[3 * n : 15 * n] = np.r_[
            self._max_speed - free_velocity,
            self._max_speed + free_velocity,
            np.tile(self._planner.upper, n) - free_position,
            -np.tile(self._planner.lower, n) + free_position,
        ]
        if len(self._segments):
            # Once the reference crosses the current gate, keep its predicted positions
            # beyond the crossing plane until actual passage/clearance advances the target.
            # Otherwise an upcoming reversal can pull the whole prediction back in front
            # of the gate and leave a stable equilibrium that never registers a pass.
            gate_id = self._gate_sequence[self._target_pass]
            normal = (
                self._planner.rotations[gate_id, :, 0] * self._gate_directions[self._target_pass]
            )
            crossing = self._gate_windows[0][1] - self._exit_distance
            progress = np.asarray(self._reference_progress) - crossing
            stage = np.flatnonzero(progress >= 0)
            clearance = np.minimum(progress[stage], min(0.20, max(0.0, self._exit_distance - 0.02)))
            rows = 15 * n + stage * self._planes_per_stage + self._planes_per_stage - 1
            prediction = self._position_matrix.reshape(n, 3, -1)[stage]
            matrix[rows, : 3 * n] = -np.einsum("i,kij->kj", normal, prediction)
            matrix[rows, 3 * n + stage] = -1
            upper[rows] = (
                free_position.reshape(n, 3)[stage] - self._planner.gates[gate_id]
            ) @ normal - clearance
        stage, normal, offset = self._planner.separating_planes(position)
        if len(stage):
            order = np.argsort(stage, kind="stable")
            stage, normal, offset = stage[order], normal[order], offset[order]
            group_starts = np.r_[0, np.flatnonzero(np.diff(stage)) + 1]
            within_stage = np.arange(len(stage)) - np.repeat(
                group_starts, np.diff(np.r_[group_starts, len(stage)])
            )
            prediction = self._position_matrix.reshape(n, 3, -1)[stage]
            rows = 15 * n + stage * self._planes_per_stage + within_stage
            matrix[rows, : 3 * n] = -np.einsum("ki,kij->kj", normal, prediction)
            matrix[rows, 3 * n + stage] = -1
            upper[rows] = np.sum(normal * free_position.reshape(n, 3)[stage], axis=-1) - offset
        try:
            result = self._qp_solver(
                h=self._qp_hessian,
                g=np.r_[gradient, np.full(n, 10.0)],
                a=matrix,
                lba=lower,
                uba=upper,
                lbx=self._qp_lower,
                ubx=self._qp_upper,
                x0=self._qp_warm_start,
                lam_a0=self._qp_dual,
                lam_x0=self._qp_bound_dual,
            )
            stats = self._qp_solver.stats()
            solution = np.asarray(result["x"]).reshape(-1)
            self.last_solver_status = str(stats["return_status"])
            if stats["success"] and np.all(np.isfinite(solution)):
                self._qp_warm_start = np.r_[
                    solution[3 : 3 * n],
                    solution[3 * n - 3 : 3 * n],
                    solution[3 * n + 1 :],
                    solution[-1],
                ]
                self._qp_dual = np.asarray(result["lam_a"]).reshape(-1)
                self._qp_bound_dual = np.asarray(result["lam_x"]).reshape(-1)
                self.predicted_positions = (
                    free_position + self._position_matrix @ solution[: 3 * n]
                ).reshape(n, 3)
                return solution[:3]
        except RuntimeError as error:
            self.last_solver_status = str(error)
        self.solver_failures += 1
        if self.solver_failures == 1:
            logger.debug("State MPC uses braking fallback: %s", self.last_solver_status)
        self._qp_warm_start.fill(0)
        self._qp_dual.fill(0)
        self._qp_bound_dual.fill(0)
        return np.clip(-2 * obs["vel"], -self._max_acceleration, self._max_acceleration)

    def compute_control(
        self, obs: dict[str, NDArray], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Return measured position/velocity and the MPC's compensated acceleration.

        Matching position and velocity references to the measured state lets MPC provide the
        outer feedback through acceleration feedforward. The existing inner attitude loop
        converts this input to orientation and thrust. Acceleration excludes gravity.
        """
        if self._needs_reset:
            self.__init__(obs, info or {}, self._config)
        measured_acceleration = (obs["vel"] - self._previous_velocity) / self._dt
        self._previous_velocity = np.array(obs["vel"], dtype=float, copy=True)
        if self._tick:
            self._estimated_acceleration = np.clip(
                0.5 * self._estimated_acceleration + 0.5 * measured_acceleration, -15, 15
            )
            self._disturbance = np.clip(
                self._disturbance + 0.025 * (measured_acceleration - self._previous_acceleration),
                -3,
                3,
            )
        self.current_gate_pass_idx = int(obs["n_gates_passed"])
        gate_id = self._gate_sequence[self._target_pass]
        normal = self._planner.rotations[gate_id, :, 0] * self._gate_directions[self._target_pass]
        axial = np.dot(obs["pos"] - self._planner.gates[gate_id], normal)
        advance = self.current_gate_pass_idx > self._target_pass and axial >= min(
            0.20, self._exit_distance - 0.02
        )
        discovered = self._planner.changed(obs)
        if advance:
            self._target_pass = min(self.current_gate_pass_idx, len(self._gate_sequence) - 1)
        if discovered:
            self._planner.update(obs)
        if discovered or advance or (self._route_blocked and self._tick - self._replan_tick >= 10):
            self._replan(obs)
        if self.current_gate_pass_idx >= len(self._gate_sequence):
            self._path = np.array(obs["pos"], dtype=float)[None, :]
            self._segments = np.empty((0, 3))
        position, velocity = self._references(obs)
        acceleration = self._solve(obs, position, velocity)
        command = np.clip(acceleration - self._disturbance, -6, 6)
        command = np.clip(
            command,
            self._previous_command - self._max_jerk * self._dt,
            self._previous_command + self._max_jerk * self._dt,
        )
        self._previous_command = command.copy()
        self._previous_acceleration = command + self._disturbance
        return np.concatenate((obs["pos"], obs["vel"], command, [0, 0, 0, 1], np.zeros(3))).astype(
            np.float32
        )

    def step_callback(
        self,
        action: NDArray,
        obs: dict[str, NDArray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the sample counter; race termination remains with the environment."""
        self._tick += 1
        return False

    def render_callback(self, sim: Sim) -> None:
        """Draw the local route and predicted positions."""
        if len(self._path) > 1:
            draw_line(sim, self._path, rgba=(0, 1, 0, 1))
        draw_line(sim, self.predicted_positions, rgba=(0, 0.5, 1, 1))
        draw_points(sim, self.predicted_positions[:1], rgba=(1, 0, 0, 1), size=0.02)

    def reset(self) -> None:
        """Reinitialize from the next observation instead of reusing the previous race map."""
        self._needs_reset = True

    def episode_callback(self) -> None:
        """Discard episode-specific state when the controller is reused."""
        self.reset()

    def episode_reset(self) -> None:
        """Request a fresh map, estimator, and optimization state for the next episode."""
        self.reset()
