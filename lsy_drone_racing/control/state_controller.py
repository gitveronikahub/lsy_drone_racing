"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class StateController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._target_speed = 0.7  # Target speed in m/s
        
        self.start_pos = obs["pos"]
        self.known_gates = np.copy(obs["gates_pos"])
        self.known_rpy = R.from_quat(obs["gates_quat"]).as_euler('xyz')
        self.known_obstacles = np.copy(obs["obstacles_pos"])
        
        self.current_gate_pass_idx = 0
        self._tick = 0
        
        # Build initial full-course trajectory anchored from episode start (t=0)
        self._update_spline()

    def _update_spline(self):
        pole_obstacles = np.array(self.known_obstacles)

        # Full gate execution sequence: Gate 1 -> Gate 2 -> Gate 3 -> Gate 4 -> Gate 2
        full_ordered_gates = self.known_gates[[0, 1, 2, 3, 1]]
        full_ordered_rpy = self.known_rpy[[0, 1, 2, 3, 1]]

        # Always start from initial spawn position to preserve full global curve shape
        waypoints = [self.start_pos]

        # 1. Build waypoints for ALL gates in sequence
        for i in range(len(full_ordered_gates)):
            g_pos = full_ordered_gates[i]
            g_yaw = full_ordered_rpy[i][2]
            
            dir_vec = np.array([np.cos(g_yaw), np.sin(g_yaw), 0.0])
            
            waypoints.append(g_pos - dir_vec * 0.20)
            waypoints.append(g_pos)
            waypoints.append(g_pos + dir_vec * 0.20)
            
            # Midpoint between Gate 3 (pass 2) and Gate 4 (pass 3)
            if i == 2:
                g3_pos = full_ordered_gates[2]
                g4_pos = full_ordered_gates[3]
                
                mid_pt = (g3_pos + g4_pos) / 2.0
                dir_2d = g4_pos[:2] - g3_pos[:2]
                perp_2d = np.array([-dir_2d[1], dir_2d[0]])
                norm = np.linalg.norm(perp_2d)
                
                if norm > 1e-5:
                    perp_unit = perp_2d / norm
                    stretched_mid = np.array([
                        mid_pt[0] + perp_unit[0] * -0.6,
                        mid_pt[1] + perp_unit[1] * -0.6,
                        mid_pt[2]
                    ])
                else:
                    stretched_mid = mid_pt
                    
                waypoints.append(stretched_mid)

        # 2. Refinement loop for pole avoidance across full path
        pole_safety = 0.30
        for _ in range(3):
            wp_arr = np.array(waypoints)
            distances = np.linalg.norm(np.diff(wp_arr, axis=0), axis=1)
            total_dist = np.sum(distances)
            if total_dist < 1e-5:
                break
                
            cum_dist = np.insert(np.cumsum(distances), 0, 0)
            est_duration = total_dist / self._target_speed
            t_pts = (cum_dist / total_dist) * est_duration
            
            spline = CubicSpline(t_pts, wp_arr, bc_type='clamped')
            
            t_eval = np.linspace(0, est_duration, 100)
            eval_pts = spline(t_eval)

            worst_penetration = 0.0
            best_fix_pt = None
            insert_t = 0.0

            if len(pole_obstacles) > 0:
                dists_2d = np.linalg.norm(eval_pts[:, None, :2] - pole_obstacles[None, :, :2], axis=2)
                penetrations = pole_safety - dists_2d
                max_idx = np.unravel_index(np.argmax(penetrations), penetrations.shape)
                
                if penetrations[max_idx] > worst_penetration:
                    worst_penetration = penetrations[max_idx]
                    pt_idx, pole_idx = max_idx
                    pt = eval_pts[pt_idx]
                    pole = pole_obstacles[pole_idx]
                    v = pt[:2] - pole[:2]
                    v_norm = np.linalg.norm(v)
                    push_dir = v / v_norm if v_norm > 1e-5 else np.array([1.0, 0.0])
                    safe_2d = pole[:2] + push_dir * pole_safety
                    best_fix_pt = np.array([safe_2d[0], safe_2d[1], pt[2]])
                    insert_t = t_eval[pt_idx]

            if best_fix_pt is not None:
                seg_idx = np.clip(np.searchsorted(t_pts, insert_t) - 1, 0, len(waypoints) - 1)
                waypoints.insert(seg_idx + 1, best_fix_pt)
            else:
                break

        # 3. Final Parameterization over global course domain [0, t_total]
        waypoints = np.array(waypoints)
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        total_dist = np.sum(distances)
        
        self._t_total = total_dist / self._target_speed
        
        cum_dist = np.insert(np.cumsum(distances), 0, 0)
        t_domain = (cum_dist / max(total_dist, 1e-5)) * self._t_total
        
        # CubicSpline built over the entire course domain keeps the shape locked globally
        self._des_pos_spline = CubicSpline(t_domain, waypoints, bc_type='clamped')

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        full_ordered_gates = self.known_gates[[0, 1, 2, 3, 1]]
        
        # Advance gate tracking index when within 0.25m of current target gate center
        if self.current_gate_pass_idx < len(full_ordered_gates):
            target_gate = full_ordered_gates[self.current_gate_pass_idx]
            if np.linalg.norm(obs["pos"] - target_gate) < 0.25:
                self.current_gate_pass_idx += 1

        # Check ONLY upcoming target gate for position discovery (> 8cm threshold)
        recalculate = False
        if self.current_gate_pass_idx < len(self.known_gates):
            upcoming_gate_idx = [0, 1, 2, 3, 1][self.current_gate_pass_idx]
            gate_shift = np.linalg.norm(self.known_gates[upcoming_gate_idx] - obs["gates_pos"][upcoming_gate_idx])
            if gate_shift > 0.08:
                recalculate = True

        # Check obstacle position shift
        obstacle_shift = np.max(np.linalg.norm(self.known_obstacles - obs["obstacles_pos"], axis=1)) \
                         if len(self.known_obstacles) > 0 else 0.0
        if obstacle_shift > 0.08:
            recalculate = True

        # Recalculate full global trajectory when a position update occurs
        if recalculate:
            self.known_gates = np.copy(obs["gates_pos"])
            self.known_rpy = R.from_quat(obs["gates_quat"]).as_euler('xyz')
            self.known_obstacles = np.copy(obs["obstacles_pos"])
            
            # Re-run full global spline build (preserves tick time and overall course geometry)
            self._update_spline()
            
        t = min(self._tick / self._freq, self._t_total)

        des_pos = np.array(self._des_pos_spline(t)).reshape(-1)
        des_vel = np.zeros(3)
        des_acc = np.zeros(3)
        
        des_yaw = 0.0
        des_quat = R.from_euler("z", des_yaw).as_quat().reshape(-1)
        des_rpy_rate = np.zeros(3)
        
        return np.concatenate((des_pos, des_vel, des_acc, des_quat, des_rpy_rate), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return False

    def episode_callback(self):
        self._tick = 0
        self.current_gate_pass_idx = 0

    def render_callback(self, sim: Sim):
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
