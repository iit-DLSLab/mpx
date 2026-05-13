import os
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mpx.dynamics import (
    ControlLayout,
    GridDynamicsBackend,
    MJXDynamicsBackend,
    RobotSpec,
    StateLayout,
)
from mpx.costs import Z1GridCost
import mpx.utils.mpc_wrapper as base_mpc_wrapper
from mujoco.mjx._src import math, smooth

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/unitree_z1/z1.xml"
grid_ffi_library_path = os.path.abspath(
    os.path.join(dir_path, "..", "..", "build", "grid", "z1_fixed", "libmpx_grid_z1.so")
)
grid_urdf_path = os.path.abspath(
    os.path.join(dir_path, "..", "..", "build", "grid", "z1_fixed", "z1_grid.urdf")
)
grid_ffi_prefix = "mpx_grid_z1"
grid_reference_fallback = False
dynamics_backend = "grid"

contact_frame = ["end_effector"]
body_name = ["link06"]

dt = 0.02
N = 12
mpc_frequency = 50
solver_mode = "primal_dual"

timer_t = jnp.zeros(len(contact_frame))
duty_factor = 0.0
step_freq = 0.0
step_height = 0.0
use_terrain_estimation = False

q0 = jnp.array([0.0, 0.785, -0.261, -0.523, 0.0, 0.0])
ee0 = jnp.array([0.13888362, 0.0, 0.29099588, 1.0, 0.0, 0.0, 0.0])

n_joints = 6
n_contact = len(contact_frame)
n = 2 * n_joints
m = n_joints
u_ref = jnp.zeros(m)
initial_state = jnp.concatenate([q0, jnp.zeros(n_joints)])

Qq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qee = jnp.diag(jnp.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
W = jax.scipy.linalg.block_diag(Qq, Qdq, Qee, Qtau)

max_torque = 30.0
min_torque = -30.0


def z1_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    del body_id, t, parameter

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=x[:n_joints], qvel=x[n_joints : 2 * n_joints])
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    mass_matrix = mjx_data.qM + jnp.eye(n_joints) * dt * 0.1
    bias = mjx_data.qfrc_bias
    tau = u - 0.1 * x[n_joints : 2 * n_joints]

    mass_matrix_cho = jax.scipy.linalg.cho_factor(mass_matrix)
    dq_next = x[n_joints : 2 * n_joints] + jax.scipy.linalg.cho_solve(
        mass_matrix_cho, tau - bias
    ) * dt
    q_next = x[:n_joints] + dq_next * dt
    return jnp.concatenate([q_next, dq_next])


def penalty(constraint, alpha=0.1, sigma=5.0, transition_width=0.2):
    def safe_log(x):
        return jnp.log(jnp.clip(x, 1e-10, 1e6))

    quadratic_barrier = alpha / 2 * (
        jnp.square((constraint - 2 * sigma) / sigma) - jnp.ones_like(constraint)
    )
    log_barrier = -alpha * safe_log(constraint)
    combined_barrier = quadratic_barrier + log_barrier
    weight = 0.5 * (1 + jnp.tanh((constraint - sigma) / transition_width))
    smooth_result = weight * log_barrier + (1 - weight) * combined_barrier
    return jnp.clip(smooth_result, 0.0, 1e8)


def joint_limits_penalty(n_joints, q):
    joint_limits = jnp.array([
        2.61799,
        2.61799,
        2.96706,
        0.0,
        0.0,
        2.87979,
        1.51844,
        1.51844,
        1.3439,
        1.3439,
        2.79253,
        2.79253,
    ])
    margins = (
        jnp.kron(jnp.eye(n_joints), jnp.array([-1.0, 1.0])).T @ q
        + joint_limits
        + jnp.ones_like(joint_limits) * 1e-2
    )
    return jnp.sum(penalty(margins, alpha=1.0, sigma=0.01, transition_width=0.05))


def z1_obj(model, mjx_model, contact_id, body_id, n_joints, n_contact, N, W, reference, x, u, t):
    del n_contact

    q = x[:n_joints]
    dq = x[n_joints : 2 * n_joints]
    tau = u[:n_joints]

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=q, qvel=dq)
    mjx_data = smooth.kinematics(mjx_model, mjx_data)

    ee = mjx_data.geom_xpos[contact_id[0]]
    # ee_quat = mjx_data.xquat[body_id[0]]
    q_ref = reference[t, :n_joints]
    ee_ref = reference[t, n_joints : n_joints + 3]
    # ee_quat_ref = reference[t, n_joints + 3 : n_joints + 7]

    # ee_pose = SE3(wxyz_xyz=jnp.concatenate([ee_quat, ee]))
    # ee_pose_ref = SE3(wxyz_xyz=jnp.concatenate([ee_quat_ref, ee_ref]))
    # ee_error = (ee_pose_ref.inverse() @ ee_pose).log()
    ee_error = ee - ee_ref
    tau_ref = reference[t, -n_joints :]

    stage_cost = (
        (q - q_ref).T @ W[:n_joints, :n_joints] @ (q - q_ref)
        + dq.T @ W[n_joints : 2 * n_joints, n_joints : 2 * n_joints] @ dq
        + ee_error.T
        @ W[2 * n_joints : 2 * n_joints + 3, 2 * n_joints : 2 * n_joints + 3]
        @ ee_error 
        + (tau - tau_ref).T
        @ W[
            2 * n_joints + 6 : 3 * n_joints + 6,
            2 * n_joints + 6 : 3 * n_joints + 6,
        ]
        @ (tau - tau_ref)
        + joint_limits_penalty(n_joints, q)
    )
    terminal_cost = (
        (q - q_ref).T @ W[:n_joints, :n_joints] @ (q - q_ref)
        + dq.T @ W[n_joints : 2 * n_joints, n_joints : 2 * n_joints] @ dq
        + 1e2
        * ee_error.T
        @ W[2 * n_joints : 2 * n_joints + 3, 2 * n_joints : 2 * n_joints + 3]
        @ ee_error
    )
    return jnp.where(t == N, 0.5 * terminal_cost, 0.5 * stage_cost)


def _compute_ee_residuals(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t):
    qpos = x[:n_joints]
    qvel = x[n_joints : 2 * n_joints]
    md = mjx.make_data(model)
    md = md.replace(qpos=qpos, qvel=qvel)
    md = smooth.kinematics(mjx_model, md)
    ee = md.geom_xpos[contact_id[0]]

    def _ee_forward(xx, uu):
        md2 = mjx.make_data(model)
        md2 = md2.replace(qpos=xx[:n_joints], qvel=xx[n_joints : 2 * n_joints])
        md2 = smooth.kinematics(mjx_model, md2)
        return md2.geom_xpos[contact_id[0]]

    ee_jac_x = jax.jacobian(_ee_forward, argnums=0)(x, u)
    ee_jac_u = jax.jacobian(_ee_forward, argnums=1)(x, u)
    return {"ee": ee, "ee_jac_x": ee_jac_x, "ee_jac_u": ee_jac_u}

def reference_tau(model, mjx_model, n_joints, x):
    qpos = x[: n_joints]
    qvel =  x[-n_joints:]

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = smooth.kinematics(mjx_model, mjx_data)
    mjx_data = smooth.com_pos(mjx_model, mjx_data)
    mjx_data = smooth.crb(mjx_model, mjx_data)
    mjx_data = smooth.factor_m(mjx_model, mjx_data)
    mjx_data = smooth.com_vel(mjx_model, mjx_data)
    mjx_data = smooth.rne(mjx_model, mjx_data)
    return mjx_data.qfrc_bias

cost = z1_obj
dynamics = z1_dynamics
hessian_approx = None


@partial(jax.jit, static_argnums=(0, 1))
def _update_warm_start(horizon, shift, u_ref, x0, U_prev, V_prev, X, U, V):
    u_fallback_idx = 1 if horizon > 1 else 0

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate([trajectory[shift:], tail], axis=0)

    def safe_update():
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            U[0, :n_joints],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(V_prev),
            U_prev[u_fallback_idx, :n_joints],
        )

    return jax.lax.cond(jnp.isnan(U[0, 0]), unsafe_update, safe_update)


class MPCWrapper:
    def __init__(self, config, limited_memory=False):
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = max(1, int(round(1 / (config.dt * config.mpc_frequency))))
        self.qpos_slice = slice(0, config.n_joints)
        self.qvel_slice = slice(config.n_joints, 2 * config.n_joints)

        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.mjx_model = mjx.put_model(self.model)
        self.contact_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]
        self.body_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.body_name
        ]

        self.hessian_approx = None
        dynamics_fn = partial(
            config.dynamics,
            self.model,
            self.mjx_model,
            self.contact_id,
            self.body_id,
            config.n_joints,
            config.dt,
        )
        backend_name = getattr(config, "dynamics_backend", "mjx")
        if backend_name == "mjx":
            self.dynamics = MJXDynamicsBackend(dynamics_fn)
            self.cost = partial(
                config.cost,
                self.model,
                self.mjx_model,
                self.contact_id,
                self.body_id,
                config.n_joints,
                config.n_contact,
                config.N,
            )
            def _precompute_fn(x, u, t):
                return _compute_ee_residuals(self.model, self.mjx_model, self.contact_id, self.body_id, config.n_joints, config.dt, x, u, t)

            self.cost.precompute_residuals = _precompute_fn
        elif backend_name == "grid":
            self.dynamics = GridDynamicsBackend(
                RobotSpec(
                    name="z1",
                    urdf_path=getattr(config, "grid_urdf_path", None),
                    nq=config.n_joints,
                    nv=config.n_joints,
                    nu=config.n_joints,
                    floating_base=False,
                    actuated_dofs=tuple(range(config.n_joints)),
                    contact_frames=(),
                    contact_body_names=(),
                    dt=config.dt,
                    state_layout=StateLayout(
                        q=slice(0, config.n_joints),
                        v=slice(config.n_joints, 2 * config.n_joints),
                    ),
                    control_layout=ControlLayout(
                        actuator_torques=slice(0, config.n_joints),
                    ),
                ),
                model=self.model,
                mjx_model=self.mjx_model,
                ffi_library_path=getattr(config, "grid_ffi_library_path", None),
                ffi_prefix=getattr(config, "grid_ffi_prefix", "mpx_grid_z1"),
                reference_fallback=getattr(config, "grid_reference_fallback", False),
            )
            self.cost = Z1GridCost(self.dynamics, config.n_joints, config.N)
        else:
            raise ValueError(f"Unsupported dynamics_backend: {backend_name}")

        self.initial_state = jnp.asarray(config.initial_state)
        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))

        _, solve = base_mpc_wrapper.build_solver_step(
            config,
            self.cost,
            self.dynamics,
            self.hessian_approx,
            limited_memory,
        )
        self._solve = jax.jit(solve)
        self._update_warm_start = partial(
            _update_warm_start,
            config.N,
            self.shift,
            config.u_ref,
        )
        self._reference_tau = jax.vmap(
            partial(reference_tau, self.model, self.mjx_model, config.n_joints)
        )

    def make_data(self):
        return base_mpc_wrapper.MPCData(
            dt=self.config.dt,
            duty_factor=0.0,
            step_freq=0.0,
            step_height=0.0,
            contact_time=jnp.zeros(self.config.n_contact),
            liftoff=jnp.zeros(3 * self.config.n_contact),
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            W=self.config.W,
        )

    def state_from_measurement(self, qpos, qvel):
        qpos = jnp.ravel(jnp.asarray(qpos))
        qvel = jnp.ravel(jnp.asarray(qvel))
        return jnp.concatenate([qpos[: self.config.n_joints], qvel[: self.config.n_joints]])

    def reset(self, data, qpos, qvel, foot=None):
        del foot
        initial_state = self.state_from_measurement(qpos, qvel)
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(initial_state, (self.config.N + 1, 1)),
            V0=self.initial_V0,
        )

    def build_reference(self, data, target_ee):
        target_ee = jnp.ravel(jnp.asarray(target_ee))
        reference_k = jnp.concatenate(
            [self.config.q0, target_ee, jnp.zeros(self.config.n_joints)]
        )
        reference = jnp.tile(reference_k, (self.config.N + 1, 1))
        tau_ref = self._reference_tau(data.X0)
        return reference.at[:, -self.config.n_joints :].set(tau_ref)

    def run(self, data, x0, input, contact=None):
        del contact
        reference = self.build_reference(data, input)
        parameter = jnp.zeros((self.config.N + 1, self.config.n_contact))
        X, U, V = self._solve(
            reference,
            parameter,
            data.W,
            x0,
            data.X0,
            data.U0,
            data.V0,
        )
        U0, X0, V0, tau = self._update_warm_start(x0, data.U0, data.V0, X, U, V)
        data = data.replace(X0=X0, U0=U0, V0=V0)
        return data, tau

    def limit_reference(self, reference_ee, ee_pose):
        reference_ee = jnp.asarray(reference_ee)
        ee_pose = jnp.asarray(ee_pose)
        ref_delta = reference_ee[:3] - ee_pose[:3]
        ref_distance = jnp.linalg.norm(ref_delta)
        ref_direction = ref_delta / jnp.maximum(ref_distance, 1e-6)
        max_step_lin = 0.8 * self.config.dt * (self.config.N + 1)
        max_step_ang = 1.0 * self.config.dt * (self.config.N + 1)
        limited_position = ee_pose[:3] + jnp.clip(ref_distance, 0.0, max_step_lin) * ref_direction

        def normalize_quat(quat):
            return quat / jnp.maximum(jnp.linalg.norm(quat), 1e-6)

        def quat_slerp_limited(q0, q1, max_angle):
            q0 = normalize_quat(q0)
            q1 = normalize_quat(q1)
            dot = jnp.dot(q0, q1)
            q1 = jnp.where(dot < 0.0, -q1, q1)
            dot = jnp.clip(jnp.abs(dot), 0.0, 1.0)
            theta0 = jnp.arccos(dot)
            max_t = jnp.where(theta0 > 1e-8, jnp.clip(max_angle / theta0, 0.0, 1.0), 1.0)

            def lerp():
                return normalize_quat((1.0 - max_t) * q0 + max_t * q1)

            def slerp():
                sin_theta0 = jnp.sin(theta0)
                theta = theta0 * max_t
                sin_theta = jnp.sin(theta)
                s0 = jnp.cos(theta) - dot * sin_theta / jnp.maximum(sin_theta0, 1e-6)
                s1 = sin_theta / jnp.maximum(sin_theta0, 1e-6)
                return s0 * q0 + s1 * q1

            return jnp.where(dot > 0.9995, lerp(), slerp())

        limited_quat = quat_slerp_limited(ee_pose[3:], reference_ee[3:], max_step_ang)
        return jnp.concatenate([limited_position, limited_quat])
