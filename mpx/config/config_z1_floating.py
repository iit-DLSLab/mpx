import os
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from jaxlie import SE3
from mujoco import mjx
from mujoco.mjx._src import math, smooth

from mpx.costs import Z1GridCost
from mpx.dynamics import ControlLayout, GridDynamicsBackend, RobotSpec, StateLayout
import mpx.utils.mpc_wrapper as base_mpc_wrapper

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/unitree_z1/z1_floating.xml"
grid_ffi_library_path = os.path.abspath(
    os.path.join(
        dir_path,
        "..",
        "..",
        "build",
        "grid",
        "z1_floating",
        "libmpx_grid_z1_floating.so",
    )
)
grid_urdf_path = os.path.abspath(
    os.path.join(
        dir_path,
        "..",
        "..",
        "build",
        "grid",
        "z1_floating",
        "z1_floating_grid.urdf",
    )
)
grid_ffi_prefix = "mpx_grid_z1_floating"
grid_kinematics_ffi_library_path = os.path.abspath(
    os.path.join(dir_path, "..", "..", "build", "grid", "z1_fixed", "libmpx_grid_z1.so")
)
grid_kinematics_urdf_path = os.path.abspath(
    os.path.join(dir_path, "..", "..", "build", "grid", "z1_fixed", "z1_grid.urdf")
)
grid_kinematics_ffi_prefix = "mpx_grid_z1"
grid_reference_fallback = False
cost_kinematics_backend = "grid"
dynamics_backend = "grid"

contact_frame = ["end_effector"]
body_name = ["link06"]
self_collision_frames = [
    "link01_collision",
    "link02_collision",
    "link04_collision",
    "link05_collision",
]

dt = 0.02
N = 12
mpc_frequency = 50.0
solver_mode = "primal_dual"

timer_t = jnp.zeros(len(contact_frame))
duty_factor = 0.0
step_freq = 0.0
step_height = 0.0
use_terrain_estimation = False

q0 = jnp.array([
    0.0,
    0.0,
    0.33,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.68,
    -1.4,
    0.0,
    0.0,
    0.0,
])
ee0 = jnp.array([0.13888362, 0.0, 0.33, 1.0, 0.0, 0.0, 0.0])

n_joints = 6
n_contact = len(contact_frame)
n = 7 + 2 * n_joints
m = 6 + n_joints
u_ref = jnp.zeros(m)
initial_state = jnp.concatenate([q0, jnp.zeros(n_joints)])

Qpos = jnp.diag(jnp.array([0.0, 0.0, 1.0])) * 1e2
Qrot = jnp.diag(jnp.array([1.0, 1.0, 0.0])) * 1e1
Qdpos = jnp.diag(jnp.ones(3)) * 1e0
Qomega = jnp.diag(jnp.array([1e1, 1e1, 1e0]))
Qq = jnp.diag(jnp.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]))
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qee = jnp.eye(6)
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
W = jax.scipy.linalg.block_diag(Qpos, Qrot, Qdpos, Qomega, Qq, Qdq, Qee, Qtau)

max_torque = 20.0
min_torque = -20.0
max_base_velocity = 0.5


def z1_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    del contact_id, body_id, t, parameter

    qpos = x[: 7 + n_joints]
    # qvel = jnp.concatenate([u[-6:], x[-n_joints:]])
    qvel = jnp.concatenate([jnp.zeros(6), x[-n_joints:]])

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = smooth.kinematics(mjx_model, mjx_data)
    mjx_data = smooth.com_pos(mjx_model, mjx_data)
    mjx_data = smooth.crb(mjx_model, mjx_data)
    mjx_data = smooth.factor_m(mjx_model, mjx_data)
    mjx_data = smooth.com_vel(mjx_model, mjx_data)
    mjx_data = smooth.rne(mjx_model, mjx_data)

    mass_matrix = mjx_data.qM[-n_joints:, -n_joints:] + jnp.eye(n_joints) * 0.1
    bias = mjx_data.qfrc_bias[-n_joints:]

    tau = u[:n_joints]

    mass_matrix_cho = jax.scipy.linalg.cho_factor(mass_matrix)
    dq_next = x[7 + n_joints :] + jax.scipy.linalg.cho_solve(
        mass_matrix_cho, tau - bias
    ) * dt
    p_next = x[:3] + u[n_joints : n_joints + 3] * dt
    quat_next = math.quat_integrate(x[3:7], u[-3:], dt)
    q_next = x[7 : 7 + n_joints] + dq_next * dt
    return jnp.concatenate([p_next, quat_next, q_next, dq_next])




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
    return jnp.sum(penalty(margins, alpha=0.5, sigma=0.1, transition_width=0.05))


def z1_obj(model, mjx_model, contact_id, body_id, n_joints, n_contact, N, W, reference, x, u, t, residuals=None):
    del n_contact

    qpos = x[: 7 + n_joints]
    p = x[:3]
    quat = x[3:7]
    q = x[7 : 7 + n_joints]
    dq = x[7 + n_joints :]
    dp = u[n_joints : n_joints + 3]
    omega = u[-3:]
    tau = u[:n_joints]

    # default: compute kinematics here
    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(
        qpos=qpos,
        qvel=jnp.concatenate([jnp.zeros(6), dq]),
    )
    mjx_data = smooth.kinematics(mjx_model, mjx_data)
    ee = mjx_data.geom_xpos[contact_id[0]]

    # ee_quat = mjx_data.xquat[body_id[0]]

    # ee_pose = SE3(wxyz_xyz=jnp.concatenate([ee_quat, ee]))
    q_ref = reference[t, :n_joints]
    ee_ref = reference[t, n_joints : n_joints + 3]
    # ee_quat_ref = reference[t, n_joints + 3 : -n_joints]
    tau_ref = reference[t, -n_joints:]
    # ee_pose = SE3(wxyz_xyz=jnp.concatenate([ee_quat, ee]))
    # ee_pose_ref = SE3(wxyz_xyz=jnp.concatenate([ee_quat_ref, ee_ref]))
    # ee_error = (ee_pose_ref.inverse() @ ee_pose).log()
    ee_error = ee - ee_ref

    base_pose = SE3(wxyz_xyz=jnp.concatenate([quat, p]))
    base_pose_ref = SE3(wxyz_xyz=jnp.concatenate([q0[3:7], q0[:3]]))
    base_error = (base_pose_ref.inverse() @ base_pose).log()

    stage_cost = (
        (p - q0[:3]) @ W[0:3, 0:3] @ (p - q0[:3])
        + base_error[3:6].T @ W[3:6, 3:6] @ base_error[3:6]
        + dp.T @ W[6:9, 6:9] @ dp
        + omega.T @ W[9:12, 9:12] @ omega
        + (q - q_ref).T @ W[12 : 12 + n_joints, 12 : 12 + n_joints] @ (q - q_ref)
        + dq.T
        @ W[
            12 + n_joints : 12 + 2 * n_joints,
            12 + n_joints : 12 + 2 * n_joints,
        ]
        @ dq
        + ee_error.T
        @ W[
            12 + 2 * n_joints : 12 + 2 * n_joints + 3,
            12 + 2 * n_joints : 12 + 2 * n_joints + 3,
        ]
        @ ee_error
        + (tau - tau_ref).T
        @ W[
            12 + 2 * n_joints + 6 : 12 + 3 * n_joints + 6,
            12 + 2 * n_joints + 6 : 12 + 3 * n_joints + 6,
        ]
        @ (tau - tau_ref)
        + joint_limits_penalty(n_joints, q)
    )
    terminal_cost = (
        (p - q0[:3]) @ W[0:3, 0:3] @ (p - q0[:3])
        + base_error[3:6].T @ W[3:6, 3:6] @ base_error[3:6]
        + (q - q_ref).T @ W[12 : 12 + n_joints, 12 : 12 + n_joints] @ (q - q_ref)
        + dq.T
        @ W[
            12 + n_joints : 12 + 2 * n_joints,
            12 + n_joints : 12 + 2 * n_joints,
        ]
        @ dq
        + 1e3
        * ee_error.T
        @ W[
            12 + 2 * n_joints : 12 + 2 * n_joints + 3,
            12 + 2 * n_joints : 12 + 2 * n_joints + 3,
        ]
        @ ee_error
    )
    return jnp.where(t == N, 0.5 * terminal_cost, 0.5 * stage_cost)


def reference_tau(model, mjx_model, n_joints, x, u):
    qpos = x[: 7 + n_joints]
    qvel = jnp.concatenate([u[-6:], x[-n_joints:]])

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = smooth.kinematics(mjx_model, mjx_data)
    mjx_data = smooth.com_pos(mjx_model, mjx_data)
    mjx_data = smooth.crb(mjx_model, mjx_data)
    mjx_data = smooth.factor_m(mjx_model, mjx_data)
    mjx_data = smooth.com_vel(mjx_model, mjx_data)
    mjx_data = smooth.rne(mjx_model, mjx_data)
    return mjx_data.qfrc_bias[-n_joints:]


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
        new_U0 = shift_trajectory(U)
        new_U0 = new_U0.at[:, :n_joints].set(
            jnp.clip(new_U0[:, :n_joints], min_torque, max_torque)
        )
        new_U0 = new_U0.at[:, n_joints:].set(
            jnp.clip(new_U0[:, n_joints:], -max_base_velocity, max_base_velocity)
        )
        return (
            new_U0,
            shift_trajectory(X),
            shift_trajectory(V),
            U[0],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(V_prev),
            U_prev[u_fallback_idx],
        )

    return jax.lax.cond(jnp.isnan(U[0, 0]), unsafe_update, safe_update)


class MPCWrapper:
    def __init__(self, config, limited_memory=False):
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = max(1, int(round(1 / (config.dt * config.mpc_frequency))))
        self.qpos_slice = slice(0, 7 + config.n_joints)
        self.qvel_slice = slice(7 + config.n_joints, 7 + 2 * config.n_joints)

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

        if getattr(config, "cost_kinematics_backend", "mjx") == "grid":
            arm_kinematics = GridDynamicsBackend(
                RobotSpec(
                    name="z1",
                    urdf_path=getattr(config, "grid_kinematics_urdf_path", None),
                    nq=config.n_joints,
                    nv=config.n_joints,
                    nu=config.n_joints,
                    floating_base=False,
                    actuated_dofs=tuple(range(config.n_joints)),
                    dt=config.dt,
                    state_layout=StateLayout(
                        q=slice(0, config.n_joints),
                        v=slice(config.n_joints, 2 * config.n_joints),
                    ),
                    control_layout=ControlLayout(
                        actuator_torques=slice(0, config.n_joints),
                    ),
                ),
                model=None,
                mjx_model=None,
                ffi_library_path=getattr(config, "grid_kinematics_ffi_library_path", None),
                ffi_prefix=getattr(config, "grid_kinematics_ffi_prefix", "mpx_grid_z1"),
                reference_fallback=getattr(config, "grid_reference_fallback", False),
            )
            self.cost = Z1GridCost(
                arm_kinematics,
                config.n_joints,
                config.N,
                floating_base=True,
                q0=config.q0,
                local_kinematics_backend=arm_kinematics,
                base_to_arm_offset=jnp.array([0.22, 0.0, 0.055]),
            )
        else:
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
            # attach precompute hook that returns residuals and jacobians per (x,u,t)

        self.hessian_approx = config.hessian_approx
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
            self.dynamics = dynamics_fn
        elif backend_name == "grid":
            self.dynamics = GridDynamicsBackend(
                RobotSpec(
                    name="z1_floating",
                    urdf_path=getattr(config, "grid_urdf_path", None),
                    nq=7 + config.n_joints,
                    nv=6 + config.n_joints,
                    nu=config.n_joints,
                    floating_base=True,
                    actuated_dofs=tuple(range(6, 6 + config.n_joints)),
                    contact_frames=(),
                    contact_body_names=(),
                    dt=config.dt,
                    state_layout=StateLayout(
                        q=slice(0, 7 + config.n_joints),
                        v=slice(7 + config.n_joints, 7 + 2 * config.n_joints),
                    ),
                    control_layout=ControlLayout(
                        actuator_torques=slice(0, config.n_joints),
                        contact_forces=slice(config.n_joints, config.m),
                    ),
                ),
                model=self.model,
                mjx_model=self.mjx_model,
                ffi_library_path=getattr(config, "grid_ffi_library_path", None),
                ffi_prefix=getattr(config, "grid_ffi_prefix", "mpx_grid_z1_floating"),
                reference_fallback=getattr(config, "grid_reference_fallback", False),
            )
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
        return jnp.concatenate([qpos[: 7 + self.config.n_joints], qvel[-self.config.n_joints :]])

    def reset(self, data, qpos, qvel=None, foot=None):
        del foot
        if qvel is None:
            qvel = jnp.zeros(self.config.n_joints)
        initial_state = self.state_from_measurement(qpos, qvel)
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(initial_state, (self.config.N + 1, 1)),
            V0=self.initial_V0,
        )

    def build_reference(self, data, target_ee):
        target_ee = jnp.ravel(jnp.asarray(target_ee))
        reference_k = jnp.concatenate(
            [self.config.q0[7:], target_ee, jnp.zeros(self.config.n_joints)]
        )
        reference = jnp.tile(reference_k, (self.config.N + 1, 1))
        padded_u = jnp.pad(data.U0, ((0, 1), (0, 0)))
        tau_ref = self._reference_tau(data.X0, padded_u)
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
        U0, X0, V0, control = self._update_warm_start(
            x0,
            data.U0,
            data.V0,
            X,
            U,
            V,
        )
        data = data.replace(X0=X0, U0=U0, V0=V0)
        return data, control

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
