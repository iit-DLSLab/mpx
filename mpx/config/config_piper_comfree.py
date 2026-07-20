"""Full-dynamics complementarity-free MPC for Piper pushing a free box.

The transcription follows the full-dynamics extension in ComFree MPC.  The
next generalized velocity and arm torque are stage decision variables.  The
rollout integrates configuration, while an equality constraint enforces the
implicit momentum balance with smooth spring-damper contact forces.
"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math, smooth
from mujoco.mjx._src.dataclasses import PyTreeNode
import numpy as np

from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers

# ----------------------------- Model and task ----------------------------- #

model_path = str(Path(__file__).resolve().parents[1] / "data" / "piper_l" / "scene_cube.xml")
contact_frame = ["end_effector"]

model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.005
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

nq = model.nq
nv = model.nv
nx = nq + nv

arm_dof = 6
n_joints = arm_dof
n = nx
box_qpos_adr = 6
box_dof_adr = 6
equality_dim = nv
inequality_dim = 2 * arm_dof + 2

ee_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, contact_frame[0])
box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_body")
box_geom_id = int(model.body_geomadr[box_body_id])
floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

ee_radius = float(model.geom_size[ee_geom_id, 0])
box_half_size = jnp.asarray(model.geom_size[box_geom_id, :3])
box_corner_signs = jnp.asarray(
    [
        [sx, sy, sz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
)
friction_direction_count = 4

dt = 0.01
forward_dt = 0.005
N = 20
mpc_frequency = 100
equality_num_alpha = 10
regularization = jnp.array(1e-3)
barrier_parameter = jnp.array(1e-4, dtype=jnp.float32)
multiplier_regularization = jnp.array(1e-8, dtype=jnp.float32)
box_sdf_smoothing = 0.005

# u = [v_next, tau_arm].  v_next makes the full dynamics an implicit equality.
m = nv + arm_dof
vnext_slice = slice(0, nv)
tau_slice = slice(nv, nv + arm_dof)
min_torque = jnp.asarray(model.actuator_ctrlrange[:, 0])
max_torque = jnp.asarray(model.actuator_ctrlrange[:, 1])
torque_limit = jnp.minimum(
    jnp.maximum(jnp.abs(min_torque), jnp.abs(max_torque)),
    jnp.array([3.0, 5.0, 5.0, 2.0, 2.0, 2.0]),
)

initial_box_x = 0.2342375717


def _initial_configuration():
    return np.array(
        [
            0.2507536545,
            1.22739269,
            -0.180199871,
            1.56976470e-08,
            0.0305185308,
            7.11176140e-09,
            initial_box_x,
            0.0,
            0.04,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )


q0 = jnp.asarray(_initial_configuration())
mujoco.mj_resetData(model, data)
data.qpos[:] = q0
mujoco.mj_forward(model, data)

x0 = jnp.concatenate((jnp.asarray(q0), jnp.zeros(nv)))
initial_state = x0
arm_nominal = jnp.asarray(q0[:arm_dof])
W = {
    "Qgoal": jnp.eye(2) * 50.0,
    "Qq": jnp.eye(arm_dof) * 0.05,
    "Qv": jnp.eye(nv) * 0.1,
    "Qacc": jnp.eye(nv) * 1e-4,
    "Qtau": jnp.eye(arm_dof) * 1e-3,
    "Qside": jnp.array(20.0),
    "Qgoal_final": jnp.eye(2) * 1000.0,
    "Qcontact_final": jnp.eye(3) * 200.0,
    "Qq_final": jnp.eye(arm_dof) * 0.1,
    "Qv_final": jnp.eye(nv) * 0.1,
}

if not model.jnt_limited[:arm_dof].all():
    raise ValueError("All controlled Piper joints must define position limits")
joint_position_min = jnp.asarray(model.jnt_range[:arm_dof, 0])
joint_position_max = jnp.asarray(model.jnt_range[:arm_dof, 1])
base_exclusion_radius = 0.2


# ------------------------ Differentiable box distance --------------------- #

@jax.custom_jvp
def _box_signed_distance(local_point, half_size):
    """Exact point-box signed distance with a smooth boundary derivative."""
    d = jnp.abs(local_point) - half_size
    outside = jnp.linalg.norm(jnp.maximum(d, 0.0))
    inside = jnp.minimum(jnp.max(d), 0.0)
    return outside + inside


def _smooth_box_normal(local_point, smoothing=box_sdf_smoothing):
    """Return a smooth transition between adjacent box face normals."""

    smoothing = jnp.maximum(jnp.asarray(smoothing, dtype=local_point.dtype), 1e-12)
    smooth_absolute_position = jnp.sqrt(local_point**2 + 1e-12)
    axis_distances = smooth_absolute_position - box_half_size
    positive_distances = smoothing * jax.nn.softplus(axis_distances / smoothing)
    outside_distance = jnp.sqrt(jnp.dot(positive_distances, positive_distances) + 1e-12)
    maximum_axis_distance = smoothing * jax.scipy.special.logsumexp(
        axis_distances / smoothing
    )
    distance_gradient = (
        positive_distances
        / outside_distance
        * jax.nn.sigmoid(axis_distances / smoothing)
        + jax.nn.sigmoid(-maximum_axis_distance / smoothing)
        * jax.nn.softmax(axis_distances / smoothing)
    )
    local_normal = distance_gradient * local_point / smooth_absolute_position
    return local_normal / jnp.sqrt(jnp.dot(local_normal, local_normal) + 1e-12)


@_box_signed_distance.defjvp
def _box_signed_distance_jvp(primals, tangents):
    local_point, half_size = primals
    point_dot, half_dot = tangents
    grad_point = _smooth_box_normal(local_point)
    grad_half = -jnp.abs(grad_point)
    primal = _box_signed_distance(local_point, half_size)
    tangent = jnp.dot(grad_point, point_dot) + jnp.dot(grad_half, half_dot)
    return primal, tangent


def _box_distance_and_normal(point, center, rotation):
    local_point = rotation.T @ (point - center)
    phi = _box_signed_distance(local_point, box_half_size)
    local_normal = _smooth_box_normal(local_point)
    normal = rotation @ local_normal
    normal /= jnp.maximum(jnp.linalg.norm(normal), 1e-12)
    return phi, normal


def _box_corner_positions(center, rotation):
    local_corners = box_corner_signs * box_half_size
    return center + local_corners @ rotation.T


def _tangent_directions(normal):
    """Return four unit directions spanning the plane normal to ``normal``."""
    sign = jnp.where(normal[2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + normal[2])
    b = normal[0] * normal[1] * a
    tangent_1 = jnp.array(
        [1.0 + sign * normal[0] ** 2 * a, sign * b, -sign * normal[0]]
    )
    tangent_2 = jnp.array([b, sign + normal[1] ** 2 * a, -normal[1]])
    return jnp.stack((tangent_1, -tangent_1, tangent_2, -tangent_2))


def _friction_pyramid(relative_jacobian, normal, phi, friction):
    tangents = _tangent_directions(normal)
    force_directions = normal[None, :] - friction * tangents
    rows = force_directions @ relative_jacobian.T
    return rows, jnp.full(friction_direction_count, phi), force_directions


def _contact_kinematics(mjx_data):
    """Build polyhedral contact Jacobians for gripper-box and box-floor."""
    ee_center = mjx_data.geom_xpos[ee_geom_id]
    box_center = mjx_data.geom_xpos[box_geom_id]
    box_rotation = mjx_data.geom_xmat[box_geom_id].reshape(3, 3)

    box_phi, box_normal = _box_distance_and_normal(
        ee_center, box_center, box_rotation
    )
    arm_phi = box_phi - ee_radius
    arm_point = ee_center - ee_radius * box_normal

    ee_jacobian, _ = mjx.jac(
        mjx_model, mjx_data, arm_point, model.geom_bodyid[ee_geom_id]
    )
    box_jacobian, _ = mjx.jac(
        mjx_model, mjx_data, arm_point, box_body_id
    )
    arm_rows, arm_phis, arm_directions = _friction_pyramid(
        ee_jacobian - box_jacobian, box_normal, arm_phi, friction=0.7
    )

    floor_position = mjx_data.geom_xpos[floor_geom_id]
    floor_rotation = mjx_data.geom_xmat[floor_geom_id].reshape(3, 3)
    floor_normal = floor_rotation[:, 2]
    floor_rows = []
    floor_phis = []
    floor_directions = []
    for corner in _box_corner_positions(box_center, box_rotation):
        corner_jacobian, _ = mjx.jac(
            mjx_model, mjx_data, corner, box_body_id
        )
        rows, phis, directions = _friction_pyramid(
            corner_jacobian,
            floor_normal,
            jnp.dot(corner - floor_position, floor_normal),
            friction=0.5,
        )
        floor_rows.append(rows)
        floor_phis.append(phis)
        floor_directions.append(directions)

    contact_rows = jnp.concatenate((arm_rows, *floor_rows), axis=0)
    contact_phis = jnp.concatenate((arm_phis, *floor_phis))
    contact_directions = jnp.concatenate(
        (arm_directions, *floor_directions), axis=0
    )
    floor_direction_count = friction_direction_count * len(box_corner_signs)
    stiffness = jnp.concatenate(
        (
            jnp.full(friction_direction_count, 35.0),
            jnp.full(floor_direction_count, 5.0),
        )
    )
    damping = jnp.concatenate(
        (
            jnp.full(friction_direction_count, 0.8),
            jnp.full(floor_direction_count, 0.08),
        )
    )
    return contact_rows, contact_phis, contact_directions, stiffness, damping


def _smooth_positive(value, sharpness):
    return jax.nn.softplus(sharpness * value) / sharpness


def _contact_force_terms(mjx_data, velocity, integration_dt=None):
    rows, phi, directions, stiffness, damping = _contact_kinematics(mjx_data)
    contact_velocity = rows @ velocity
    predicted_gap = phi
    if integration_dt is not None:
        predicted_gap = predicted_gap + integration_dt * contact_velocity
    sharpness = jnp.concatenate(
        (
            jnp.full(friction_direction_count, 100.0),
            jnp.full(
                friction_direction_count * len(box_corner_signs), 20.0
            ),
        )
    )
    beta = _smooth_positive(
        -stiffness * predicted_gap - damping * contact_velocity, sharpness
    )
    return rows, phi, directions, beta


def _contact_force(mjx_data, velocity, integration_dt=None):
    """ComFree spring-damper force, corresponding to paper Eq. (24)."""
    rows, _, _, beta = _contact_force_terms(
        mjx_data, velocity, integration_dt
    )
    return rows.T @ beta


class ContactForceInfo(PyTreeNode):
    arm_force_on_box: jnp.ndarray
    ground_force_on_box: jnp.ndarray
    ground_corner_forces: jnp.ndarray
    arm_contact_position: jnp.ndarray
    ground_contact_positions: jnp.ndarray
    arm_gap: jnp.ndarray
    ground_gaps: jnp.ndarray
    beta: jnp.ndarray


def _contact_force_info_from_terms(state_data, phi, directions, beta):
    ee_center = state_data.geom_xpos[ee_geom_id]
    box_center = state_data.geom_xpos[box_geom_id]
    box_rotation = state_data.geom_xmat[box_geom_id].reshape(3, 3)
    arm_normal = _box_distance_and_normal(ee_center, box_center, box_rotation)[1]
    arm_contact_position = ee_center - ee_radius * arm_normal
    ground_contact_positions = _box_corner_positions(box_center, box_rotation)
    arm_force_on_box = -jnp.sum(
        beta[:friction_direction_count, None]
        * directions[:friction_direction_count],
        axis=0,
    )
    ground_corner_forces = jnp.sum(
        beta[friction_direction_count:].reshape(
            len(box_corner_signs), friction_direction_count, 1
        )
        * directions[friction_direction_count:].reshape(
            len(box_corner_signs), friction_direction_count, 3
        ),
        axis=1,
    )
    return ContactForceInfo(
        arm_force_on_box=arm_force_on_box,
        ground_force_on_box=jnp.sum(ground_corner_forces, axis=0),
        ground_corner_forces=ground_corner_forces,
        arm_contact_position=arm_contact_position,
        ground_contact_positions=ground_contact_positions,
        arm_gap=phi[0],
        ground_gaps=phi[
            friction_direction_count::friction_direction_count
        ],
        beta=beta,
    )


# ---------------------------- OCP transcription --------------------------- #

def _split_state(state):
    return state[:nq], state[nq:]


def _state_data(q, v):
    state_data = mjx.make_data(mjx_model).replace(qpos=q, qvel=v)
    state_data = mjx.fwd_position(mjx_model, state_data)
    return mjx.fwd_velocity(mjx_model, state_data)


def _integrate_configuration(q, velocity_next, integration_dt=dt):
    return jnp.concatenate(
        (
            q[:arm_dof] + integration_dt * velocity_next[:arm_dof],
            q[box_qpos_adr : box_qpos_adr + 3]
            + integration_dt * velocity_next[box_dof_adr : box_dof_adr + 3],
            math.quat_integrate(
                q[box_qpos_adr + 3 : box_qpos_adr + 7],
                velocity_next[box_dof_adr + 3 : box_dof_adr + 6],
                integration_dt,
            ),
        )
    )


def dynamics(state, control, t, parameter):
    del t, parameter
    q, _ = _split_state(state)
    velocity_next = control[vnext_slice]
    q_next = _integrate_configuration(q, velocity_next)
    return jnp.concatenate((q_next, velocity_next))


def _applied_torque(torque_command):
    return torque_limit * jnp.tanh(torque_command / torque_limit)


def _momentum_residual(state, torque, velocity_next):
    q, velocity = _split_state(state)
    q_next = _integrate_configuration(q, velocity_next)
    endpoint_data = _state_data(q_next, velocity_next)
    generalized_actuation = jnp.zeros(nv).at[:arm_dof].set(
        _applied_torque(torque)
    )
    contact_force = _contact_force(endpoint_data, velocity_next)
    return (
        endpoint_data.qM @ ((velocity_next - velocity) / dt)
        + endpoint_data.qfrc_bias
        - endpoint_data.qfrc_passive
        - generalized_actuation
        - contact_force
    )


def equality(state, control, t, parameter):
    """Implicit full-system momentum balance at the end of a stage."""
    del t, parameter
    return _momentum_residual(
        state, control[tau_slice], control[vnext_slice]
    )


def inequality(state, control, t, parameter):
    """State limits on each stage successor, including the terminal state."""
    next_state = dynamics(state, control, t, parameter)
    q, velocity = _split_state(next_state)
    robot_q = q[:arm_dof]
    state_data = smooth.kinematics(
        mjx_model,
        mjx.make_data(mjx_model).replace(qpos=q, qvel=velocity),
    )
    box_position = state_data.geom_xpos[box_geom_id]
    end_effector_position = state_data.geom_xpos[ee_geom_id]
    return jnp.concatenate(
        (
            jnp.array(
                [
                    base_exclusion_radius**2
                    - jnp.dot(box_position[:2], box_position[:2]),
                ]
            ),
        )
    )


class ForwardDynamicsInfo(PyTreeNode):
    acceleration: jnp.ndarray
    momentum_residual: jnp.ndarray
    contact: ContactForceInfo


@jax.jit
def forward_dynamics(state, torque):
    """Step MJX full dynamics with semi-implicit Euler integration."""
    q, velocity = _split_state(state)
    torque = jnp.asarray(torque, dtype=state.dtype)
    current_data = _state_data(q, velocity)
    generalized_actuation = jnp.zeros(nv).at[:arm_dof].set(
        _applied_torque(torque)
    )
    free_acceleration = jnp.linalg.solve(
        current_data.qM,
        generalized_actuation
        + current_data.qfrc_passive
        - current_data.qfrc_bias,
    )
    free_velocity = velocity + forward_dt * free_acceleration
    rows, phi, directions, beta = _contact_force_terms(
        current_data, free_velocity, forward_dt
    )
    generalized_contact_force = rows.T @ beta
    acceleration = jnp.linalg.solve(
        current_data.qM,
        generalized_actuation
        + generalized_contact_force
        + current_data.qfrc_passive
        - current_data.qfrc_bias,
    )
    velocity_next = velocity + forward_dt * acceleration
    q_next = _integrate_configuration(q, velocity_next, forward_dt)
    next_state = jnp.concatenate((q_next, velocity_next))
    residual = (
        current_data.qM @ acceleration
        + current_data.qfrc_bias
        - current_data.qfrc_passive
        - generalized_actuation
        - generalized_contact_force
    )
    info = ForwardDynamicsInfo(
        acceleration=acceleration,
        momentum_residual=residual,
        contact=_contact_force_info_from_terms(
            current_data,
            phi,
            directions,
            beta,
        ),
    )
    return next_state, info


def _task_positions(state):
    q, v = _split_state(state)
    state_data = smooth.kinematics(
        mjx_model, mjx.make_data(mjx_model).replace(qpos=q, qvel=v)
    )
    ee_position = state_data.geom_xpos[ee_geom_id]
    box_position = state_data.geom_xpos[box_geom_id]
    return ee_position, box_position


def cost(weights, reference, state, control, t):
    q, velocity = _split_state(state)
    velocity_next = control[vnext_slice]
    torque = control[tau_slice]
    ee_position, box_position = _task_positions(state)
    goal = reference["goal"][t]

    goal_error = box_position[:2] - goal
    contact_error = ee_position - box_position
    posture_error = q[:arm_dof] - arm_nominal
    acceleration = (velocity_next - velocity) / dt
    goal_direction = goal[:2] - box_position[:2]
    goal_direction = goal_direction / jnp.maximum(jnp.linalg.norm(goal_direction), 1e-4)
    side_projection = jnp.dot(ee_position[:2] - box_position[:2], goal_direction)

    stage_cost = (
        goal_error.T @ weights["Qgoal"] @ goal_error
        + posture_error.T @ weights["Qq"] @ posture_error
        + velocity.T @ weights["Qv"] @ velocity
        + acceleration.T @ weights["Qacc"] @ acceleration
        + torque.T @ weights["Qtau"] @ torque
        + weights["Qside"] * jax.nn.relu(side_projection + 0.025) ** 2
    )
    terminal_cost = (
        goal_error.T @ weights["Qgoal_final"] @ goal_error
        + contact_error.T @ weights["Qcontact_final"] @ contact_error
        + posture_error.T @ weights["Qq_final"] @ posture_error
        + velocity.T @ weights["Qv_final"] @ velocity
    )
    return jnp.where(t == N, terminal_cost, stage_cost)


mujoco.mj_forward(model, data)
initial_bias = jnp.asarray(data.qfrc_bias[:arm_dof])
initial_torque_command = torque_limit * jnp.arctanh(
    jnp.clip(initial_bias / torque_limit, -0.99, 0.99)
)
u_ref = jnp.concatenate((jnp.zeros(nv), initial_torque_command))


class ComFreeMPCData(PyTreeNode):
    dt: float
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    Veq0: jnp.ndarray
    Vineq0: jnp.ndarray
    slack0: jnp.ndarray
    W: dict
    regularization: jnp.ndarray
    barrier_parameter: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 1))
def _update_warm_start(
    horizon,
    shift,
    state,
    X_previous,
    X,
    U,
    V,
    Veq,
    Vineq,
    slack,
    current_barrier_parameter,
    accepted,
):
    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate((trajectory[shift:], tail), axis=0)

    def safe_update():
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            shift_trajectory(Veq),
            shift_trajectory(Vineq),
            shift_trajectory(slack),
            X[1, :arm_dof],
            X[1, nq : nq + arm_dof],
        )

    def unsafe_update():
        q, velocity = _split_state(state)
        state_data = _state_data(q, velocity)
        holding_torque = jnp.clip(
            state_data.qfrc_bias[:arm_dof]
            - state_data.qfrc_passive[:arm_dof]
            - 0.2 * velocity[:arm_dof],
            -0.99 * torque_limit,
            0.99 * torque_limit,
        )
        holding_command = torque_limit * jnp.arctanh(
            holding_torque / torque_limit
        )
        reset_control = jnp.concatenate((velocity, holding_command))
        reset_X = jnp.tile(state, (horizon + 1, 1))
        reset_U = jnp.tile(reset_control, (horizon, 1))
        reset_g = jax.vmap(inequality)(
            reset_X[:-1],
            reset_U,
            jnp.arange(horizon),
            jnp.zeros((horizon, 1), dtype=state.dtype),
        )
        reset_slack = jnp.maximum(-reset_g, 1e-2)
        return (
            reset_U,
            reset_X,
            jnp.zeros_like(X_previous),
            jnp.zeros((horizon, equality_dim), dtype=state.dtype),
            current_barrier_parameter / reset_slack,
            reset_slack,
            X_previous[1, :arm_dof],
            X_previous[1, nq : nq + arm_dof],
        )

    valid_update = (
        jnp.isfinite(U).all()
        & jnp.isfinite(Vineq).all()
        & jnp.isfinite(slack).all()
        & (Vineq > 0.0).all()
        & (slack > 0.0).all()
        & accepted
    )
    return jax.lax.cond(valid_update, safe_update, unsafe_update)


class MPCWrapper:
    def __init__(self, config, limited_memory=False):
        self.mpc_frequency = config.mpc_frequency
        self.shift = max(1, int(1 / (config.dt * config.mpc_frequency)))
        self.qpos_slice = slice(0, nq)
        self.qvel_slice = slice(nq, nq + nv)
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.model.opt.timestep = config.dt

        self.initial_state = jnp.asarray(config.initial_state)
        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_Veq0 = jnp.zeros((config.N, config.equality_dim))
        initial_g = jax.vmap(config.inequality)(
            self.initial_X0[:-1],
            self.initial_U0,
            jnp.arange(config.N),
            jnp.zeros((config.N, 1)),
        )
        self.initial_slack0 = jnp.maximum(-initial_g, 1e-2)
        self.initial_Vineq0 = config.barrier_parameter / self.initial_slack0

        solver = partial(
            optimizers.ip_mpc,
            config.cost,
            config.dynamics,
            None,
            limited_memory,
            equality=config.equality,
            inequality=config.inequality,
            num_alpha=config.equality_num_alpha,
        )

        def solve(
            reference,
            parameter,
            weights,
            state,
            X0,
            U0,
            V0,
            Veq0,
            Vineq0,
            slack0,
            current_regularization,
            current_barrier_parameter,
        ):
            return solver(
                reference,
                parameter,
                weights,
                state,
                X0,
                U0,
                V0,
                Veq_in=Veq0,
                Vineq_in=Vineq0,
                slack_in=slack0,
                regularization=current_regularization,
                barrier_parameter=current_barrier_parameter,
                multiplier_regularization=config.multiplier_regularization,
            )

        self._solve = solve
        self._update_warm_start = partial(
            _update_warm_start, config.N, self.shift
        )

    def make_data(self):
        return ComFreeMPCData(
            dt=dt,
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            Vineq0=self.initial_Vineq0,
            slack0=self.initial_slack0,
            W=W,
            regularization=regularization,
            barrier_parameter=barrier_parameter,
        )

    def _reference(self, command):
        goal = jnp.asarray(command[:2])
        return {"goal": jnp.tile(goal, (N + 1, 1))}

    def control_output(self, U):
        return _applied_torque(U[0, tau_slice])

    def fallback_output(self, state):
        q, velocity = _split_state(state)
        state_data = _state_data(q, velocity)
        holding_torque = (
            state_data.qfrc_bias[:arm_dof]
            - state_data.qfrc_passive[:arm_dof]
            - 0.2 * velocity[:arm_dof]
        )
        return jnp.clip(holding_torque, -torque_limit, torque_limit)

    def run(self, mpc_data, state, command, contact=None):
        del contact
        reference = self._reference(command)
        parameter = jnp.zeros((N + 1, 1), dtype=state.dtype)
        nominal_g = jax.vmap(inequality)(
            mpc_data.X0[:-1],
            mpc_data.U0,
            jnp.arange(N),
            jnp.zeros((N, 1), dtype=state.dtype),
        )
        slack_in = jnp.maximum(-nominal_g, 1e-2)
        vineq_in = barrier_parameter / slack_in
        (
            X,
            U,
            V,
            Veq,
            Vineq,
            slack,
            reg,
            _,
            alpha_best,
            accepted,
        ) = self._solve(
            reference,
            parameter,
            mpc_data.W,
            state,
            mpc_data.X0,
            mpc_data.U0,
            mpc_data.V0,
            mpc_data.Veq0,
            vineq_in,
            slack_in,
            mpc_data.regularization,
            barrier_parameter,
        )
        valid_update = (
            jnp.isfinite(U).all()
            & jnp.isfinite(Vineq).all()
            & jnp.isfinite(slack).all()
            & (Vineq > 0.0).all()
            & (slack > 0.0).all()
            & accepted
        )
        tau = jax.lax.cond(
            valid_update,
            lambda: self.control_output(U),
            lambda: self.fallback_output(state),
        )
        U0, X0, V0, Veq0, Vineq0, slack0, q, dq = self._update_warm_start(
            state,
            mpc_data.X0,
            X,
            U,
            V,
            Veq,
            Vineq,
            slack,
            barrier_parameter,
            accepted,
        )
        next_data = mpc_data.replace(
            X0=X0,
            U0=U0,
            V0=V0,
            Veq0=Veq0,
            Vineq0=Vineq0,
            slack0=slack0,
            regularization=jnp.where(valid_update, reg, regularization),
            barrier_parameter=barrier_parameter,
        )
        return next_data, tau, q, dq, alpha_best, accepted

    def reset(self, mpc_data, qpos, qvel, foot=None):
        del foot
        qpos = jnp.ravel(qpos)
        qvel = jnp.ravel(qvel)
        state = (
            self.initial_state.at[self.qpos_slice]
            .set(qpos)
            .at[self.qvel_slice]
            .set(qvel)
        )
        reset_control = u_ref.at[vnext_slice].set(qvel)
        X0 = jnp.tile(state, (N + 1, 1))
        U0 = jnp.tile(reset_control, (N, 1))
        reset_g = jax.vmap(inequality)(
            X0[:-1],
            U0,
            jnp.arange(N),
            jnp.zeros((N, 1), dtype=state.dtype),
        )
        slack0 = jnp.maximum(-reset_g, 1e-2)
        return mpc_data.replace(
            X0=X0,
            U0=U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            Vineq0=barrier_parameter / slack0,
            slack0=slack0,
            regularization=regularization,
            barrier_parameter=barrier_parameter,
        )


def state_to_qpos(state):
    return state[:nq]
