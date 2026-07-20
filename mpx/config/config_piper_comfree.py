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
body_name = ["link6"]

model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.005
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

nq = model.nq
nv = model.nv
nu = model.nu
nx = nq + nv

arm_dof = 6
n_joints = arm_dof
n = nx
box_qpos_adr = 6
box_dof_adr = 6
equality_dim = nv

ee_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, contact_frame[0])
box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_body")
box_geom_id = int(model.body_geomadr[box_body_id])

ee_radius = float(model.geom_size[ee_geom_id, 0])
box_half_size = jnp.asarray(model.geom_size[box_geom_id, :3])

dt = 0.02
forward_dt = 0.005
N = 20
mpc_frequency = 50
sim_steps = max(1, round(1.0 / (mpc_frequency * model.opt.timestep)))
equality_num_alpha = 10
solver_iterations = 1
regularization = jnp.array(1e-3)
merit_penalty = jnp.array(1e-2)
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
goal_default = jnp.array([initial_box_x, -0.4])
object_goal = goal_default


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
q0_init = q0
mujoco.mj_resetData(model, data)
data.qpos[:] = q0
mujoco.mj_forward(model, data)

x0 = jnp.concatenate((jnp.asarray(q0), jnp.zeros(nv)))
initial_state = x0
arm_nominal = jnp.asarray(q0[:arm_dof])
W = {"scale": jnp.array(1.0)}


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
    return rows, jnp.full(4, phi), force_directions


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

    floor_rows = []
    floor_phis = []
    floor_directions = []
    floor_normal = jnp.array([0.0, 0.0, 1.0])
    for sx, sy in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
        local_corner = box_half_size * jnp.array([sx, sy, -1.0])
        corner = box_center + box_rotation @ local_corner
        corner_jacobian, _ = mjx.jac(
            mjx_model, mjx_data, corner, box_body_id
        )
        rows, phis, directions = _friction_pyramid(
            corner_jacobian, floor_normal, corner[2], friction=0.5
        )
        floor_rows.append(rows)
        floor_phis.append(phis)
        floor_directions.append(directions)

    contact_rows = jnp.concatenate((arm_rows, *floor_rows), axis=0)
    contact_phis = jnp.concatenate((arm_phis, *floor_phis))
    contact_directions = jnp.concatenate(
        (arm_directions, *floor_directions), axis=0
    )
    stiffness = jnp.concatenate((jnp.full(4, 35.0), jnp.full(16, 5.0)))
    damping = jnp.concatenate((jnp.full(4, 0.8), jnp.full(16, 0.08)))
    return contact_rows, contact_phis, contact_directions, stiffness, damping


def _smooth_positive(value, sharpness):
    return jax.nn.softplus(sharpness * value) / sharpness


def _contact_force_terms(mjx_data, velocity, integration_dt=None):
    rows, phi, directions, stiffness, damping = _contact_kinematics(mjx_data)
    contact_velocity = rows @ velocity
    predicted_gap = phi
    if integration_dt is not None:
        predicted_gap = predicted_gap + integration_dt * contact_velocity
    sharpness = jnp.concatenate((jnp.full(4, 100.0), jnp.full(16, 20.0)))
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
    ground_contact_positions = jnp.stack(
        [
            box_center
            + box_rotation
            @ (box_half_size * jnp.array([sx, sy, -1.0]))
            for sx, sy in (
                (-1.0, -1.0),
                (-1.0, 1.0),
                (1.0, -1.0),
                (1.0, 1.0),
            )
        ]
    )
    arm_force_on_box = -jnp.sum(
        beta[:4, None] * directions[:4], axis=0
    )
    ground_corner_forces = jnp.sum(
        beta[4:].reshape(4, 4, 1) * directions[4:].reshape(4, 4, 3),
        axis=1,
    )
    return ContactForceInfo(
        arm_force_on_box=arm_force_on_box,
        ground_force_on_box=jnp.sum(ground_corner_forces, axis=0),
        ground_corner_forces=ground_corner_forces,
        arm_contact_position=arm_contact_position,
        ground_contact_positions=ground_contact_positions,
        arm_gap=phi[0],
        ground_gaps=phi[4::4],
        beta=beta,
    )


def contact_force_info(q, velocity):
    """Return Cartesian forces and gaps from the smooth contact model."""
    state_data = _state_data(q, velocity)
    _, phi, directions, beta = _contact_force_terms(state_data, velocity)
    return _contact_force_info_from_terms(state_data, phi, directions, beta)


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


def _task_geometry(state):
    q, v = _split_state(state)
    state_data = smooth.kinematics(
        mjx_model, mjx.make_data(mjx_model).replace(qpos=q, qvel=v)
    )
    ee_position = state_data.geom_xpos[ee_geom_id]
    box_position = state_data.geom_xpos[box_geom_id]
    box_rotation = state_data.geom_xmat[box_geom_id].reshape(3, 3)
    box_distance, _ = _box_distance_and_normal(
        ee_position, box_position, box_rotation
    )
    return ee_position, box_position, box_distance - ee_radius


def cost(weights, reference, state, control, t):
    del weights
    q, velocity = _split_state(state)
    velocity_next = control[vnext_slice]
    torque = control[tau_slice]
    ee_position, box_position, _ = _task_geometry(state)
    goal = reference["goal"][t]

    goal_error = box_position[:2] - goal
    contact_error = ee_position - box_position
    posture_error = q[:arm_dof] - arm_nominal
    acceleration = (velocity_next - velocity) / dt
    goal_direction = goal[:2] - box_position[:2]
    goal_direction = goal_direction / jnp.maximum(jnp.linalg.norm(goal_direction), 1e-4)
    side_projection = jnp.dot(ee_position[:2] - box_position[:2], goal_direction)

    stage_cost = (
        50.0 * jnp.dot(goal_error, goal_error)
        + 0.05 * jnp.dot(posture_error, posture_error)
        + 0.1 * jnp.dot(velocity, velocity)
        + 1e-4 * jnp.dot(acceleration, acceleration)
        + 1e-3 * jnp.dot(torque, torque)
        + 20*jax.nn.relu(side_projection + 0.025) ** 2
    )
    terminal_cost = (
        1000.0 * jnp.dot(goal_error, goal_error)
        + 200.0 * jnp.dot(contact_error, contact_error)
        + 0.1 * jnp.dot(posture_error, posture_error)
        + 0.1 * jnp.dot(velocity, velocity)
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
    W: dict
    regularization: jnp.ndarray
    merit_penalty: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 1))
def _update_warm_start(
    horizon,
    shift,
    u_reference,
    state,
    X_previous,
    U_previous,
    X,
    U,
    V,
    Veq,
    accepted,
):
    del u_reference, U_previous

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate((trajectory[shift:], tail), axis=0)

    def safe_update():
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            shift_trajectory(Veq),
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
        return (
            jnp.tile(reset_control, (horizon, 1)),
            jnp.tile(state, (horizon + 1, 1)),
            jnp.zeros_like(X_previous),
            jnp.zeros((horizon, equality_dim), dtype=state.dtype),
            X_previous[1, :arm_dof],
            X_previous[1, nq : nq + arm_dof],
        )

    valid_update = jnp.isfinite(U).all() & accepted
    return jax.lax.cond(valid_update, safe_update, unsafe_update)


class MPCWrapper:
    def __init__(self, config, limited_memory=False):
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = max(1, int(1 / (config.dt * config.mpc_frequency)))
        self.qpos_slice = slice(0, nq)
        self.qvel_slice = slice(nq, nq + nv)
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.model.opt.timestep = config.dt
        self.mjx_model = mjx.put_model(self.model)

        self.initial_state = jnp.asarray(config.initial_state)
        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_Veq0 = jnp.zeros((config.N, config.equality_dim))

        solver = partial(
            optimizers.mpc_equality,
            config.cost,
            config.dynamics,
            None,
            limited_memory,
            equality=config.equality,
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
            current_regularization,
        ):
            initial_values = (
                X0,
                U0,
                V0,
                Veq0,
                current_regularization,
                jnp.array(0.0, dtype=state.dtype),
                jnp.array(False),
            )

            def solver_iteration(_, values):
                X, U, V, Veq, reg, _, accepted_before = values
                result = solver(
                    reference,
                    parameter,
                    weights,
                    state,
                    X,
                    U,
                    V,
                    Veq_in=Veq,
                    regularization=reg,
                )
                return (*result[:-1], accepted_before | result[-1])

            return jax.lax.fori_loop(
                0, config.solver_iterations, solver_iteration, initial_values
            )

        self._solve = solve
        self._update_warm_start = partial(
            _update_warm_start, config.N, self.shift, config.u_ref
        )

    def make_data(self):
        return ComFreeMPCData(
            dt=dt,
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            W=W,
            regularization=regularization,
            merit_penalty=merit_penalty,
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
        X, U, V, Veq, reg, alpha_best, accepted = self._solve(
            reference,
            parameter,
            mpc_data.W,
            state,
            mpc_data.X0,
            mpc_data.U0,
            mpc_data.V0,
            mpc_data.Veq0,
            mpc_data.regularization,
        )
        valid_update = jnp.isfinite(U).all() & accepted
        tau = jax.lax.cond(
            valid_update,
            lambda: self.control_output(U),
            lambda: self.fallback_output(state),
        )
        U0, X0, V0, Veq0, q, dq = self._update_warm_start(
            state, mpc_data.X0, mpc_data.U0, X, U, V, Veq, accepted
        )
        next_data = mpc_data.replace(
            X0=X0,
            U0=U0,
            V0=V0,
            Veq0=Veq0,
            regularization=jnp.where(valid_update, reg, regularization),
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
        return mpc_data.replace(
            X0=jnp.tile(state, (N + 1, 1)),
            U0=jnp.tile(reset_control, (N, 1)),
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            regularization=regularization,
            merit_penalty=merit_penalty,
        )


def state_to_qpos(state):
    return state[:nq]
