import os
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src import math, smooth

from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/piper_l/scene_flat.xml"

contact_frame = ["end_effector"]
body_name = ["link6"]#, "link8"]

dt = 0.01
N = 50
mpc_frequency = 100.0
solver_mode = "equality"
equality_num_alpha = 5

regularization = jnp.array(1e-6,dtype=jnp.float32)
merit_penalty = jnp.array(1e-6,dtype=jnp.float32)

contact_stiffness = 1000.0
terrain_contact_stiffness = 1000.0
contact_smoothing = 0.005
terrain_contact_smoothing = 0.001
contact_dissipation_velocity = 0.1
friction_coefficient = 0.6
friction_coefficient_terrain = 0.7
stiction_velocity = 0.05
cube_half_extent = 0.035
cube_contact_radius = cube_half_extent

q0 = jnp.array([0, 0, -0., -0., 0, 0, 0.3, 0.0,0.11,1.0,0,0,0])
object_goal = jnp.array([0.8, 0.0])
# q0 = jnp.concatenate([q_grasp, jnp.array([0.3, 0.0, 0.11, 1.0, 0.0, 0.0, 0.0])])
q0_init = q0


n_joints = 6
n_contact = len(contact_frame)
nq = n_joints + 7
nv = n_joints + 6
nx_error = 2 * n_joints + 3
n = nq + nv
m = nv + nv
equality_dim = nv

qacc_slice = slice(0, nv)
tau_slice = slice(nv, nv + nv)

Qpos = jnp.diag(jnp.ones(2))*100
Qq = jnp.diag(jnp.ones(n_joints)) * 0.1
Qdq = jnp.diag(jnp.ones(n_joints)) * 0.1
Qacc = jnp.diag(jnp.ones(nv)) * 0.01
Q_acc_tau = jnp.ones(n_joints) * 1e-2
Q_unacc_tau = jnp.ones(nv - n_joints) * 1e4
Qtau = jnp.diag(jnp.concatenate([Q_acc_tau, Q_unacc_tau]))
Qee = jnp.diag(jnp.ones(3)) * 100.0
W = {
    "Qpos": Qpos,
    "Qq": Qq,
    "Qdq": Qdq,
    "Qacc": Qacc,
    "Qtau": Qtau,
    "Qee": Qee
}

initial_state = jnp.concatenate([q0, jnp.zeros(nv)])

max_torque = 30.0
min_torque = -30.0
u_ref = jnp.zeros(m)

_MODEL = mujoco.MjModel.from_xml_path(model_path)
_MODEL.opt.timestep = dt
_MJX_MODEL = mjx.put_model(_MODEL)
_CONTACT_ID = [
    mjx.name2id(_MJX_MODEL, mujoco.mjtObj.mjOBJ_GEOM, name)
    for name in contact_frame
]
_CONTACT_HALF_SIZE = jnp.asarray([_MODEL.geom_size[contact_id] for contact_id in _CONTACT_ID])
_BODY_ID = [
    mjx.name2id(_MJX_MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in body_name
]


def _state_parts(x):
    qpos = x[:nq]
    qvel = x[nq : nq + nv]
    return qpos, qvel


def _integrate_state(qpos, qvel, qacc):
    qvel_next = qvel + qacc * dt
    qpos_joints = qpos[:n_joints] + qvel_next[:n_joints] * dt
    # qpos_next = qvel_next * dt + qpos
    qpos_next = jnp.concatenate(
        [
            qpos_joints,
            qpos[n_joints:n_joints+3] + qvel_next[n_joints:n_joints+3] * dt,
            math.quat_integrate(qpos[n_joints+3:n_joints+7], qvel_next[n_joints+3:n_joints+6], dt),
        ]
    )
    return qpos_next, qvel_next


def dynamics(x, u, t, parameter):
    del t, parameter
    qpos, qvel = _state_parts(x)
    qacc = u[qacc_slice]
    qpos_next, qvel_next = _integrate_state(qpos, qvel, qacc)
    return jnp.concatenate([qpos_next, qvel_next])

def _normal_dissipation(normal_velocity):
    velocity_ratio = normal_velocity / contact_dissipation_velocity
    separating = 0.25 * (velocity_ratio - 2.0) ** 2
    approaching = 1.0 - velocity_ratio
    return jnp.where(
        velocity_ratio < 0.0,
        approaching,
        jnp.where(velocity_ratio < 2.0, separating, 0.0),
    )

@jax.jit
def contact_force_from_state(foot_position, foot_velocity,body_position,body_velocity):
    delta_world = foot_position - body_position
    signed_distance = jnp.sqrt(jnp.sum(delta_world**2) + 1e-6) - cube_contact_radius
    normal_vector = delta_world / (math.norm(delta_world) + 1e-6)
    relative_velocity = foot_velocity-body_velocity
    normal_velocity = jnp.dot(relative_velocity, normal_vector)
    tangential_velocity_world = relative_velocity - normal_velocity * normal_vector
    compliance = (
        contact_smoothing
        * contact_stiffness
        * jax.nn.softplus(-signed_distance / contact_smoothing)
    )
    normal_force = compliance* _normal_dissipation(normal_velocity)
    vt2 = jnp.dot(tangential_velocity_world, tangential_velocity_world)
    denom = jnp.sqrt(stiction_velocity**2 + vt2)
    tangential_force_world = (
        -friction_coefficient
        * normal_force
        * tangential_velocity_world 
        / denom
    )
    total_force = normal_force * normal_vector + tangential_force_world
    return total_force


def contact_with_terrain(body_position,body_velocity):
    signed_distance = body_position[2] - cube_half_extent
    normal_velocity = body_velocity[2]
    tangential_velocity = jnp.array([body_velocity[0], body_velocity[1]])
    compliance = (
        terrain_contact_smoothing
        * terrain_contact_stiffness
        * jax.nn.softplus(-signed_distance / contact_smoothing)
    )
    normal_force = compliance * _normal_dissipation(normal_velocity)
    tangential_force = -friction_coefficient_terrain * normal_force * tangential_velocity
    tangential_force /= jnp.sqrt(
        stiction_velocity**2 + jnp.dot(tangential_velocity, tangential_velocity)
    )
    return jnp.array([tangential_force[0], tangential_force[1], normal_force])

def _contact_kinematics(data):
    jacobians = []
    contact_positions = []
    for contact_id, body_id, half_size in zip(_CONTACT_ID, _BODY_ID, _CONTACT_HALF_SIZE):
        contact_position = data.geom_xpos[contact_id]
        jacobian, _ = mjx.jac(_MJX_MODEL, data, contact_position, body_id)
        jacobians.append(jacobian)
        contact_positions.append(contact_position)
    return jnp.stack(jacobians, axis=0), jnp.stack(contact_positions, axis=0)


def equality(x, u, t, parameter):
    # del t
    qpos, qvel = _state_parts(x)
    qacc = u[qacc_slice]
    qpos_next, qvel_next = _integrate_state(qpos, qvel, qacc)    
    tau = u[tau_slice]
    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos_next, qvel=qvel_next)
    data = mjx.fwd_position(_MJX_MODEL, data)
    data = mjx.fwd_velocity(_MJX_MODEL, data)
    M = data.qM
    D = data.qfrc_bias
    qfrc_inverse = M @ qacc + D

    object_position = data.qpos[n_joints:n_joints+3]
    object_velocity = data.qvel[n_joints:n_joints+3]
    contact_jacobians, contact_positions = _contact_kinematics(data)
    contact_velocities = jax.vmap(lambda jacobian: jacobian.T @ qvel_next)(contact_jacobians)
    arm_box_forces = jax.vmap(
        lambda contact_position, contact_velocity: contact_force_from_state(
            contact_position,
            contact_velocity,
            object_position,
            object_velocity,
        )
    )(contact_positions, contact_velocities)
    contact_jacobian = jnp.concatenate(contact_jacobians, axis=1)
    arm_wrench = (contact_jacobian @ arm_box_forces.reshape(-1))[:-6]

    box_terrain_force = contact_with_terrain(object_position, object_velocity)
    object_force = box_terrain_force + jnp.sum(-arm_box_forces, axis=0)
    # jax.debug.print("arm_box_forces: {}, time: {}", arm_box_forces,t)
    contact_wrench = jnp.concatenate([arm_wrench, object_force, jnp.zeros(3, dtype=x.dtype)])
    # contact_wrench = jnp.concatenate([arm_wrench, object_force, jnp.zeros(3)])
    generalized_actuation = tau
    return qfrc_inverse - contact_wrench - generalized_actuation

class ForwardDynamicsInfo(PyTreeNode):
    qacc: jnp.ndarray
    contact_positions: jnp.ndarray
    arm_box_forces: jnp.ndarray
    box_terrain_force: jnp.ndarray
    box_terrain_position: jnp.ndarray
    contact_wrench: jnp.ndarray


def forward_dynamics(x, tau):
    qpos, qvel = _state_parts(x)
    tau = jnp.asarray(tau, dtype=x.dtype)
    if tau.shape[0] == n_joints:
        generalized_actuation = jnp.concatenate(
            [tau, jnp.zeros(nv - n_joints, dtype=x.dtype)]
        )
    elif tau.shape[0] == nv:
        generalized_actuation = tau
    else:
        raise ValueError(f"Expected tau to have shape ({n_joints},) or ({nv},), got {tau.shape}.")

    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos, qvel=qvel)
    data = mjx.fwd_position(_MJX_MODEL, data)
    data = mjx.fwd_velocity(_MJX_MODEL, data)

    mass_matrix = data.qM
    bias = data.qfrc_bias

    object_position = qpos[n_joints : n_joints + 3]
    object_velocity = qvel[n_joints : n_joints + 3]
    contact_jacobians, contact_positions = _contact_kinematics(data)
    contact_velocities = jax.vmap(lambda jacobian: jacobian.T @ qvel)(contact_jacobians)

    arm_box_forces = jax.vmap(
        lambda contact_position, contact_velocity: contact_force_from_state(
            contact_position,
            contact_velocity,
            object_position,
            object_velocity,
        )
    )(contact_positions, contact_velocities)
    contact_jacobian = jnp.concatenate(contact_jacobians, axis=1)
    arm_wrench = (contact_jacobian @ arm_box_forces.reshape(-1))[:-6]

    box_terrain_force = contact_with_terrain(object_position, object_velocity)
    object_force = box_terrain_force + jnp.sum(-arm_box_forces, axis=0)
    contact_wrench = jnp.concatenate([arm_wrench, object_force, jnp.zeros(3, dtype=x.dtype)])

    rhs = generalized_actuation + contact_wrench - bias
    qacc = jnp.linalg.solve(mass_matrix + 1e-6 * jnp.eye(nv, dtype=x.dtype), rhs)
    qpos_next, qvel_next = _integrate_state(qpos, qvel, qacc)
    x_next = jnp.concatenate([qpos_next, qvel_next])

    info = ForwardDynamicsInfo(
        qacc=qacc,
        contact_positions=contact_positions,
        arm_box_forces=arm_box_forces,
        box_terrain_force=box_terrain_force,
        box_terrain_position=object_position + jnp.array([0.0, 0.0, -cube_half_extent], dtype=x.dtype),
        contact_wrench=contact_wrench,
    )
    return x_next, info

def cost(W, reference, x, u, t):
    qpos, qvel = _state_parts(x)
    body_position = qpos[n_joints:n_joints+3]  # Adjusted for 2D case
    q = qpos[:n_joints]
    dq = qvel[:n_joints]
    acc = u[qacc_slice]
    tau = u[tau_slice]

    q_ref = reference[t, : n_joints]
    dq_ref = reference[t, n_joints : 2*n_joints]
    p_ref = reference[t, 2*n_joints: 2*n_joints+2]

    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos, qvel=qvel)
    data = smooth.kinematics(_MJX_MODEL, data)

    contact_positions = []
    for contact_id in zip(_CONTACT_ID):
        contact_position = data.geom_xpos[contact_id]
        contact_positions.append(contact_position)
    end_effector_position = jnp.stack(contact_positions, axis=0)


    stage_cost = (
        (body_position[:2] - p_ref).T @ W["Qpos"] @ (body_position[:2] - p_ref)
        + (q - q_ref).T @ W["Qq"] @ (q - q_ref)
        + (dq - dq_ref).T
        @ W["Qdq"] @ (dq - dq_ref)
        + acc.T @ W["Qacc"] @ acc
        + (tau ).T
        @ W["Qtau"] @ (tau) +
        (end_effector_position.flatten()-body_position).T @ W["Qee"] @ (end_effector_position.flatten()-body_position)
    )
    terminal_cost = (
        (body_position[:2] - p_ref).T @ W["Qpos"] @ (body_position[:2] - p_ref)
        + (q - q_ref).T @ W["Qq"] @ (q - q_ref)
        + (dq - dq_ref).T
        @ W["Qdq"] @ (dq - dq_ref) + 
        (end_effector_position.flatten()-body_position).T @ W["Qee"] @ (end_effector_position.flatten()-body_position)
    )
    return jnp.where(t == N, 0.5 * terminal_cost, 0.5 * stage_cost)


class InverseDynamicsMPCData(PyTreeNode):
    dt: float
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    Veq0: jnp.ndarray
    W: jnp.ndarray
    regularization: jnp.ndarray
    merit_penalty: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 1))
def _update_warm_start(horizon, shift, u_ref, x0, X_prev, U_prev, X, U, V, Veq):
    q_slice = slice(0, n_joints)
    dq_slice = slice(nq, nq + n_joints)
    u_fallback_idx = 1 if horizon > 1 else 0

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate([trajectory[shift:], tail], axis=0)

    def safe_update():
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            shift_trajectory(Veq),
            X[1, q_slice],
            X[1, dq_slice],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(X_prev),
            jnp.zeros((horizon, equality_dim), dtype=X_prev.dtype),
            X_prev[1, q_slice],
            X_prev[1, dq_slice],
        )

    valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
    return jax.lax.cond(valid_solution, safe_update, unsafe_update)


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
        
        self.dynamics = config.dynamics
        solver = partial(
            optimizers.mpc_equality_fddp,
            config.cost,
            self.dynamics,
            None,
            limited_memory,
            equality=config.equality,
            num_alpha=config.equality_num_alpha,
        )

        def solve(reference, parameter, W, x0, X0, U0, V0, Veq0, regularization,merit_penalty):
            return solver(reference, parameter, W, x0, X0, U0, V0, Veq_in=Veq0, regularization=regularization, merit_penalty=merit_penalty)

        self._solve = solve
        self._update_warm_start = partial(
            _update_warm_start,
            config.N,
            self.shift,
            config.u_ref,
        )

    def make_data(self):
        return InverseDynamicsMPCData(
            dt=dt,
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            W=W,
            regularization=regularization,
            merit_penalty=merit_penalty,
        )

    def _reference(self, x, command):
        # times = jnp.arange(N + 1, dtype=x.dtype) * dt
        p_ref = jnp.tile(object_goal, (N + 1, 1))
        q_refs = jnp.tile(q0[:n_joints], (N + 1, 1))
        dq_refs = jnp.zeros((N + 1, n_joints))

        return jnp.concatenate([q_refs, dq_refs, p_ref], axis=1)

    def control_output(self, x0, X, U, reference, parameter):
        del x0, X, reference, parameter
        return jnp.clip(U[0, tau_slice], min_torque, max_torque)

    def run(self, data, x, command, contact=None):
        del contact
        reference = self._reference(x, command)
        parameter = jnp.zeros((N + 1, 1), dtype=x.dtype)
        X, U, V, Veq, regularization, merit_penalty, alpha_best, any_accepted = self._solve(
            reference,
            parameter,
            data.W,
            x,
            data.X0,
            data.U0,
            data.V0,
            data.Veq0,
            data.regularization,
            data.merit_penalty
        )
        valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
        tau = jax.lax.cond(
            valid_solution,
            lambda _: self.control_output(x, X, U, reference, parameter),
            lambda _: self.control_output(x, data.X0, data.U0, reference, parameter),
            operand=None,
        )
        U0, X0, V0, Veq0, q, dq = self._update_warm_start(
            x,
            data.X0,    
            data.U0,
            X,
            U,
            V,
            Veq,
        )
        # jax.debug.print("====================" )
        return data.replace(X0=X0, U0=U0, V0=V0, Veq0=Veq0, regularization=regularization, merit_penalty=merit_penalty), tau, q, dq, alpha_best, any_accepted

    def reset(self, data, qpos, qvel, foot=None):
        del foot
        x = (
            self.initial_state.at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
        )
        return data.replace(
            X0=jnp.tile(x, (N + 1, 1)),
            U0=self.initial_U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
        )


def state_to_qpos(x):
    return x[:nq]
