import os
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src import smooth

from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/piper_l/scene_flat.xml"

contact_frame = ["end_effector"]
body_name = ["link6"]#, "link8"]

dt = 0.02
N = 50
mpc_frequency = 50.0
solver_mode = "equality"
equality_num_alpha = 8

regularization = jnp.array(1e-6,dtype=jnp.float32)
merit_penalty = jnp.array(1e-6,dtype=jnp.float32)



n_joints = 6
n_contact = len(contact_frame)
nq = n_joints
nv = n_joints
n = nq + nv
m = nv + n_joints
equality_dim = nv
inequality_dim = 2 * n_joints

qacc_slice = slice(0, nv)
tau_slice = slice(nv, nv + n_joints)

Qq = jnp.diag(jnp.ones(n_joints)) * 0.1
Qdq = jnp.diag(jnp.ones(n_joints)) * 0.2
Qacc = jnp.diag(jnp.ones(nv)) * 0.01
Qtau = jnp.diag(jnp.ones(n_joints)) * 0.0001
Qee = jnp.diag(jnp.ones(3)) * 200.0
Qee_final = jnp.diag(jnp.ones(3)) * 1000.0
W = {
    "Qq": Qq,
    "Qdq": Qdq,
    "Qacc": Qacc,
    "Qtau": Qtau,
    "Qee": Qee,
    "Qee_final": Qee_final,
}


obstacle_center = jnp.array([0.3, 0.0, 0.6])
obstacle_radius = 0.2
max_torque = jnp.array([32.0, 32.0, 32.0, 8.0, 8.0, 8.0])
min_torque = -max_torque
_MODEL = mujoco.MjModel.from_xml_path(model_path)
_MODEL.opt.timestep = dt
if not _MODEL.jnt_limited[:n_joints].all():
    raise ValueError("All controlled Piper joints must define position limits")
joint_position_min = jnp.asarray(_MODEL.jnt_range[:n_joints, 0])
joint_position_max = jnp.asarray(_MODEL.jnt_range[:n_joints, 1])
q0 = joint_position_min + 0.5 * (joint_position_max - joint_position_min)
q0_init = q0
initial_state = jnp.concatenate([q0, jnp.zeros(nv)])
barrier_parameter = jnp.array(1.0, dtype=jnp.float32)
multiplier_regularization = jnp.array(1e-8, dtype=jnp.float32)
_INITIAL_DATA = mujoco.MjData(_MODEL)
_INITIAL_DATA.qpos[:] = q0
mujoco.mj_forward(_MODEL, _INITIAL_DATA)
u_ref = jnp.concatenate(
    [
        jnp.zeros(nv),
        jnp.asarray(_INITIAL_DATA.qfrc_bias - _INITIAL_DATA.qfrc_passive),
    ]
)
_MJX_MODEL = mjx.put_model(_MODEL)
_CONTACT_ID = [
    mjx.name2id(_MJX_MODEL, mujoco.mjtObj.mjOBJ_GEOM, name)
    for name in contact_frame
]

def _state_parts(x):
    qpos = x[:nq]
    qvel = x[nq : nq + nv]
    return qpos, qvel


def _backward_euler_state(qpos, qvel, qacc):
    """Integrates a stage using its endpoint acceleration and velocity."""
    qvel_next = qvel + qacc * dt
    qpos_joints = qpos + qvel_next * dt
    return qpos_joints, qvel_next


def dynamics(x, u, t, parameter):
    del t, parameter
    qpos, qvel = _state_parts(x)
    qacc = u[qacc_slice]
    qpos_next, qvel_next = _backward_euler_state(qpos, qvel, qacc)
    return jnp.concatenate([qpos_next, qvel_next])

def equality(x, u, t, parameter):
    del t, parameter
    qpos, qvel = _state_parts(x)
    qacc = u[qacc_slice]
    qpos_next, qvel_next = _backward_euler_state(qpos, qvel, qacc)
    tau = u[tau_slice]
    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos_next, qvel=qvel_next)
    data = mjx.fwd_position(_MJX_MODEL, data)
    data = mjx.fwd_velocity(_MJX_MODEL, data)

    residual = data.qM @ qacc + data.qfrc_bias - data.qfrc_passive - tau

    return residual


def _equilibrium_control(qpos, qvel):
    qacc = jnp.zeros(nv, dtype=qpos.dtype)
    qpos_next, qvel_next = _backward_euler_state(qpos, qvel, qacc)
    data = mjx.make_data(_MJX_MODEL).replace(qpos=qpos_next, qvel=qvel_next)
    data = mjx.fwd_position(_MJX_MODEL, data)
    data = mjx.fwd_velocity(_MJX_MODEL, data)
    tau = data.qfrc_bias - data.qfrc_passive
    return jnp.concatenate([qacc, tau])


def inequality(x, u, t, parameter):
    """Joint position bounds, feasible when values are <= 0."""
    del u, t, parameter
    qpos, _ = _state_parts(x)
    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos)
    data = smooth.kinematics(_MJX_MODEL, data)
    end_effector_position = data.geom_xpos[_CONTACT_ID[0]]
    ## obstacle avoidance constraints - sphere centered at (0.3, 0.0, 0.4) with radius 0.2
    distance_to_obstacle = jnp.linalg.norm(end_effector_position - obstacle_center)
    obstacle_constraint = jnp.array([obstacle_radius - distance_to_obstacle])
    return jnp.concatenate(
        [
            joint_position_min - qpos,
            qpos - joint_position_max,
            obstacle_constraint
        ]
    )

def pseudo_huber(error, weight, delta):
    squared_error = error.T @ weight @ error
    return delta**2 * (
        jnp.sqrt(1.0 + squared_error / delta**2) - 1.0
    )

def cost(W, reference, x, u, t):
    qpos, qvel = _state_parts(x)
    q = qpos
    dq = qvel
    acc = u[qacc_slice]
    tau = u[tau_slice]

    q_ref = reference["q"][t]
    dq_ref = reference["dq"][t]
    p_ref = reference["p"][t]

    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos, qvel=qvel)
    data = smooth.kinematics(_MJX_MODEL, data)

    end_effector_position = data.geom_xpos[_CONTACT_ID[0]]
    end_effector_error = end_effector_position - p_ref
    ee_tracking_cost = pseudo_huber(end_effector_error, W["Qee"], delta=0.1)
    joint_error = q - q_ref

    stage_cost = (
          joint_error.T @ W["Qq"] @ joint_error
        + (dq - dq_ref).T @ W["Qdq"] @ (dq - dq_ref)
        + acc.T @ W["Qacc"] @ acc
        + tau.T @ W["Qtau"] @ tau
        + ee_tracking_cost
    )
    terminal_cost = (
        joint_error.T @ W["Qq"] @ joint_error
        + (dq - dq_ref).T @ W["Qdq"] @ (dq - dq_ref)
        + end_effector_error.T @ W["Qee_final"] @ end_effector_error
    )
    return jnp.where(t == N, 0.5 * terminal_cost, 0.5 * stage_cost)


class InverseDynamicsMPCData(PyTreeNode):
    dt: float
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    Veq0: jnp.ndarray
    Vineq0: jnp.ndarray
    slack0: jnp.ndarray
    W: jnp.ndarray
    regularization: jnp.ndarray
    merit_penalty: jnp.ndarray
    barrier_parameter: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 1))
def _update_warm_start(
    horizon, shift, u_ref, x0, X_prev, U_prev, X, U, V, Veq, Vineq, slack
):
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
            shift_trajectory(Vineq),
            shift_trajectory(slack),
            X[1, q_slice],
            X[1, dq_slice],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(X_prev),
            jnp.zeros((horizon, equality_dim), dtype=X_prev.dtype),
            shift_trajectory(Vineq),
            shift_trajectory(slack),
            X_prev[1, q_slice],
            X_prev[1, dq_slice],
        )

    valid_solution = jnp.isfinite(U).all()
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
        initial_g = jax.vmap(config.inequality)(
            self.initial_X0[:-1],
            self.initial_U0,
            jnp.arange(config.N),
            jnp.zeros((config.N, 1)),
        )
        self.initial_slack0 = jnp.maximum(-initial_g, 1e-2)
        self.initial_Vineq0 = config.barrier_parameter / self.initial_slack0
        
        self.dynamics = config.dynamics
        solver = partial(
            optimizers.ip_mpc,
            config.cost,
            self.dynamics,
            None,
            limited_memory,
            equality=config.equality,
            inequality=config.inequality,
            num_alpha=config.equality_num_alpha,
        )

        def solve(
            reference,
            parameter,
            W,
            x0,
            X0,
            U0,
            V0,
            Veq0,
            Vineq0,
            slack0,
            regularization,
            current_barrier_parameter,
        ):
            return solver(
                reference,
                parameter,
                W,
                x0,
                X0,
                U0,
                V0,
                Veq_in=Veq0,
                Vineq_in=Vineq0,
                slack_in=slack0,
                regularization=regularization,
                barrier_parameter=current_barrier_parameter,
                multiplier_regularization=config.multiplier_regularization,
            )

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
            Vineq0=self.initial_Vineq0,
            slack0=self.initial_slack0,
            W=W,
            regularization=regularization,
            merit_penalty=merit_penalty,
            barrier_parameter=barrier_parameter,
        )

    def _reference(self, x, command):
        qpos, qvel = _state_parts(x)
        goal = command["goal"]
        return {
            "p": jnp.tile(goal, (N + 1, 1)),
            "q": jnp.tile(qpos, (N + 1, 1)),
            "dq": jnp.zeros((N + 1, n_joints)),
        }

    def control_output(self, x0, X, U, reference, parameter):
        del x0, X, reference, parameter
        return jnp.clip(U[0, tau_slice], min_torque, max_torque)

    def run(self, data, x, command, contact=None):
        del contact
        reference = self._reference(x, command)
        parameter = jnp.zeros((N + 1, 1), dtype=x.dtype)
        (
            X,
            U,
            V,
            Veq,
            Vineq,
            slack,
            regularization,
            next_barrier_parameter,
            alpha_best,
            any_accepted,
        ) = self._solve(
            reference,
            parameter,
            data.W,
            x,
            data.X0,
            data.U0,
            data.V0,
            data.Veq0,
            data.Vineq0,
            data.slack0,
            data.regularization,
            data.barrier_parameter,
        )
        valid_solution = jnp.isfinite(U).all()
        tau = jax.lax.cond(
            valid_solution,
            lambda _: self.control_output(x, X, U, reference, parameter),
            lambda _: self.control_output(x, data.X0, data.U0, reference, parameter),
            operand=None,
        )
        U0, X0, V0, Veq0, Vineq0, slack0, q, dq = self._update_warm_start(
            x,
            data.X0,    
            data.U0,
            X,
            U,
            V,
            Veq,
            Vineq,
            slack,
        )
        # jax.debug.print("====================" )
        return data.replace(
            X0=X0,
            U0=U0,
            V0=V0,
            Veq0=Veq0,
            Vineq0=Vineq0,
            slack0=slack0,
            regularization=regularization,
            barrier_parameter=jnp.maximum(next_barrier_parameter, 1e-8),
        ), tau, q, dq, alpha_best, any_accepted

    def reset(self, data, qpos, qvel, foot=None):
        del foot
        x = (
            self.initial_state.at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
        )
        nominal_control = _equilibrium_control(
            x[self.qpos_slice], x[self.qvel_slice]
        )
        X0 = jnp.tile(x, (N + 1, 1))
        U0 = jnp.tile(nominal_control, (N, 1))
        parameters = jnp.zeros((N, 1), dtype=x.dtype)
        g = jax.vmap(inequality)(X0[:-1], U0, jnp.arange(N), parameters)
        slack0 = jnp.maximum(-g, 1e-2)
        return data.replace(
            X0=X0,
            U0=U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            Vineq0=barrier_parameter / slack0,
            slack0=slack0,
            regularization=regularization,
            merit_penalty=merit_penalty,
            barrier_parameter=barrier_parameter,
        )


def state_to_qpos(x):
    return x[:nq]
