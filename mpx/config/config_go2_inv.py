import os
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco.mjx._src.dataclasses import PyTreeNode

from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers
import mpx.utils.mpc_utils as mpc_utils


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/go2/go2_mjx.xml"

contact_frame = ["FL", "FR", "RL", "RR"]
body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

dt = 0.02
N = 12
mpc_frequency = 50.0
solver_mode = "equality"
equality_num_alpha = 10
regularization = jnp.array(1e-6)

timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])
duty_factor = jnp.array([0.65])
step_freq = jnp.array([1.35])
step_height = jnp.array([0.065])
initial_height = 0.1
robot_height = 0.27
use_terrain_estimation = False
clearance_speed = 0.2

p0 = jnp.array([0.0, 0.0, robot_height])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array([0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8])
q0_init = q0
p_legs0 = jnp.array(
    [
        0.192,
        0.142,
        0.0,
        0.192,
        -0.142,
        0.0,
        -0.195,
        0.142,
        0.0,
        -0.195,
        -0.142,
        0.0,
    ]
)

n_joints = 12
n_contact = len(contact_frame)
nq = 7 + n_joints
nv = 6 + n_joints
nx_error = 2 * nv
n = nq + nv
m = nv + n_joints + 3 * n_contact
equality_dim = nv

qacc_slice = slice(0, nv)
tau_slice = slice(nv, nv + n_joints)
grf_slice = slice(nv + n_joints, m)

Qp = jnp.diag(jnp.array([0.0, 0.0, 1.0e6]))
Qrot = jnp.diag(jnp.array([1.0e3, 1.0e3, 0.0]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1.0e2
Qdp = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 5.0e3
Qomega = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1.0e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1.0e-1
Qcontact = jnp.diag(jnp.tile(jnp.array([1.0e5, 1.0e5, 1.0e5]), n_contact))
Qacc = jnp.diag(jnp.ones(nv)) * 1.0e-1
Qtau = jnp.diag(jnp.ones(n_joints)* 1.0e-1) 
Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1.0e-2
W = {
    "pos": Qp,
    "rot": Qrot,
    "q": Qq,
    "vel": Qdp,
    "omega": Qomega,
    "dq": Qdq,
    "contact": Qcontact,
    "acc": Qacc,
    "tau": Qtau,
    "grf": Qgrf,
}

initial_state = jnp.concatenate([p0, quat0, q0, jnp.zeros(nv)])

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
    qpos_next = jnp.concatenate(
        [
            qpos[:3] + qvel_next[:3] * dt,
            math.quat_integrate(qpos[3:7], qvel_next[3:6], dt),
            qpos[7:] + qvel_next[6:] * dt,
        ]
    )
    return qpos_next, qvel_next


def dynamics(x, u, t, parameter):
    del t, parameter
    qpos, qvel = _state_parts(x)
    qacc = u[qacc_slice]
    qpos_next, qvel_next = _integrate_state(qpos, qvel, qacc)
    return jnp.concatenate([qpos_next, qvel_next])

@jax.jit
def _foot_positions(qpos, qvel):
    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos, qvel=qvel)
    data = mjx.fwd_position(_MJX_MODEL, data)
    return jnp.concatenate([data.geom_xpos[contact_id] for contact_id in _CONTACT_ID])

def equality(x, u, t, parameter):
    
    qpos, qvel = _state_parts(x)
    qacc = u[qacc_slice]
    tau = u[tau_slice]
    grf = u[grf_slice]
    data = mjx.make_data(_MJX_MODEL)
    data = data.replace(qpos=qpos, qvel=qvel, qacc=qacc)
    data = mjx.fwd_position(_MJX_MODEL, data)
    data = mjx.fwd_velocity(_MJX_MODEL, data)
    M = data.qM
    D = data.qfrc_bias
    qfrc_inverse = M @ qacc + D
    jacobians = []
    for contact_id, body_id in zip(_CONTACT_ID, _BODY_ID):
        jac, _ = mjx.jac(_MJX_MODEL, data, data.geom_xpos[contact_id], body_id)
        jacobians.append(jac)
    contact_jacobian = jnp.concatenate(jacobians, axis=1)
    contact = parameter[t, :4]
    grf = jnp.concatenate([grf[:3]*contact[0], grf[3:6]*contact[1], grf[6:9]*contact[2], grf[9:12]*contact[3]])
    contact_wrench = contact_jacobian @ grf
    generalized_actuation = jnp.concatenate([jnp.zeros(6, dtype=u.dtype), tau])
    return qfrc_inverse - contact_wrench - generalized_actuation


def cost(W, reference, x, u, t):
    qpos, qvel = _state_parts(x)
    p = qpos[:3]
    quat = qpos[3:7]
    q = qpos[7:]
    dp = qvel[:3]
    omega = qvel[3:6]
    dq = qvel[6:]
    acc = u[qacc_slice]
    tau = u[tau_slice]
    grf = u[grf_slice]
    p_leg = _foot_positions(qpos, qvel)
    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    p_leg_ref = reference["foot"][t]
    grf_ref = reference["grf"][t]

    quat_err = math.quat_sub(quat, quat_ref)
    
    stage_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + quat_err.T @ W["rot"] @ quat_err
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
        + (p_leg-p_leg_ref).T @ W["contact"] @ (p_leg-p_leg_ref)
        + acc.T @ W["acc"] @ acc
        + tau.T @ W["tau"] @ tau
        + (grf - grf_ref).T @ W["grf"] @ (grf - grf_ref)
    )
    terminal_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + quat_err.T @ W["rot"] @ quat_err
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
    )
    return jnp.where(t == N, 0.5 * terminal_cost, 0.5 * stage_cost)


class InverseDynamicsMPCData(PyTreeNode):
    dt: float
    time: jnp.ndarray
    duty_factor: jnp.ndarray
    step_freq: jnp.ndarray
    step_height: jnp.ndarray
    contact_time: jnp.ndarray
    liftoff: jnp.ndarray
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    Veq0: jnp.ndarray
    W: jnp.ndarray
    regularization: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 1))
def _update_warm_start(horizon, shift, u_ref, x0, X_prev, U_prev, X, U, V, Veq):
    q_slice = slice(7, 7 + n_joints)
    dq_slice = slice(nq + 6, nq + nv)
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
        data = mujoco.MjData(self.model)
        mujoco.mj_fwdPosition(self.model, data)
        self.initial_state = jnp.asarray(config.initial_state)
        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_Veq0 = jnp.zeros((config.N, config.equality_dim))
        self.initial_liftoff = config.p_legs0

        self.dynamics = config.dynamics
        solver = partial(
            optimizers.mpc_equality,
            config.cost,
            self.dynamics,
            None,
            limited_memory,
            equality=config.equality,
            num_alpha=config.equality_num_alpha,
        )

        def solve(reference, parameter, W, x0, X0, U0, V0, Veq0, regularization):
            return solver(
                reference,
                parameter,
                W,
                x0,
                X0,
                U0,
                V0,
                Veq_in=Veq0,
                regularization=regularization,
            )

        self._solve = jax.jit(solve)
        robot_mass = data.qM[0]
        self._ref_gen = partial(config.reference_generator, mass=robot_mass)
        self._timer_run = jax.jit(mpc_utils.timer_run)
        self._update_warm_start = partial(
            _update_warm_start,
            config.N,
            self.shift,
            config.u_ref,
        )

    def make_data(self):
        return InverseDynamicsMPCData(
            dt=dt,
            time=jnp.asarray(0.0, dtype=jnp.float32),
            duty_factor=duty_factor,
            step_freq=step_freq,
            step_height=step_height,
            contact_time=timer_t,
            liftoff=self.initial_liftoff,
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            W=W,
            regularization=regularization,
        )

    def control_output(self, x0, X, U, reference, parameter):
        del x0, X, reference, parameter
        return jnp.clip(U[0, tau_slice], min_torque, max_torque)

    def run(self, data, x, command, contact=None):
        if contact is None:
            contact = jnp.zeros(n_contact, dtype=x.dtype)
        current_time = data.time + jnp.asarray(1 / self.mpc_frequency, dtype=data.time.dtype)
        _, contact_time = self._timer_run(
            data.duty_factor,
            data.step_freq,
            data.contact_time,
            1 / self.mpc_frequency,
        )
        qpos, qvel = _state_parts(x)
        foot = _foot_positions(qpos, qvel)
        reference_state = jnp.concatenate([x, foot, jnp.zeros(3 * n_contact, dtype=x.dtype)])
        reference, parameter, liftoff = self._ref_gen(
            duty_factor=data.duty_factor,
            step_freq=data.step_freq,
            step_height=data.step_height,
            t_timer=data.contact_time,
            x=reference_state,
            foot=foot,
            input=command,
            liftoff=data.liftoff,
            contact=contact,
            current_time=current_time,
        )
        X, U, V, Veq, regularization, alpha_best, any_accepted = self._solve(
            reference,
            parameter,
            data.W,
            x,
            data.X0,
            data.U0,
            data.V0,
            data.Veq0,
            data.regularization,
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
        return (
            data.replace(
                time=current_time,
                X0=X0,
                U0=U0,
                V0=V0,
                Veq0=Veq0,
                contact_time=contact_time,
                liftoff=liftoff,
                regularization=regularization,
            ),
            tau,
            q,
            dq,
            alpha_best,
            any_accepted,
            reference,
        )

    def reset(self, data, qpos, qvel, foot=None):
        x = (
            self.initial_state.at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
        )
        liftoff = _foot_positions(jnp.ravel(qpos), jnp.ravel(qvel)) if foot is None else jnp.ravel(foot)
        return data.replace(
            time=jnp.asarray(0.0, dtype=jnp.float32),
            X0=jnp.tile(x, (N + 1, 1)),
            U0=self.initial_U0,
            V0=self.initial_V0,
            Veq0=self.initial_Veq0,
            liftoff=liftoff,
            contact_time=timer_t,
        )


def state_to_qpos(x):
    return x[:nq]


reference_generator = partial(
    mpc_utils.reference_generator,
    use_terrain_estimation,
    N,
    dt,
    n_joints,
    n_contact,
    foot0=p_legs0,
    q0=q0,
    clearence_speed=clearance_speed,
)
