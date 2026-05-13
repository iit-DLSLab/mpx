import argparse
import os
import sys
import time
from timeit import default_timer as timer
from types import SimpleNamespace

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


scene_path = os.path.abspath(os.path.join(dir_path, "..", "data", "acrobot", "scene.xml"))

N = 100
n = 4
m = 1
dt = 0.01

m1 = 1.0
m2 = 1.0
l1 = 0.5
l2 = 0.5
lc1 = 0.5 * l1
lc2 = 0.5 * l2
g = 9.81
I1 = m1 * l1 * l1 / 12.0
I2 = m2 * l2 * l2 / 12.0

parameter = jnp.zeros(N + 1)
reference = jnp.zeros(N + 1)


def dynamics(x, u, t, parameter):
    del t, parameter
    theta1_dot = x[0]
    theta2_dot = x[1]
    theta1 = x[2]
    theta2 = x[3]

    d11 = (
        I1
        + I2
        + m1 * lc1 * lc1
        + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * jnp.cos(theta2))
    )
    d12 = I2 + m2 * (lc2 * lc2 + l1 * lc2 * jnp.cos(theta2))
    d21 = d12
    d22 = I2 + m2 * lc2 * lc2

    c11 = -2.0 * m2 * l1 * lc2 * jnp.sin(theta2) * theta2_dot
    c12 = -m2 * l1 * lc2 * jnp.sin(theta2) * theta2_dot
    c21 = m2 * l1 * lc2 * jnp.sin(theta2) * theta1_dot
    c22 = 0.0

    g1 = m1 * g * lc1 * jnp.sin(theta1) + m2 * g * (
        l1 * jnp.sin(theta1) + lc2 * jnp.sin(theta1 + theta2)
    )
    g2 = m2 * lc2 * g * jnp.sin(theta1 + theta2)

    D = jnp.array([[d11, d12], [d21, d22]])
    C = jnp.array([[c11, c12], [c21, c22]])
    G = jnp.array([g1, g2])

    theta_dot = jnp.array([theta1_dot, theta2_dot])
    theta_dot_new = theta_dot + dt * jnp.linalg.inv(D) @ (
        jnp.array([0.0, u[0]]) - C @ theta_dot - G
    )
    theta_new = jnp.array([theta1, theta2]) + dt * theta_dot_new
    return jnp.concatenate([theta_dot_new, theta_new])


x_init = jnp.array([0.0, 0.0, 0.1, 0.0])
x_ref = jnp.array([0.0, 0.0, jnp.pi, 0.0])
u_ref = jnp.array([0.0])

Q = jnp.diag(jnp.array([1e-5 / dt, 1e-5 / dt, 1e-5 / dt, 1e-5 / dt]))
R = jnp.diag(jnp.array([1e-4 / dt]))
Q_f = jnp.diag(jnp.array([10.0, 10.0, 100.0, 100.0]))
W = jnp.zeros((N, 1))


@jax.jit
def cost(W, reference, x, u, t):
    del W, reference
    stage_cost = (x - x_ref).T @ Q @ (x - x_ref) + (u - u_ref).T @ R @ (u - u_ref)
    term_cost = (x - x_ref).T @ Q_f @ (x - x_ref)
    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)


hessian_x = jax.hessian(cost, argnums=2)
hessian_u = jax.hessian(cost, argnums=3)
hessian_x_u = jax.jacobian(jax.grad(cost, argnums=2), argnums=3)


def hessian_approx(*args):
    return hessian_x(*args), hessian_u(*args), hessian_x_u(*args)


@jax.jit
def state_to_qpos(x):
    return x[2:]


@jax.jit
def state_to_qvel(x):
    return x[:2]


@jax.jit
def qpos_qvel_to_state(qpos, qvel):
    return jnp.concatenate([qvel, qpos])


def _shift_trajectory(trajectory):
    return jnp.concatenate([trajectory[1:], trajectory[-1:]], axis=0)


def _ghost_qpos_sequence(X, stride):
    return np.asarray([state_to_qpos(state) for state in X[::stride]])


def _sim_state(data):
    return qpos_qvel_to_state(jnp.asarray(data.qpos), jnp.asarray(data.qvel))


def main(headless=False, steps=500, solver_mode="primal_dual"):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    config = SimpleNamespace(solver_mode=solver_mode)
    _, solve = mpc_wrapper.build_solver_step(
        config,
        cost,
        dynamics,
        hessian_approx,
        limited_memory=True,
    )
    solve = jax.jit(solve)
    x = x_init
    X0 = jnp.tile(x, (N + 1, 1))
    U0 = jnp.tile(u_ref, (N, 1))
    V0 = jnp.zeros((N + 1, n))

    qpos = np.asarray(state_to_qpos(x))
    qvel = np.asarray(state_to_qvel(x))
    data.qpos = qpos
    data.qvel = qvel
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    x = _sim_state(data)

    X_warm, U_warm, V_warm = solve(reference, parameter, W, x, X0, U0, V0)
    X_warm.block_until_ready()
    X0 = X_warm
    U0 = U_warm
    V0 = V_warm

    ghost_geoms = None
    ghost_scratch = mujoco.MjData(model)
    step_idx = 0

    def step_controller(viewer=None):
        nonlocal x, X0, U0, V0, ghost_geoms, ghost_scratch, step_idx

        start = timer()
        X, U, V = solve(reference, parameter, W, x, X0, U0, V0)
        X.block_until_ready()
        stop = timer()

        u = U[0]
        data.ctrl[0] = float(u[0])
        mujoco.mj_step(model, data)
        x = _sim_state(data)
        X0 = _shift_trajectory(X)
        U0 = _shift_trajectory(U)
        V0 = _shift_trajectory(V)

        if viewer is not None:
            qpos_sequence = _ghost_qpos_sequence(np.asarray(X), stride=5)
            if qpos_sequence.shape[0] == 0:
                qpos_sequence = np.asarray([np.asarray(state_to_qpos(x))])
            alphas = np.linspace(0.15, 0.55, qpos_sequence.shape[0], dtype=np.float64)
            ghost_geoms, ghost_scratch = sim_utils.render_ghost_trajectory(
                viewer,
                model,
                qpos_sequence,
                alphas,
                ghost_geoms=ghost_geoms,
                scratch_data=ghost_scratch,
                subsample=5
            )

        print(
            f"MPC time: {1e3 * (stop - start):.2f} ms | "
            f"u: {float(u[0]):+.3f} | "
            f"x: {np.asarray(x)}"
        )
        step_idx += 1

    if headless:
        for _ in range(steps):
            step_controller()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            step_controller(viewer)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--solver", choices=("primal_dual", "fddp", "lipa"), default="primal_dual")
    args = parser.parse_args()
    main(headless=args.headless, steps=args.steps, solver_mode=args.solver)
