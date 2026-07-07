import os

import jax
import jax.numpy as jnp

task_name = "acrobot_swingup"
benchmark_mode = "direct"

dir_path = os.path.dirname(os.path.realpath(__file__))
scene_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/acrobot/scene.xml"

N = 200
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
reference = {"dummy": jnp.zeros(N + 1)}


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

    g1 = m1 * g * lc1 * jnp.sin(theta1) + m2 * g * (l1 * jnp.sin(theta1) + lc2 * jnp.sin(theta1 + theta2))
    g2 = m2 * lc2 * g * jnp.sin(theta1 + theta2)

    D = jnp.array([[d11, d12], [d21, d22]])
    C = jnp.array([[c11, c12], [c21, c22]])
    G = jnp.array([g1, g2])

    theta_dot = jnp.array([theta1_dot, theta2_dot])
    theta_dot_new = theta_dot + dt * jnp.linalg.inv(D) @ (jnp.array([0.0, u[0]]) - C @ theta_dot - G)
    theta_new = jnp.array([theta1, theta2]) + dt * theta_dot_new
    return jnp.concatenate([theta_dot_new, theta_new])


x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
x_ref = jnp.array([0.0, 0.0, jnp.pi, 0.0])
u_ref = jnp.array([0.0])

Q = jnp.diag(jnp.array([1e-5 / dt, 1e-5 / dt, 1e-5 / dt, 1e-5 / dt]))
R = jnp.diag(jnp.array([1e-4 / dt]))
Q_f = jnp.diag(jnp.array([10.0, 10.0, 100.0, 100.0]))
W = {"dummy": jnp.zeros((N, 1))}


def cost(W, reference, x, u, t):
    del W, reference
    stage_cost = (x - x_ref).T @ Q @ (x - x_ref) + (u - u_ref).T @ R @ (u - u_ref)
    term_cost = (x - x_ref).T @ Q_f @ (x - x_ref)
    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)


hessian_approx = None


initial_X0 = jnp.tile(x0, (N + 1, 1))
initial_U0 = jnp.tile(u_ref, (N, 1))
initial_V0 = jnp.zeros((N + 1, n))

solver_mode = "fddp"


def state_to_qpos(x):
    return x[2:]


def state_to_qvel(x):
    return x[:2]
