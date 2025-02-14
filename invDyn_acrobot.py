import os
# os.environ['XLA_FLAGS'] = (
    # '--xla_gpu_enable_triton_softmax_fusion=true '
    # '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_highest_priority_async_stream=true '
# )
os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })
import jax.numpy as jnp
import jax

import numpy as np

from trajax import integrators
from trajax.experimental.sqp import util

import  primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from functools import partial

from jax import grad, jvp

# Problem dimensions
N = 100  # Number of stages
n = 4    # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = 3    # Number of controls (F)

# Doble pendulum parameters
m1 = 1  # Mass of the cart
m2 = 1  # Mass of the pendulum
l1 = 0.5  # Length first pendulum
l2 = 0.5  # Length second pendulum
lc1 = 0.5*l1  # Length to the center of mass of the first pendulum
lc2 = 0.5*l2  # Length to the center of mass of the second pendulum
g = 9.81  # Acceleration due to gravity
I1 = m1 * (l1 * l1) / 3  # Moment of inertia of the first pendulum
I2 = m2 * (l2 * l2) / 3  # Moment of inertia of the second pendulum
dt = 0.01  # Time step
parameter = jnp.zeros(N+1)
reference = jnp.zeros(N+1)

def dynamics(x, u,t,parameter):
    del t
    theta1_dot = x[0]
    theta2_dot = x[1]
    theta1 = x[2]
    theta2 = x[3]

    theta1_acc = u[0]
    theta2_acc = u[1]
    
    theta_dot_new = jnp.array([theta1_dot,theta2_dot]) + dt * jnp.array([theta1_acc,theta2_acc])
    theta_new = jnp.array([theta1,theta2]) + dt * theta_dot_new

    return jnp.concatenate([theta_dot_new,theta_new])

def equality_constraints(x, u, t,parameter):

    del t
    theta1_dot = x[0]
    theta2_dot = x[1]
    theta1 = x[2]
    theta2 = x[3]
    theta1_acc = u[0]
    theta2_acc = u[1]
    tau = u[2]

    d11 = I1 + I2 + m2 * l1*l1 + 2 * m2 * l1 * lc2 * jnp.cos(theta2)
    d12 = I2 + m2 * l1 * lc2 * jnp.cos(theta2)
    d21 = d12
    d22 = I2

    c11 = -2 * m2 * l1 * lc2 * jnp.sin(theta2) * theta2_dot
    c12 = -m2 * l1 * lc2 * jnp.sin(theta2) * theta2_dot
    c21 = m2 * l1 * lc2 * jnp.sin(theta2) * theta1_dot
    c22 = 0

    g1 = m1*g*lc1*jnp.sin(theta1)+m2*g*(l1*jnp.sin(theta1)+lc2*jnp.sin(theta1+theta2))
    g2 = m2 * lc2 * g * jnp.sin(theta1 + theta2)

    D = jnp.array([[d11, d12], [d21, d22]])
    C = jnp.array([[c11, c12], [c21, c22]])
    G = jnp.array([g1, g2])
    acc = jnp.array([theta1_acc,theta2_acc])

    return D @ acc + C @ jnp.array([theta1_dot,theta2_dot]) + G - jnp.array([0,tau])

pos_0 = jnp.array([0.0,0.0,0.1, 0.0])
# pos_0 = jnp.array([-3., 0.5, 0., 0])
pos_g = jnp.array([0.0, 0.0 ,0.0, 0.0])

x_ref = jnp.array([0, 0,3.14,3.14])
u_ref = jnp.array([0.0,0.0,0.0])

# Define the cost function
Q = jnp.diag(jnp.array([1e-5/dt, 1e-5/dt, 1e-5/dt, 1e-5/dt]))
R =  jnp.diag(jnp.array([1e-4/dt, 1e-4/dt, 1e-4/dt]))
Q_f = jnp.diag(jnp.array([10.0, 10.0, 100.0, 100.0]))

@jax.jit
def cost(x, u,t,reference):
    alpha = 0.1
    sigma = 5
    # torque_limit = 5*jnp.ones_like(u) - u
    # quadratic_barrier = alpha/2*(jnp.square((u-2*sigma)/sigma)-jnp.ones_like(torque_limit))
    # log_barrier = -alpha*jnp.log(torque_limit)
    # torque_limit = jnp.where(torque_limit>sigma,log_barrier,quadratic_barrier+log_barrier)
    stage_cost = (x-x_ref).T @ Q @ (x-x_ref) + (u-u_ref).T @ R @ (u-u_ref) #+ jnp.sum(torque_limit)
    term_cost = (x-x_ref).T @ Q_f @ (x-x_ref)
    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
x0 = pos_0
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N+ 1, n))
Veq0 = jnp.zeros((N+1, 2))
from timeit import default_timer as timer


@jax.jit
def work(x0,X0,U0,V0,Veq0,rho,tol):
    return optimizers.eq_con_mpc(
        cost,
        dynamics,
        equality_constraints,
        False,
        reference,
        parameter,
        x0,
        X0,
        U0,
        V0,
        Veq0,
        rho,
        tol
    )

import numpy as np
from IPython.display import display, clear_output
def fwd_dynamics(x, u,t,parameter):
    del t
    theta1_dot = x[0]
    theta2_dot = x[1]
    theta1 = x[2]
    theta2 = x[3]

    d11 = I1 + I2 + m2 * l1*l1 + 2 * m2 * l1 * lc2 * jnp.cos(theta2)
    d12 = I2 + m2 * l1 * lc2 * jnp.cos(theta2)
    d21 = d12
    d22 = I2

    c11 = -2 * m2 * l1 * lc2 * jnp.sin(theta2) * theta2_dot
    c12 = -m2 * l1 * lc2 * jnp.sin(theta2) * theta2_dot
    c21 = m2 * l1 * lc2 * jnp.sin(theta2) * theta1_dot
    c22 = 0

    g1 = m1*g*lc1*jnp.sin(theta1)+m2*g*(l1*jnp.sin(theta1)+lc2*jnp.sin(theta1+theta2))
    g2 = m2 * lc2 * g * jnp.sin(theta1 + theta2)

    D = jnp.array([[d11, d12], [d21, d22]])
    C = jnp.array([[c11, c12], [c21, c22]])
    G = jnp.array([g1, g2])

    theta_dot_new = jnp.array([theta1_dot,theta2_dot]) + dt * jnp.linalg.inv(D)@(jnp.array([0,u[0]]) - C@(jnp.array([theta1_dot,theta2_dot])) - G)
    theta_new = jnp.array([theta1,theta2]) + dt * theta_dot_new

    return jnp.concatenate([theta_dot_new,theta_new])
jittedDynamics = jax.jit(fwd_dynamics)
import matplotlib.pyplot as plt

plt.ion()
fig, (ax, ax_u) = plt.subplots(2, 1)
line, = ax.plot([], [], 'o-', lw=2)
line.set_color('b')
line1, = ax.plot([], [], 'o-', lw=2)
line1.set_color('r')
line1.set_alpha(0.5)
line2, = ax.plot([], [], 'o-', lw=2)
line2.set_color('r')
ax.set_xlim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
ax.set_ylim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
ax.set_aspect('equal')
plt.grid()

def update_plot1(x1, y1, x2, y2, x1f, y1f, x2f, y2f, x1hf, y1hf, x2hf, y2hf):
    line.set_data([0, x1, x2], [0, y1, y2])
    line1.set_data([0, x1hf, x2hf], [0, y1hf, y2hf])
    line2.set_data([0, x1f, x2f], [0, y1f, y2f])
    display(fig)
    clear_output(wait=True)
    plt.pause(dt)
def update_plot(x1, y1, x2, y2, x1f, y1f, x2f, y2f):
    line.set_data([0, x1, x2], [0, y1, y2])
    line2.set_data([0, x1f, x2f], [0, y1f, y2f])
    display(fig)
    clear_output(wait=True)
    plt.pause(dt)
u_history = []
counter = 0
rho = 10
tol = np.pow(rho,-0.1)
while True:
    x0 = jittedDynamics(x0,U0[0,2:],0,parameter)
    X0,U0,V0,Veq0,rho,tol = work(x0,X0,U0,V0,Veq0,rho,tol)
    u_history.append(U0[0,2])
    x1 = l1 * jnp.sin(x0[2])
    y1 = -l1 * jnp.cos(x0[2])
    x2 = x1 + l2 * jnp.sin(x0[3])
    y2 = y1 - l2 * jnp.cos(x0[3])

    x1f = l1*jnp.sin(X0[-1,2])
    y1f = -l1*jnp.cos(X0[-1,2])
    x2f = x1f + l2*jnp.sin(X0[-1,3])
    y2f = y1f - l2*jnp.cos(X0[-1,3])
    if counter == N/2 or counter == 0:
        x1hf = l1*jnp.sin(X0[int(N/2),2])
        y1hf = -l1*jnp.cos(X0[int(N/2),2])
        x2hf = x1hf + l2*jnp.sin(X0[int(N/2),3])
        y2hf = y1hf - l2*jnp.cos(X0[int(N/2),3])
        update_plot1(x1, y1, x2, y2, x1f, y1f, x2f, y2f, x1hf, y1hf, x2hf, y2hf)
        counter = 0
    update_plot(x1, y1, x2, y2, x1f, y1f, x2f, y2f)
    counter += 1
    
    ax_u.clear()
    ax_u.plot(u_history)
    ax_u.set_title('Control Input Over Time')
    ax_u.set_xlabel('Time Step')
    ax_u.set_ylabel('Control Input (u)')
    clear_output(wait=True)
    plt.pause(dt)

