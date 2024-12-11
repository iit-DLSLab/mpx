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
m = 1    # Number of controls (F)

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
parameter = []
reference = []

def dynamics(x, u,t,parameter):
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


pos_0 = jnp.array([0.0,0.0,0.0, 0.0])
# pos_0 = jnp.array([-3., 0.5, 0., 0])
pos_g = jnp.array([0.0, 0.0 ,0.0, 0.0])

x_ref = jnp.array([0, 0,3.14,0])
u_ref = jnp.array([0.0])

# Define the cost function
Q = jnp.diag(jnp.array([1e-5/dt, 1e-5/dt, 1e-5/dt, 1e-5/dt]))
R = jnp.diag(jnp.array([ 1e-4/dt]))
Q_f = jnp.diag(jnp.array([10.0, 10.0, 100.0, 100.0]))

@jax.jit
def cost(x, u,t,reference):
    stage_cost = (x-x_ref).T @ Q @ (x-x_ref) + (u-u_ref).T @ R @ (u-u_ref)
    term_cost = (x-x_ref).T @ Q_f @ (x-x_ref)
    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
x0 = pos_0
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
from timeit import default_timer as timer


@jax.jit
def work():
    return optimizers.primal_dual_ilqr(
        cost,
        dynamics,
        reference,
        parameter,
        x0,
        X0,
        U0,
        np.zeros((N + 1, 4)),
        max_iterations=10000,
        psd_delta=1e-3,
    )


X, U, V, num_iterations, g, c, no_errors = work()
X.block_until_ready()

# @partial(jax.jit, static_argnums=(0, 1))
# def workWS(cost,dynamics,x0,X,U,V):
#     return optimizers.primal_dual_ilqr(
#         cost,
#         dynamics,
#         x0,
#         X,
#         U,
#         V,
#         max_iterations=10000,
#         psd_delta=1e-3,
#     )



n = 10
# Define a function to perform the work and return the state trajectory
start = timer()
for i in range(n):
    X, _, _, _, _, _, _ = work()
    X.block_until_ready()
# Execute the vectorized function
end = timer()

t = (end - start)/n

print(f"{t=},{num_iterations=}, {g=}, {no_errors=}")
# start = timer()
# for i in range(n):
#     X, _, _, _, _, _, _ = work()
#     X.block_until_ready()
# # Execute the vectorized function
# end = timer()

# t = (end - start)

# print(f"{t=},{num_iterations=}, {g=}, {no_errors=}")
# Initialize arrays to store positions
x1 = np.zeros(N)
y1 = np.zeros(N)
x2 = np.zeros(N)
y2 = np.zeros(N)

# Integrate the equations of motion
for i in range(N):
    
    theta1 = X[i,2]
    theta2 = X[i,3]
    x1[i] = l1 * np.sin(theta1)
    y1[i] = -l1 * np.cos(theta1)
    x2[i] = x1[i] + l2 * np.sin(theta1+theta2)
    y2[i] = y1[i] - l2 * np.cos(theta1+theta2)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
line, = ax.plot([], [], 'o-', lw=2)

# Initialization function
def init():
    line.set_data([], [])
    return line,

# Animation function
def update(frame):
    thisx = [0, x1[frame], x2[frame]]
    thisy = [0, y1[frame], y2[frame]]
    line.set_data(thisx, thisy)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=int(dt*10000))
# Create subplots
fig, axis = plt.subplots(3, 1)

axis[0].plot(x2, y2, 'r')
axis[0].set_title('Double Pendulum')
axis[1].plot(np.linspace(0,(N+1)*dt,N+1),X[:,0])
axis[1].plot(np.linspace(0,(N+1)*dt,N+1),X[:,1])
axis[2].plot(np.linspace(0,(N+1)*dt,N),U[:,0])
# Show the plot
plt.show()