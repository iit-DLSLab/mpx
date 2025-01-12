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


from jax.scipy.spatial.transform import Rotation


from primal_dual_ilqr.utils.rotation import quaternion_integration,rpy_intgegration
import primal_dual_ilqr.utils.mpc_utils as mpc_utils
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)





model_path = './urdfs/aliengo.urdf'

joints_name = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]
contact_frame = ['FL_foot','FR_foot','RL_foot','RR_foot']

n_joints = len(joints_name)
n_contact = len(contact_frame)

w_H_b0 = jnp.block([
    [jnp.eye(3), jnp.array([[0], [0], [0.33]])],
    [jnp.zeros((1, 3)), jnp.array([[1]])]
])
q0 = jnp.array([0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8])

mass = 24
inertia = jnp.array([[ 2.5719824e-01,  1.3145953e-03, -1.6161108e-02],[ 1.3145991e-03,  1.0406910e+00,  1.1957530e-04],[-1.6161105e-02,  1.1957530e-04,  1.0870107e+00]])
inertia_inv = jnp.linalg.inv(inertia)
p_legs0 = jnp.array([ 0.2717287,   0.13780001,  0.02074774,  0.2717287,  -0.13780001,  0.02074774, -0.20967132,  0.13780001,  0.02074774, -0.20967132, -0.13780001,  0.02074774])

@jax.jit
def dynamics(x, u, t,parameter):
    # Extract state variables
    p = x[:3]
    quat = x[3:6]
    # p_legs = x[6:6+n_joints]
    dp = x[6:9]
    omega = x[9:12]
    # dp_leg = u[:n_joints]
    grf = u

    contact = parameter[t,:4]
    p_legs = parameter[t,4:]

    # Convert quaternion to rotation matrix
    # R = Rotation.from_quat(quat).as_matrix()

    # w_H_b = jnp.block([
    #     [R, p.reshape((3, 1))],
    #     [jnp.zeros((1, 3)), jnp.array([[1]])]
    # ])

    dp_next = dp + (jnp.array([0, 0, -9.81]) + (1 / mass) * (grf[:3]*contact[0] + grf[3:6]*contact[1] + grf[6:9]*contact[2] + grf[9:12]*contact[3])) * dt

    p0 = p_legs[:3]
    p1 = p_legs[3:6]
    p2 = p_legs[6:9]
    p3 = p_legs[9:]

    omega_next = omega + inertia_inv@((jnp.cross(p0 - p, grf[:3])*contact[0] + jnp.cross(p1 - p, grf[3:6])*contact[1] + jnp.cross(p2 - p, grf[6:9])*contact[2] + jnp.cross(p3 - p, grf[9:12])*contact[3]))*dt

    # Semi-implicit Euler integration
    p_new = p + dp_next * dt
    rpy_new = rpy_intgegration(omega_next, quat, dt)
    # p_legs_new = p_legs# + dp_leg * dt

    x_next = jnp.concatenate([p_new, rpy_new, dp_next, omega_next])

    return x_next

Qp = jnp.diag(jnp.array([0, 0, 1000]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1e2
Qdp = jnp.diag(jnp.array([100, 100, 100]))
Qomega = jnp.diag(jnp.array([10, 10, 10]))
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-2
Rgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
Qquat = jnp.diag(jnp.ones(4)) * 1e-1
Qrpy = jnp.diag(jnp.array([100,100,0]))

# Define the cost function
@jax.jit
def cost(x, u, t, reference):

    p = x[:3]
    dp = x[6:9]
    omega = x[9:12]
    grf = u
    rpy = x[3:6]

    p_ref = reference[t,:3]
    rpy_ref = reference[t,3:6]
    dp_ref = reference[t,6:9]
    omega_ref = reference[t,9:12]

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref) + grf.T @ Rgrf @ grf #+ dq.T @ Rgrf @ dq + (q - q_ref).T @ Qq @ (q - q_ref)
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref) #+ omega.T @ Qomega @ omega + (q - q_ref).T @ Qq @ (q - q_ref)

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
# Problem dimensions
N = 100  # Number of stages
n = 12   # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = 12    # Number of controls (F)
dt = 0.01  # Time step

p0 = jnp.array([0, 0, 0.33])
quat0 = jnp.array([1, 0, 0, 0])
rpy0 = jnp.array([0, 0, 0])
x0 = jnp.concatenate([p0, rpy0, jnp.zeros(3), jnp.array([0, 0, 0])])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.4])
quat_ref = jnp.array([0, 0, 0, 1])
rpy_ref = jnp.array([0, 0, 0])
q_ref = jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8])
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

grf_ref = jnp.zeros(3 * n_contact)

u_ref = grf_ref

U0 = jnp.tile(grf_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))
reference = jnp.tile(jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref]), (N + 1, 1))
parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))

from timeit import default_timer as timer

mu = 1e-3  
@jax.jit
def work(reference,parameter,x0,X0,U0,V0,mu):
    return optimizers.mpc(
        cost,
        dynamics,
        reference,
        parameter,
        x0,
        X0,
        U0,
        V0,
        mu,
    )
# Ns = [20,50,100,200,300,400,500]
# for N in Ns:
#     U0 = jnp.tile(grf_ref, (N, 1))
#     X0 = jnp.tile(x0, (N + 1, 1))
#     V0 = jnp.zeros((N + 1, n ))
#     reference = jnp.tile(jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref]), (N + 1, 1))
#     parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))
#     X,U,V, _,_,_ = work(reference,parameter,x0,X0,U0,V0,mu)
#     start = timer()
#     for i in range(100):
#         X,U,V, _,_,_ = work(reference,parameter,x0,X0,U0,V0,mu)
#     end = timer()
#     print(f"Time for N = {N} is {(end-start)/100}")
Ns = [1,2,4,8,16,32,64,128,256,512,1024,2048]
N = 20
times = []
for val in Ns:
    U0 = jnp.tile(grf_ref, (N, 1))
    X0 = jnp.tile(x0, (N + 1, 1))
    V0 = jnp.zeros((N + 1, n ))
    reference = jnp.tile(jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref]), (N + 1, 1))
    parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))
    batch_x0 = jnp.tile(x0, (val, 1))
    batch_U0 = jnp.tile(U0, (val, 1, 1))
    batch_X0 = jnp.tile(X0, (val, 1, 1))
    batch_V0 = jnp.tile(V0, (val, 1, 1))
    mu_batch = jnp.tile(mu,val)
    batch_reference = jnp.tile(reference,(val,1,1))
    batch_parameter = jnp.tile(parameter,(val,1,1))
    
    
    vmap_work = jax.vmap(work)
    X,U,V, _,_,_ = vmap_work(batch_reference,batch_parameter,batch_x0,batch_X0,batch_U0,batch_V0,mu_batch)
    start = timer()
    for i in range(10):
        X,U,V, _,_,_ = vmap_work(batch_reference,batch_parameter,batch_x0,batch_X0,batch_U0,batch_V0,mu_batch)
        X.block_until_ready()
    end = timer()
    print(f"Time for n = {val} is {(end-start)/10}")
    times.append((end-start)/10)

import matplotlib.pyplot as plt
plt.plot(Ns, times, marker='o')
plt.xlabel('Batch Size (n)')
plt.ylabel('Average Time (s)')
plt.title('Average Time vs Batch Size')
plt.xscale('log')
plt.grid(True)
plt.show()
    