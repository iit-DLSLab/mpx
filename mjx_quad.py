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

import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import mujoco.viewer
import matplotlib.pyplot as plt

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

class  timer_class :
    def __init__(self,duty_factor, step_freq, delta):
        self.duty_factor = duty_factor
        self.step_freq = step_freq
        self.delta = delta
        self.t = np.zeros(len(delta))
        self.n_contact = len(delta)
        self.init = [True]*len(delta)
    def run(self,dt):
        contact = np.zeros(self.n_contact)
        for leg in range(self.n_contact):
            if self.t[leg] == 1.0:
                self.t[leg] = 0 #restart
            self.t[leg] += dt*self.step_freq
            if self.delta[leg] == -1:
                contact[leg] = 0
            else:
                if self.init[leg] :
                    if self.t[leg] < self.delta[leg]:
                        contact[leg] = 1
                    else :
                        self.init[leg] = False
                        contact[leg] = 1
                        self.t[leg] = 0
                else:
                    if self.t[leg] < self.duty_factor:
                        contact[leg] = 1
                    else:
                        contact[leg] = 0
                if self.t[leg]>1 :
                    self.t[leg] = 1
        return contact
    def set(self,t,init):
        self.t = t
        self.init = init
    #add a restart function to make it usable to start and stop the motion
def refGenerator(timer_class,initial_state,input,param,terrain_height):

    n_contact = param["n_contact"]
    N = param["N"]
    dt = param["dt"]
    foot_0 = param['foot_0']
    des_speed = input['des_speeds']
    # des_orientation =  input['des_orientation']
    des_height = input['des_height']

    contact = np.zeros((n_contact,N+1))

    ref = {}

    ref['p'] = np.zeros((3,N+1))
    ref['dp'] = np.zeros((3,N+1))

    ref['rpy'] = np.zeros((3,N+1))
    ref['omega'] = np.zeros((3,N+1))

    ref['foot'] = np.zeros((n_contact*3,N+1))
    ref['grf'] = np.zeros((3*n_contact,N))

    ref['p'][:,0] = initial_state['p']
    ref['p'][2,0] = des_height
    ref['dp'][:,0] = des_speed[:3]

    ref['omega'][:,0] = np.array([0,0,0])
    # ref['dq'][:,0] = initial_state['dq']

    step_height = 0.06

    contact[:,0] = initial_state['contact']


    for leg in range(n_contact):

        ref['grf'][3*leg:3+3*leg,0] =  (np.array([0.0,0.0,220.0])/(max(contact[:,0].sum(),1)))*contact[leg,0]
        ref['foot'][3*leg:3+3*leg,0] = initial_state['foot'][3*leg:3+3*leg]

        if contact[leg,0]:
            terrain_height[leg] = initial_state['foot'][3*leg+2]

    foot_speed = np.zeros((3,n_contact))
    foot_speed_out = np.zeros((3*n_contact))

    step_height = 0.06

    for k in range(N):

        contact[:,k+1] = timer_class.run(dt = dt)

        ref['p'][:,k+1] = ref['p'][:,k]  + des_speed[:3]*dt
        ref['dp'][:,k+1] = des_speed[:3]

        # ref['quat'][:,k+1] = np.array([0,0,0,1]) ###add the rotation propagation considering omega
        ref['omega'][:,k+1] = np.array([0,0,0])

        for leg in range(n_contact):
            ref['grf'][3*leg:3+3*leg,k] = (np.array([0.0,0.0,220.0])/(max(contact[:,k].sum(),1)))*contact[leg,k]
            if (not contact[leg,k+1] and contact[leg,k]) or (not contact[leg,k] and k == 0): #lift off event
                foothold = ref['p'][:,k] + des_speed[:3]*(1-timer_class.duty_factor)/timer_class.step_freq + foot_0[3*leg:3+3*leg] + 0.5*timer_class.duty_factor/timer_class.step_freq*des_speed[:3]
                foot_speed[:,leg ] = (foothold - ref['foot'][3*leg:3+3*leg,k])*timer_class.step_freq/(1-timer_class.duty_factor)
                if k == 0:
                    foot_speed_out[3*leg:3+3*leg] = foot_speed[:,leg]
            if not contact[leg,k+1]:
                ref['foot'][3*leg:3+3*leg,k+1] = ref['foot'][3*leg:3+3*leg,k] + foot_speed[:,leg]*dt
                ref['foot'][3*leg+2,k+1] = terrain_height[leg] + step_height * np.sin(3.14*(timer_class.t[leg]-timer_class.duty_factor)/(1-timer_class.duty_factor))

            else :
                ref['foot'][3*leg:3+3*leg,k+1] = ref['foot'][3*leg:3+3*leg,k]
    reference = jnp.concatenate([ref['p'],ref['rpy'],ref['dp'],ref['omega']],axis=0)
    parameter = jnp.concatenate([contact.T,ref['foot'].T],axis=1)
    return parameter,reference,terrain_height,foot_speed_out


model_path = './data/aliengo.xml'

joints_name = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]
contact_frame = ['FL','FR','RL','RR']
body_name = ['FL_hip', 'FL_thigh', 'FL_calf',
    'FR_hip', 'FR_thigh', 'FR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf'
]
n_joints = len(joints_name)
n_contact = len(contact_frame)

# Problem dimensions
N = 100  # Number of stages
n =  13 + 2*n_joints  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints + 3*n_contact  # Number of controls (F)
dt = 0.01  # Time step
param = {}

param["N"] = N
param["n"] = n
param["m"] = m
param["dt"] = dt
param["n_contact"] = 4

mass = 24
print('mass:\n',mass)
inertia = jnp.array([[ 2.5719824e-01,  1.3145953e-03, -1.6161108e-02],[ 1.3145991e-03,  1.0406910e+00,  1.1957530e-04],[-1.6161105e-02,  1.1957530e-04,  1.0870107e+00]])
print('inertia',inertia)

inertia_inv = jnp.linalg.inv(inertia)
p_legs0 = jnp.array([ 0.2717287,   0.13780001,  0.02074774,  0.2717287,  -0.13780001,  0.02074774, -0.20967132,  0.13780001,  0.02074774, -0.20967132, -0.13780001,  0.02074774])
print('leg:\n',p_legs0)
param['foot_0'] = p_legs0

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
print(contact_id)





np.random.seed(0)


# mujoco.mj_step1(model, data)
# dx = mjx.make_data(model)
# mjx.forward(mjx_model, mjx_data)
# point = np.random.randn(3)
# jacp, jacr = jax.jit(mjx.jac)(mjx_model, dx, point, 4)
# jacp_expected, jacr_expected = np.zeros((3, model.nv)), np.zeros((3, model.nv))
# mujoco.mj_jac(model, data, jacp_expected, jacr_expected, point, 4)
# print(jacp_expected)
# print(jacp)
# print(np.testing.assert_almost_equal(jacp, jacp_expected.T, 6))
# print(np.testing.assert_almost_equal(jacr, jacr_expected.T, 6))
@jax.jit
def dynamics(x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    
    contact = parameter[t,:4]

    tau = jnp.concatenate([jnp.zeros(6),u[:n_joints]])
    grf = u[n_joints:]

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    J_T_lamda_FL = mjx.apply_ft(mjx_model, mjx_data, contact[0]*grf[:3], jnp.zeros(3), FL_leg, body_id[2])
    J_T_lamda_FR = mjx.apply_ft(mjx_model, mjx_data, contact[1]*grf[3:6], jnp.zeros(3), FR_leg, body_id[5])
    J_T_lamda_RL = mjx.apply_ft(mjx_model, mjx_data, contact[2]*grf[6:9], jnp.zeros(3), RL_leg, body_id[8])
    J_T_lamda_RR = mjx.apply_ft(mjx_model, mjx_data, contact[3]*grf[9:], jnp.zeros(3), RR_leg, body_id[11])
    J_T_lamda = J_T_lamda_FR + J_T_lamda_FL + J_T_lamda_RL + J_T_lamda_RR

    # Solve for the ground reaction forces (grf) using the constraint dynamics
    J_FL = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[2])
    J_FR = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[5])
    J_RL = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[8])
    J_RR = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[11])
    J = jnp.concatenate([J_FL,J_FR,J_RL,J_RR],axis=0)
    v = x[n_joints+7:] + jax.scipy.linalg.cho_solve((M,False),tau + J_T_lamda - D)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v])

    return x_next

p0 = jnp.array([0, 0, 0.33])
quat0 = jnp.array([1, 0, 0, 0])
rpy0 = jnp.array([0, 0, 0])
q0 = jnp.array([0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8])
x0 = jnp.concatenate([p0, quat0,q0, jnp.zeros(6+n_joints)])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.4])
quat_ref = jnp.array([1, 0, 0, 0])
rpy_ref = jnp.array([0, 0, 0])
q_ref = jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8])
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

grf_ref = jnp.zeros(3 * n_contact)
tau_ref = jnp.zeros(n_joints)

u_ref = jnp.concatenate([tau_ref, grf_ref])

Qp = jnp.diag(jnp.array([0, 0, 1000]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qdp = jnp.diag(jnp.array([100, 100, 100]))
Qomega = jnp.diag(jnp.array([10, 10, 10]))
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-2
Rgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
Qquat = jnp.diag(jnp.ones(4)) * 1e-1
Qrpy = jnp.diag(jnp.array([100,100,0]))
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-2  
Qleg = jnp.diag(jnp.ones(12)) * 1e4

# Define the cost function
@jax.jit
def cost(x, u, t, reference):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:]
    grf = u[n_joints:]
    tau = u[:n_joints]

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    # dq_ref = reference[t,13+n_joints:]
    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    # mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]
    
    p_leg = jnp.concatenate([FL_leg,FR_leg,RL_leg,RR_leg],axis=0)

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) +  (q - q_ref).T @ Qq @ (q - q_ref) + \
                 (dp - dp_ref).T @ Qdp @ (dp - dp_ref) + (omega - omega_ref).T @ Qomega @ (omega - omega_ref) + \
                 + (p_leg - p_legs0).T @ Qleg @ (p_leg - p_legs0) + \
                 dq.T @ Qdq @ dq + grf.T @ Rgrf @ grf + tau.T @ Qtau @ tau
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))
reference = jnp.tile(jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref]), (N + 1, 1))
parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))

from timeit import default_timer as timer


# # # Integrate the dynamics using the initial condition and U_ref
# res = np.zeros((n, N + 1))
# res[:, 0] = x0
# import time
# for i in range(N):
#     res[:, i + 1] = dynamics(res[:, i], u_ref, i, parameter)
#     pd = 10*(q0-res[7:n_joints+7,i+1]) + 2*(-res[13+n_joints:,i+1])
#     u_ref = jnp.concatenate([pd,grf0])
#     # Create a Mujoco viewer instance
# counter = 0
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         if counter == N:
#             counter = 0
#         data.qpos[:] = res[:7+n_joints, counter]
#         data.qvel[:] = res[7+n_joints, counter]
#         mujoco.mj_step(model, data)
#         time.sleep(dt*10)
#         viewer.sync()
#         counter += 1


# for i in range(N):

#     viewer.render()

# viewer.close()
# # Plot the linear position (p)
# plt.figure(figsize=(12, 8))
# for i in range(3):
#     plt.plot(res[i, :], label=f'p {i+1}')
# plt.xlabel('Time step')
# plt.ylabel('Position')
# plt.legend()
# plt.grid(True)

# # Plot the linear velocity (dp)
# plt.figure(figsize=(12, 8))
# for i in range(3):
#     plt.plot(res[7 + n_joints + i, :], label=f'dp {i+1}')
# plt.xlabel('Time step')
# plt.ylabel('Velocity')
# plt.legend()
# plt.grid(True)
# plt.show()


@jax.jit
def work(reference,parameter,x0,X0,U0,V0):
    return optimizers.mpc(
        cost,
        dynamics,
        reference,
        parameter,
        x0,
        X0,
        U0,
        V0,
    )

# Vectorize the work function to solve multiple instances in parallel
# Solve in parallel
start = timer()
X,U,V, g, c =  work(reference,parameter,x0,X0,U0,V0)
end = timer()
print(f"Compilation time: {end-start}")
#calculate the jacobian of the work function
# start = timer()
# X,U,V, g, c =  work_jac(reference,parameter,x0,X0,U0,V0)
# end = timer()
# print(f"Execution time: {end-start}")
# work_vmap = jax.vmap(work, in_axes=(0,0,0,0,0,0))
# work_vmap_jit = jax.jit(work_vmap)
# times = []
# for batch_size in [2,4,8,16,32,64,128,256,512,1024]:
#     # Generate different initial states x0 for the batch
#     print(batch_size)
#     key = jax.random.PRNGKey(0)
#     x0_batch = jnp.array([x0 + jax.random.normal(key, shape=x0.shape) * 0.01 for _ in range(batch_size)])
#     # Initialize instances of X0 and U0
#     X0_batch = jnp.tile(X0, (batch_size, 1, 1))
#     U0_batch = jnp.tile(U0, (batch_size, 1, 1))
#     V0_batch = jnp.tile(V0, (batch_size, 1, 1))
#     reference_batch = jnp.tile(reference, (batch_size, 1, 1))
#     parameter_batch = jnp.tile(parameter, (batch_size, 1, 1))

#     X_batch, U_batch, V_batch, temp1 , temp2 = work_vmap_jit(reference_batch, parameter_batch, x0_batch, X0_batch, U0_batch,V0_batch)
# #
#     start = timer()
#     X_batch, U_batch, V_batch, temp1 , temp2 = work_vmap_jit(reference_batch,parameter_batch,x0_batch, X0_batch, U0_batch,V0_batch)
#     X_batch.block_until_ready()
#     end = timer()
#     times.append((end - start))
# # Plot the times over the batch size
# batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# plt.figure(figsize=(12, 8))
# plt.plot(batch_sizes, times, marker='o')
# plt.xlabel('Batch Size')
# plt.ylabel('Execution Time (seconds)')
# plt.title('Execution Time vs Batch Size')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True)
# plt.show()

# print("aaaaa",delta_time)
# Extract the first solution for further use
# X, U, V, num_iterations, g, c, no_errors = X_batch[0], U_batch[0], V_batch[0], num_iterations_batch[0], g_batch[0], c_batch[0], no_errors_batch[0]
# print(f"{num_iterations=}, {g=}, {no_errors=}")
# #INTEGRATE THE DYNAMICS
# res = np.zeros((n,N+1))
# res[:,0] = x0
# for i in range(N+1):
#     x0 = dynamics(x0,U[i,:],i)
#     # x0 = dynamics(x0,u_ref,i)
#     # print(U[i,:])
#     res[:,i] = x0
# import matplotlib.pyplot as plt

# # Plot the control inputs (U)
# plt.figure(figsize=(12, 8))
# for i in range(3):
#     plt.plot(res[i, :], label=f'p {i+1}')
# plt.xlabel('Time step')
# plt.ylabel('position')
# plt.legend()
# plt.grid(True)
# #plot quaternion
# plt.figure(figsize=(12, 8))
# # rpy = np.zeros((3,N+1))
# # for i in range(N+1):
# #     rpy[:,i] = Rotation.from_quat(res[3:7,i]).as_euler('xyz', degrees=True)
# for i in range(3):
#     plt.plot(res[i+3,:], label=f'eul {i+1}')
# plt.xlabel('Time step')
# plt.ylabel('eurler')
# plt.legend()
# plt.grid(True)
# #plot control
# plt.figure(figsize=(12, 8))
# for i in range(12):
#     plt.plot(U[:,i+12], label=f'u {i+1}')
# plt.show()


# Plot the reference control inputs (u_ref)
plt.figure(figsize=(12, 8))
for i in range(3*n_contact):
    plt.plot(U[:, n_joints+i], label=f'grf {i+1}')
plt.xlabel('Time step')
plt.ylabel('GRF')
plt.legend()
plt.grid(True)
plt.figure(figsize=(12, 8))
for i in range(n_joints):
    plt.plot(U[:, i], label=f'torque {i+1}')
plt.xlabel('Time step')
plt.ylabel('Torque')
plt.legend()
plt.grid(True)
print(U)
# Plot the linear position (p)
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.plot(X[:, i], label=f'p {i+1}')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

# Plot the linear velocity (dp)
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.plot(X[:, 7+n_joints+i], label=f'dp {i+1}')
plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.show()

from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
import copy



robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = 100.0
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

sim_frequency = 1000.0

env = QuadrupedEnv(robot=robot_name,
                   hip_height=0.25,
                   legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
                   feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=1.5,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
# breakpoint()
obs = env.reset(random=False)
env.render()
t = timer_class(duty_factor=0.6,step_freq= 1.5,delta=[0000.5,0000.0,0000,0000.5])
t_sim = copy.deepcopy(t)
terrain_height = np.zeros(n_contact)

init = {}
input = {}

Kp = 10
Kd = 2

Kp_c = 500
Kd_c = 5
counter = 0



while True:

    qpos = env.mjData.qpos
    qvel = env.mjData.qvel

    if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

        foot_op = np.array([env.feet_pos('world').FL, env.feet_pos('world').FR, env.feet_pos('world').RL, env.feet_pos('world').RR],order="F")
        contact_op = t_sim.run(dt = 1/mpc_frequency)
        t.set(t_sim.t.copy(),t_sim.init.copy())

        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        p = qpos[:3].copy()
        q = qpos[7:].copy()

        dp = qvel[:3].copy()
        omega = qvel[3:6]
        dq = qvel[6:].copy()

        rpy = env.base_ori_euler_xyz.copy()
        foot_op_vec = foot_op.flatten()
        x0 = jnp.concatenate([qpos, qvel])

        init['p'] = p
        init['q'] = q
        init['dp'] = dp
        init['omega'] = omega
        init['rpy'] = rpy
        init['contact'] = contact_op
        init['foot'] = foot_op_vec

        input['des_speeds'] = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2]])
        input['des_height'] = 0.36

        _ , _, terrain_height, foot_ref_dot = refGenerator(timer_class = t,initial_state = init,input = input,param=param, terrain_height=terrain_height)

        # reference = jnp.tile(jnp.concatenate([p_ref, rpy_ref, ref_base_lin_vel, ref_base_ang_vel]), (N + 1, 1))
        start = timer()
        X,U,V, _,c =  work(reference,parameter,x0,X0,U0,V0)
        X.block_until_ready()
        stop = timer()
        print(f"Execution time: {stop-start}")

        U0 = U
        X0 = X
        V0 = V
        tau = U[0,:n_joints]
        grf = U[0,n_joints:]
        print("grf: \n",grf)
        print("tau: \n",tau)
    
    # feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
    action = np.zeros(env.mjModel.nu)
    feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
    action[env.legs_tau_idx.FL] = -(feet_jac['FL'].T @ (grf[:3]))[6:9]
    action[env.legs_tau_idx.FR] = -(feet_jac['FR'].T @ (grf[3:6]))[9:12]
    action[env.legs_tau_idx.RL] = -(feet_jac['RL'].T @ (grf[6:9]))[12:15]
    action[env.legs_tau_idx.RR] = -(feet_jac['RR'].T @ (grf[9:] ))[15:18]
    print("why not zeros: \n",action-tau)
    #PD
    # print(catisian_space_action.shape)
    state, reward, is_terminated, is_truncated, info = env.step(action=tau)
    counter += 1
    if is_terminated:
        pass
        # Do some stuff
    env.render()
env.close()