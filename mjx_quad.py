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
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

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

from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere
from mujoco.mjx._src.dataclasses import PyTreeNode 
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

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
class timer_data(PyTreeNode):
    duty_factor: float
    step_freq: float
    delta: jnp.array
    t: jnp.array
    n_contact: int
    init: jnp.array
@jax.jit
def timer_run(timer_data,dt):
    contact = jnp.zeros(timer_data.n_contact)
    leg_time = timer_data.t
    delta = timer_data.delta
    duty_factor = timer_data.duty_factor
    init_flag = timer_data.init
    def integrate(input):
        leg_time,delta,init_flag = input
        leg_time = jnp.where(leg_time == 1.0,0,leg_time)
        leg_time = leg_time + dt*timer_data.step_freq
        def init_operation(leg_time):
            return 1 , leg_time < delta,jnp.where(leg_time < delta,leg_time,0)
        def runtime_operation(leg_time):
            return jnp.where(leg_time < duty_factor,1,0),False,leg_time
        contact , init_flag , leg_time = jax.lax.cond(init_flag,init_operation, runtime_operation, leg_time)
        leg_time = jnp.where(leg_time > 1.0,1.0,leg_time)
        return contact, leg_time, init_flag
    jax.debug.print("delta: {}",delta)
    jax.debug.print("leg_time: {}",leg_time)
    jax.debug.print("init_flag: {}",init_flag)
    inputs = jnp.array([(leg_time[i], delta[i], init_flag[i]) for i in range(timer_data.n_contact)])
    contact , leg_time, init_flag = jax.vmap(integrate)(inputs)
    jax.debug.print("contact: {}",contact)
    jax.debug.print("leg_time: {}",leg_time)
    jax.debug.print("init_flag: {}",init_flag)
    timer_data = timer_data.replace(t = leg_time,init = init_flag)
    return contact , timer_data
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
    return parameter,ref['foot'].T,terrain_height,foot_speed_out
# @jax.jit
# def reference_generator(timer_class,x,input):
#     p = x[:3]
#     quat = x[3:7]
#     q = x[7:7+n_joints]
#     dp = x[7+n_joints:10+n_joints]
#     omega = x[10+n_joints:13+n_joints]
#     dq = x[13+n_joints:13+2*n_joints]
#     ref_base_lin_vel,ref_base_ang_vel, robot_height = input
#     t , step_freq, duty_f, delay = timer
#     p = jnp.array([p[0], p[1], robot_height])
#     p_ref = jnp.arange(N+1)*dt*ref_base_lin_vel + p
#     quat_ref = jnp.tile(jnp.array([1,0,0,0]),(N+1,1))
#     q_ref = jnp.tile(jnp.array([0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8]),(N+1,1)) 
#     dp_ref = jnp.tile(ref_base_lin_vel,(N+1,1))
#     omega_ref = jnp.tile(ref_base_ang_vel,(N+1,1))
#     foot0 = jnp.tile(jnp.array([0.2717287,   0.13780001,  0.2717287,  -0.13780001, -0.20967132,  0.13780001, -0.20967132, -0.13780001]),(N+1,1))
#     fh1 = 0.5*stance_time*dt*ref_base_lin_vel[:2]
#     fh2 = jnp.sqrt(robot_height/9.81)*(dp[:2]-ref_base_lin_vel[:2])
#     foothold = jnp.tile(fh1+fh2,4)+foot0
#     for i in range(N+1):
#         for leg in range(4):
#             if jnp.sin((t[leg] + i*dt)*step_freq+delay[leg]) > :
#                 foothold = jnp.tile(foot0,(N+1,1)) > 
    
    


#     return ref
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
n =  13 + 2*n_joints + 3*n_contact # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
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

model = env.mjModel
data = env.mjData
mjx_model = mjx.put_model(model)

contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
print(contact_id)



t = timer_data(duty_factor=0.6,step_freq= 1.35,delta=[0000.5,0000.0,0000,0000.5],t=jnp.zeros(4),n_contact=4,init=jnp.ones(4))
contact, t = timer_run(t,0.01)
for i in range(1000):
    contact, t = timer_run(t,0.01)
    print(contact)
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
# Constraint drift terms
alpha = 10
beta = 2*np.sqrt(alpha)

@jax.jit
def dynamics(x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:2*n_joints+13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t,:4]
    p_legs = parameter[t,4:]

    tau = jnp.concatenate([jnp.zeros(6),u])

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[2])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[5])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[8])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[11])

    J = jnp.concatenate([J_FL,J_FR,J_RL,J_RR],axis=1)

    g = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0) - p_legs # position-level constraint violation
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]  # Velocity-level constraint violation

    # Stabilization term
    baumgarte_term = - 2*alpha * g_dot - beta * beta * g

    JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)
    # Finate diference Jdot
    # #integrate qpos with a really small dt
    # h = 1e-6
    # delta_p = jnp.concatenate([x[:3] + x[7 + n_joints:10 + n_joints]*h, math.quat_integrate(x[3:7], x[10 + n_joints:13 + n_joints], h), x[7:7+n_joints] + x[13 + n_joints:]*h])
    # mjx_data = mjx_data.replace(qpos = delta_p[:n_joints+7])
    # mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    # delta_J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[2])
    # delta_J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[5])
    # delta_J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[8])
    # delta_J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[11])
    # delta_J = jnp.concatenate([delta_J_FL,delta_J_FR,delta_J_RL,delta_J_RR],axis=1)
    # Jdot = (delta_J - J)/h

    rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False),tau - D) + baumgarte_term #+ Jdot.T@x[n_joints+7:] 
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ,rhs)
    grf = jnp.concatenate([grf[:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M,False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v, grf])

    return x_next

p0 = jnp.array([0, 0, 0.33])
quat0 = jnp.array([1, 0, 0, 0])
q0 = jnp.array([0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8])
x0 = jnp.concatenate([p0, quat0,q0, jnp.zeros(6+n_joints),jnp.zeros(3*n_contact)])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.36])
quat_ref = jnp.array([1, 0, 0, 0])
rpy_ref = jnp.array([0, 0, 0])
q_ref = jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8])
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

grf_ref = jnp.zeros(3 * n_contact)
tau_ref = jnp.zeros(n_joints)

u_ref = jnp.concatenate([tau_ref])

Qp = jnp.diag(jnp.array([1, 1, 1e4]))
Qq = jnp.diag(jnp.ones(n_joints)) * 5e0
Qdp = jnp.diag(jnp.array([1, 1, 1]))*1e3
Qomega = jnp.diag(jnp.array([10, 10, 10]))
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Rgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
Qrot = jnp.diag(jnp.array([500,500,0]))
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qleg = jnp.diag(jnp.tile(jnp.array([1e3,1e3,1e5]),n_contact))
Qpenalty = jnp.diag(jnp.ones(5*n_contact)) * 1
# Define the cost function
@jax.jit
def cost(x, u, t, reference):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    grf = x[13+2*n_joints:]
    tau = u[:n_joints]

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    p_leg_ref = reference[t,13+n_joints:]
    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:13+2*n_joints])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    p_leg = jnp.concatenate([FL_leg,FR_leg,RL_leg,RR_leg],axis=0)

    # J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[2])
    # J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[5])
    # J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[8])
    # J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[11])

    # J = jnp.concatenate([J_FL,J_FR,J_RL,J_RR],axis=1) 
    #impose the friction cone constraint as a penalty at the torque level

    mu = 0.4
    friction_cone = jnp.array([[0,0,1],[-1,0,mu],[1,0,mu],[0,-1,mu],[0,1,mu]])
    friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    friction_cone = friction_cone @ grf
    alpha = 0.1
    #use ln(1+exp(x)) as a smooth approximation of max(0,x)
    friction_cone = 1/alpha*(jnp.log1p(jnp.exp(-alpha*friction_cone)))

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) +  (q - q_ref).T @ Qq @ (q - q_ref) + math.quat_sub(quat,quat_ref).T@Qrot@math.quat_sub(quat,quat_ref) +\
                 (dp - dp_ref).T @ Qdp @ (dp - dp_ref) + (omega - omega_ref).T @ Qomega @ (omega - omega_ref) + dq.T @ Qdq @ dq +\
                 tau.T @ Qtau @ tau+\
                 friction_cone.T @ Qpenalty @ friction_cone+\
                (p_leg - p_leg_ref).T @ Qleg @ (p_leg - p_leg_ref)
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))
reference = jnp.tile(jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref,p_legs0]), (N + 1, 1))
parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))

from timeit import default_timer as timer

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
# start = timer()
# X,U,V, g, c =  work(reference,parameter,x0,X0,U0,V0)
# end = timer()
# print(f"Compilation time: {end-start}")

env.render()
t = timer_data(duty_factor=0.6,step_freq= 1.5,delta=[0000.5,0000.0,0000,0000.5])
t_sim = copy.deepcopy(t)
terrain_height = np.zeros(n_contact)

init = {}
input = {}

Kp = 10
Kd = 2

counter = 0
ids = []
for i in range(N):
     ids.append(render_sphere(viewer=env.viewer,
              position = np.array([0,0,0]),
              diameter = 0.01,
              color=[1,0,0,1]))

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
        x0 = jnp.concatenate([qpos, qvel,np.zeros(3*n_contact)])
        
        init['p'] = p
        init['q'] = q
        init['dp'] = dp
        init['omega'] = omega
        init['rpy'] = rpy
        init['contact'] = contact_op
        init['foot'] = foot_op_vec

        input['des_speeds'] = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2]])
        input['des_height'] = 0.36

        parameter, foot_ref, terrain_height, foot_ref_dot = refGenerator(timer_class = t,initial_state = init,input = input,param=param, terrain_height=terrain_height)

        reference = jnp.tile(jnp.concatenate([p_ref, quat_ref,q0, ref_base_lin_vel, ref_base_ang_vel]), (N + 1, 1))
        reference = jnp.concatenate([reference,foot_ref],axis=1)
        start = timer()
        X,U,V, _,c =  work(reference,parameter,x0,X0,U0,V0)
        X.block_until_ready()
        stop = timer()
        print(f"Execution time: {stop-start}")
        # move the prediction one step forward
        U0 = jnp.concatenate([U[1:],U[-1:]])
        X0 = jnp.concatenate([X[1:],X[-1:]])
        V0 = jnp.concatenate([V[1:],V[-1:]])
        for i in range(N):
            render_sphere(viewer=env.viewer,
                      position = X[i,:3],
                      diameter = 0.01,
                      color=[1,0,0,1],
                      geom_id = ids[i])
        tau = U[0,:n_joints]
    action = np.zeros(env.mjModel.nu)

    #PD
    # print(catisian_space_action.shape)
    # env.mjData.qpos = res[:n_joints+7]
    # env.mjData.qvel = res[n_joints+7:]
    state, reward, is_terminated, is_truncated, info = env.step(action=tau)
    
    counter += 1
    if is_terminated:
        pass
        # Do some stuff
    env.render()
env.close()