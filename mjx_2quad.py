import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=true '
    # '--xla_gpu_deterministic_ops=true'
)
# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })
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
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere
import primal_dual_ilqr.utils.mpc_utils as mpc_utils

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = 50.0
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

sim_frequency = 200.0
# env = QuadrupedEnv(robot=robot_name,
#                    hip_height=0.25,
#                    legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
#                    feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
#                    scene=scene_name,
#                    sim_dt = 1/sim_frequency,  # Simulation time step [s]
#                    ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
#                    ground_friction_coeff=1.5,  # pass a float for a fixed value
#                    base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
#                    state_obs_names=state_observables_names,  # Desired quantities in the 'state'
#                    )
# env.mjModel = mujoco.MjModel.from_xml_path('./data/go2/scene_mjx.xml')
# env.mjModel.opt.timestep = 1/sim_frequency
# env.mjData = mujoco.MjData(env.mjModel)
# obs = env.reset(random=False)
class SimInput:
    def __init__(self, hip_height):
        self.hip_height = hip_height
        self._ref_base_lin_vel_H = np.zeros(3)
        self._ref_base_ang_yaw_dot = 0.0

    def key_callback(self, keycode):
        if keycode == 262:  # arrow right
            self._ref_base_ang_yaw_dot -= np.pi / 6
        elif keycode == 263:  # arrow left
            self._ref_base_ang_yaw_dot += np.pi / 6
        elif keycode == 265:  # arrow up
            self._ref_base_lin_vel_H[0] += 0.25 * self.hip_height  # % of (hip_height / second)
        elif keycode == 264:  # arrow down
            self._ref_base_lin_vel_H[0] -= 0.25 * self.hip_height  # % of (hip_height / second)
        elif keycode == 345:  # ctrl
            self._ref_base_lin_vel_H *= 0.0
            self._ref_base_ang_yaw_dot = 0.0

        self._ref_base_ang_yaw_dot = np.clip(self._ref_base_ang_yaw_dot, -2 * np.pi, 2 * np.pi)
        self._ref_base_lin_vel_H[0] = np.clip(self._ref_base_lin_vel_H[0], -6 * self.hip_height, 6 * self.hip_height)
    def get_ref_base_lin_vel_W(self,quat):
        yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
        Ryaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw), np.cos(yaw), 0],[0, 0, 1]])
        return Ryaw@self._ref_base_lin_vel_H
@jax.jit
def reference_generator(t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    ref_lin_vel, ref_ang_vel, robot_height = input
    p = jnp.array([p[0], p[1], robot_height])
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * robot_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    q_ref = jnp.tile(jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8]), (N+1, 1))
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot0 = jnp.tile(p,n_contact) + jnp.array([ 0.192, 0.142, 0.024,  0.192, -0.142, 0.024,-0.195,  0.142,  0.024, -0.195, -0.142, 0.024])@jax.scipy.linalg.block_diag(Ryaw,Ryaw,Ryaw,Ryaw).T
    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = mpc_utils.timer_run(duty_factor, step_freq, new_t, dt)
        
        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)
        
        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(robot_height/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0[direction::3]
            return f
        
        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,val):
            a0 = current_foot
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2 
            return a0 + a1*val + a2*val**2 + a3*val**3
        
        def cubic_splineZ(current_foot, foothold, step_height,val):
            # a0 = current_foot
            # a1 = 0
            # a2 = 8*(step_height) - foothold + current_foot
            # a3 = 8*(step_height) - 2*a2
            a0 = current_foot
            a3 = 8*step_height - 6*foothold -2*a0
            a2 = -foothold +a0 -2*a3
            a1 = +2*foothold -2*a0 +a3
            return a0 + a1*val + a2*val**2 + a3*val**3
        new_foot_x = jnp.where(new_contact_sequence>0, new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(new_contact_sequence>0, new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(new_contact_sequence>0, new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        return (new_t, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z)
    
    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    init_carry = (t_timer, contact_sequence,foot_ref,liftoff_x,liftoff_y,liftoff_z)
    _, contact_sequence,foot_ref, liftoff_x,liftoff_y,liftoff_z = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    # foot to world frame
    # foot_ref = foot_ref@jax.scipy.linalg.block_diag(Ryaw,Ryaw,Ryaw,Ryaw) + jnp.tile(p,(N+1,n_contact))
    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1), liftoff

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
N = 15  # Number of stages
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
dt = 0.02  # Time step
p_legs0 = jnp.array([ 0.192, 0.142, 0.024,  0.192, -0.142, 0.024,-0.195,  0.142,  0.024, -0.195, -0.142, 0.024])
model = mujoco.MjModel.from_xml_path('./data/go2/scene_mjx.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))



alpha = 25
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
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0)
    g = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0) - p_legs # position-level constraint violation
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]  # Velocity-level constraint violation

    # Stabilization term
    baumgarte_term = - 2*alpha * g_dot #- beta * beta * g

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
    x_next = jnp.concatenate([p, quat, q, v, current_leg,grf])

    return x_next

p0 = jnp.array([0, 0, 0.28])
quat0 = jnp.array([1, 0, 0, 0])
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
x0 = jnp.concatenate([p0, quat0,q0, jnp.zeros(6+n_joints),p_legs0,jnp.zeros(3*n_contact)])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.28])
quat_ref = jnp.array([1, 0, 0, 0])
rpy_ref = jnp.array([0, 0, 0])
q_ref = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

grf_ref = jnp.zeros(3 * n_contact)
tau_ref = jnp.zeros(n_joints)

u_ref = jnp.concatenate([tau_ref])

Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qdp = jnp.diag(jnp.array([1, 1, 1]))*1e3
Qomega = jnp.diag(jnp.array([1, 1, 1]))*1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Rgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
Qrot = jnp.diag(jnp.array([500,500,0]))
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qleg = jnp.diag(jnp.tile(jnp.array([1e3,1e3,1e5]),n_contact))
Qpenalty = jnp.diag(jnp.ones(5*n_contact))
QpenaltyZ = jnp.diag(jnp.ones(3*n_contact))*10
# Define the cost function
@jax.jit
def cost(x, u, t, reference):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
    grf = x[13+2*n_joints+3*n_contact:]
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

    mu = 0.4
    friction_cone = jnp.array([[0,0,1],[-1,0,mu],[1,0,mu],[0,-1,mu],[0,1,mu]])
    friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    friction_cone = friction_cone @ grf
    alpha = 0.1
    #use ln(1+exp(x)) as a smooth approximation of max(0,x)
    friction_cone = 1/alpha*(jnp.log1p(jnp.exp(-alpha*friction_cone)))
    # delta = 0.0001
    # alpha_swing = 0.1
    # swing_z_plus = p_leg-p_leg_ref #+ jnp.ones(3*n_contact)*delta
    # swing_z_plus = 1/alpha_swing*(jnp.log1p(jnp.exp(-alpha_swing*swing_z_plus)))
    # swing_z_minus = p_leg-p_leg_ref #- jnp.ones(3*n_contact)*delta
    # swing_z_minus = 1/alpha_swing*(jnp.log1p(jnp.exp(-alpha_swing*swing_z_minus)))

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) +  (q - q_ref).T @ Qq @ (q - q_ref) + math.quat_sub(quat,quat_ref).T@Qrot@math.quat_sub(quat,quat_ref) +\
                 (dp - dp_ref).T @ Qdp @ (dp - dp_ref) + (omega - omega_ref).T @ Qomega @ (omega - omega_ref) + dq.T @ Qdq @ dq +\
                 tau.T @ Qtau @ tau +\
                 (p_leg - p_leg_ref).T @ Qleg @ (p_leg - p_leg_ref) #+\
                #  friction_cone.T @ Qpenalty @ friction_cone
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
# reference = jnp.tile(jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref,p_legs0]), (N + 1, 1))
# parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))
from timeit import default_timer as timer
param_size = n_contact + 3*n_contact
reference_size = 13 + n_joints + 3*n_contact
n_robot = 2
@jax.jit
def multi_robot_dynamics(x, u, t,parameter):
    return jnp.concatenate([dynamics(x[:61], u[:12], t,parameter[:,:param_size]),dynamics(x[61:], u[12:], t,parameter[:,param_size:])])

@jax.jit
def multi_robot_cost(x, u, t, reference):
    p1 = x[:3]
    p2 = x[61:64]
    distance = jnp.sum((p1 - p2) ** 2)
    distance_penalty = -jnp.log(distance - 0.64)
    return cost(x[:61], u[:12], t, reference[:,:reference_size]) + cost(x[61:], u[12:], t, reference[:,reference_size:]) + distance_penalty
U0 = jnp.tile(jnp.tile(u_ref,(1,n_robot)), (N, 1))
X0 = jnp.tile(jnp.tile(x0,(1,n_robot)), (N + 1, 1))
V0 = jnp.zeros((N + 1, n*n_robot ))
reference = jnp.tile(jnp.tile(jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref,p_legs0]), (1, n_robot)), (N + 1, 1))
parameter = jnp.tile(jnp.tile(jnp.concatenate([jnp.ones(4), p_legs0]), (1, n_robot)), (N + 1, 1))

@jax.jit
def work(reference,parameter,x0,X0,U0,V0):
    return optimizers.mpc(
        multi_robot_cost,
        multi_robot_dynamics,
        False,
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
# env.render()
# Timer
timer_t = jnp.array([0000.5,0000.0,0000,0000.5])
timer_t_sim = timer_t.copy()
contact, timer_t = mpc_utils.timer_run(duty_factor = 0.6, step_freq = 1.35,leg_time=timer_t, dt=dt)
liftoff_1 = p_legs0.copy()
liftoff_2 = p_legs0.copy()
terrain_height = np.zeros(n_contact)

Kp = 10
Kd = 2

counter = 0
ref = []
ref_history = []
actual = []
tau_new = np.zeros(n_joints)
flag = False
sim_model = mujoco.MjModel.from_xml_path('./data/go2/scene_mjx_two.xml')
sim_data = mujoco.MjData(sim_model)
sim_data.qpos = jnp.concatenate([p0, quat0, q0,p0+jnp.array([1,0,0]), quat0, q0])
sim_model.opt.timestep = 1 / sim_frequency
contact_list = ['FL','FR','RL','RR','second_FL','second_FR','second_RL','second_RR']
contact_list_id = []
for name in contact_list:
    contact_list_id.append(mujoco.mj_name2id(sim_model,mujoco.mjtObj.mjOBJ_GEOM,name))
sim_input = SimInput(0.28)
def key_callback_wrapper(keycode):
    sim_input.key_callback(keycode)
with mujoco.viewer.launch_passive(sim_model, sim_data, key_callback=key_callback_wrapper) as viewer:
    while viewer.is_running():
        qpos = sim_data.qpos
        qvel = sim_data.qvel
        if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

            foot_op = np.array([sim_data.geom_xpos[id] for id in contact_list_id], order="F")
            contact_op, timer_t_sim = mpc_utils.timer_run(duty_factor=0.6, step_freq=1.35, leg_time=timer_t_sim, dt=dt)
            timer_t = timer_t_sim.copy()
            foot_op_vec = foot_op.flatten()
            x0 = jnp.concatenate([qpos[:19], qvel[:18],foot_op_vec[:12],jnp.zeros(12),qpos[19:], qvel[18:],foot_op_vec[12:],jnp.zeros(12)])   
            input = (sim_input.get_ref_base_lin_vel_W(qpos[3:7]),np.array([0,0,sim_input._ref_base_ang_yaw_dot]) , 0.28)
            reference_1 , parameter_1 , liftoff_1 = reference_generator(timer_t, jnp.concatenate([qpos[:19],qvel[:18]]), foot_op_vec[:12], input, duty_factor = 0.6,  step_freq= 1.35 ,step_height=0.08,liftoff=liftoff_1)
            input = (np.array([0,0,0]), np.zeros(3), 0.28)
            reference_2 , parameter_2 , liftoff_2 = reference_generator(timer_t, jnp.concatenate([qpos[19:],qvel[18:]]), foot_op_vec[12:], input, duty_factor = 0.6,  step_freq= 1.35 ,step_height=0.08,liftoff=liftoff_2)
            
            reference = jnp.concatenate([reference_1,reference_2],axis=1)
            parameter = jnp.concatenate([parameter_1,parameter_2],axis=1)
            start = timer()
            X,U,V =  work(reference,parameter,x0,X0,U0,V0)
            X.block_until_ready()
            stop = timer()
            # print(f"Time elapsed: {stop-start}")
            distance = jnp.linalg.norm(x0[:3] - x0[61:64])
            print(f"Distance between the two COMs: {distance}")
            # # move the prediction one step forward
            U0 = jnp.concatenate([U[1:],U[-1:]])
            X0 = jnp.concatenate([X[1:],X[-1:]])
            V0 = jnp.concatenate([V[1:],V[-1:]])
            tau = U[0,:]
        sim_data.ctrl = tau
        mujoco.mj_step(sim_model, sim_data)
        viewer.sync()
        counter += 1

