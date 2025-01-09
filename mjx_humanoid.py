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
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere
import time
import primal_dual_ilqr.utils.mpc_utils as mpc_utils

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

model = mujoco.MjModel.from_xml_path('./data/unitree_g1/scene.xml')
d = mujoco.MjData(model)
mpc_frequency = 100.0
sim_frequency = 1000.0
@jax.jit
def reference_generator(t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    # dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    ref_lin_vel, ref_ang_vel, robot_height = input
    p = jnp.array([p[0], p[1], robot_height])
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * robot_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    q_ref = jnp.tile(jnp.zeros(n_joints), (N+1, 1))
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    foot_ref = jnp.tile(foot-jnp.tile(p,(1,n_contact)), (N+1, 1))
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

        foot0 = jnp.array([0.09500057, 0.11787131, 0.00032407, 0.09500057, -0.11787131, 0.00032407])
        
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
    foot_ref = foot_ref + jnp.tile(p,(N+1,n_contact))
    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1), liftoff

# contact_frame = ['0_left','1_left','2_left','3_left','0_right','1_right','2_right','3_right']
contact_frame = ['foot_left','foot_right']
body_name = ['right_hip_pitch_link','right_hip_yaw_link','right_knee_link','right_ankle_pitch_link','right_ankle_roll_link',
              'left_hip_pitch_link','left_hip_yaw_link','left_knee_link','left_ankle_pitch_link','left_ankle_roll_link']
n_joints = 23
n_contact = len(contact_frame)

# # Problem dimensions
N = 50  # Number of stages
n =  13 + 2*n_joints + 6*n_contact # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
dt = 0.01 # Time step
# p_legs0 = jnp.array([ 0.2717287,   0.13780001,  0.02074774,  0.2717287,  -0.13780001,  0.02074774])
mjx_model = mjx.put_model(model)

contact_id = []
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

    contact = parameter[t,:8]
    # p_legs = parameter[t,8:]v

    tau = jnp.concatenate([jnp.zeros(6),u])

    # left_0 = mjx_data.geom_xpos[contact_id[0]]
    # left_1 = mjx_data.geom_xpos[contact_id[1]]
    # left_2 = mjx_data.geom_xpos[contact_id[2]]
    # left_3 = mjx_data.geom_xpos[contact_id[3]]

    # right_0 = mjx_data.geom_xpos[contact_id[4]]
    # right_1 = mjx_data.geom_xpos[contact_id[5]]
    # right_2 = mjx_data.geom_xpos[contact_id[6]]
    # right_3 = mjx_data.geom_xpos[contact_id[7]]

    left_foot = mjx_data.geom_xpos[contact_id[0]]
    right_foot = mjx_data.geom_xpos[contact_id[1]]


    # J_left_0, _ = mjx.jac(mjx_model, mjx_data, left_0, body_id[4])
    # J_left_1, _ = mjx.jac(mjx_model, mjx_data, left_1, body_id[4])
    # J_left_2, _ = mjx.jac(mjx_model, mjx_data, left_2, body_id[4])
    # J_left_3, _ = mjx.jac(mjx_model, mjx_data, left_3, body_id[4])

    # J_right_0, _ = mjx.jac(mjx_model, mjx_data, right_0, body_id[8])
    # J_right_1, _ = mjx.jac(mjx_model, mjx_data, right_1, body_id[8])
    # J_right_2, _ = mjx.jac(mjx_model, mjx_data, right_2, body_id[8])
    # J_right_3, _ = mjx.jac(mjx_model, mjx_data, right_3, body_id[8])

    J_left_l,J_left_a = mjx.jac(mjx_model, mjx_data, left_foot, body_id[4])
    J_right_l,J_right_a = mjx.jac(mjx_model, mjx_data, right_foot, body_id[8])

    # J = jnp.concatenate([J_left_0,J_left_1,J_left_2,J_left_3,J_right_0,J_right_1,J_right_2,J_right_3],axis=1)
    J = jnp.concatenate([J_left_l,J_left_a,J_right_l,J_right_a],axis=1)
    # g = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0) - p_legs # position-level constraint violation
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
    # grf = jnp.concatenate([grf[:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3],grf[12:15]*contact[4],grf[15:18]*contact[5],grf[18:21]*contact[6],grf[21:24]*contact[7]])
    grf = jnp.concatenate([grf[:6]*contact[0],grf[6:12]*contact[1]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M,False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v, grf])

    return x_next
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))

# # Initial state
pos0 = jnp.array([0, 0, 0.75,
      1, 0, 0, 0,
      0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0,
      0, 
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0])
vel0 = jnp.zeros(6 + n_joints)  
x0 = jnp.concatenate([pos0, vel0])
u0 = jnp.zeros(m)
Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1e2
Qdp = jnp.diag(jnp.array([1, 1, 1]))*1e3
Qomega = jnp.diag(jnp.array([1, 1, 1]))*1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qrot = jnp.diag(jnp.array([500,500,0]))
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e5]),n_contact))
Qpenalty = jnp.diag(jnp.ones(5*n_contact))
QpenaltyZ = jnp.diag(jnp.ones(3*n_contact))*10
Rreg = jnp.diag(jnp.ones(6 + n_joints)) * 1
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
    p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
    grf_ref = jnp.array([0,0,343.35/2,0,0,0,0,0,343.35/2,0,0,0])
    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = jnp.zeros(6+n_joints))

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    D = mjx_data.qfrc_bias

    left_foot = mjx_data.geom_xpos[contact_id[0]]
    right_foot = mjx_data.geom_xpos[contact_id[1]]

    J_left_l,J_left_a = mjx.jac(mjx_model, mjx_data, left_foot, body_id[4])
    J_right_l,J_right_a = mjx.jac(mjx_model, mjx_data, right_foot, body_id[8])

    J = jnp.concatenate([J_left_l,J_left_a,J_right_l,J_right_a],axis=1)

    torque_reg = jnp.concatenate([jnp.zeros(6),tau])-D+J@grf
    
    p_leg = jnp.concatenate([left_foot,right_foot],axis=0)


    # mu = 0.4
    # friction_cone = jnp.array([[0,0,1],[-1,0,mu],[1,0,mu],[0,-1,mu],[0,1,mu]])
    # friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    # friction_cone = friction_cone @ grf
    # alpha = 0.1
    # #use ln(1+exp(x)) as a smooth approximation of max(0,x)
    # friction_cone = 1/alpha*(jnp.log1p(jnp.exp(-alpha*friction_cone)))
    # delta = 0.0001
    # alpha_swing = 0.1
    # swing_z_plus = p_leg-p_leg_ref #+ jnp.ones(3*n_contact)*delta
    # swing_z_plus = 1/alpha_swing*(jnp.log1p(jnp.exp(-alpha_swing*swing_z_plus)))
    # swing_z_minus = p_leg-p_leg_ref #- jnp.ones(3*n_contact)*delta
    # swing_z_minus = 1/alpha_swing*(jnp.log1p(jnp.exp(-alpha_swing*swing_z_minus)))

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) +  (q - q_ref).T @ Qq @ (q - q_ref) +\
                 (dp - dp_ref).T @ Qdp @ (dp - dp_ref) + (omega - omega_ref).T @ Qomega @ (omega - omega_ref) + dq.T @ Qdq @ dq +\
                 + tau.T@Qtau@tau + torque_reg.T @ Rreg @ torque_reg
                #  (p_leg - p_leg_ref).T @ Qleg @ (p_leg - p_leg_ref)#
                 #+friction_cone.T @ Qpenalty @ friction_cone
                # swing_z_plus.T @ QpenaltyZ @ swing_z_plus + swing_z_minus.T @ QpenaltyZ @ swing_z_minus
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)
# Solve
p0 = jnp.array([0, 0, 0.755])
quat0 = jnp.array([1, 0, 0, 0])
q0 = jnp.zeros(n_joints)
x0 = jnp.concatenate([p0, quat0,q0, jnp.zeros(6+n_joints),jnp.zeros(6*n_contact)])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.755])
quat_ref = jnp.array([1, 0, 0, 0])
q_ref = jnp.zeros(n_joints)
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

grf_ref = jnp.zeros(6 * n_contact)
tau_ref = jnp.zeros(n_joints)

u_ref = jnp.concatenate([tau_ref])
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))
p_legs0 = jnp.array([0.09500057, 0.11787131, 0.00032407, 0.09500057, -0.11787131, 0.00032407])
reference = jnp.tile(jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref,p_legs0]), (N + 1, 1))
parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))
from timeit import default_timer as timer
mu = 1e-3
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
# Timer
duty_factor = 2
step_freq = 0.7
timer_t = jnp.array([0000.5,0000.0])
timer_t_sim = timer_t.copy()
contact, timer_t = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq,leg_time=timer_t, dt=dt)
liftoff = p_legs0.copy()
terrain_height = np.zeros(n_contact)
counter = 0
jitdyn = jax.jit(dynamics)
with mujoco.viewer.launch_passive(model, d) as viewer:
    while viewer.is_running():
        qpos = d.qpos.copy()
        qvel = d.qvel.copy()
        if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:
            foot_op = np.array([d.geom_xpos[contact_id[0]],d.geom_xpos[contact_id[1]]])
            contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq ,leg_time=timer_t_sim, dt=dt)
            print(contact_op)
            timer_t = timer_t_sim.copy()

            ref_base_lin_vel = jnp.array([0.0,0,0])
            ref_base_ang_vel = jnp.array([0,0,0])

            p = qpos[:3].copy()
            q = qpos[7:].copy()

            dp = qvel[:3].copy()
            omega = qvel[3:6]
            dq = qvel[6:].copy()

        
            foot_op_vec = foot_op.flatten()
            x0 = jnp.concatenate([qpos, qvel,np.zeros(6*n_contact)])
            input = (ref_base_lin_vel, ref_base_ang_vel, 0.36)
            start = timer()
            _ , parameter , liftoff = reference_generator(timer_t, jnp.concatenate([qpos,qvel]), foot_op_vec, input, duty_factor = duty_factor,  step_freq= step_freq ,step_height=0.08,liftoff=liftoff)
            X,U,V, _,_ =  work(reference,parameter,x0,X0,U0,V0)
            X.block_until_ready()
            stop = timer()
            print(f"Time elapsed: {stop-start}")
            # move the prediction one step forward
            U0 = jnp.concatenate([U[1:],U[-1:]])
            X0 = jnp.concatenate([X[1:],X[-1:]])
            V0 = jnp.concatenate([V[1:],V[-1:]])
            tau = U[0,:n_joints]
            # tau = 60*(np.zeros(n_joints)-qpos[7:7+n_joints]) + 4*(np.zeros(n_joints)-qvel[6:6+n_joints])
            # time.sleep(0.1)
            # xnew = jitdyn(x0,tau,0,parameter)
            # d.qpos = xnew[:n_joints+7] 
            # d.qvel = xnew[n_joints+7:13+2*n_joints] 

        d.ctrl = tau
        mujoco.mj_step(model, d)
        
        viewer.sync()
        
    
    

