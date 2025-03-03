import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=true '
    # 'XLA_PYTHON_CLIENT_PREALLOCATE=false'
    # '--xla_gpu_deterministic_ops=true'
)
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


from utils.rotation import quaternion_integration,rpy_intgegration,quaternion_to_rpy,rotation_matrix_to_quaternion

import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import mujoco.viewer
from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere ,render_vector
import time
import utils.mpc_utils as mpc_utils

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

model = mujoco.MjModel.from_xml_path('./data/unitree_h1/mjx_scene_h1_walk.xml')
data = mujoco.MjData(model)
mpc_frequency = 50.0
sim_frequency = 1000.0
model.opt.timestep = 1/sim_frequency
contact_frame = ['FL','RL','FR','RR']

body_name = ['left_ankle_link','right_ankle_link']

n_joints = 19
n_contact = 4
# # # Problem dimensions
N = 150  # Number of stages
n =  13 + 2*n_joints + 3*n_contact + 3*n_contact # Number of states
m = n_joints  # Number of controls (F)
dt = 0.01 # Time step
p_legs0 = jnp.array([ 0.14738185,  0.20541158,  0.01398883,  
                    -0.00253908,  0.2102815,   0.01398485,
                    0.14787466, -0.20581408,  0.01399987,
                    -0.00203967, -0.21088305,  0.0139761 ])
mjx_model = mjx.put_model(model)

alpha = 5
beta = 2*np.sqrt(alpha)


def dynamics(x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:2*n_joints+13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t,:n_contact]

    tau = jnp.concatenate([jnp.zeros(6),u])

    FL = mjx_data.geom_xpos[contact_id[0]]
    RL = mjx_data.geom_xpos[contact_id[1]]
    FR = mjx_data.geom_xpos[contact_id[2]]
    RR = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL, body_id[0])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR,  body_id[1])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR,  body_id[1])

    # left = mjx_data.xpos[body_id[0]]
    # right = mjx_data.xpos[body_id[1]]

    # left_a = mjx_data.xquat[body_id[0]]
    # right_a = mjx_data.xquat[body_id[1]]

    # J_left_l,J_left_a = mjx.jac(mjx_model, mjx_data, left, body_id[0])[0]
    # J_right_l,J_right_a = mjx.jac(mjx_model, mjx_data, right, body_id[0])[0]

    

    # J = jnp.concatenate([J_left_l,J_left_a,J_right_l,J_right_a],axis=1)
    J = jnp.concatenate([J_FL,J_RL,J_FR,J_RR],axis=1)
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
    epsilon = 1e-3
    JT_M_invJ_reg = JT_M_invJ + epsilon * jnp.eye(JT_M_invJ.shape[0])
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ_reg)
    
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ,rhs)
    grf = jnp.concatenate([grf[0:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v,FL,RL,FR,RR,grf])
    
    return x_next

contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))

# # # Initial state
p0 = jnp.array([0, 0, 0.98,
    1, 0, 0, 0,
    0, 0, -0.4, 0.8, -0.4,
    0, 0, -0.4, 0.8, -0.4,
    0,
    0, 0, 0, 0,
    0, 0, 0, 0])
x0 = jnp.concatenate([p0, jnp.zeros(6 + n_joints),p_legs0,jnp.zeros(3*n_contact)])
vel0 = jnp.zeros(6 + n_joints)  
u0 = jnp.zeros(m)
Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qq = jnp.diag(jnp.array([ 4e-1, 4e-1, 4e-1, 4e-1, 4e-1,
                          4e-1, 4e-1, 4e-1, 4e-1, 4e-1,
                          4e1, 
                          4e2, 4e2, 4e2, 4e2,
                          4e2, 4e2, 4e2, 4e2])) 
Qdp = jnp.diag(jnp.array([1, 1, 1]))*1e3
Qomega = jnp.diag(jnp.array([1, 1, 1]))*1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qrot = jnp.diag(jnp.array([1,1,1]))*1e3
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e4]),n_contact))
Qgrf = jnp.diag(jnp.ones(n_contact))*1e-2
tau0 = jnp.array([
    2.5319746e+00,  2.3485034e+00, -4.5697699e+00, -3.3193874e+01,
 -4.4357371e-01, -2.5355515e+00, -2.6617262e+00, -4.6011033e+00,
 -3.2992393e+01, -5.6599975e-01,  9.7643714e-03, -1.3478606e+00,
  5.9931871e-02,  5.6568943e-03, -1.0474428e+00, -1.3432151e+00,
 -2.2387700e-02,  2.4983226e-03, -1.0475020e+00
])
# grf0 = jnp.array([0,0,198,0,0,50,0,0,198,0,0,50])
# # Define the cost function
@jax.jit
def cost(x, u, t, reference):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    grf = x[13+2*n_joints+n_contact*3:]
    tau = u[:n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
    grf_ref = reference[t,13+n_joints+3*n_contact:]

    mu = 0.7
    friction_cone = jnp.array([0,0,1])
    friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    # scaled_grf = grf/500.0 
    friction_cone = friction_cone @ (grf - grf_ref)
    # friction_cone = friction_cone + jnp.ones_like(friction_cone)*1e-2
    alpha = 0.1
    sigma = 5
    # quadratic_barrier_friction = alpha/2*(jnp.square((grf[2::3]-2*sigma)/sigma)-jnp.ones_like(grf[2::3]))
    # log_barrier_friction = -alpha*jnp.log(grf[2::3])
    # grf_bar = jnp.where(grf[2::3]>sigma,log_barrier_friction,quadratic_barrier_friction+log_barrier_friction)

    joints_limits = jnp.array([
    0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    2.35, 2.35, 
    2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
    2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
    joints_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    quadratic_barrier_joint = alpha/2*(jnp.square((joints_limits-2*sigma)/sigma)-jnp.ones_like(joints_limits))
    log_barrierr_joint = -alpha*jnp.log(joints_limits)
    joints_limits = jnp.where(joints_limits>sigma,log_barrierr_joint,quadratic_barrier_joint+log_barrierr_joint)
    torque_limits = jnp.array([
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200,
        40, 40, 40, 40, 18, 18, 18, 18,
        40, 40, 40, 40, 18, 18, 18, 18])
    torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    quadratic_barrier_torque = alpha/2*(jnp.square((torque_limits-2*sigma)/sigma)-jnp.ones_like(torque_limits))
    log_barrierr_torque = -alpha*jnp.log(torque_limits)
    torque_limits = jnp.where(torque_limits>sigma,log_barrierr_torque,quadratic_barrier_torque+log_barrierr_torque)
    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) +  (q - q_ref).T @ Qq @ (q - q_ref)+ math.quat_sub(quat,quat_ref).T@Qrot@math.quat_sub(quat,quat_ref) +\
                 (dp - dp_ref).T @ Qdp @ (dp - dp_ref) + (omega - omega_ref).T @ Qomega @ (omega - omega_ref) + dq.T @ Qdq @ dq +\
                 (tau-tau0).T@Qtau@(tau-tau0) + (p_leg-p_leg_ref).T @ Qleg @ (p_leg-p_leg_ref) + jnp.sum(joints_limits) + jnp.sum(torque_limits)    + friction_cone.T@Qgrf@friction_cone 
                
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)
# # Solve

p_ref = jnp.array([0, 0, 0.98])
quat_ref = jnp.array([1, 0, 0, 0])
q_ref = jnp.array([0, 0, -0.4, 0.8, -0.4,
                   0, 0, -0.4, 0.8, -0.4,
                   0,
                   0, 0, 0, 0,
                   0, 0, 0, 0])
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

u_ref = jnp.zeros(n_joints)
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))

@jax.jit
def work(reference,parameter,x0,X0,U0,V0):
    return optimizers.mpc(
        cost,
        dynamics,
        False,
        reference,
        parameter,
        x0,
        X0,
        U0,
        V0,
    )
from timeit import default_timer as timer
# # Timer
duty_factor = 0.7
step_freq = 1
step_height = 0.08
timer_t = jnp.array([0000.5,0000.5,0000.0,0000.0])
timer_t_sim = timer_t.copy()
contact, timer_t = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq,leg_time=timer_t, dt=dt)
liftoff = p_legs0.copy()
counter = 0
high_freq_counter = 0
@jax.jit
def jitted_reference_generator(foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    return mpc_utils.reference_generator(N,dt,n_joints,n_contact,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff)
ids = []
jitted_dynamics = jax.jit(dynamics)
data.qpos = x0[:7+n_joints]
with mujoco.viewer.launch_passive(model, data) as viewer:
    for c in range(n_contact):
        ids.append(render_vector(viewer,
              np.zeros(3),
              np.zeros(3),
              0.1,
              np.array([1, 0, 0, 1])))
    for c in range(n_contact):
        for k in range(N):
            ids.append(render_sphere(viewer,
                         np.zeros(3),
                         diameter = 0.01,
                  color=[0,1,0,1]))
            ids.append(render_sphere(viewer,
                         np.zeros(3),
                         diameter = 0.01,
                  color=[0,1,0,1]))
    # for k in range(N):
    #         ids.append(render_sphere(viewer,
    #                      np.zeros(3),
    #                      diameter = 0.01,
    #               color=[0,1,0,1]))
    avg_tau = jnp.zeros(n_joints)
    while viewer.is_running():
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

            foot_op = np.array([data.geom_xpos[contact_id[i]] for i in range(n_contact)])
            contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq ,leg_time=timer_t_sim, dt=1/mpc_frequency)
            timer_t = timer_t_sim.copy()
            ref_base_lin_vel = jnp.array([0.3,0,0])
            ref_base_ang_vel = jnp.array([0,0,0])

        
            foot_op_vec = foot_op.flatten()
            x0 = jnp.concatenate([qpos, qvel,foot_op_vec,np.zeros(3*n_contact)])
            input = (ref_base_lin_vel, ref_base_ang_vel, 0.98)
            start = timer()
            reference , parameter , liftoff = jitted_reference_generator(p_legs0,p0[7:7+n_joints],timer_t, jnp.concatenate([qpos,qvel]), foot_op_vec, input, duty_factor = duty_factor,  step_freq= step_freq ,step_height=step_height,liftoff=liftoff)
            # reference = jnp.concatenate([jnp.tile(p_ref,(N+1,1)),jnp.tile(quat_ref,(N+1,1)),jnp.tile(q_ref,(N+1,1)),jnp.tile(dp_ref,(N+1,1)),jnp.tile(omega_ref,(N+1,1)),jnp.tile(foot_op_vec,(N+1,1))], axis=1)
            # parameter = jnp.concatenate([jnp.ones((N,2)),jnp.ones((N,2))])
            # print(reference[:,13+n_joints+3*n_contact:].T)
            
            # X0 = X0.at[0].set(x0)
            X,U,V =  work(reference,parameter,x0,X0,U0,V0)
            X.block_until_ready()
            stop = timer()
            print(f"Time elapsed: {stop-start}")
            tau_val = U[:4,:n_joints]
            grf = X[1,13+2*n_joints+n_contact*3:]
            # print("grf",grf)    
            high_freq_counter = 0
            if jnp.any(jnp.isnan(tau_val)):
                print('Nan detected')
                U0 = jnp.tile(u_ref, (N, 1))
                V0 = jnp.zeros((N + 1,n ))
                x0 = jnp.concatenate([p0, jnp.zeros(6 + n_joints),p_legs0,jnp.zeros(3*n_contact)])
                X0 = jnp.tile(x0, (N + 1, 1))
            else:
                shift = int(1/(dt*mpc_frequency))
                U0 = jnp.concatenate([U[shift:],jnp.tile(U[-1:],(shift,1))])
                X0 = jnp.concatenate([X[shift:],jnp.tile(X[-1:],(shift,1))])
                V0 = jnp.concatenate([V[shift:],jnp.tile(V[-1:],(shift,1))])
            # tau = 500*(p0[7:7+n_joints]-qpos[7:7+n_joints]) - 20*qvel[6:6+n_joints]
            # x_new = jitted_dynamics(x0,tau,0,parameter)
            # data.qpos = x_new[:7+n_joints]
            # data.qvel = x_new[7+n_joints:13+2*n_joints]
            # mujoco.mj_step(model, data)
            # grf = x_new[13+2*n_joints+n_contact*3:]
            
            for c in range(n_contact):
                print(np.linalg.norm(grf[3*c:3*(c+1)]))
                render_vector(viewer,
                      grf[3*c:3*(c+1)],
                      data.geom_xpos[contact_id[c]],
                      np.linalg.norm(grf[3*c:3*(c+1)])/500,
                      np.array([1, 0, 0, 1]),
                      ids[c])
            n_sphere = n_contact
            for c in range(n_contact):
                for k in range(N):
                    render_sphere(viewer,
                                 reference[k,13+n_joints+3*c:13+n_joints+3*(c+1)],
                                 diameter = 0.01,
                          color=[0,1,0,1],
                          geom_id = ids[n_sphere])
                    n_sphere += 1
            
                    render_sphere(viewer,
                                 X[k,13+2*n_joints+3*c:13+2*n_joints+3*(c+1)],
                                 diameter = 0.01,
                          color=[1,0,0,1],
                          geom_id = ids[n_sphere])
                    n_sphere += 1
            # for k in range(N):
            #         render_sphere(viewer,
            #                      X[k,:3],
            #                      diameter = 0.01,
            #               color=[1,0,0,1],
            #               geom_id = ids[n_sphere])
            #         n_sphere += 1
        if counter % (sim_frequency * dt) == 0 or counter == 0:
            tau = tau_val[high_freq_counter,:]
            high_freq_counter += 1
        counter += 1        
        data.ctrl = tau + 20*(X[high_freq_counter,7:7+n_joints]-qpos[7:7+n_joints]) + 5*(X[high_freq_counter,13+n_joints:13+2*n_joints] - qvel[6:6+n_joints])
        # if counter > 100:
            # avg_tau += data.ctrl
            # print(avg_tau/(counter-100))
        # print("grf",grf)
        mujoco.mj_step(model, data)
        
        viewer.sync()
        
    
    

