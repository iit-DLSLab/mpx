import os
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=true '
#     # 'XLA_PYTHON_CLIENT_PREALLOCATE=false'
#     # '--xla_gpu_deterministic_ops=true'
# )
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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
import numpy as np
import copy
# from gym_quadruped.utils.mujoco.visual import render_sphere ,render_vector
import time
import utils.mpc_utils as mpc_utils
import utils.objectives as mpc_objectives
import utils.models as mpc_dyn_model

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)
model = mujoco.MjModel.from_xml_path('./data/unitree_h1/mjx_scene_h1_walk.xml')
data = mujoco.MjData(model)
mpc_frequency = 50.0
sim_frequency = 500.0
model.opt.timestep = 1/sim_frequency
contact_frame = ['FL','RL','FR','RR']

body_name = ['left_ankle_link','right_ankle_link']

n_joints = 19
n_contact = 4
# # # Problem dimensions
N = 100  # Number of stages
n =  13 + 2*n_joints + 3*n_contact + 3*n_contact # Number of states
m = n_joints  # Number of controls (F)
dt = 0.01 # Time step
p_legs0 = jnp.array([ 0.14738185,  0.20541158,  0.01398883,  
                    -0.00253908,  0.2102815,   0.01398485,
                    0.14787466, -0.20581408,  0.01399987,
                    -0.00203967, -0.21088305,  0.0139761 ])
mjx_model = mjx.put_model(model)

contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))

# # # Initial state
p0 = jnp.array([0, 0, 0.91,
    1, 0, 0, 0,
    0, 0, -0.54, 1.2, -0.68,
    0, 0, -0.54, 1.2, -0.68,
    0,
    0.5, 0.25, 0.0, 0.5,
    0.5, -0.25, 0.0, 0.5])
x0 = jnp.concatenate([p0, jnp.zeros(6 + n_joints),p_legs0,jnp.array([0,0,125,0,0,125,0,0,125,0,0,125])])
vel0 = jnp.zeros(6 + n_joints)  
u0 = jnp.zeros(m)
Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qq = jnp.diag(jnp.array([ 4e-1, 4e-1, 4e-1, 4e-1, 4e-1,
                          4e-1, 4e-1, 4e-1, 4e-1, 4e-1,
                          4e1, 
                          4e1, 4e1, 4e1, 4e1,
                          4e1, 4e1, 4e1, 4e1])) 
Qdp = jnp.diag(jnp.array([1, 1, 1]))*1e3
Qomega = jnp.diag(jnp.array([1, 1, 1]))*1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e0
Qrot = jnp.diag(jnp.array([1,1,1]))*1e3
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qleg = jnp.diag(jnp.tile(jnp.array([1e3,1e3,1e5]),n_contact))
Qgrf = jnp.diag(jnp.ones(3*n_contact))*1e-2
tau0 = jnp.array([
    -3.8866019e-01,  8.2269782e-01, -6.9408727e+00, -5.7233673e+01,
  9.7760363e+00,  3.9106184e-01, -1.3329812e+00, -6.8945923e+00,
 -5.7180595e+01,  9.7612352e+00, -4.5000316e-04, -1.1907737e+00,
  3.1719621e-02, -4.7352805e-04, -1.1461809e+00, -1.1899408e+00,
 -3.3747468e-02, -7.4449526e-05, -1.1465545e+00
])
# grf0 = jnp.array([0,0,198,0,0,50,0,0,198,0,0,50])
# # Define the cost function
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau,Qgrf)
cost = partial(mpc_objectives.humanoid_wb_obj, n_joints, n_contact, N)
hessian_approx = partial(mpc_objectives.humanoid_wb_hessian_gn, n_joints, n_contact)
dynamics = partial(mpc_dyn_model.humanoid_wb_dynamics,model,mjx_model,contact_id, body_id,n_joints,dt)
# # Solve
p_ref = jnp.array([0, 0, 0.9])
quat_ref = jnp.array([1, 0, 0, 0])
q_ref = jnp.array([0, 0, -0.54, 1.2, -0.68,
                   0, 0, -0.54, 1.2, -0.68,
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
        hessian_approx,
        False,
        reference,
        parameter,
        W,
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
timer_t = jnp.array([0.4,0.5,-0.1,0.0])
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
    # for c in range(n_contact):
    #     ids.append(render_vector(viewer,
    #           np.zeros(3),
    #           np.zeros(3),
    #           0.1,
    #           np.array([1, 0, 0, 1])))
    # for c in range(n_contact):
    #     for k in range(N):
    #         ids.append(render_sphere(viewer,
    #                      np.zeros(3),
    #                      diameter = 0.01,
    #               color=[0,1,0,1]))
    #         ids.append(render_sphere(viewer,
    #                      np.zeros(3),
    #                      diameter = 0.01,
    #               color=[0,1,0,1]))
    # for k in range(N):
    #         ids.append(render_sphere(viewer,
    #                      np.zeros(3),
    #                      diameter = 0.01,
    #               color=[0,1,0,1]))
    while viewer.is_running():
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

            foot_op = np.array([data.geom_xpos[contact_id[i]] for i in range(n_contact)])
            contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq ,leg_time=timer_t_sim, dt=1/mpc_frequency)
            timer_t = timer_t_sim.copy()
            ref_base_lin_vel = jnp.array([0.5,0,0])
            ref_base_ang_vel = jnp.array([0,0,0])
        
            foot_op_vec = foot_op.flatten()
            x0 = jnp.concatenate([qpos, qvel,foot_op_vec,np.zeros(3*n_contact)])
            input = jnp.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           0.9])
            start = timer()
            reference , parameter , liftoff = jitted_reference_generator(p_legs0,p0[7:7+n_joints],timer_t, jnp.concatenate([qpos,qvel]), foot_op_vec, input, duty_factor = duty_factor,  step_freq= step_freq ,step_height=step_height,liftoff=liftoff)
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
                x0 = jnp.concatenate([p0, jnp.zeros(6 + n_joints),p_legs0,jnp.array([0,0,125,0,0,125,0,0,125,0,0,125])])
                X0 = jnp.tile(x0, (N + 1, 1))
            else:
                shift = int(1/(dt*mpc_frequency))
                U0 = jnp.concatenate([U[shift:],jnp.tile(U[-1:],(shift,1))])
                X0 = jnp.concatenate([X[shift:],jnp.tile(X[-1:],(shift,1))])
                V0 = jnp.concatenate([V[shift:],jnp.tile(V[-1:],(shift,1))])
            
            # for c in range(n_contact):
            #     render_vector(viewer,
            #           grf[3*c:3*(c+1)],
            #           data.geom_xpos[contact_id[c]],
            #           np.linalg.norm(grf[3*c:3*(c+1)])/500,
            #           np.array([1, 0, 0, 1]),
            #           ids[c])
            # n_sphere = n_contact
            # for c in range(n_contact):
            #     for k in range(N):
            #         render_sphere(viewer,
            #                      reference[k,13+n_joints+3*c:13+n_joints+3*(c+1)],
            #                      diameter = 0.01,
            #               color=[0,1,0,1],
            #               geom_id = ids[n_sphere])
            #         n_sphere += 1
            
            #         render_sphere(viewer,
            #                      X[k,13+2*n_joints+3*c:13+2*n_joints+3*(c+1)],
            #                      diameter = 0.01,
            #               color=[1,0,0,1],
            #               geom_id = ids[n_sphere])
            #         n_sphere += 1
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
        mujoco.mj_step(model, data)
        
        viewer.sync()
        
    
    

