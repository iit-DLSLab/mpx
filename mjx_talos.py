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
import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
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
from gym_quadruped.utils.mujoco.visual import render_sphere ,render_vector
import time
import utils.mpc_utils as mpc_utils
import utils.objectives as mpc_objectives
import utils.models as mpc_dyn_model

model = mujoco.MjModel.from_xml_path('./data/pal_talos/scene_motor.xml')
data = mujoco.MjData(model)
mpc_frequency = 50.0
sim_frequency = 500.0
model.opt.timestep = 1/sim_frequency
contact_frame = ['foot_left_1','foot_left_2','foot_left_3','foot_left_4',
                'foot_right_1','foot_right_2','foot_right_3','foot_right_4']

body_name = ['leg_left_6_link','leg_right_6_link']

n_joints = 22
n_contact = len(contact_frame)
# # # Problem dimensions
N = 50  # Number of stages
n =  13 + 2*n_joints + 3*n_contact # Number of states
m = n_joints  + 3*n_contact # Number of controls (F)
dt = 0.01 # Time step
p_legs0 = jnp.array([ 0.08592681,  0.145, 0.01690434,
                      0.08592681,  0.025, 0.01690434,
                     -0.11407319,  0.145, 0.01690434,
                     -0.11407319,  0.025, 0.01690434,
                      0.08592681, -0.025, 0.01690434,
                      0.08592681, -0.145, 0.01690434,
                     -0.11407319, -0.025, 0.01690434,
                     -0.11407319, -0.145, 0.01690434 ])
mjx_model = mjx.put_model(model)

contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))

# # # Initial state
p0 = np.array([0, 0, 1.01,
            1, 0, 0, 0,
            0.0, 0.006761,
            0.25847, 0.173046, 0.0002,-0.525366,
            -0.25847, -0.173046, -0.0002,-0.525366,
            0, 0, -0.411354, 0.859395, -0.448041, -0.001708, 
            0, 0, -0.411354, 0.859395, -0.448041, -0.001708])
x0 = jnp.concatenate([p0, jnp.zeros(6 + n_joints),p_legs0])
vel0 = jnp.zeros(6 + n_joints)  
u0 = jnp.zeros(m)
Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qq = jnp.diag(jnp.array([ 1e3, 1e3,
                          1e1, 1e1, 1e1, 1e1,
                          1e1, 1e1, 1e1, 1e1, 
                          1e0, 1e0, 1e0, 1e0, 1e0, 1e0,
                          1e0, 1e0, 1e0, 1e0, 1e0, 1e0
                          ])) 
Qdp = jnp.diag(jnp.array([1, 1, 1]))*1e3
Qomega = jnp.diag(jnp.array([1, 1, 1]))*1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e0
Qrot = jnp.diag(jnp.array([1,1,1]))*1e3
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qleg = jnp.diag(jnp.tile(jnp.array([1e5,1e5,1e5]),n_contact))
Qgrf = jnp.diag(jnp.ones(3*n_contact))*1e-3
tau0 = jnp.array([
     0.00000000e+00,  9.54602083e+00,
  2.10570849e+00,  3.16683082e+00,  1.70595736e-01,  3.07892969e-03,
 -2.15571230e+00, -3.45877652e+00, -2.21616569e-01,  3.07892969e-03,
  0.00000000e+00,  5.87571065e+00, -2.16026768e+00, -7.01326068e+01, -1.56904880e+00, -8.07436111e-03,
  0.00000000e+00, -5.87571065e+00, -2.16136096e+00, -7.01246970e+01, -1.56883900e+00,  8.07436111e-03,
])
# tau0 = jnp.zeros(n_joints)
# # Define the cost function
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau,Qgrf)
cost = partial(mpc_objectives.humanoid_wb_obj, n_joints, n_contact, N)
hessian_approx = partial(mpc_objectives.humanoid_wb_hessian_gn, n_joints, n_contact)
dynamics = partial(mpc_dyn_model.talos_wb_dynamics,model,mjx_model,contact_id, body_id,n_joints,dt)
# # # Solve

u_ref = jnp.zeros(n_joints + 3*n_contact)
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))

# @jax.jit
def work(reference,parameter,W,x0,X0,U0,V0):
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
step_freq = 1.0
step_height = 0.08
timer_t = jnp.array([0.5,0.5,0.5,0.5,0.0,0.0,0.0,0.0])
timer_t_sim = timer_t.copy()
contact, timer_t = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq,leg_time=timer_t, dt=dt)
liftoff = p_legs0.copy()
counter = 0
high_freq_counter = 0
@jax.jit
def jitted_reference_generator(foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    return mpc_utils.reference_generator(N,dt,n_joints,n_contact,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff)
ids = []
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
    #         ids.append(render_sphere(viewer,
    #                      np.zeros(3),
    #                      diameter = 0.01,
    #               color=[0,1,0,1]))
    # for k in range(N):
    #         ids.append(render_sphere(viewer,
    #                      np.zeros(3),
    #                      diameter = 0.01,
    #               color=[0,1,0,1]))
    data.qpos = p0
    mujoco.mj_step(model, data)
    jitted_dynamics = jax.jit(dynamics)
    viewer.sync()
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
            x0 = jnp.concatenate([qpos, qvel,foot_op_vec])
            input = jnp.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           1.0])
            start = timer()
            reference , parameter , liftoff = jitted_reference_generator(p_legs0,p0[7:7+n_joints],timer_t, jnp.concatenate([qpos,qvel]), foot_op_vec, input, duty_factor = duty_factor,  step_freq= step_freq ,step_height=step_height,liftoff=liftoff)
            X,U,V,_ =  work(reference,parameter,W,x0,X0,U0,V0)
            X.block_until_ready()
            stop = timer()
            print(f"Time elapsed: {stop-start}")
            tau_val = U[:4,:n_joints]
            # grf = X[1,13+2*n_joints+n_contact*3:]
            grf = U[0,n_joints:]
            high_freq_counter = 0
            if jnp.any(jnp.isnan(tau_val)):
                print('Nan detected')
                U0 = jnp.tile(u_ref, (N, 1))
                V0 = jnp.zeros((N + 1,n ))
                x0 = jnp.concatenate([p0, jnp.zeros(6+n_joints),p_legs0,np.zeros(3*n_contact)])
                X0 = jnp.tile(x0, (N + 1, 1))
                U = U0.copy()
            else:
                shift = int(1/(dt*mpc_frequency))
                U0 = jnp.concatenate([U[shift:],jnp.tile(U[-1:],(shift,1))])
                X0 = jnp.concatenate([X[shift:],jnp.tile(X[-1:],(shift,1))])
                V0 = jnp.concatenate([V[shift:],jnp.tile(V[-1:],(shift,1))])
            
            for c in range(n_contact):
                render_vector(viewer,
                      grf[3*c:3*(c+1)],
                      data.geom_xpos[contact_id[c]],
                      np.linalg.norm(grf[3*c:3*(c+1)])/800,
                      np.array([1, 0, 0, 1]),
                      ids[c])
            n_sphere = n_contact
            # tau = tau_val[0,:]
            # x0 = jnp.concatenate([qpos, qvel,foot_op_vec,np.zeros(3*n_contact)])
            # x_new = jitted_dynamics(x0, U[0,:],0,parameter)
            # for c in range(n_contact):
            #     ref_leg = reference[:,13+n_joints+3*c:13+n_joints+3*(c+1)]
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
            for k in range(N):
                    render_sphere(viewer,
                                 X[k,:3],
                                 diameter = 0.01,
                          color=[1,0,0,1],
                          geom_id = ids[n_sphere])
                    n_sphere += 1
            # time.sleep(0.5)
        if counter % (sim_frequency * dt) == 0 or counter == 0:
            tau = tau_val[high_freq_counter,:]
            high_freq_counter += 1
        counter += 1        
        data.ctrl = tau - 3*qvel[6:6+n_joints]#+ 20*(p0[7:7+n_joints]-qpos[7:7+n_joints]) + 5*(- qvel[6:6+n_joints])
        # data.qpos = x_new[:7+n_joints]
        # data.qvel = x_new[7+n_joints:13+2*n_joints]
        # time.sleep(1)
        # time.sleep(0.1)
        # height_offsets = np.linspace(-0.001, 0.001, 2001)
        # print(height_offsets)
        # vertical_forces = []
        # for offset in height_offsets:
        #   mujoco.mj_resetDataKeyframe(model, data, 1)
        #   mujoco.mj_forward(model, data)
        #   data.qacc = 0
        #   # Offset the height by `offset`.
        #   data.qpos = p0
        # #   data.qpos[2] += offset
          
        #   mujoco.mj_inverse(model, data)
        #   print("**********************")
        #   print(data.qfrc_inverse)
        #   print(data.qvel)
        #   print("**********************")
        mujoco.mj_step(model, data)
        viewer.sync()
        
    
    

