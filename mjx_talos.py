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
# from gym_quadruped.utils.mujoco.visual import render_sphere ,render_vector
import utils.mpc_wrapper as mpc_wrapper
import config.config_talos as config

model = mujoco.MjModel.from_xml_path('./data/pal_talos/scene_motor.xml')
data = mujoco.MjData(model)
sim_frequency = 500.0
model.opt.timestep = 1/sim_frequency

# contact_id = []
# for name in config.contact_frame:
#     contact_id.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name))
# body_id = []
# for name in config.body_name:
#     body_id.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name))

mpc = mpc_wrapper.MPCControllerWrapper(config)
data.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])

from timeit import default_timer as timer

ids = []
tau = jnp.zeros(config.n_joints)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # for c in range(config.n_contact):
    #     ids.append(render_vector(viewer,
    #           np.zeros(3),
    #           np.zeros(3),
    #           0.1,
    #           np.array([1, 0, 0, 1])))
    # for c in range(config.n_contact):
    #     for k in range(config.N):
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
    mujoco.mj_step(model, data)
    viewer.sync()
    delay = int(0*sim_frequency)
    print('Delay: ',delay)
    mpc.robot_height = config.robot_height
    mpc.reset(data.qpos.copy(),data.qvel.copy())
    counter = 0
    while viewer.is_running():
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        if counter % (sim_frequency / config.mpc_frequency) == 0 or counter == 0:
            
            if counter != 0:
                for i in range(delay):
                    qpos = data.qpos.copy()
                    qvel = data.qvel.copy()
                    tau_fb = -3*(qvel[6:6+config.n_joints])
                    data.ctrl = tau + tau_fb
                    mujoco.mj_step(model, data)
                    counter += 1
            start = timer()
            # foot_op = np.array([data.geom_xpos[contact_id[i]] for i in range(config.n_contact)])
            # foot_op_vec = foot_op.flatten()
            ref_base_lin_vel = jnp.array([0.3,0,0])
            ref_base_ang_vel = jnp.array([0,0,0.2])
            
            # x0 = jnp.concatenate([qpos, qvel,jnp.zeros(3*config.n_contact)])
            input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           1.0])
            start = timer()
            tau, q, dq = mpc.run(qpos,qvel,input)   
            stop = timer()
            print(f"Time elapsed: {stop-start}")            
            # for c in range(config.n_contact):
            #     render_vector(viewer,
            #           grf[3*c:3*(c+1)],
            #           data.geom_xpos[contact_id[c]],
            #           np.linalg.norm(grf[3*c:3*(c+1)])/800,
            #           np.array([1, 0, 0, 1]),
            #           ids[c])
            # n_sphere = config.n_contact
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
            # for k in range(N):
            #         render_sphere(viewer,
            #                      X[k,:3],
            #                      diameter = 0.01,
            #               color=[1,0,0,1],
            #               geom_id = ids[n_sphere])
            #         n_sphere += 1
            # time.sleep(0.5)
        counter += 1        
        data.ctrl = tau - 3*qvel[6:6+config.n_joints]
        mujoco.mj_step(model, data)
        viewer.sync()
        
    
    

