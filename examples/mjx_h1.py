import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
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
import config.config_h1 as config

model = mujoco.MjModel.from_xml_path(dir_path + '../data/unitree_h1/mjx_scene_h1_walk.xml')
data = mujoco.MjData(model)
mpc_frequency = 50.0
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
            ref_base_lin_vel = jnp.array([0.3,0,0])
            ref_base_ang_vel = jnp.array([0,0,0.2])
            

            input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           1.0])
            start = timer()
            tau, q, dq = mpc.run(qpos,qvel,input)   
            stop = timer()
            print(f"Time elapsed: {stop-start}")            
        counter += 1        
        data.ctrl = tau - 3*qvel[6:6+config.n_joints]
        mujoco.mj_step(model, data)
        viewer.sync()
        
    
    

