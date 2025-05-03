import os
 
# Set environment variables for XLA flags
#os.environ['XLA_FLAGS'] = (
#    '--xla_gpu_enable_triton_softmax_fusion=true '
#    '--xla_gpu_triton_gemm_any=true '
    # '--xla_gpu_deterministic_ops=true'
#)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np
import primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from functools import partial
from gym_quadruped.quadruped_env import QuadrupedEnv
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
 
import utils.mpc_wrapper as mpc_wrapper
import config.config_barrel_roll as config

from timeit import default_timer as timer
import time
import pickle
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)
 
# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = 100.0
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 200.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=1.5,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)
n_env = 2
# Define the MPC wrapper
mpc = mpc_wrapper.MPCControllerWrapper(config)
env.mjData.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])
env.render()
ids = []
for i in range(config.N*12):
     ids.append(render_vector(env.viewer,
              np.zeros(3),
              np.zeros(3),
              0.1,
              np.array([1, 0, 0, 1])))
counter = 0
# Main simulation loop
tau = jnp.zeros(config.n_joints)
tau_old = jnp.zeros(config.n_joints)
delay = int(0.015*sim_frequency)
print('Delay: ',delay)
q = config.q0.copy()
dq = jnp.zeros(config.n_joints)
mpc_time = 0
mpc.robot_height = config.robot_height
mpc.reset(env.mjData.qpos.copy(),env.mjData.qvel.copy())

X,U,reference = mpc.runOffline(jnp.concatenate([config.p0, config.quat0,config.q0]),jnp.zeros(6+config.n_joints))

import matplotlib.pyplot as plt
iteration = 0

while env.viewer.is_running():
    
    env.mjData.qpos = X[counter,:7+config.n_joints]
    env.mjData.qvel = X[counter,7+config.n_joints:13+2*config.n_joints]
    grf = X[counter,13+2*config.n_joints + 3*config.n_contact:13+2*config.n_joints+6*config.n_contact]
    foot_op_vec = X[counter,13+2*config.n_joints:13+2*config.n_joints+3*config.n_contact]
    for c in range(config.n_contact):
            render_vector(env.viewer,
                  grf[3*c:3*(c+1)],
                  foot_op_vec[3*c:3*(c+1)],
                  np.linalg.norm(grf[3*c:3*(c+1)])/220,
                  np.array([1, 0, 0, 1]),
                  ids[c])
    mujoco.mj_step(env.mjModel, env.mjData)
    counter += 1
    time.sleep(1/mpc_frequency)
    if counter > config.N:
        counter = 0
    env.render()