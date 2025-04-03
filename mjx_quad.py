import os
 
# Set environment variables for XLA flags
#os.environ['XLA_FLAGS'] = (
#    '--xla_gpu_enable_triton_softmax_fusion=true '
#    '--xla_gpu_triton_gemm_any=true '
    # '--xla_gpu_deterministic_ops=true'
#)
 
import jax.numpy as jnp
import jax
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
import utils.config as config

from timeit import default_timer as timer
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)
 
# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "stairs"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = 50.0
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 200.0
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
while env.viewer.is_running():
 
    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
    
 
        ref_base_lin_vel = env._ref_base_lin_vel_H
        ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])
 
        
        input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])
        
        if counter != 0:
            for i in range(delay):
                qpos = env.mjData.qpos.copy()
                qvel = env.mjData.qvel.copy()
                # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                tau_fb = -3*(qvel[6:6+config.n_joints])
                mpc_time += 1
                state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
                counter += 1
        start = timer()
        tau_old = tau
        tau, q, dq = mpc.run(qpos,qvel,input)   
        stop = timer()
        print("Time taken for MPC: ", stop-start)   

        mpc_time = 0
        stop = timer()

        # tau = U[0,:config.n_joints]
        # for leg in range(config.n_contact):
        #     pleg = reference[:,13+config.n_joints:]
        #     for i in range(config.N):
        #         render_sphere(viewer=env.viewer,
        #                   position = pleg[i,3*leg:3+3*leg],
        #                   diameter = 0.01,
        #                   color=[0,1,0,1],
        #                   geom_id = ids[leg*config.N+i])
        # for i in range(config.N):
        #     render_sphere(viewer=env.viewer,
        #                   position = reference[i,:3],
        #                   diameter = 0.01,
        #                   color=[0,1,0,1],
        #                   geom_id = ids[config.N*config.n_contact+config.N+i])
        # time.sleep(1)
        # grf = X[1,13+2*config.n_joints+3*config.n_contact:]
        # for c in range(config.n_contact):
        #         render_vector(env.viewer,
        #               grf[3*c:3*(c+1)],
        #               foot_op_vec[3*c:3*(c+1)],
        #               np.linalg.norm(grf[3*c:3*(c+1)])/500,
        #               np.array([1, 0, 0, 1]),
        #               ids[c])
        # tau_val = U[:4,:config.n_joints]
        # high_freq_counter = 0
        # if jnp.any(jnp.isnan(tau_val)):
        #     print('Nan detected')
        #     U0 = jnp.tile(config.u_ref, (config.N, 1))
        #     X0 = jnp.tile(x0, (config.N + 1, 1))
        #     V0 = jnp.zeros((config.N + 1, config.n ))
        # else:
        #     shift = int(1/(config.dt*mpc_frequency))
        #     U0 = jnp.concatenate([U[shift:],jnp.tile(U[-1:],(shift,1))])
        #     X0 = jnp.concatenate([X[shift:],jnp.tile(X[-1:],(shift,1))])
        #     V0 = jnp.concatenate([V[shift:],jnp.tile(V[-1:],(shift,1))])
        
    # if counter % (sim_frequency * config.dt) == 0 or counter == 0:
    #         tau = tau_val[high_freq_counter,:]
    #         high_freq_counter += 1

    tau_fb = -3*(qvel[6:6+config.n_joints])
    # tau_fb = K@(x-np.concatenate([qpos,qvel]))
    mpc_time += 1
    state, reward, is_terminated, is_truncated, info = env.step(action= tau + tau_fb)
    # mujoco.mj_step(env.mjModel, env.mjData)
    counter += 1
    
    # if False:
    #     env.reset(random=False)
    #     timer_t = jnp.array([0000.5,0000.0,0000,0000.5])
    #     liftoff = config.p_legs0.copy()
    #     counter = 0
    #     x0 = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
    #     U0 = jnp.tile(config.u_ref, (config.N, 1))
    #     X0 = jnp.tile(x0, (config.N + 1, 1))
    #     V0 = jnp.zeros((config.N + 1, config.n ))
    env.render()