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
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import numpy as np
import mujoco

import utils.mpc_wrapper as mpc_wrapper
import utils.config as config

from gym_quadruped.quadruped_env import QuadrupedEnv

import pickle

# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
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
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=1.5,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )



obs = env.reset(random=False)

# Define the MPC wrapper
mpc = mpc_wrapper.MPCControllerWrapper(config)
env.mjData.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])
env.render()


# Dataset Parameters
total_counter = 0
run_length_time = 10.0
run_counter = 0
n_runs = 50
dataset = {}
env_id = 0
run_id = -1
dataset_frequency = 100.0


def reset(env, mpc, config, key):
    key, key_ang, key_lin, key_height = jax.random.split(key, 4)

    ## Resample Reference
    ref_base_ang_vel_min = -0.3
    ref_base_ang_vel_max = 0.3
    ref_base_ang_vel = jnp.array([0,
                                  0,
                                  jax.random.uniform(key_ang, shape=(), minval=ref_base_ang_vel_min, maxval=ref_base_ang_vel_max)])

    ref_base_lin_vel_min = jnp.array([-0.3, -0.3, 0])
    ref_base_lin_vel_max = jnp.array([0.5, 0.3, 0])
    ref_base_lin_vel = jax.random.uniform(key_lin, shape=(3,), minval=ref_base_lin_vel_min, maxval=ref_base_lin_vel_max)

    ref_base_height_lim = 0.01
    ref_base_height = config.robot_height
    ref_base_height += jax.random.uniform(key_height, minval=-ref_base_height_lim, maxval=ref_base_height_lim)


    ref_param = (ref_base_lin_vel, ref_base_ang_vel, ref_base_height)
    print(f'ref_param\n {ref_param}')

    # TODO: Find a way to randomize the initial configuration, maybe use 
    q_init = jnp.concatenate([config.p0, config.quat0,config.q0])
    q_rand = 0.1
    # q_arm = jax.random.uniform(key_init, shape=(8,), minval=-q_rand, maxval=q_rand)
    # q_init = q_init.at[6:].add(q_arm)
    # print(f'q_init\n {q_init}')

    qd_init = jnp.zeros(config.n_joints + 6)

    # Reset Environment
    # env.reset(qpos=q_init,
    #           qvel=qd_init)
    env.reset(random=True)

    print(f'q_init\n {env.mjData.qpos.copy()}')
    # Reset MPC
    mpc.robot_height = ref_base_height
    mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())

    return env, mpc, ref_param, key


def init_run_dataset():
    training_labels = [
        "labels", "t", "q_full",
        "qd_full", "qdd_full",
        "tau_mj_real", "tau_m_mj_real", "tau_cg_mj_real",
        "tau_g_mj_real",
    ]
    dataset = {label: [] for label in training_labels}
    return dataset


def add_sim_data_to_dataset(dataset, time, mj_data, mj_model):
    # Joint
    q = np.copy(mj_data.qpos[7:])

    # Base
    base_lin_pos = np.copy(mj_data.qpos[:3])
    base_lin_vel = np.copy(mj_data.qvel[:3])
    base_lin_acc = np.copy(mj_data.qacc[:3])

    base_quat = np.roll((np.copy(mj_data.qpos[3:7])),-1) # Get in xyzw order

    q_full = np.concatenate((base_lin_pos,
                             base_quat,
                             q,
                            ))
    qd_full = np.copy(mj_data.qvel)
    qdd_full = np.copy(mj_data.qacc)

    # Mujoco Simulation model
    M_full_mj_real = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M_full_mj_real, mj_data.qM)
    tau_m_mj_real = M_full_mj_real @ qdd_full
    tau_cg_mj_real = np.copy(mj_data.qfrc_bias.reshape((mj_model.nv,)))
    tau_mj_real = tau_m_mj_real + tau_cg_mj_real
    tau_g_mj_real = np.zeros(mj_model.nv)

    # Save to Dictionary
    dataset['t'].append(time)
    dataset['q_full'].append(q_full)
    dataset['qd_full'].append(qd_full)
    dataset['qdd_full'].append(qdd_full)

    dataset['tau_mj_real'].append(tau_mj_real)
    dataset['tau_m_mj_real'].append(tau_m_mj_real)
    dataset['tau_cg_mj_real'].append(tau_cg_mj_real)
    dataset['tau_g_mj_real'].append(tau_g_mj_real)

    # Labels and, env_id

    return dataset, tau_mj_real


# Init Data
tau = jnp.zeros(config.n_joints)
tau_old = jnp.zeros(config.n_joints)
nan_flag = False
rng_key = jax.random.PRNGKey(0)

while env.viewer.is_running():

    # Check for end of Run and reset
    if total_counter % int(run_length_time * sim_frequency) == 0 or total_counter == 0 or nan_flag:
        counter = 0
        env, mpc, ref_param, rng_key = reset(env, mpc, config, rng_key)
        ref_base_lin_vel, ref_base_ang_vel, ref_height = ref_param
        # print(f'ref_base_lin_vel\n{ref_base_lin_vel}')

        if run_id >= 0:
            for key in run_dataset.keys():
                if key not in ['labels']:
                    run_dataset[key] = np.array(run_dataset[key])

        # Reset Run Dataset
        if nan_flag is False:
            env_id_array = env_id*np.ones(run_dataset['q_full'].shape[0]) if run_id >= 0 else np.ones(1)
            if run_id > 0:
                dataset['env_id'].append(env_id_array)
                for key in run_dataset.keys():
                    dataset[key].append(run_dataset[key])

            elif run_id == 0:
                dataset['env_id'] = [env_id_array]
                for key in run_dataset.keys():
                    dataset[key] = [run_dataset[key]]
                
            run_id += 1

        else:
            total_counter = run_id * int(run_length_time * sim_frequency)
        
        run_dataset = init_run_dataset()
        full_label = 'env_' + str(env_id) + '_run_' + str(run_id)
        print(f'Starting run: {full_label}')
        run_dataset['labels'] = full_label

        nan_flag = False

    # Check if end of the dataset
    if total_counter == int(n_runs * run_length_time * sim_frequency):
        break

    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    current_time = counter / sim_frequency
    
    # Add data to dataset
    if counter % (sim_frequency / dataset_frequency) == 0:
        run_dataset, tau_mj_real = add_sim_data_to_dataset(run_dataset, current_time, env.mjData, env.mjModel)

        if (np.any(np.abs(tau_mj_real[3:5]) > 800) or np.any(np.abs(qvel[3:5]) > 2.5)) and current_time > 0.5:
            print(tau_mj_real[3:5])
            print(qvel[3:5])
            nan_flag = True

    # MPC Update
    if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

        # ref_base_lin_vel = env._ref_base_lin_vel_H
        # ref_base_ang_vel = np.array([0., 0., env._ref_base_ang_yaw_dot])

        # input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
        #                     ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
        #                     config.robot_height])

        input = np.concatenate((ref_base_lin_vel, ref_base_ang_vel, np.array([ref_height])))

        tau_old = tau
        tau, q, dq = mpc.run(qpos,qvel,input)

        # Check NaN
        if jnp.any(jnp.isnan(tau)):
            nan_flag = True
            print('Nan detected')

    counter += 1
    total_counter += 1
    tau_fb = -3*(qvel[6:6+config.n_joints])
    
    # Step Simulation
    state, reward, is_terminated, is_truncated, info = env.step(action= tau + tau_fb)
    env.render()
        
# Create folder and save dataset
folder_name = 'datasets/' + robot_name + '/'
os.makedirs(folder_name, exist_ok=True)

# Add foot data
dataset['duty_factor'] = config.duty_factor
dataset['step_freq'] = config.step_freq
dataset['step_height'] = config.step_height

Nsamples = int(n_runs * run_length_time * sim_frequency)
filename = robot_name + '_samples_' + str(Nsamples) + '_n_runs_' + str(n_runs) + '_data_freq_' + str(int(dataset_frequency)) + '_sim_freq_' + str(int(sim_frequency)) + '_total_time_' + str(int(run_length_time)) + '_base_full_arm.pkl'
with open(folder_name + filename, 'wb') as fp:
    pickle.dump(dataset, fp)
    print(f'Dictionary saved successfully to file {filename} | Nsamples {total_counter}')