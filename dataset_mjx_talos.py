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
import time
import utils.mpc_utils as mpc_utils
import utils.objectives as mpc_objectives
import utils.models as mpc_dyn_model


import pickle

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
N = 25  # Number of stages
n =  13 + 2*n_joints + 3*n_contact # Number of states
m = n_joints  + 3*n_contact # Number of controls (F)
dt = 0.02 # Time step
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
            -0.25847, 0.773046, 0.0002,-1.225366,
            0.25847, -0.773046, -0.0002,-1.225366,
            0, 0, -0.411354, 0.859395, -0.448041, -0.001708, 
            0, 0, -0.411354, 0.859395, -0.448041, -0.001708])
p0_home = p0.copy()
x0 = jnp.concatenate([p0, jnp.zeros(6 + n_joints),p_legs0])
vel0 = jnp.zeros(6 + n_joints)  
u0 = jnp.zeros(m)
Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qq = jnp.diag(jnp.array([ 1e3, 1e3,
                          1e3, 1e3, 1e3, 1e3,
                          1e3, 1e3, 1e3, 1e3,
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
cost = partial(mpc_objectives.talos_wb_obj, n_joints, n_contact, N)
hessian_approx = partial(mpc_objectives.talos_wb_hessian_gn, n_joints, n_contact)
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
step_freq = 1.2
step_height = 0.08
timer_t = jnp.array([0.5,0.5,0.5,0.5,0.0,0.0,0.0,0.0])
timer_t_sim = timer_t.copy()
contact, timer_t = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq,leg_time=timer_t, dt=dt)
liftoff = p_legs0.copy()
counter = 0
high_freq_counter = 0
mpc_time = 0
use_terrain_estimator = False
@jax.jit
def jitted_reference_generator(foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    return mpc_utils.reference_generator(use_terrain_estimator,N,dt,n_joints,n_contact,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff)

# Dataset Parameters
total_counter = 0
run_length_time = 10.0
run_counter = 0
n_runs = 1
dataset = {}
env_id = 0
run_id = -1
dataset_frequency = 250.0

arm_reference = True

@jax.jit
def jitted_reference_generator_arm(foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff, current_time, arm_amp_ref, arm_freq_ref):
    return mpc_utils.reference_generator_arm(use_terrain_estimator,N,dt,n_joints,n_contact,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff, current_time, arm_amp_ref, arm_freq_ref)

def reset(mj_model, mj_data, q_home, key):
    key, key_ang, key_lin, key_amp, key_freq, key_init = jax.random.split(key, 6)

    ## Resample Reference
    # ref_base_lin_vel = jnp.array([-0.3,0.0,0])
    ref_base_ang_vel_lim = 0.4
    ref_base_ang_vel = jnp.array([0,
                                  0,
                                  jax.random.uniform(key_ang, shape=(), minval=-ref_base_ang_vel_lim, maxval=ref_base_ang_vel_lim)])

    ref_base_lin_vel_lim = jnp.array([0.3, 0.05, 0])
    ref_base_lin_vel = jax.random.uniform(key_lin, shape=(3,), minval=-ref_base_lin_vel_lim, maxval=ref_base_lin_vel_lim)

    ## Arm reference
    arm_amp_ref_max = jnp.concatenate([
        jnp.array([0.3, 0.4, 0.3, 0.8]),
        jnp.array([0.3, 0.4, 0.3, 0.8])
    ])
    arm_amp_ref_min = 0.1 * jnp.ones(8)

    arm_freq_ref_max = 2*jnp.concatenate([
        jnp.array([0.2, 0.2, 0.2, 0.3]),
        jnp.array([0.2, 0.2, 0.2, 0.3])
    ])
    arm_freq_ref_min = 0.2 * jnp.ones(8)
    arm_amp_ref = jax.random.uniform(key_amp, shape=(8,), minval=arm_amp_ref_min, maxval=arm_amp_ref_max)
    arm_freq_ref = jax.random.uniform(key_freq, shape=(8,), minval=arm_freq_ref_min, maxval=arm_freq_ref_max)

    ref_param = (ref_base_lin_vel, ref_base_ang_vel, arm_amp_ref, arm_freq_ref)

    q_init = q_home
    q_rand = 0.1
    q_torso_arm = jax.random.uniform(key_init, shape=(10,), minval=-q_rand, maxval=q_rand)
    q_init[7:17] = q_init[7:17] + q_torso_arm

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    mj_data.qpos[:] = np.copy(q_init)
    mj_data.qvel[:] = np.zeros_like(mj_data.qvel)
    mujoco.mj_forward(mj_model, mj_data)
    mujoco.mj_step(mj_model, mj_data)

    print(ref_param)
    
    return mj_model, mj_data, q_init, ref_param, key


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

    return dataset


ids = []
tau = jnp.zeros(n_joints)
rng_key = jax.random.PRNGKey(0)
nan_flag = False
with mujoco.viewer.launch_passive(model, data) as viewer:


    data.qpos = p0
    mujoco.mj_step(model, data)
    # jitted_dynamics = jax.jit(dynamics)
    viewer.sync()

    while viewer.is_running():

        # Check for end of Run and reset
        if total_counter % int(run_length_time * sim_frequency) == 0 or total_counter == 0 or nan_flag:
            counter = 0
            model, data, p0, ref_param, rng_key = reset(model, data, p0_home, rng_key)
            ref_base_lin_vel, ref_base_ang_vel, arm_amp_ref, arm_freq_ref = ref_param
            # print(f'ref_base_lin_vel\n{ref_base_lin_vel}')
            timer_t = jnp.array([0.5,0.5,0.5,0.5,0.0,0.0,0.0,0.0])
            timer_t_sim = timer_t.copy()
            print(f'total_counter {total_counter}')

            ## Init MPC Data
            U0 = jnp.tile(u_ref, (N, 1))
            V0 = jnp.zeros((N + 1,n ))
            x0 = jnp.concatenate([p0, jnp.zeros(6+n_joints),p_legs0])
            X0 = jnp.tile(x0, (N + 1, 1))
            U = U0.copy()


            if run_id >= 0:
                for key in run_dataset.keys():
                    if key not in ['labels']:
                        run_dataset[key] = np.array(run_dataset[key])

            # Reset Run Dataset
            if nan_flag is False:
                env_id_array = env_id*np.ones(run_dataset['q_full'].shape[0]) if run_id >= 0 else np.ones(1)
                print(f'size dataset {env_id_array.shape}')
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
            print(f'run_name {full_label}')
            run_dataset['labels'] = full_label

            nan_flag = False

        if total_counter == int(n_runs * run_length_time * sim_frequency):
            break

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        current_time = counter / sim_frequency
        if counter % (sim_frequency / dataset_frequency) == 0:
            run_dataset = add_sim_data_to_dataset(run_dataset, current_time, data, model)

        if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

            start = timer()
            foot_op = np.array([data.geom_xpos[contact_id[i]] for i in range(n_contact)])
            foot_op_vec = foot_op.flatten()
    
            contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq ,leg_time=timer_t_sim, dt=1/mpc_frequency)
            timer_t = timer_t_sim.copy()
            # ref_base_lin_vel = jnp.array([-0.3,0.0,0])
            # ref_base_ang_vel = jnp.array([0,0,0])
            
            ## Update Reference
            x0 = jnp.concatenate([qpos, qvel,foot_op_vec])
            input = jnp.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           1.0])
            start = timer()
            if arm_reference:
                reference , parameter , liftoff = jitted_reference_generator_arm(p_legs0,p0[7:7+n_joints],timer_t, jnp.concatenate([qpos,qvel]), foot_op_vec, input, duty_factor = duty_factor, step_freq=step_freq, step_height=step_height,liftoff=liftoff,
                                                                             current_time=current_time, arm_amp_ref=arm_amp_ref, arm_freq_ref=arm_freq_ref)
            else:
                reference , parameter , liftoff = jitted_reference_generator(p_legs0,p0[7:7+n_joints],timer_t, jnp.concatenate([qpos,qvel]), foot_op_vec, input, duty_factor = duty_factor,  step_freq= step_freq ,step_height=step_height,liftoff=liftoff)
            
            
            X,U,V,_ =  work(reference,parameter,W,x0,X0,U0,V0)
            X.block_until_ready()
            stop = timer()
            
            # print(f"Time elapsed: {stop-start}")
            mpc_time = 0
            tau_val = U[:8,:n_joints]
            # grf = X[1,13+2*n_joints+n_contact*3:]
            grf = U[0,n_joints:]
            high_freq_counter = 0
            if jnp.any(jnp.isnan(tau_val)):
                nan_flag = True
                print('Nan detected')
                U0 = jnp.tile(u_ref, (N, 1))
                V0 = jnp.zeros((N + 1,n ))
                x0 = jnp.concatenate([p0, jnp.zeros(6+n_joints),p_legs0])
                X0 = jnp.tile(x0, (N + 1, 1))
                U = U0.copy()
            else:
                shift = int(1/(dt*mpc_frequency))
                U0 = jnp.concatenate([U[shift:],jnp.tile(U[-1:],(shift,1))])
                X0 = jnp.concatenate([X[shift:],jnp.tile(X[-1:],(shift,1))])
                V0 = jnp.concatenate([V[shift:],jnp.tile(V[-1:],(shift,1))])
            
  
        if counter % (sim_frequency * dt) == 0 or counter == 0:
            tau = tau_val[high_freq_counter,:]
            high_freq_counter += 1
        mpc_time += 1
        counter += 1
        total_counter += 1
        data.ctrl = tau - 3*qvel[6:6+n_joints]#+ 20*(p0[7:7+n_joints]-qpos[7:7+n_joints]) + 5*(- qvel[6:6+n_joints])

        mujoco.mj_step(model, data)
        viewer.sync()
        
    

# Create folder and save dataset
folder_name = 'datasets/talos/'
os.makedirs(folder_name, exist_ok=True)

filename = 'samples_' + str(total_counter) + '_data_freq_' + str(int(dataset_frequency)) + '_sim_time_' + str(int(run_length_time)) + '.pkl'
with open(folder_name + filename, 'wb') as fp:
    pickle.dump(dataset, fp)
    print(f'Dictionary saved successfully to file {filename} | Nsamples {total_counter}')