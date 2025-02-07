import os

# Set environment variables for XLA flags
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=true '
    # '--xla_gpu_deterministic_ops=true'
)

import jax.numpy as jnp
import jax

# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import numpy as np
import primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from functools import partial
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere

import utils.mpc_utils as mpc_utils
import utils.models as mpc_dyn_model
import utils.objectives as mpc_objectives
import utils.config as config

from mujoco import mjx
from mujoco.mjx._src import math

# Set GPU device for JAX
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

# Define robot and scene parameters
robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
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

# Load Mujoco model and data
model = mujoco.MjModel.from_xml_path('./data/go2/go2_mjx.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(model)

# Get contact and body IDs from configuration
contact_id = []
for name in config.contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in config.body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))

# Initial state and control inputs
x0 = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
U0 = jnp.tile(config.u_ref, (config.N, 1))
X0 = jnp.tile(x0, (config.N + 1, 1))
V0 = jnp.zeros((config.N + 1, config.n ))

from timeit import default_timer as timer

# Define cost and dynamics functions
grf_scaling = 220
cost = partial(mpc_objectives.quadruped_wb_obj, config.W, config.n_joints, config.n_contact, config.N, grf_scaling)
dynamics = partial(mpc_dyn_model.quadruped_wb_dynamics,model,mjx_model,contact_id, body_id,config.n_joints,config.dt)

# Define JAX jitted functions for MPC and reference generation
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

@jax.jit
def jitted_reference_generator(foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    return mpc_utils.reference_generator(config.N,config.dt,config.n_joints,config.n_contact,foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff)

# Timer initialization
timer_t_sim = config.timer_t.copy()
contact, timer_t = mpc_utils.timer_run(duty_factor = config.duty_factor, step_freq = config.step_freq,leg_time=config.timer_t, dt=config.dt)
liftoff = config.p_legs0.copy()

counter = 0
high_freq_counter = 0
env.render()
# ids = []
# for i in range(N*4):
#      ids.append(render_sphere(viewer=env.viewer,
#               position = np.array([0,0,0]),
#               diameter = 0.01,
#               color=[1,0,0,1]))

# Main simulation loop
while env.viewer.is_running():

    qpos = env.mjData.qpos
    qvel = env.mjData.qvel
    if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

        foot_op = np.array([env.feet_pos('world').FL, env.feet_pos('world').FR, env.feet_pos('world').RL, env.feet_pos('world').RR],order="F")
        contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = config.duty_factor, step_freq = config.step_freq,leg_time=timer_t_sim, dt=1/mpc_frequency)
        timer_t = timer_t_sim.copy()

        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        foot_op_vec = foot_op.flatten()
        x0 = jnp.concatenate([qpos, qvel,foot_op_vec,jnp.zeros(3*config.n_contact)])
        input = (ref_base_lin_vel, ref_base_ang_vel, config.robot_height)
        
        reference , parameter , liftoff = jitted_reference_generator(config.p_legs0,timer_t, jnp.concatenate([qpos,qvel]), foot_op_vec, input, duty_factor = config.duty_factor,  step_freq= config.step_freq ,step_height=config.step_height,liftoff=liftoff)
        
        start = timer()
        X,U,V =  work(reference,parameter,x0,X0,U0,V0)
        X.block_until_ready()
        stop = timer()

        print(f"Time elapsed: {stop-start}")       
        
        # for leg in range(n_contact):
        #     pleg = reference[:,13+n_joints:]
        #     for i in range(N):
        #         render_sphere(viewer=env.viewer,
        #                   position = pleg[i,3*leg:3+3*leg],
        #                   diameter = 0.01,
        #                   color=[parameter[i,leg],1,0,1],
        #                   geom_id = ids[leg*N+i])
       
        tau_val = U[:4,:config.n_joints]
        high_freq_counter = 0
        if jnp.any(jnp.isnan(tau_val)):
            print('Nan detected')
            U0 = jnp.tile(config.u_ref, (config.N, 1))
            X0 = jnp.tile(x0, (config.N + 1, 1))
            V0 = jnp.zeros((config.N + 1, config.n ))
        else:
            shift = int(1/(config.dt*mpc_frequency))
            U0 = jnp.concatenate([U[shift:],jnp.tile(U[-1:],(shift,1))])
            X0 = jnp.concatenate([X[shift:],jnp.tile(X[-1:],(shift,1))])
            V0 = jnp.concatenate([V[shift:],jnp.tile(V[-1:],(shift,1))])
    if counter % (sim_frequency * config.dt) == 0 or counter == 0:
            tau = tau_val[high_freq_counter,:]
            high_freq_counter += 1
    state, reward, is_terminated, is_truncated, info = env.step(action=tau)
    counter += 1
    if is_terminated:
        env.reset(random=False)
        timer_t = jnp.array([0000.5,0000.0,0000,0000.5])
        liftoff = config.p_legs0.copy()
        counter = 0
        x0 = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(x0, (config.N + 1, 1))
        V0 = jnp.zeros((config.N + 1, config.n ))
    env.render()
