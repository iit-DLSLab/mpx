import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import mujoco.viewer
from timeit import default_timer as timer
import utils.mpc_wrapper as mpc_wrapper
import config.config_go2 as config
from gym_quadruped.quadruped_env import QuadrupedEnv
from functools import partial
import math
# -- JAX setup --------------------------------------------------------------
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
# --------------------------------------------------------------------------
robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 200.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=0.7,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)
env.render()

# Simulation parameters
n_env = 64
sim_frequency = 200.0
mpc_frequency = config.mpc_frequency
episode_length = 10.0  # seconds
n_episodes = 10
robots_per_row = math.ceil(math.sqrt(n_env))
offset_x = jnp.tile(jnp.arange(robots_per_row),(1,robots_per_row)).flatten()
offset_y = jnp.tile(jnp.arange(robots_per_row),(robots_per_row,1)).T.flatten()
offset = jnp.concatenate([offset_x[:, None], offset_y[:, None], jnp.zeros((n_env, 5 + config.n_joints))], axis=-1)
# Build model and data
model = mujoco.MjModel.from_xml_path(dir_path + '/../data/go2/scene_mjx.xml')
data = mujoco.MjData(model)
model.opt.timestep = 1/sim_frequency

# Contact IDs
contact_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
              for name in config.contact_frame]

# MPC wrapper
mpc = mpc_wrapper.BatchedMPCControllerWrapper(config, n_env)

# Initialize state
data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
mujoco.mj_kinematics(model, data)
foot_op = np.array([data.geom_xpos[i] for i in contact_id])
mpc.liftoff = jnp.tile(foot_op.flatten(), (n_env, 1))

# Put into MJX
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)
mjx_contact_id = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
                  for name in config.contact_frame]

# JIT-compiled stepping
@jax.jit
def _mjx_step(model, data, action):
    tau_fb = -3 * (data.qvel[6:6+config.n_joints])
    data = data.replace(ctrl=tau_fb + action)
    return mjx.step(model,data)


def _set_inputs_helper(data, command,mjx_contact_id):
    foot_pos = jnp.array([data.geom_xpos[mjx_contact_id[i]] for i in range(config.n_contact)]).flatten()
    if config.grf_as_state:
        x0 = jnp.concatenate([data.qpos, data.qvel, foot_pos, jnp.zeros(3*config.n_contact)])
    else:
        x0 = jnp.concatenate([data.qpos, data.qvel, foot_pos])
    inp = jnp.array([command[0], command[1], 0., 0., 0., command[2], config.robot_height])
    return x0, inp, foot_pos

step = jax.jit(jax.vmap(_mjx_step, in_axes=(None,0,0)))

_set_inputs = partial(_set_inputs_helper, mjx_contact_id=mjx_contact_id)
set_inputs = jax.vmap(_set_inputs)

# Random command generator
def set_random_command(n_env, limits, key):
    keys = jax.random.split(key, len(limits))
    cmds = [jax.random.uniform(k, (n_env,), minval=low, maxval=high)
            for k, (low, high) in zip(keys, limits)]
    return jnp.stack(cmds, axis=1)
# Prepare initial batch data
qpos0 = jnp.tile(jnp.concatenate([config.p0, config.quat0, config.q0]), (n_env, 1))
batch_data = jax.vmap(lambda x: mjx_data.replace(qpos=x))(qpos0)
command_limits = [(-0.2, 0.5), (-0.1, 0.1), (-0.2, 0.2)]
rng_key = jax.random.PRNGKey(0)
# Containers for collected data
action = jnp.zeros((n_env, config.n_joints))
# Launch viewer and simulate
while env.viewer.is_running():
    for _ in range(n_episodes):
        # Generate random command
        rng_key, subkey_cmd, subkey_gait = jax.random.split(rng_key,num=3)
        batch_command = set_random_command(n_env, command_limits, subkey_cmd)        
        # reset sim
        batch_data = jax.vmap(lambda x: mjx_data.replace(qpos=x, qvel=jnp.zeros(6+config.n_joints), ctrl = jnp.zeros(config.n_joints)))(qpos0)
        batch_data = step(mjx_model, batch_data, action)
        _, _, batch_foot = set_inputs(batch_data, batch_command)
        mpc.reset(jnp.arange(n_env),batch_foot)
        start = timer()
        for t in range(int(episode_length * sim_frequency)):
            if t % int(sim_frequency/mpc_frequency) == 0:
                # compute inputs and run MPC
                batch_x0, batch_input, batch_foot = set_inputs(batch_data, batch_command)
                start_mpc = timer()
                _, U, _ = mpc.run(batch_x0, batch_input, batch_foot)
                U.block_until_ready()
                stop_mpc = timer()
                print(f"MPC time: {stop_mpc - start_mpc:.4f} seconds")
                
                action = U[:, 0, :config.n_joints]
            
                
            # step env
            batch_data = step(mjx_model, batch_data, action)
            if t % int(sim_frequency/mpc_frequency) == 0:
                start_render = timer()
                batch_robots = batch_data.qpos + offset
                env._render_ghost_robots(batch_robots,1.0)
                env.render()
                stop_render = timer()
                print(f"Render time: {stop_render - start_render:.4f} seconds")
        stop = timer()
        print(f"Episode time: {stop - start:.2f} seconds")
        print(f"Real time factor: {episode_length * n_env / (stop - start):.2f}")