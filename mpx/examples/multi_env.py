import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import mujoco.viewer
from timeit import default_timer as timer
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_talos as config
# from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_ghost_robot
from functools import partial
import math
# -- JAX setup --------------------------------------------------------------
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
# --------------------------------------------------------------------------

# Simulation parameters
n_env = 64
sim_frequency = 500.0
mpc_frequency = config.mpc_frequency
episode_length = 10.0  # seconds
n_episodes = 10
robots_per_row = math.ceil(math.sqrt(n_env))
offset_x = jnp.tile(jnp.arange(robots_per_row),(1,robots_per_row)).flatten()
offset_y = jnp.tile(jnp.arange(robots_per_row),(robots_per_row,1)).T.flatten()
offset = jnp.concatenate([offset_x[:, None], offset_y[:, None], jnp.zeros((n_env, 5 + config.n_joints))], axis=-1)
# Build model and data
model = mujoco.MjModel.from_xml_path(dir_path + '/../data/pal_talos/scene_motor.xml')
data = mujoco.MjData(model)
model.opt.timestep = 1/sim_frequency

# Contact IDs
contact_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
              for name in config.contact_frame]

# MPC wrapper
mpc = mpc_wrapper.BatchedMPCControllerWrapper(config, n_env,limited_memory=True)
batch_mpc_data = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))
def _solve_mpc(mpc_data, x0, input):
    """Run MPC for a batch of environments."""
    return mpc.run(mpc_data,  x0, input)
solve_mpc = jax.jit(jax.vmap(_solve_mpc))
mpc_reset = jax.jit(jax.vmap(mpc.reset, in_axes = (0,0,0,0)))
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
set_inputs = jax.jit(jax.vmap(_set_inputs))

# Random command generator
def set_random_command(n_env, limits, key):
    keys = jax.random.split(key, len(limits))
    cmds = [jax.random.uniform(k, (n_env,), minval=low, maxval=high)
            for k, (low, high) in zip(keys, limits)]
    return jnp.stack(cmds, axis=1)
# Prepare initial batch data
qpos0 = jnp.tile(jnp.concatenate([config.p0, config.quat0, config.q0]), (n_env, 1))
batch_data = jax.vmap(lambda x: mjx_data.replace(qpos=x))(qpos0)
command_limits = [(-0.2, 0.5), (-0.1, 0.1), (-0.3, 0.3)]
rng_key = jax.random.PRNGKey(0)
# Containers for collected data
action = jnp.zeros((n_env, config.n_joints))
# Launch viewer and simulate
viewer = mujoco.viewer.launch_passive(model, data)
temp_data = data
ids = []
# Initialize ghost robots
for idx in range(n_env):
    temp_data.qpos = temp_data.qpos
    mujoco.mj_forward(model, temp_data)
    # Render each robot
    ids.append(render_ghost_robot(
        viewer,
        model,
        temp_data,
        alpha = 1,
    )
    )
viewer.sync()
while viewer.is_running():
    for _ in range(n_episodes):
        # Generate random command
        rng_key, subkey_cmd, subkey_gait = jax.random.split(rng_key,num=3)
        batch_command = set_random_command(n_env, command_limits, subkey_cmd)        
        # reset sim
        batch_data = jax.vmap(lambda x: mjx_data.replace(qpos=x, qvel=jnp.zeros(6+config.n_joints), ctrl = jnp.zeros(config.n_joints)))(qpos0)
        batch_data = step(mjx_model, batch_data, action)
        _, _, batch_foot = set_inputs(batch_data, batch_command)
        batch_mpc_data = mpc_reset(batch_mpc_data,batch_data.qpos,batch_data.qvel,batch_foot)
        for t in range(int(episode_length * sim_frequency)):
            if t % int(sim_frequency/mpc_frequency) == 0:
                # compute inputs and run MPC
                batch_x0, batch_input, _ = set_inputs(batch_data, batch_command)
                start_mpc = timer()
                batch_mpc_data, tau = solve_mpc(
                    batch_mpc_data, batch_x0, batch_input)
                tau.block_until_ready()
                stop_mpc = timer()
                print(f"MPC time: {stop_mpc - start_mpc:.4f} seconds") 
                action = tau    
            # step env
            batch_data = step(mjx_model, batch_data, action)
            if t % int(sim_frequency/mpc_frequency) == 0:
                batch_robots = batch_data.qpos + offset
                for idx in range(n_env):
                    temp_data.qpos = batch_robots[idx]
                    mujoco.mj_forward(model, temp_data)
                    # Render each robot
                    render_ghost_robot(
                        viewer,
                        model,
                        temp_data,
                        alpha = 1,
                        ghost_geoms = ids[idx],
                    )
                viewer.sync()
