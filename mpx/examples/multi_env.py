import argparse
import math
import os
import sys
from functools import partial
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

import mpx.config.config_talos as config
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils

gpus = [device for device in jax.devices() if device.platform == "gpu"]
if gpus:
    jax.default_device(gpus[0])
jax.config.update("jax_compilation_cache_dir", "./jax_cache")


def main(headless=False, n_env=64, episode_length=10.0, n_episodes=10, dynamics_backend=None):
    if dynamics_backend is not None:
        config.dynamics_backend = dynamics_backend
    sim_frequency = 500.0
    mpc_frequency = config.mpc_frequency
    robots_per_row = math.ceil(math.sqrt(n_env))
    grid = jnp.arange(robots_per_row**2)
    offset_x = (grid % robots_per_row)[:n_env]
    offset_y = (grid // robots_per_row)[:n_env]
    offset = jnp.concatenate(
        [offset_x[:, None], offset_y[:, None], jnp.zeros((n_env, 5 + config.n_joints))],
        axis=-1,
    )

    model = mujoco.MjModel.from_xml_path(dir_path + "/../data/pal_talos/scene_motor.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = 1 / sim_frequency

    mpc = mpc_wrapper.MPCWrapper(config, limited_memory=True)
    batch_mpc_data = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))

    def _solve_mpc(mpc_data, x0, command):
        return mpc.run(mpc_data, x0, command)

    solve_mpc = jax.jit(jax.vmap(_solve_mpc))
    mpc_reset = jax.jit(jax.vmap(mpc.reset, in_axes=(0, 0, 0, 0)))

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_kinematics(model, data)

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    mjx_contact_id = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in config.contact_frame
    ]

    def _mjx_step(model, data, action):
        tau_fb = -3.0 * data.qvel[6 : 6 + config.n_joints]
        data = data.replace(ctrl=tau_fb + action)
        return mjx.step(model, data)

    def _set_inputs_helper(data, command, mjx_contact_id):
        foot_pos = jnp.array(
            [data.geom_xpos[mjx_contact_id[idx]] for idx in range(config.n_contact)]
        ).flatten()
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(data.qpos)
            .at[mpc.qvel_slice].set(data.qvel)
            .at[mpc.foot_slice].set(foot_pos)
        )
        inp = jnp.array([command[0], command[1], 0.0, 0.0, 0.0, command[2], config.robot_height])
        return x0, inp, foot_pos

    step = jax.jit(jax.vmap(_mjx_step, in_axes=(None, 0, 0)))
    set_inputs = jax.jit(jax.vmap(partial(_set_inputs_helper, mjx_contact_id=mjx_contact_id)))

    def set_random_command(n_env, limits, key):
        keys = jax.random.split(key, len(limits))
        cmds = [
            jax.random.uniform(k, (n_env,), minval=low, maxval=high)
            for k, (low, high) in zip(keys, limits)
        ]
        return jnp.stack(cmds, axis=1)

    qpos0 = jnp.tile(jnp.concatenate([config.p0, config.quat0, config.q0]), (n_env, 1))
    batch_data = jax.vmap(lambda x: mjx_data.replace(qpos=x))(qpos0)
    command_limits = [(-0.2, 0.5), (-0.1, 0.1), (-0.3, 0.3)]
    rng_key = jax.random.PRNGKey(0)
    action = jnp.zeros((n_env, config.n_joints))

    # Warm up the batched jitted MPC path before timing.
    rng_key, subkey = jax.random.split(rng_key)
    warm_command = set_random_command(n_env, command_limits, subkey)
    batch_data = step(mjx_model, batch_data, action)
    batch_x0, batch_input, batch_foot = set_inputs(batch_data, warm_command)
    batch_mpc_data = mpc_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, batch_foot)
    batch_mpc_data, tau = solve_mpc(batch_mpc_data, batch_x0, batch_input)
    tau.block_until_ready()

    viewer = None
    temp_data = data
    ids = []
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        for _ in range(n_env):
            mujoco.mj_forward(model, temp_data)
            ids.append(sim_utils.render_ghost_robot(viewer, model, temp_data, alpha=1))
        viewer.sync()

    for _ in range(n_episodes):
        rng_key, subkey = jax.random.split(rng_key)
        batch_command = set_random_command(n_env, command_limits, subkey)
        batch_data = jax.vmap(
            lambda x: mjx_data.replace(
                qpos=x,
                qvel=jnp.zeros(6 + config.n_joints),
                ctrl=jnp.zeros(config.n_joints),
            )
        )(qpos0)
        batch_data = step(mjx_model, batch_data, action)
        _, _, batch_foot = set_inputs(batch_data, batch_command)
        batch_mpc_data = mpc_reset(batch_mpc_data, batch_data.qpos, batch_data.qvel, batch_foot)

        for t in range(int(episode_length * sim_frequency)):
            if t % int(sim_frequency / mpc_frequency) == 0:
                batch_x0, batch_input, _ = set_inputs(batch_data, batch_command)
                start = timer()
                batch_mpc_data, tau = solve_mpc(batch_mpc_data, batch_x0, batch_input)
                tau.block_until_ready()
                stop = timer()
                print(f"Batched MPC time: {1e3 * (stop - start):.2f} ms")
                action = tau

            batch_data = step(mjx_model, batch_data, action)
            if viewer is not None and t % int(sim_frequency / mpc_frequency) == 0:
                batch_robots = batch_data.qpos + offset
                for idx in range(n_env):
                    temp_data.qpos = batch_robots[idx]
                    mujoco.mj_forward(model, temp_data)
                    sim_utils.render_ghost_robot(
                        viewer,
                        model,
                        temp_data,
                        alpha=1,
                        ghost_geoms=ids[idx],
                    )
                viewer.sync()

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--n-env", type=int, default=64)
    parser.add_argument("--episode-length", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dynamics-backend", choices=("mjx", "grid"), default=None)
    args = parser.parse_args()
    main(
        headless=args.headless,
        n_env=args.n_env,
        episode_length=args.episode_length,
        n_episodes=args.episodes,
        dynamics_backend=args.dynamics_backend,
    )
