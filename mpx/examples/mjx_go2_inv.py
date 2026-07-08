import argparse
import os
import sys
import time
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_go2_inv as config
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, command):
        x0 = (
            mpc.initial_state.at[mpc.qpos_slice]
            .set(qpos)
            .at[mpc.qvel_slice]
            .set(qvel)
        )
        return mpc.run(mpc_data, x0, command)

    return solve_mpc





def main(headless=False, steps=500):
    model = mujoco.MjModel.from_xml_path(dir_path + "/../data/go2/scene_mjx.xml")
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    mpc = config.MPCWrapper(config, limited_memory=False)
    command_handle = sim_utils.KeyboardVelocityCommand()
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)

    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy())

    warm_command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    mpc_data, tau, _, _, _, _,_ = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        warm_command,
    )
    tau.block_until_ready()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy())

    period = int(sim_frequency / config.mpc_frequency)
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    tau = jnp.zeros(config.n_joints)
    q = jnp.zeros(config.n_joints)
    dq = jnp.zeros(config.n_joints)
    kp = 30.0
    kd = 1.5
    trajectory_geom_ids = []
    grf_geom_ids = []
    reference_geom_ids = []
    reference = None

    def step_controller():
        nonlocal counter, tau, mpc_data, q, dq, reference

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % period == 0:
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
           
            start = timer()
            mpc_data, tau, q, dq, alpha_best, any_accepted, reference = solve_mpc(mpc_data, qpos, qvel, command)
            tau.block_until_ready()
            stop = timer()
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")
            print("regularization:", mpc_data.regularization)
            print("alpha_best:", alpha_best)
            print("line_search_accepted:", any_accepted)
        feedback_tau = (
            np.asarray(tau)
            + kp * (np.asarray(q) - data.qpos[7:19])
            + kd * (np.asarray(dq) - data.qvel[6:18])
        )
        data.ctrl = np.clip(feedback_tau, config.min_torque, config.max_torque)
        mujoco.mj_step(model, data)
        counter += 1

    if headless:
        for _ in range(steps):
            step_controller()
        return

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            tic = timer()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            step_controller()
            
            # Render predicted robot states as spheres
            mpc_trajectory = np.asarray(mpc_data.X0[: config.N, :])  # shape (N, state_dim)
            if mpc_trajectory.size > 0:
                positions = mpc_trajectory[:, :3]  # Extract position (x, y, z)
                alphas = np.linspace(1.0, 0.2, len(positions))  # Fade out towards future
                trajectory_geom_ids = sim_utils.render_sphere_trajectory(
                    viewer,
                    positions,
                    alphas,
                    diameter=0.1,
                    color=np.array([0.0, 1.0, 0.0, 1.0]),  # green
                    geom_ids=trajectory_geom_ids if len(trajectory_geom_ids) == len(positions) else None,
                )
            
            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))
            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(headless=args.headless, steps=args.steps)
