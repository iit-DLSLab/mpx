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

import mpx.config.config_piper as config
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
    model = mujoco.MjModel.from_xml_path(dir_path + "/../data/piper_l/scene_flat.xml")
    data = mujoco.MjData(model)
    data_vis = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    mpc = config.MPCWrapper(config, limited_memory=False)
    command_handle = sim_utils.KeyboardVelocityCommand()
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = np.asarray(config.q0)
    mujoco.mj_forward(model, data)
    mpc_data = mpc.make_data()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy())

    warm_command = jnp.zeros(5)
    mpc_data, tau, _, _, _, _ = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        warm_command,
    )
    tau.block_until_ready()
    # mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy())

    period = int(sim_frequency / config.mpc_frequency)
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    tau = jnp.zeros(config.n_joints)
    q = jnp.asarray(data.qpos[:config.n_joints])
    dq = jnp.asarray(data.qvel[:config.n_joints])
    # kp = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 1.0])
    # kd = np.array([1.0, 1.0, 1.0, 0.4, 0.4, 0.2])
    # q_min = np.array(
    #     [
    #         model.jnt_range[i, 0] if model.jnt_limited[i] else -np.inf
    #         for i in range(6)
    #     ]
    # )
    # q_max = np.array(
    #     [
    #         model.jnt_range[i, 1] if model.jnt_limited[i] else np.inf
    #         for i in range(6)
    #     ]
    # )
    ctrl_min = model.actuator_ctrlrange[:, 0].copy()
    ctrl_max = model.actuator_ctrlrange[:, 1].copy()
    trajectory_geom_ids = []
    contact_geom_ids = []
    contact_force_geom_ids = []

    def step_controller():
        nonlocal counter, tau, mpc_data, q, dq

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % period == 0:
            # command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            command = jnp.zeros(5)
            start = timer()
            mpc_data, tau, q, dq, alpha_best, any_accepted = solve_mpc(mpc_data, qpos, qvel, command)
            tau.block_until_ready()
            stop = timer()
            # time.sleep(0.1)
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")
            print("regularization:", mpc_data.regularization)
            print("alpha_best:", alpha_best)
            print("line_search_accepted:", any_accepted)
            print("tau:", tau)
            print("object position", qpos[config.n_joints : config.n_joints + 3])
        # q_des = np.clip(np.asarray(q[:6]), q_min, q_max)
        # dq_des = np.asarray(dq[:6])
        ctrl = np.asarray(tau)[:config.n_joints]
        data.ctrl = np.clip(ctrl, ctrl_min, ctrl_max)
        mujoco.mj_step(model, data)
        counter += 1
    if counter % 2000 == 0:
        data.qpos[config.n_joints : config.n_joints + 3] = np.random.uniform(low=0.2, high=0.5, size=3)
        # mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy())
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
            mpc_trajectory = np.asarray(mpc_data.X0[: config.N, config.n_joints:config.n_joints + 3])  # shape (N, state_dim)
            if mpc_trajectory.size > 0:
                positions = mpc_trajectory[:, :3]  # Extract position (x, y, z)
                alphas = np.linspace(1.0, 0.2, len(positions))  # Fade out towards future
                trajectory_geom_ids = sim_utils.render_sphere_trajectory(
                    viewer,
                    positions,
                    alphas,
                    diameter=0.01,
                    color=np.array([0.0, 1.0, 0.0, 1.0]),  # green
                    geom_ids=trajectory_geom_ids if len(trajectory_geom_ids) == len(positions) else None,
                )
            geoms_id = sim_utils.geom_ids(model,config.contact_frame)
            contact_positions_list = []
            # contact_forces_list = []
            for i in range(0,config.N,5):
                predicted_qpos = np.asarray(mpc_data.X0[i, : config.nq])
                predicted_qvel = np.asarray(mpc_data.X0[i, config.nq : config.nq + config.nv])
                # body_position = predicted_qpos[config.n_joints : config.n_joints + 3]
                # body_quat = predicted_qpos[config.n_joints + 3 : config.n_joints + 7]
                # body_velocity = predicted_qvel[config.n_joints : config.n_joints + 3]
                data_vis.qpos = predicted_qpos
                data_vis.qvel = predicted_qvel
                mujoco.mj_forward(model, data_vis)
                contact_pos = sim_utils.geom_positions(data_vis,geoms_id,True)
                contact_positions_list.append(contact_pos)
                # for geom_idx, geom_id in enumerate(geoms_id):
                #     jacp = np.zeros((3, model.nv))
                #     jacr = np.zeros((3, model.nv))
                #     mujoco.mj_jacGeom(model, data_vis, jacp, jacr, geom_id)
                #     contact_velocity = jacp @ predicted_qvel
                #     contact_force = np.asarray(
                #         config.contact_force_from_state(
                #             jnp.asarray(contact_pos[geom_idx]),
                #             jnp.asarray(contact_velocity),
                #             jnp.asarray(body_position),
                #             jnp.asarray(body_velocity),
                #         )
                #     )
                #     contact_forces_list.append(contact_force)
            contact_positions = np.stack(contact_positions_list, axis=0)  # shape (N, num_geoms, 3)
            if contact_positions.size > 0:
                contact_geom_ids = sim_utils.render_sphere_trajectory(
                    viewer,
                    contact_positions.reshape(-1, 3),
                    np.tile(np.linspace(1.0, 0.2, config.N), len(geoms_id)),
                    diameter=0.01,
                    color=np.array([1.0, 0.0, 0.0, 1.0]),  # red
                    geom_ids=contact_geom_ids,
                )
            #     flat_contact_positions = contact_positions.reshape(-1, 3)
            #     force_alphas = np.repeat(np.linspace(1.0, 0.2, config.N), len(geoms_id))
            #     if len(contact_force_geom_ids) != len(contact_forces_list):
            #         contact_force_geom_ids = [-1] * len(contact_forces_list)
            #     contact_force_geom_ids = [
            #         sim_utils.render_vector(
            #             viewer,
            #             vector=contact_force,
            #             pos=contact_position,
            #             scale=np.linalg.norm(contact_force)/5.0,
            #             color=np.array([1.0, 0.5, 0.0, alpha]),
            #             geom_id=geom_id,
            #         )
            #         for contact_force, contact_position, alpha, geom_id in zip(
            #             contact_forces_list,
            #             flat_contact_positions,
            #             force_alphas,
            #             contact_force_geom_ids,
            #         )
            #     ]
            
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
