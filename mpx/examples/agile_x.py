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

def sample_reachable_position(rng, model, data, end_effector_id, qpos):
    joint_range = model.jnt_range[: config.n_joints]
    sample_qpos = rng.uniform(
        low=joint_range[:, 0],
        high=joint_range[:, 1],
    )
    data.qpos[:] = sample_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if data.geom_xpos[end_effector_id][2] < 0.4:
        return sample_reachable_position(rng, model, data, end_effector_id, qpos)
    return data.geom_xpos[end_effector_id].copy()

def main():
    model = mujoco.MjModel.from_xml_path(dir_path + "/../data/piper_l/scene_flat.xml")
    data = mujoco.MjData(model)
    target_data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    mpc = config.MPCWrapper(config, limited_memory=False)
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    rng = np.random.default_rng()

    end_effector_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, config.contact_frame[0]
    )
    desired_position = sample_reachable_position(
        rng, model, target_data, end_effector_id, config.q0
    )

    data.qpos = np.asarray(config.q0)
    mujoco.mj_forward(model, data)

    mpc_data = mpc.make_data()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy())
    max_steps_per_target = int(5.0 * sim_frequency)
    period = max(1, round(sim_frequency / config.mpc_frequency))
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    tau = jnp.zeros(config.n_joints)
    ctrl_min = model.actuator_ctrlrange[:, 0].copy()
    ctrl_max = model.actuator_ctrlrange[:, 1].copy()

    target_start_counter = 0
    q = jnp.zeros(config.n_joints)
    dq = jnp.zeros(config.n_joints)
    def step_controller():
        nonlocal counter, tau, mpc_data, desired_position, target_start_counter, q, dq

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % period == 0:
            command = {
                "goal": desired_position,
            }
            start = timer()
            mpc_data, tau, q, dq, _, _ = solve_mpc(mpc_data, qpos, qvel, command)
            tau.block_until_ready()
            stop = timer()
            if counter % 100 == 0:
                print(f"MPC time: {1e3 * (stop - start):.2f} ms")
        ctrl = np.asarray(tau)[:config.n_joints] + 5*(q-qpos) + 0.8*(dq-qvel)
        data.ctrl = np.clip(ctrl, ctrl_min, ctrl_max)
        mujoco.mj_step(model, data)
        counter += 1
        timed_out = counter - target_start_counter >= max_steps_per_target
        if timed_out:
            desired_position = sample_reachable_position(
                rng, model, target_data, end_effector_id, data.qpos
            )
            target_start_counter = counter
            mpc_data = reset_mpc(
                mpc.make_data(), data.qpos.copy(), data.qvel.copy()
            )
            print("New desired position:", desired_position)

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:
        viewer.sync()
        desired_position_geom_id = sim_utils.render_sphere(
                viewer,
                desired_position,
                diameter=0.05,
                color=np.array([0.0, 0.0, 1.0, 0.5]))

        while viewer.is_running():
            tic = timer()
            step_controller()
            # geoms_id = sim_utils.geom_ids(model,config.contact_frame)
            # contact_positions_list = []
            # for i in range(0,config.N,5):
            #     predicted_qpos = np.asarray(mpc_data.X0[i, : config.nq])
            #     data_vis.qpos = predicted_qpos
            #     mujoco.mj_forward(model, data_vis)
            #     contact_pos = sim_utils.geom_positions(data_vis,geoms_id,True)
            #     contact_positions_list.append(contact_pos)
            # contact_positions = np.stack(contact_positions_list, axis=0)  # shape (N, num_geoms, 3)
            # if contact_positions.size > 0:
            #     contact_geom_ids = sim_utils.render_sphere_trajectory(
            #         viewer,
            #         contact_positions.reshape(-1, 3),
            #         np.tile(np.linspace(1.0, 0.2, config.N), len(geoms_id)),
            #         diameter=0.01,
            #         color=np.array([1.0, 0.0, 0.0, 1.0]),  # red
            #         geom_ids=contact_geom_ids,
            #     )
            desired_position_geom_id = sim_utils.render_sphere(
                viewer,
                desired_position,
                diameter=0.05,
                color=np.array([0.0, 0.0, 1.0, 0.5]),  # blue
                geom_id=desired_position_geom_id
            )
            
            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))
            viewer.sync()


if __name__ == "__main__":
    result = main()
