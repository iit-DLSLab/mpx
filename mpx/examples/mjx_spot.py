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

import mpx.config.config_spot as config
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact):
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(qpos)
            .at[mpc.qvel_slice].set(qvel)
            .at[mpc.foot_slice].set(foot)
        )
        return mpc.run(mpc_data, x0, command, contact)

    return solve_mpc


def main(headless=False, steps=500, scene="flat"):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/boston_dynamics_spot/scene.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    mpc = mpc_wrapper.MPCWrapper(config, limited_memory=True)
    command_handle = sim_utils.KeyboardVelocityCommand()
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    warm_command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    warm_contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        foot,
        warm_command,
        warm_contact,
    )
    tau.block_until_ready()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)

    period = int(sim_frequency / config.mpc_frequency)
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    tau = jnp.zeros(config.n_joints)
    q_ref = config.q0.copy()

    def step_controller():
        nonlocal counter, tau, q_ref, mpc_data

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        
        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
           
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
            print(f"Contact: {contact}")
            print(foot)
            print(f"Command: {command}")
            
            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data,
                qpos,
                qvel,
                foot,
                command,
                contact,
            )
            tau.block_until_ready()
            stop = timer()

            # tau = jnp.clip(tau, config.min_torque, config.max_torque)
            # The shifted warm start is the next joint target used by the PD stabilizer.
            q_ref = mpc_data.X0[0, 7 : 7 + config.n_joints]
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = np.asarray(tau)
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
            toc = timer()
            if toc - tic < model.opt.timestep:
                sleep_time = model.opt.timestep - (toc - tic)
                time.sleep(sleep_time)

            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
    )
