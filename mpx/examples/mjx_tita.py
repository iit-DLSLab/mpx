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

import mpx.config.config_tita as config
import mpx.utils.mpc_wrapper_dfcip as mpc_wrapper_dfcip
import mpx.utils.sim as sim_utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _reset_to_initial_state(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _gather_raw_state(model, data, base_body_id, contact_ids):
    mujoco.mj_subtreeVel(model, data)
    pcom = data.subtree_com[base_body_id].copy()
    vcom = data.subtree_linvel[base_body_id].copy()

    centers = data.geom_xpos[contact_ids].copy()               # (2, 3)
    Rs = data.geom_xmat[contact_ids].reshape(2, 3, 3).copy()   # (2, 3, 3)
    radii = model.geom_size[contact_ids, 0].copy()             # (2,)

    feet_vel = np.zeros((2, 3))
    vel = np.zeros(6)
    for i, g in enumerate(contact_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, int(g), vel, 0)
        feet_vel[i] = vel[3:6]
    return pcom, vcom, centers, Rs, radii, feet_vel


def main(headless=False, steps=500, scene="flat"):
    model = mujoco.MjModel.from_xml_path(dir_path + f"/../data/tita/scene_{scene}.xml")
    data = mujoco.MjData(model)
    sim_frequency = float(config.simulation_frequency)
    model.opt.timestep = config.dt_sim

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    command_handle = sim_utils.KeyboardVelocityCommand(
        vx=0.0,
        vy=0.0,
        wz=0.0,
        height=config.com_z_to_track,
        forward_step=0.1,
        yaw_step=0.2,
        height_step=0.01,
        forward_limits=(-10.0, 10.0),
        yaw_limits=(-1.5, 1.5),
        height_limits=(config.h_min, config.h_max),
    )

    mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)

    _reset_to_initial_state(model, data)

    theta_prev = 0.0
    raw = _gather_raw_state(model, data, base_body_id, contact_ids)
    tita_state, x0, theta_prev = mpc.process_state(*raw, theta_prev)
    command = jnp.asarray(command_handle.mpc_wheeled_input())
    mpc_state = mpc.init_state()
    mpc_state, _ = mpc.run(mpc_state, x0[None, :], command[None, :])
    mpc_state, tau, qddot, _, _, _ = mpc.whole_body_run(
        mpc_state,
        x0,
        jnp.asarray(data.qpos)[None, :],
        jnp.asarray(data.qvel)[None, :],
        tita_state[6:9][None, :],
        tita_state[9:12][None, :],
        tita_state[12:15][None, :],
        tita_state[15:18][None, :],
    )
    tau.block_until_ready()
    mpc_state = mpc.reset()

    period = config.mpc_period_steps
    wbc_period = config.wbc_period_steps
    print(f"sim_frequency: {sim_frequency} Hz, "
          f"whole_body_frequency: {config.whole_body_frequency} Hz, "
          f"mpc_frequency: {config.mpc_frequency} Hz")
    counter = 0

    def step_controller(mpc_state, tau, qddot, theta_prev):
        nonlocal counter

        if counter % period == 0 or counter % wbc_period == 0:
            raw = _gather_raw_state(model, data, base_body_id, contact_ids)
            tita_state, x0, theta_prev = mpc.process_state(*raw, theta_prev)

        if counter % period == 0:
            command = jnp.asarray(command_handle.mpc_wheeled_input())
            mpc_state, _ = mpc.run(mpc_state, x0[None, :], command[None, :])

        if counter % wbc_period == 0:
            mpc_state, tau, qddot, _, _, _ = mpc.whole_body_run(
                mpc_state,
                x0,
                jnp.asarray(data.qpos.copy())[None, :],
                jnp.asarray(data.qvel.copy())[None, :],
                tita_state[6:9][None, :],
                tita_state[9:12][None, :],
                tita_state[12:15][None, :],
                tita_state[15:18][None, :],
            )
            data.ctrl = np.asarray(tau[0])

        mujoco.mj_step(model, data)
        counter += 1

        return mpc_state, tau, qddot, theta_prev

    if headless:
        for _ in range(steps):
            mpc_state, tau, qddot, theta_prev = step_controller(mpc_state, tau, qddot, theta_prev)
        return mpc_state

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        viewer.cam.distance *= 5.5
        viewer.sync()
        overlay_text = None
        while viewer.is_running():
            updated_overlay_text = command_handle.consume_overlay_text_dfcip()
            if updated_overlay_text is not None:
                overlay_text = updated_overlay_text
            tic = timer()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            mpc_state, tau, qddot, theta_prev = step_controller(mpc_state, tau, qddot, theta_prev)
            viewer.sync()
            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))


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
