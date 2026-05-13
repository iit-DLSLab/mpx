import argparse
import os
import sys
import time
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_srbd as config
import mpx.utils.mpc_wrapper_srbd as mpc_wrapper_srbd
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _reset_to_initial_state(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
    else:
        data.qpos = np.asarray(jnp.concatenate([config.p0, config.quat0, config.q0]))
        data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _srbd_state(qpos, qvel):
    return jnp.concatenate(
        [
            jnp.asarray(qpos[:3]),
            jnp.asarray(qpos[3:7]),
            jnp.asarray(qvel[:3]),
            jnp.asarray(qvel[3:6]),
        ]
    )


def main(
    headless=False,
    steps=500,
    scene="flat",
    video=None,
    vx=0.0,
    vy=0.0,
    wz=0.0,
    fps=30,
):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/aliengo/scene_{scene}.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = float(config.whole_body_frequency)
    model.opt.timestep = 1.0 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    command_handle = sim_utils.KeyboardVelocityCommand(vx=vx, vy=vy, wz=wz)
    mpc = mpc_wrapper_srbd.BatchedMPCControllerWrapper(config, n_env=1)

    _reset_to_initial_state(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    x0 = _srbd_state(data.qpos, data.qvel)
    command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

    mpc.run(x0[None, :], command[None, :], foot[None, :], contact[None, :])
    tau_warm, _ = mpc.whole_body_run(
        jnp.asarray(data.qpos)[None, :],
        jnp.asarray(data.qvel)[None, :],
    )
    tau_warm.block_until_ready()
    mpc.reset()

    period = int(sim_frequency / config.mpc_frequency)
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0

    def step_controller():
        nonlocal counter

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
            x0 = _srbd_state(qpos, qvel)

            print(f"Contact: {contact}")
            print(foot)
            print(f"Command: {command}")

            start = timer()
            mpc.run(x0[None, :], command[None, :], foot[None, :], contact[None, :])
            stop = timer()
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        tau_cmd, _ = mpc.whole_body_run(
            jnp.asarray(qpos)[None, :],
            jnp.asarray(qvel)[None, :],
        )
        data.ctrl = np.asarray(tau_cmd[0])
        mujoco.mj_step(model, data)
        counter += 1

    if headless or video is not None:
        recorder = None
        capture_period = max(1, int(round(sim_frequency / fps)))
        if video is not None:
            os.makedirs(os.path.dirname(os.path.abspath(video)) or ".", exist_ok=True)
            recorder = sim_utils.VideoRecorder(model, video, fps=fps)
        p_start = np.asarray(data.qpos[:3]).copy()
        try:
            for i in range(steps):
                step_controller()
                if recorder is not None and i % capture_period == 0:
                    recorder.capture(data)
        finally:
            if recorder is not None:
                recorder.close()
                print(f"Wrote video: {video}")
        p_end = np.asarray(data.qpos[:3])
        delta = p_end - p_start
        print(f"Base position: start={p_start} end={p_end} delta={delta}")
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
                time.sleep(model.opt.timestep - (toc - tic))
            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", type=str, default=None,
                        help="Write an mp4 of the run to this path (forces headless).")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
        video=args.video,
        vx=args.vx,
        vy=args.vy,
        wz=args.wz,
        fps=args.fps,
    )
