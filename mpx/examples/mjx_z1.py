import argparse
import os
import sys
import time
import types
from timeit import default_timer as timer

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(DIR_PATH, "..", "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")


def _disable_mjx_warp_import():
    """Force MJX onto its JAX path when an installed Warp package is incompatible."""
    warp_pkg = types.ModuleType("mujoco.mjx.warp")
    warp_types = types.ModuleType("mujoco.mjx.warp.types")

    for name in (
        "TileSet",
        "BlockDim",
        "StatisticWarp",
        "OptionWarp",
        "ModelWarp",
        "DataWarp",
    ):
        setattr(warp_types, name, type(name, (), {}))

    warp_types.GraphMode = int
    warp_types.DATA_NON_VMAP = ()
    warp_types._BATCH_DIM = "_batch"

    warp_pkg.WARP_INSTALLED = False
    warp_pkg.types = warp_types
    warp_pkg.mjwp_types = types.SimpleNamespace(
        TileSet=warp_types.TileSet,
        BlockDim=warp_types.BlockDim,
        Callback=type("Callback", (), {}),
    )
    warp_pkg.mujoco_warp = None
    warp_pkg.warp = types.SimpleNamespace(
        types=types.SimpleNamespace(warp_type_to_np_dtype={}),
        array=(),
    )

    sys.modules.setdefault("mujoco.mjx.warp", warp_pkg)
    sys.modules.setdefault("mujoco.mjx.warp.types", warp_types)


_disable_mjx_warp_import()

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_z1 as config

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, target_ee):
        x0 = mpc.state_from_measurement(qpos, qvel)
        return mpc.run(mpc_data, x0, target_ee)

    return solve_mpc


def _init_marker(viewer, idx, rgba, size):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0.0, 0.0], dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )


def main(headless=False, steps=500):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/unitree_z1/scene_z1.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    vis_data = mujoco.MjData(model)
    sim_frequency = 500.0
    model.opt.timestep = 1.0 / sim_frequency

    ee_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config.contact_frame[0])
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.body_name[0])
    reference_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference")
    limited_reference_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "limited_reference",
    )
    reference_mocap_id = model.body_mocapid[reference_body_id]
    limited_reference_mocap_id = model.body_mocapid[limited_reference_body_id]

    mpc = config.MPCWrapper(config, limited_memory=False)
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos[:] = np.asarray(config.q0, dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    reference_pos = data.geom_xpos[ee_geom_id].copy()
    reference_quat = data.xquat[ee_body_id].copy()
    data.mocap_pos[reference_mocap_id] = reference_pos
    data.mocap_quat[reference_mocap_id] = reference_quat
    data.mocap_pos[limited_reference_mocap_id] = reference_pos
    data.mocap_quat[limited_reference_mocap_id] = reference_quat
    mujoco.mj_forward(model, data)

    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy())
    warm_target = jnp.concatenate(
        [
            jnp.asarray(data.mocap_pos[reference_mocap_id]),
            jnp.asarray(data.mocap_quat[reference_mocap_id]),
        ]
    )
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        warm_target,
    )
    tau.block_until_ready()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy())

    period = max(1, int(round(sim_frequency / config.mpc_frequency)))
    prediction_indices = list(range(0, config.N + 1, 3))
    counter = 0
    tau = jnp.zeros(config.n_joints)

    def update_markers(viewer, target_ee):
        viewer.user_scn.ngeom = 1 + len(prediction_indices)
        _init_marker(viewer, 0, [1.0, 0.2, 0.2, 0.9], 0.02)
        viewer.user_scn.geoms[0].pos = np.asarray(target_ee[:3], dtype=np.float64)

        for slot, idx in enumerate(prediction_indices, start=1):
            vis_data.qpos[:] = np.asarray(mpc_data.X0[idx, : config.n_joints], dtype=np.float64)
            vis_data.qvel[:] = 0.0
            mujoco.mj_forward(model, vis_data)
            _init_marker(viewer, slot, [0.8, 0.8, 0.8, 0.5], 0.01)
            viewer.user_scn.geoms[slot].pos = vis_data.geom_xpos[ee_geom_id]

    def step_controller(viewer=None):
        nonlocal counter, tau, mpc_data

        target_ee = np.concatenate(
            [
                data.mocap_pos[reference_mocap_id].copy(),
                data.mocap_quat[reference_mocap_id].copy(),
            ]
        )
        current_ee = np.concatenate(
            [
                data.geom_xpos[ee_geom_id].copy(),
                data.xquat[ee_body_id].copy(),
            ]
        )
        limited_target = np.asarray(
            mpc.limit_reference(jnp.asarray(target_ee), jnp.asarray(current_ee)),
            dtype=np.float64,
        )
        data.mocap_pos[limited_reference_mocap_id] = limited_target[:3]
        data.mocap_quat[limited_reference_mocap_id] = limited_target[3:]

        if counter % period == 0:
            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data,
                data.qpos.copy(),
                data.qvel.copy(),
                jnp.asarray(limited_target),
            )
            tau.block_until_ready()
            stop = timer()
            tau = jnp.clip(tau, config.min_torque, config.max_torque)
            ee_error = np.linalg.norm(current_ee[:3] - limited_target[:3])
            print(f"MPC time: {1e3 * (stop - start):.2f} ms | ee_err={ee_error:.3f}")

        data.ctrl[:] = np.asarray(tau, dtype=np.float64)
        mujoco.mj_step(model, data)
        counter += 1

        if viewer is not None and counter % 5 == 0:
            update_markers(viewer, target_ee)

    if headless:
        for _ in range(steps):
            step_controller()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        initial_target = np.concatenate(
            [
                data.mocap_pos[limited_reference_mocap_id].copy(),
                data.mocap_quat[limited_reference_mocap_id].copy(),
            ]
        )
        update_markers(viewer, initial_target)
        viewer.sync()
        while viewer.is_running():
            wall_start = time.perf_counter()
            step_controller(viewer)
            viewer.sync()
            elapsed = time.perf_counter() - wall_start
            time.sleep(max(0.0, model.opt.timestep - elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(headless=args.headless, steps=args.steps)
