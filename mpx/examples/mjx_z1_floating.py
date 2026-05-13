import argparse
import os
import sys
import time
from functools import partial
from timeit import default_timer as timer

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(DIR_PATH, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_z1_floating as config

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, target_ee):
        x0 = mpc.state_from_measurement(qpos, qvel)
        return mpc.run(mpc_data, x0, target_ee)

    return solve_mpc


def _sample_reference_pose(rng, position_center):
    roll = rng.uniform(-0.4, 0.4)
    pitch = rng.uniform(-0.4, 0.4)
    yaw = rng.uniform(-1.0, 1.0)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    position = position_center + rng.uniform(
        low=np.array([-0.12, -0.12, -0.06], dtype=np.float64),
        high=np.array([0.12, 0.12, 0.06], dtype=np.float64),
    )
    return position, quat


def _init_marker(viewer, idx, rgba, size):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0.0, 0.0], dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )

def extract_mpc_state(qpos, qvel):
    qpos_mpc = np.zeros(7 + config.n_joints)
    qpos_mpc[:7] = qpos[: 7]
    qpos_mpc[7:] = qpos[7 + 12 :7 + 12 + config.n_joints]
    qvel_mpc = np.zeros( 6 + config.n_joints)
    qvel_mpc[:6] = qvel[:6]
    qvel_mpc[6:] = qvel[6 + 12 : 6 + 12 + config.n_joints]
    return qpos_mpc, qvel_mpc

def main(steps=500, dynamics_backend=None, cost_kinematics_backend=None, headless=False):
    if dynamics_backend is not None:
        config.dynamics_backend = dynamics_backend
    if cost_kinematics_backend is not None:
        config.cost_kinematics_backend = cost_kinematics_backend

    scene_path = os.path.join(DIR_PATH, "..", "data", "aliengo_z1", "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    vis_data = mujoco.MjData(model)
    sim_frequency = 200.0
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

    mpc = config.MPCWrapper(config, limited_memory=True)
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)
    # step_dynamics = jax.jit(
    #     partial(
    #         config.dynamics,
    #         mpc.model,
    #         mpc.mjx_model,
    #         mpc.contact_id,
    #         mpc.body_id,
    #         config.n_joints,
    #         config.dt,
    #     )
    # )

    data.qpos = jnp.array([0,0,0.33,1,0,0,0,0.2, 0.8, -1.8, -0.2, 0.8, -1.8, 0.2, 0.8, -1.8, -0.2, 0.8, -1.8,0.0,
    1.68,
    -1.4,
    0.0,
    0.0,
    0.0,])
    qo_aliengo = jnp.array([0.2, 0.8, -1.8, -0.2, 0.8, -1.8, 0.2, 0.8, -1.8, -0.2, 0.8, -1.8])
    # data.qvel = 0.0
    mujoco.mj_forward(model, data)

    reference_center = data.geom_xpos[ee_geom_id].copy()
    rng = np.random.default_rng(0)
    reference_update_steps = max(1, int(sim_frequency * 5.0))
    prediction_indices = list(range(0, config.N + 1, 2))

    target_pos, target_quat = _sample_reference_pose(rng, reference_center)
    data.mocap_pos[reference_mocap_id] = target_pos
    data.mocap_quat[reference_mocap_id] = target_quat
    data.mocap_pos[limited_reference_mocap_id] = target_pos
    data.mocap_quat[limited_reference_mocap_id] = target_quat
    mujoco.mj_forward(model, data)
    mpc_qpos, mpc_qvel = extract_mpc_state(data.qpos.copy(), data.qvel.copy())
    mpc_data = reset_mpc(mpc.make_data(), mpc_qpos, mpc_qvel)
    warm_target = jnp.concatenate(
        [
            jnp.asarray(data.mocap_pos[reference_mocap_id]),
            jnp.asarray(data.mocap_quat[reference_mocap_id]),
        ]
    )
    mpc_data, control = solve_mpc(
        mpc_data,
        mpc_qpos,
        mpc_qvel,
        warm_target,
    )
    control.block_until_ready()
    mpc_data = reset_mpc(mpc_data, mpc_qpos, mpc_qvel)

    period = max(1, int(round(sim_frequency / config.mpc_frequency)))
    counter = 0
    control = jnp.zeros(config.m - 6)
    x_state = mpc.state_from_measurement(data.qpos.copy(), data.qvel.copy())

    def update_markers(viewer):
        viewer.user_scn.ngeom = len(prediction_indices)
        for slot, idx in enumerate(prediction_indices):
            vis_data.qpos[:7] = np.asarray(
                mpc_data.X0[idx, : 7],
                dtype=np.float64,
            )
            vis_data.qpos[19: 19 + config.n_joints] = np.asarray(
                mpc_data.X0[idx, 7 : 7 + config.n_joints],
                dtype=np.float64)
            vis_data.qvel[:] = 0.0
            mujoco.mj_forward(model, vis_data)
            _init_marker(viewer, slot, [0.9, 0.9, 0.9, 0.45], 0.01)
            print(f"Prediction {slot}: {vis_data.geom_xpos[ee_geom_id]}")
            viewer.user_scn.geoms[slot].pos = vis_data.geom_xpos[ee_geom_id]

    def step_controller():
        nonlocal counter, control, x_state, mpc_data
        mpc_qpos, mpc_qvel = extract_mpc_state(data.qpos.copy(), data.qvel.copy())
        if counter % reference_update_steps == 0:
            
            target_pos, target_quat = _sample_reference_pose(rng, reference_center)
            data.mocap_pos[reference_mocap_id] = target_pos
            data.mocap_quat[reference_mocap_id] = target_quat
            mujoco.mj_forward(model, data)
            mpc_data = reset_mpc(mpc_data, mpc_qpos, mpc_qvel)

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
            mpc_data, control = solve_mpc(
                mpc_data,
                mpc_qpos,
                mpc_qvel,
                jnp.asarray(target_ee),
            )
            control.block_until_ready()
            stop = timer()
            # control =jnp.clip(control_tot[: config.n_joints], config.min_torque, config.max_torque)
            ee_error = np.linalg.norm(current_ee[:3] - limited_target[:3])
            print(f"MPC time: {1e3 * (stop - start):.2f} ms | ee_err={ee_error:.3f}")
        
        data.ctrl[:12] = 60*(qo_aliengo - data.qpos[7:19]) - 5.0*data.qvel[6:18]
        data.ctrl[12:] = np.asarray(control[: config.n_joints], dtype=np.float64)
        mujoco.mj_step(model, data)
        counter += 1

    if headless:
        for _ in range(steps):
            step_controller()
        print(
            f"Headless Z1 floating MPC ({config.dynamics_backend}, "
            f"cost={config.cost_kinematics_backend}) completed {steps} steps"
        )
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        update_markers(viewer)
        viewer.sync()
        while viewer.is_running() and counter < steps:
            wall_start = time.perf_counter()
            step_controller()
            if counter % 2 == 0:
                update_markers(viewer)
            viewer.sync()
            elapsed = time.perf_counter() - wall_start
            time.sleep(max(0.0, model.opt.timestep - elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--dynamics-backend", choices=("mjx", "grid"), default=None)
    parser.add_argument(
        "--cost-kinematics-backend",
        choices=("mjx", "grid"),
        default=None,
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(
        steps=args.steps,
        dynamics_backend=args.dynamics_backend,
        cost_kinematics_backend=args.cost_kinematics_backend,
        headless=args.headless,
    )
