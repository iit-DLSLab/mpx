"""Sequential Piper box-pushing example using full-dynamics ComFree MPC."""

import argparse
import os
from timeit import default_timer as timer

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import mpx.config.config_piper_comfree as config
import mpx.utils.sim as sim_utils

GOAL_MIN_RADIUS = 0.2
GOAL_MAX_RADIUS = 0.6
GOAL_UPDATE_SECONDS = 10.0


def _build_controller(mpc):
    @jax.jit
    def controller(mpc_data, qpos, qvel, target):
        state = (
            mpc.initial_state.at[mpc.qpos_slice]
            .set(qpos)
            .at[mpc.qvel_slice]
            .set(qvel)
        )
        command = jnp.zeros(5, dtype=state.dtype).at[:2].set(target)
        return mpc.run(mpc_data, state, command)

    return controller


def _sample_goal(rng, base_xy, min_radius=GOAL_MIN_RADIUS, max_radius=GOAL_MAX_RADIUS):
    theta = rng.uniform(-np.pi, np.pi)
    radius = rng.uniform(min_radius, max_radius)
    offset = radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    return np.asarray(base_xy, dtype=np.float32) + offset


def run(steps=12000, target_count=0, tolerance=0.02, show_viewer=False):
    if target_count < 0:
        raise ValueError("target_count must be >= 0")

    model = mujoco.MjModel.from_xml_path(config.model_path)
    model.opt.timestep = 0.005
    floor_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    box_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "sphere"
    )
    ee_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, config.contact_frame[0]
    )
    physical_geom = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    robot_geom = physical_geom.copy()
    robot_geom[[floor_geom_id, box_geom_id, ee_geom_id]] = False
    model.geom_contype[robot_geom] = 2
    model.geom_conaffinity[robot_geom] = 1
    model.geom_contype[floor_geom_id] = 1
    model.geom_conaffinity[floor_geom_id] = 2 | 4 | 8
    model.geom_contype[box_geom_id] = 4
    model.geom_conaffinity[box_geom_id] = 1 | 8
    model.geom_contype[ee_geom_id] = 8
    model.geom_conaffinity[ee_geom_id] = 1 | 4
    simulation_data = mujoco.MjData(model)
    visualization_data = mujoco.MjData(model)
    simulation_data.qpos[:] = np.asarray(config.q0)
    mujoco.mj_forward(model, simulation_data)

    mpc = config.MPCWrapper(config, limited_memory=False)
    mpc_data = mpc.reset(
        mpc.make_data(), simulation_data.qpos, simulation_data.qvel
    )
    controller_dtype = np.asarray(config.initial_state).dtype
    controller = _build_controller(mpc)

    control_period = max(1, round(1.0 / (config.mpc_frequency * model.opt.timestep)))
    goal_update_steps = max(1, round(GOAL_UPDATE_SECONDS / model.opt.timestep))
    base_xy = np.zeros(2, dtype=np.float32)
    for body_name in ("base_link", "base", "link0"):
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if base_body_id >= 0:
            base_xy = np.asarray(model.body_pos[base_body_id, :2], dtype=np.float32)
            break
    rng = np.random.default_rng()
    target = _sample_goal(rng, base_xy)
    target_start_step = 0
    results = []
    torque = np.zeros(config.n_joints)
    rejected_updates = 0

    initial_qpos = np.asarray(simulation_data.qpos, dtype=controller_dtype)
    initial_qvel = np.asarray(simulation_data.qvel, dtype=controller_dtype)
    initial_target = np.asarray(target, dtype=controller_dtype)
    compile_start = timer()
    controller = controller.lower(
        mpc_data, initial_qpos, initial_qvel, initial_target
    ).compile()
    compile_elapsed = timer() - compile_start

    # Execute once to initialize the GPU without advancing the retained warm start.
    compiled_data, compiled_torque, *_ = controller(
        mpc_data,
        initial_qpos,
        initial_qvel,
        initial_target,
    )
    compiled_torque.block_until_ready()
    del compiled_data
    verification_data = mpc.reset(
        mpc.make_data(), initial_qpos, initial_qvel
    )
    verification_target = initial_target + np.array(
        [0.0, 0.01], dtype=controller_dtype
    )
    _, verification_torque, *_ = controller(
        verification_data,
        initial_qpos,
        initial_qvel,
        verification_target,
    )
    verification_torque.block_until_ready()
    print(
        f"Compiled one {jax.default_backend()} ComFree MPC executable in "
        f"{compile_elapsed:.2f} s; verified reuse after reset and target change"
    )

    def simulation_step(step):
        nonlocal mpc_data, torque, target, target_start_step
        nonlocal rejected_updates
        if step % control_period == 0:
            start = timer()
            mpc_data, next_torque, _, _, alpha, accepted = controller(
                mpc_data,
                np.asarray(simulation_data.qpos, dtype=controller_dtype),
                np.asarray(simulation_data.qvel, dtype=controller_dtype),
                np.asarray(target, dtype=controller_dtype),
            )
            next_torque.block_until_ready()
            torque = np.asarray(next_torque)
            rejected_updates = 0 if bool(accepted) else rejected_updates + 1
            if step % 200 == 0:
                elapsed_ms = 1e3 * (timer() - start)
                position = simulation_data.qpos[config.n_joints : config.n_joints + 2]
                print(f"MPC time: {elapsed_ms:.2f} ms")
                print("regularization:", mpc_data.regularization)
                print("alpha_best:", alpha)
                print("line_search_accepted:", accepted)
                print("tau:", next_torque)
                print(
                    "end-effector position:",
                    simulation_data.geom_xpos[ee_geom_id],
                )
                print("object position:", position)
                print("desired object position:", target)

        simulation_data.ctrl[:] = np.clip(
            torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, simulation_data)

        box_xy = simulation_data.qpos[config.n_joints : config.n_joints + 2].copy()
        error = float(np.linalg.norm(box_xy - target))
        box_speed = float(
            np.linalg.norm(
                simulation_data.qvel[
                    config.n_joints : config.n_joints + 2
                ]
            )
        )
        reached = error <= tolerance and box_speed <= 0.02
        goal_update_due = step + 1 - target_start_step >= goal_update_steps
        if not goal_update_due:
            return False

        results.append(
            {
                "target": target.copy(),
                "position": box_xy,
                "error": error,
                "speed": box_speed,
                "reached": reached,
            }
        )
        print("Target result:", results[-1])
        if target_count > 0 and len(results) == target_count:
            return True

        target = _sample_goal(rng, base_xy)
        print("Sampled next goal:", target)
        target_start_step = step + 1
        # A changed goal is a new OCP.  Discarding the previous goal's local
        # trajectory avoids presenting FDDP with a deliberately stale nominal.
        mpc_data = mpc.reset(
            mpc.make_data(), simulation_data.qpos, simulation_data.qvel
        )
        rejected_updates = 0
        return False

    if show_viewer:
        with mujoco.viewer.launch_passive(model, simulation_data) as viewer:
            box_trajectory_geom_ids = None
            ee_trajectory_geom_ids = None
            desired_position_geom_id = -1
            prediction_alphas = np.linspace(1.0, 0.2, config.N + 1)
            desired_height = float(model.geom_size[config.box_geom_id, 2])
            for step in range(steps):
                if not viewer.is_running():
                    break
                complete = simulation_step(step)

                if step % control_period == 0:
                    predicted_states = np.asarray(mpc_data.X0)
                    box_positions = predicted_states[
                        :, config.box_qpos_adr : config.box_qpos_adr + 3
                    ]
                    ee_positions = []
                    for predicted_state in predicted_states:
                        visualization_data.qpos[:] = predicted_state[: config.nq]
                        visualization_data.qvel[:] = predicted_state[
                            config.nq : config.nq + config.nv
                        ]
                        mujoco.mj_forward(model, visualization_data)
                        ee_positions.append(
                            visualization_data.geom_xpos[config.ee_geom_id].copy()
                        )
                    ee_positions = np.asarray(ee_positions)

                    box_trajectory_geom_ids = sim_utils.render_sphere_trajectory(
                        viewer,
                        box_positions,
                        prediction_alphas,
                        diameter=0.012,
                        color=np.array([0.1, 0.9, 0.25, 1.0]),
                        geom_ids=box_trajectory_geom_ids,
                    )
                    ee_trajectory_geom_ids = sim_utils.render_sphere_trajectory(
                        viewer,
                        ee_positions,
                        prediction_alphas,
                        diameter=0.01,
                        color=np.array([1.0, 0.25, 0.05, 1.0]),
                        geom_ids=ee_trajectory_geom_ids,
                    )

                desired_position_geom_id = sim_utils.render_sphere(
                    viewer,
                    np.array([target[0], target[1], desired_height]),
                    diameter=0.05,
                    color=np.array([0.1, 0.35, 1.0, 0.55]),
                    geom_id=desired_position_geom_id,
                )
                viewer.sync()
                if complete:
                    break
    else:
        for step in range(steps):
            if simulation_step(step):
                break

    final_position = simulation_data.qpos[
        config.n_joints : config.n_joints + 3
    ].copy()
    result = {
        "final_object_position": final_position,
        "targets": results,
        "passed": (
            len(results) == target_count if target_count > 0 else len(results) > 0
        ) and all(item["reached"] for item in results),
    }
    print("ComFree result:", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--target-count", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    result = run(
        steps=args.steps,
        target_count=args.target_count,
        tolerance=args.tolerance,
        show_viewer=args.viewer,
    )
    if not args.viewer and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
