import argparse
import importlib
import os
import sys
import time
from types import SimpleNamespace

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

# Headless video recording uses `mujoco.Renderer`, which requires an OpenGL
# backend to be configured before the first `import mujoco` in the process.
if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.utils.mpc_wrapper as base_mpc_wrapper
import mpx.utils.offline_solver as offline_solver
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


TASKS = {
    "barrel_roll": {
        "config": "mpx.config.config_barrel_roll",
        "scene_path": os.path.abspath(
            os.path.join(dir_path, "..", "data", "aliengo", "scene_flat.xml")
        ),
        "benchmark_mode": "wrapper",
    },
    "h1_jump_forward": {
        "config": "mpx.config.config_h1_jump_forward",
        "scene_path": os.path.abspath(
            os.path.join(dir_path, "..", "data", "unitree_h1", "mjx_scene_h1_walk.xml")
        ),
        "benchmark_mode": "wrapper",
    },
    "aliengo_trot_two_step": {
        "config": "mpx.config.config_aliengo_trot_two_step",
        "scene_path": os.path.abspath(
            os.path.join(dir_path, "..", "data", "aliengo", "scene_flat.xml")
        ),
        "benchmark_mode": "wrapper",
    },
    "acrobot_swingup": {
        "config": "mpx.config.config_acrobot_swingup",
        "benchmark_mode": "direct",
    },
}

SOLVERS = ("primal_dual", "fddp", "lipa")


def _clone_config(module_name, solver_mode, lipa_enforce_inequalities=None):
    module = importlib.import_module(module_name)
    attrs = {name: getattr(module, name) for name in dir(module) if not name.startswith("__")}
    config = SimpleNamespace(**attrs)
    if solver_mode is not None:
        config.solver_mode = solver_mode
    if lipa_enforce_inequalities is not None:
        config.lipa_enforce_inequalities = lipa_enforce_inequalities
    return config


def _solve_wrapper_task(config, max_iter, verbose):
    wrapper_cls = getattr(config, "MPCWrapper", base_mpc_wrapper.MPCWrapper)
    mpc = wrapper_cls(config, limited_memory=True)
    qpos0 = getattr(config, "offline_qpos0", jnp.concatenate([config.p0, config.quat0, config.q0]))
    qvel0 = getattr(config, "offline_qvel0", jnp.zeros(6 + config.n_joints))
    X, U, reference, history, stats = mpc.runOffline(
        qpos0,
        qvel0,
        return_stats=True,
        verbose=verbose,
        max_iter=max_iter,
    )
    return {
        "config": config,
        "X": X,
        "U": U,
        "reference": reference,
        "history": history,
        "stats": stats,
        "initial_state": {"qpos0": qpos0, "qvel0": qvel0},
    }


def _solve_direct_task(config, max_iter, verbose):
    if getattr(config, "solver_mode", None) == "lipa":
        from mpx.utils.lipa_solver import run_lipa_offline
        from mpx.utils.mpc_wrapper import lipa_pick_cost_and_inequalities

        (
            lipa_cost,
            lipa_inequalities,
            lipa_settings,
            lipa_warmup_cost,
            lipa_warmup_settings,
        ) = lipa_pick_cost_and_inequalities(config, config.cost)
        X, U, V, history, stats = run_lipa_offline(
            lipa_cost,
            config.dynamics,
            config.reference,
            config.parameter,
            config.W,
            config.x0,
            config.initial_X0,
            config.initial_U0,
            config.initial_V0,
            settings=lipa_settings,
            inequalities=lipa_inequalities,
            warmup_cost=lipa_warmup_cost,
            warmup_settings=lipa_warmup_settings,
            verbose=verbose,
        )
    else:
        _, solve = base_mpc_wrapper.build_solver_step(
            config,
            config.cost,
            config.dynamics,
            config.hessian_approx,
            limited_memory=False,
        )
        solve = jax.jit(solve)
        X, U, V, history, stats = offline_solver.run_offline_solve(
            solve,
            config.cost,
            config.dynamics,
            config.solver_mode,
            config.reference,
            config.parameter,
            config.W,
            config.x0,
            config.initial_X0,
            config.initial_U0,
            config.initial_V0,
            max_iter=max_iter,
            verbose=verbose,
        )
    return {
        "config": config,
        "X": X,
        "U": U,
        "V": V,
        "reference": config.reference,
        "history": history,
        "stats": stats,
        "initial_state": {"x0": config.x0},
    }


def solve_task(
    task_name, solver_mode=None, max_iter=100, verbose=True, lipa_enforce_inequalities=None
):
    task = TASKS[task_name]
    config = _clone_config(task["config"], solver_mode, lipa_enforce_inequalities)
    benchmark_mode = task["benchmark_mode"]
    if benchmark_mode == "direct":
        result = _solve_direct_task(config, max_iter=max_iter, verbose=verbose)
    else:
        result = _solve_wrapper_task(config, max_iter=max_iter, verbose=verbose)
    result["task_name"] = task_name
    result["benchmark_mode"] = benchmark_mode
    result["scene_path"] = getattr(config, "scene_path", task.get("scene_path"))
    return result


def _state_to_mujoco(config, state):
    state = jnp.asarray(state)
    if hasattr(config, "state_to_qpos"):
        return (
            np.asarray(config.state_to_qpos(state)),
            np.asarray(config.state_to_qvel(state)),
        )

    qpos_dim = 7 + config.n_joints
    qvel_start = qpos_dim
    qvel_stop = qvel_start + 6 + config.n_joints
    return (
        np.asarray(state[:qpos_dim]),
        np.asarray(state[qvel_start:qvel_stop]),
    )


def _resolve_base_body_id(config, model):
    body_name = getattr(config, "body_name", None)
    if isinstance(body_name, str):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return int(body_id)

    if model.nbody > 1:
        return 1
    return 0


def _predicted_base_positions(config, model, qpos_sequence):
    qpos_sequence = np.asarray(qpos_sequence)
    if qpos_sequence.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Floating-base tasks encode the base world position directly in qpos[:3].
    if qpos_sequence.shape[1] >= 7:
        return np.asarray(qpos_sequence[:, :3], dtype=np.float64)

    base_body_id = _resolve_base_body_id(config, model)
    scratch_data = mujoco.MjData(model)
    base_positions = np.zeros((qpos_sequence.shape[0], 3), dtype=np.float64)
    for idx, qpos in enumerate(qpos_sequence):
        scratch_data.qpos = qpos
        mujoco.mj_forward(model, scratch_data)
        base_positions[idx] = np.asarray(scratch_data.xpos[base_body_id], dtype=np.float64)
    return base_positions


def _record_offline_trajectory_video(result, video_path, fps=None, loop_count=1, width=1280, height=720):
    """Render the optimised offline trajectory `X` to an mp4 via offscreen GL.

    No viewer involved. Uses `MUJOCO_GL=egl` (auto-set when the example sees
    `--video` in argv) and the shared `sim_utils.VideoRecorder`. One frame per
    state in `X`, optionally repeated `loop_count` times so short trajectories
    aren't blink-and-miss-it. Playback is real-time by default (fps = 1/config.dt).
    """
    config = result["config"]
    scene_path = result["scene_path"]
    X = np.asarray(result["X"])
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    os.makedirs(os.path.dirname(os.path.abspath(video_path)) or ".", exist_ok=True)
    
    config_fps = 1.0 / float(config.dt)
    if fps is None:
        fps = int(round(config_fps))
        subsample = 1
    else:
        subsample = max(1, int(round(config_fps / fps)))
        fps = int(round(config_fps / subsample))

    recorder = sim_utils.VideoRecorder(model, video_path, fps=fps, width=width, height=height)
    try:
        for _ in range(loop_count):
            for i, state in enumerate(X):
                if i % subsample != 0:
                    continue
                qpos, qvel = _state_to_mujoco(config, state)
                data.qpos = np.asarray(qpos)
                data.qvel = np.asarray(qvel)
                mujoco.mj_forward(model, data)
                recorder.capture(data)
    finally:
        recorder.close()
        print(f"Wrote video: {video_path} at {fps} fps (subsample {subsample}), {width}x{height}")


def _play_mujoco_trajectory(result, headless=False, loop=True, ghost_stride=1, video=None, fps=None, width=None, height=None):
    if video is not None:
        # Recording mode: render trajectory frames to mp4 instead of opening
        # the viewer. Implies headless.
        _record_offline_trajectory_video(result, video, fps=fps, width=width or 1280, height=height or 720)
        return

    config = result["config"]
    scene_path = result["scene_path"]
    X = np.asarray(result["X"])
    history = result["history"]

    if headless:
        print(
            "Offline solve shapes:",
            X.shape,
            np.asarray(result["U"]).shape,
            np.asarray(result["reference"]).shape,
            len(history),
        )
        print(
            f"Final objective {result['stats']['final_objective']:.6f} | "
            f"Final dynamics violation {result['stats']['final_dynamics_violation']:.6f}"
        )
        return

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    ghost_geoms = None
    ghost_scratch = mujoco.MjData(model)
    base_marker_ids = None
    plan_states = np.asarray(history[-1][::ghost_stride])
    qpos_sequence = np.asarray(
        [_state_to_mujoco(config, state)[0] for state in plan_states]
    )
    if qpos_sequence.shape[0] == 0:
        qpos_sequence = np.asarray([_state_to_mujoco(config, X[0])[0]])
    display_qpos_sequence = qpos_sequence
    display_alphas = np.ones(display_qpos_sequence.shape[0], dtype=np.float64)
    base_positions = _predicted_base_positions(config, model, display_qpos_sequence)
    base_marker_diameter = float(np.clip(0.03 * model.stat.extent, 0.01, 0.06))

    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            idx = frame % X.shape[0] if loop else min(frame, X.shape[0] - 1)
            qpos, qvel = _state_to_mujoco(config, X[idx])
            data.qpos = qpos
            data.qvel = qvel
            mujoco.mj_forward(model, data)
            ghost_geoms, ghost_scratch = sim_utils.render_ghost_trajectory(
                viewer,
                model,
                display_qpos_sequence,
                display_alphas*0.1,
                ghost_geoms=ghost_geoms,
                scratch_data=ghost_scratch,
                subsample=15,
            )
            base_marker_ids = sim_utils.render_sphere_trajectory(
                viewer,
                base_positions,
                display_alphas,
                diameter=base_marker_diameter,
                color=np.array([1.0, 0.45, 0.0, 1.0]),
                geom_ids=base_marker_ids,
            )
            viewer.sync()
            frame += 1
            time.sleep(config.dt)


def run_task(
    task_name,
    solver_mode=None,
    headless=False,
    max_iter=100,
    verbose=True,
    loop=True,
    lipa_enforce_inequalities=None,
    video=None,
    fps=None,
    width=None,
    height=None,
):
    result = solve_task(
        task_name,
        solver_mode=solver_mode,
        max_iter=max_iter,
        verbose=verbose,
        lipa_enforce_inequalities=lipa_enforce_inequalities,
    )
    stats = result["stats"]
    enforce = getattr(result["config"], "lipa_enforce_inequalities", False)
    enforce_tag = " | enforce-ineq" if (result["config"].solver_mode == "lipa" and enforce) else ""
    print(
        f"{task_name} | {result['config'].solver_mode}{enforce_tag} | "
        f"iterations {stats['n_iterations']} | "
        f"avg iter time {stats['average_iteration_time_ms']:.3f} ms"
    )
    _play_mujoco_trajectory(result, headless=headless, loop=loop, video=video, fps=fps, width=width, height=height)


def build_parser(default_task=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=tuple(TASKS.keys()),
        default=default_task,
    )
    parser.add_argument(
        "--solver",
        choices=SOLVERS,
        default=None,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--video", type=str, default=None,
                        help="Render the optimised trajectory to this mp4 path (forces headless).")
    parser.add_argument("--fps", type=int, default=None,
                        help="Target FPS for video recording (e.g. 50 or 60). Subsamples the trajectory.")
    parser.add_argument("--width", type=int, default=1280, help="Video width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Video height (default: 720).")
    enforce_group = parser.add_mutually_exclusive_group()
    enforce_group.add_argument(
        "--lipa-enforce-inequalities",
        dest="lipa_enforce_inequalities",
        action="store_true",
        default=None,
        help="(LIPA only) Enforce config inequalities as true constraints; overrides config attr.",
    )
    enforce_group.add_argument(
        "--no-lipa-enforce-inequalities",
        dest="lipa_enforce_inequalities",
        action="store_false",
        default=None,
        help="(LIPA only) Disable enforcement; revert to soft-penalty cost shared with FDDP/PD.",
    )
    return parser


def main(default_task=None):
    parser = build_parser(default_task=default_task)
    args = parser.parse_args()
    if args.task is None:
        parser.error("--task is required")
    run_task(
        args.task,
        solver_mode=args.solver,
        headless=args.headless,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        loop=not args.no_loop,
        lipa_enforce_inequalities=args.lipa_enforce_inequalities,
        video=args.video,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
