import argparse
import os
import sys
import time
from timeit import default_timer as timer

from mpx.examples.offline_task import _resolve_base_body_id

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_spot_arm as config
# import mpx.utils.mpc_wrapper as mpc_wrapper
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
            .at[config.leg_slice].set(foot)
        )
        return mpc.run(mpc_data, x0, command, contact)

    return solve_mpc



def main(headless=False, steps=500, scene="flat"):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/boston_dynamics_spot/scene_arm.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    mpc = config.MpcWrapper(config, limited_memory=True)
    command_handle = sim_utils.KeyboardVelocityCommand()
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    print(f"Initial foot positions: {foot}")
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    warm_command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    warm_contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids[:4]))
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
    arm_ref_fun = jax.jit(config.extra_ref_fun)
    arm_ref_data = {"foot": jnp.zeros((config.N + 1, 3))}
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
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids[:4]))
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
                contact*0.0,
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
    arm_ref_sphere = []
    arm_pred_sphere = []
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
            arm_ref_data["foot"] = jnp.zeros((config.N + 1, 1))
            arm_ref_data_new = arm_ref_fun(arm_ref_data,counter * model.opt.timestep)
            arm_ref_sphere = sim_utils.render_sphere_trajectory(viewer, arm_ref_data_new["foot"][:,1:],np.ones(config.N+1),0.03,np.array([0.0,1.0,0.0,1.0]),geom_ids = arm_ref_sphere)
            arm_pred_sphere = sim_utils.render_sphere_trajectory(viewer, mpc_data.X0[:,13+2*config.n_joints + 12 :13+2*config.n_joints+15],np.ones(config.N+1),0.03,geom_ids = arm_pred_sphere)
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
