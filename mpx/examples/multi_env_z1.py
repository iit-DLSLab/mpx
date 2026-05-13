import argparse
import importlib
import os
from timeit import default_timer as timer

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp

import mpx.config.config_z1 as config_module


def main(n_env=16, dynamics_backend="mjx", steps=10):
    config = importlib.reload(config_module)
    config.dynamics_backend = dynamics_backend

    mpc = config.MPCWrapper(config, limited_memory=True)
    batch_data = jax.vmap(lambda _: mpc.make_data())(jnp.arange(n_env))
    qpos = jnp.tile(jnp.asarray(config.q0, dtype=jnp.float32), (n_env, 1))
    qvel = jnp.zeros((n_env, config.n_joints), dtype=jnp.float32)
    target = jnp.tile(jnp.asarray(config.ee0, dtype=jnp.float32), (n_env, 1))

    def reset_one(data, qpos_i, qvel_i):
        return mpc.reset(data, qpos_i, qvel_i)

    def solve_one(data, qpos_i, qvel_i, target_i):
        x0 = mpc.state_from_measurement(qpos_i, qvel_i)
        return mpc.run(data, x0, target_i)

    reset = jax.jit(jax.vmap(reset_one))
    solve = jax.jit(jax.vmap(solve_one))
    batch_data = reset(batch_data, qpos, qvel)

    batch_data, tau = solve(batch_data, qpos, qvel, target)
    tau.block_until_ready()

    for _ in range(steps):
        start = timer()
        batch_data, tau = solve(batch_data, qpos, qvel, target)
        tau.block_until_ready()
        print(f"Batched Z1 MPC ({dynamics_backend}, n_env={n_env}): {1e3 * (timer() - start):.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-env", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dynamics-backend", choices=("mjx", "grid"), default="mjx")
    args = parser.parse_args()
    main(n_env=args.n_env, dynamics_backend=args.dynamics_backend, steps=args.steps)
