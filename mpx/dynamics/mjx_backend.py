from __future__ import annotations

import jax

from .base import BoundDynamics


class MJXDynamicsBackend:
    """Adapter that preserves the existing MPX/MJX dynamics behavior."""

    backend_name = "mjx"

    def __init__(self, dynamics_fn):
        self.dynamics_fn = dynamics_fn

    def __call__(self, x, u, t, parameter):
        return self.step(x, u, t, parameter)

    def with_parameter(self, parameter):
        return BoundDynamics(self, parameter)

    def step(self, x, u, t, parameter):
        return self.dynamics_fn(x, u, t, parameter)

    def step_with_derivatives(self, x, u, t, parameter):
        step = lambda x_, u_: self.step(x_, u_, t, parameter)
        x_next = step(x, u)
        A = jax.jacobian(step, argnums=0)(x, u)
        B = jax.jacobian(step, argnums=1)(x, u)
        return x_next, A, B

    def linearize_trajectory(self, X, U, timesteps, parameter):
        def linearize_one(x, u, t):
            _, A, B = self.step_with_derivatives(x, u, t, parameter)
            return A, B

        return jax.vmap(linearize_one)(X, U, timesteps)
