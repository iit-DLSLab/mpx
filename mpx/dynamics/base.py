from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import jax
import jax.numpy as jnp
from mujoco.mjx._src import math


@dataclass(frozen=True)
class StateLayout:
    """Slices for MPX whole-body states."""

    q: slice
    v: slice
    contact_positions: slice | None = None
    aux: slice | None = None

    @property
    def nx(self) -> int:
        candidates = [self.q.stop, self.v.stop]
        if self.contact_positions is not None:
            candidates.append(self.contact_positions.stop)
        if self.aux is not None:
            candidates.append(self.aux.stop)
        return max(candidates)


@dataclass(frozen=True)
class ControlLayout:
    """Slices for MPX whole-body controls."""

    actuator_torques: slice
    contact_forces: slice | None = None

    @property
    def nu_total(self) -> int:
        candidates = [self.actuator_torques.stop]
        if self.contact_forces is not None:
            candidates.append(self.contact_forces.stop)
        return max(candidates)

    @property
    def n_contact_forces(self) -> int:
        if self.contact_forces is None:
            return 0
        return self.contact_forces.stop - self.contact_forces.start


@dataclass(frozen=True)
class RobotSpec:
    """Robot-specific metadata used by dynamics backends."""

    name: str
    urdf_path: str | None
    nq: int
    nv: int
    nu: int
    floating_base: bool
    actuated_dofs: tuple[int, ...]
    contact_frames: tuple[str, ...] = field(default_factory=tuple)
    contact_body_names: tuple[str, ...] = field(default_factory=tuple)
    dt: float = 0.0
    state_layout: StateLayout | None = None
    control_layout: ControlLayout | None = None

    @property
    def n_contact(self) -> int:
        return len(self.contact_frames)

    @property
    def nx(self) -> int:
        if self.state_layout is None:
            return self.nq + self.nv
        return self.state_layout.nx

    @property
    def nu_total(self) -> int:
        if self.control_layout is None:
            return self.nu
        return self.control_layout.nu_total

    @property
    def urdf_hash_key(self) -> str:
        if not self.urdf_path:
            return "no_urdf"
        path = Path(self.urdf_path)
        if not path.exists():
            return "missing_urdf"
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class DynamicsBackend(Protocol):
    """Callable dynamics backend interface used by MPX solvers."""

    def __call__(self, x, u, t, parameter):
        ...

    def step(self, x, u, t, parameter):
        ...

    def step_with_derivatives(self, x, u, t, parameter):
        ...

    def linearize_trajectory(self, X, U, timesteps, parameter):
        ...


class BoundDynamics:
    """Binds per-solve parameters while preserving backend Jacobian methods."""

    def __init__(self, backend, parameter):
        self.backend = backend
        self.parameter = parameter

    def __call__(self, x, u, t):
        return self.backend.step(x, u, t, self.parameter)

    def step_with_derivatives(self, x, u, t):
        return self.backend.step_with_derivatives(x, u, t, self.parameter)

    def linearize_trajectory(self, X, U, timesteps):
        return self.backend.linearize_trajectory(X, U, timesteps, self.parameter)


def bind_parameter(dynamics, parameter):
    if hasattr(dynamics, "with_parameter"):
        return dynamics.with_parameter(parameter)
    from functools import partial

    return partial(dynamics, parameter=parameter)


def contact_body_ids(contact_ids: Iterable[int], body_ids: Iterable[int]) -> tuple[int, ...]:
    """Map contact points to body ids, preserving legacy two-foot configs."""

    contact_ids = tuple(contact_ids)
    body_ids = tuple(body_ids)
    if len(body_ids) == len(contact_ids):
        return body_ids
    if len(body_ids) == 1:
        return body_ids * len(contact_ids)
    if len(body_ids) == 2 and len(contact_ids) % 2 == 0:
        half = len(contact_ids) // 2
        return (body_ids[0],) * half + (body_ids[1],) * half
    raise ValueError(
        "contact body mapping is ambiguous; provide one body_name per contact_frame"
    )


def integrate_q(q, v_next, dt: float, *, floating_base: bool, nq: int, nv: int):
    """Semi-implicit configuration integration for fixed and floating bases."""

    if floating_base:
        if nq < 7 or nv < 6:
            raise ValueError("floating-base layouts require nq >= 7 and nv >= 6")
        p_next = q[..., :3] + v_next[..., :3] * dt
        quat_next = math.quat_integrate(q[..., 3:7], v_next[..., 3:6], dt)
        joints_next = q[..., 7:nq] + v_next[..., 6:nv] * dt
        return jnp.concatenate([p_next, quat_next, joints_next], axis=-1)
    return q + v_next * dt


def generalized_torque(
    u,
    *,
    nv: int,
    actuated_dofs: tuple[int, ...],
    control_layout: ControlLayout,
):
    tau = jnp.zeros(u.shape[:-1] + (nv,), dtype=u.dtype)
    actuator_tau = u[..., control_layout.actuator_torques]
    return tau.at[..., jnp.asarray(actuated_dofs)].set(actuator_tau)


def vmap_leading(fn, x, u, t, parameter):
    """Apply `fn` over any leading batch dimensions without Python loops."""

    batch_shape = jnp.shape(x)[:-1]
    if not batch_shape:
        return fn(x, u, t, parameter)

    flat_x = jnp.reshape(x, (-1, x.shape[-1]))
    flat_u = jnp.reshape(u, (-1, u.shape[-1]))
    flat_t = jnp.broadcast_to(t, batch_shape).reshape((-1,))

    if parameter is None:
        flat_parameter = None
        in_axes = (0, 0, 0, None)
    else:
        p = jnp.asarray(parameter)
        if p.shape[: len(batch_shape)] == batch_shape:
            flat_parameter = jnp.reshape(p, (-1,) + p.shape[len(batch_shape) :])
            in_axes = (0, 0, 0, 0)
        else:
            flat_parameter = parameter
            in_axes = (0, 0, 0, None)

    out = jax.vmap(fn, in_axes=in_axes)(flat_x, flat_u, flat_t, flat_parameter)

    def unflatten(value):
        return jnp.reshape(value, batch_shape + value.shape[1:])

    return jax.tree.map(unflatten, out)
