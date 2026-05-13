from __future__ import annotations

import ctypes
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .base import (
    BoundDynamics,
    ControlLayout,
    RobotSpec,
    StateLayout,
    contact_body_ids,
    generalized_torque,
    integrate_q,
    vmap_leading,
)


_REGISTERED_TARGETS: set[str] = set()
_LOADED_LIBRARIES: list[ctypes.CDLL] = []


def _capsule_pointer(library: ctypes.CDLL, symbol_name: str):
    import ctypes.util

    pythonapi = ctypes.pythonapi
    pythonapi.PyCapsule_New.restype = ctypes.py_object
    pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    symbol = getattr(library, symbol_name)
    return pythonapi.PyCapsule_New(ctypes.cast(symbol, ctypes.c_void_p), b"xla._CUSTOM_CALL_TARGET", None)


def register_grid_ffi_library(path: str | os.PathLike[str], *, prefix: str):
    """Register GRiD XLA FFI targets exported by a generated shared library."""

    library_path = Path(path)
    if not library_path.exists():
        raise FileNotFoundError(library_path)

    library = ctypes.CDLL(str(library_path))
    _LOADED_LIBRARIES.append(library)

    targets = {
        f"{prefix}_step": "mpx_grid_step",
        f"{prefix}_step_with_derivatives": "mpx_grid_step_with_derivatives",
        f"{prefix}_ee_position": "mpx_grid_ee_position",
        f"{prefix}_ee_position_jacobian": "mpx_grid_ee_position_jacobian",
    }
    for ffi_name, symbol_name in targets.items():
        if ffi_name in _REGISTERED_TARGETS:
            continue
        if not hasattr(library, symbol_name):
            continue
        jax.ffi.register_ffi_target(
            ffi_name,
            _capsule_pointer(library, symbol_name),
            platform="CUDA",
            api_version=1,
        )
        _REGISTERED_TARGETS.add(ffi_name)


class GridDynamicsBackend:
    """Generic GRiD dynamics backend.

    When a generated GRiD FFI library is provided, `step` and
    `step_with_derivatives` dispatch to batched foreign calls.  If no library is
    configured, the backend can run in `reference_fallback` mode, which keeps the
    solver integration and tests executable through the existing MJX model while
    reporting that the production GRiD kernels are unavailable.
    """

    backend_name = "grid"

    def __init__(
        self,
        robot_spec: RobotSpec,
        *,
        model=None,
        mjx_model=None,
        contact_id=(),
        body_id=(),
        ffi_library_path: str | None = None,
        ffi_prefix: str | None = None,
        reference_fallback: bool = True,
    ):
        self.robot_spec = robot_spec
        self.model = model
        self.mjx_model = mjx_model
        self.contact_id = tuple(contact_id)
        self.body_id = tuple(body_id)
        self._contact_body_id = contact_body_ids(self.contact_id, self.body_id) if contact_id else ()
        self.ffi_prefix = ffi_prefix or f"mpx_grid_{robot_spec.name}_{robot_spec.urdf_hash_key}"
        self.ffi_library_path = ffi_library_path
        self.reference_fallback = reference_fallback

        if ffi_library_path:
            register_grid_ffi_library(ffi_library_path, prefix=self.ffi_prefix)

    @property
    def uses_ffi(self) -> bool:
        return bool(self.ffi_library_path)

    @property
    def using_reference_fallback(self) -> bool:
        return not self.uses_ffi

    def __call__(self, x, u, t, parameter):
        return self.step(x, u, t, parameter)

    def with_parameter(self, parameter):
        return BoundDynamics(self, parameter)

    def step(self, x, u, t, parameter):
        if self.uses_ffi:
            return self._ffi_step(x, u, parameter)
        if not self.reference_fallback:
            raise RuntimeError(
                "GRiD FFI library is not registered. Build it with "
                "`python -m mpx.grid_codegen.build ...` or enable reference_fallback."
            )
        return vmap_leading(self._reference_step_one, x, u, t, parameter)

    def step_with_derivatives(self, x, u, t, parameter):
        if self.uses_ffi:
            return self._ffi_step_with_derivatives(x, u, parameter)
        if not self.reference_fallback:
            raise RuntimeError("GRiD analytical derivative FFI is not available")
        return vmap_leading(self._reference_step_with_derivatives_one, x, u, t, parameter)

    def linearize_trajectory(self, X, U, timesteps, parameter):
        def linearize_one(x, u, t):
            _, A, B = self.step_with_derivatives(x, u, t, parameter)
            return A, B

        return jax.vmap(linearize_one)(X, U, timesteps)

    def _ffi_step(self, x, u, parameter):
        batch_shape = x.shape[:-1]
        out_shape = jax.ShapeDtypeStruct(batch_shape + (self.robot_spec.nx,), x.dtype)
        return jax.ffi.ffi_call(
            f"{self.ffi_prefix}_step",
            out_shape,
            vmap_method="broadcast_all",
        )(x, u, parameter)

    def _ffi_step_with_derivatives(self, x, u, parameter):
        batch_shape = x.shape[:-1]
        nx = self.robot_spec.nx
        nu = self.robot_spec.nu_total
        out_shapes = (
            jax.ShapeDtypeStruct(batch_shape + (nx,), x.dtype),
            jax.ShapeDtypeStruct(batch_shape + (nx, nx), x.dtype),
            jax.ShapeDtypeStruct(batch_shape + (nx, nu), x.dtype),
        )
        return jax.ffi.ffi_call(
            f"{self.ffi_prefix}_step_with_derivatives",
            out_shapes,
            vmap_method="broadcast_all",
        )(x, u, parameter)

    def ee_position(self, x):
        batch_shape = x.shape[:-1]
        out_shape = jax.ShapeDtypeStruct(batch_shape + (3,), x.dtype)
        return jax.ffi.ffi_call(
            f"{self.ffi_prefix}_ee_position",
            out_shape,
            vmap_method="broadcast_all",
        )(x)

    def ee_position_jacobian(self, x):
        batch_shape = x.shape[:-1]
        out_shapes = (
            jax.ShapeDtypeStruct(batch_shape + (3,), x.dtype),
            jax.ShapeDtypeStruct(batch_shape + (3, self.robot_spec.nx), x.dtype),
        )
        return jax.ffi.ffi_call(
            f"{self.ffi_prefix}_ee_position_jacobian",
            out_shapes,
            vmap_method="broadcast_all",
        )(x)

    def _reference_step_with_derivatives_one(self, x, u, t, parameter):
        step = lambda x_, u_: self._reference_step_one(x_, u_, t, parameter)
        x_next = step(x, u)
        A = jax.jacobian(step, argnums=0)(x, u)
        B = jax.jacobian(step, argnums=1)(x, u)
        return x_next, A, B

    def _reference_step_one(self, x, u, t, parameter):
        if self.model is None or self.mjx_model is None:
            raise RuntimeError("reference_fallback requires MuJoCo model and MJX model")
        spec = self.robot_spec
        state_layout = spec.state_layout
        control_layout = spec.control_layout
        if state_layout is None or control_layout is None:
            raise RuntimeError("RobotSpec must define state_layout and control_layout")

        q = x[state_layout.q]
        v = x[state_layout.v]

        data = mjx.make_data(self.model)
        data = data.replace(qpos=q, qvel=v)
        data = mjx.fwd_position(self.mjx_model, data)
        data = mjx.fwd_velocity(self.mjx_model, data)

        tau = generalized_torque(
            u,
            nv=spec.nv,
            actuated_dofs=spec.actuated_dofs,
            control_layout=control_layout,
        )
        contact_positions, contact_jacobian = self._contact_kinematics(data)
        contact_forces = self._contact_forces(u, t, parameter)
        generalized_contact = (
            contact_jacobian @ contact_forces
            if contact_forces.size
            else jnp.zeros(spec.nv, dtype=x.dtype)
        )

        qdd = jax.scipy.linalg.cho_solve(
            (data.qLD, False), tau - data.qfrc_bias + generalized_contact
        )
        v_next = v + qdd * spec.dt
        q_next = integrate_q(
            q,
            v_next,
            spec.dt,
            floating_base=spec.floating_base,
            nq=spec.nq,
            nv=spec.nv,
        )

        x_next = x
        x_next = x_next.at[state_layout.q].set(q_next)
        x_next = x_next.at[state_layout.v].set(v_next)
        if state_layout.contact_positions is not None and contact_positions.size:
            x_next = x_next.at[state_layout.contact_positions].set(contact_positions)
        return x_next

    def _contact_kinematics(self, data):
        if not self.contact_id:
            return jnp.zeros((0,), dtype=data.qpos.dtype), jnp.zeros(
                (self.robot_spec.nv, 0), dtype=data.qpos.dtype
            )
        positions = []
        jacobians = []
        for geom_id, body_id in zip(self.contact_id, self._contact_body_id):
            pos = data.geom_xpos[geom_id]
            jacp, _ = mjx.jac(self.mjx_model, data, pos, body_id)
            positions.append(pos)
            jacobians.append(jacp)
        return jnp.concatenate(positions, axis=0), jnp.concatenate(jacobians, axis=1)

    def _contact_forces(self, u, t, parameter):
        control_layout = self.robot_spec.control_layout
        if control_layout is None or control_layout.contact_forces is None:
            return jnp.zeros((0,), dtype=u.dtype)
        forces = u[control_layout.contact_forces]
        if self.robot_spec.n_contact == 0:
            return forces
        if parameter is None:
            contact = jnp.ones((self.robot_spec.n_contact,), dtype=u.dtype)
        else:
            contact = parameter[t, : self.robot_spec.n_contact]
        return (forces.reshape((self.robot_spec.n_contact, 3)) * contact[:, None]).reshape(-1)


def robot_spec_from_config(config, model) -> RobotSpec:
    n_joints = int(config.n_joints)
    n_contact = int(config.n_contact)
    nq = int(model.nq)
    nv = int(model.nv)
    q = slice(0, nq)
    v = slice(nq, nq + nv)
    contact = slice(nq + nv, nq + nv + 3 * n_contact) if n_contact else None
    aux_start = nq + nv + 3 * n_contact
    aux = slice(aux_start, int(config.n)) if aux_start < int(config.n) else None

    torque = slice(0, n_joints)
    force = slice(n_joints, int(config.m)) if int(config.m) > n_joints else None

    state_layout = StateLayout(
        q=q,
        v=v,
        contact_positions=contact,
        aux=aux,
    )
    control_layout = ControlLayout(
        actuator_torques=torque,
        contact_forces=force,
    )
    floating_base = nq == n_joints + 7 and nv == n_joints + 6
    actuated_offset = 6 if floating_base else 0
    actuated_dofs = tuple(range(actuated_offset, actuated_offset + n_joints))

    return RobotSpec(
        name=getattr(config, "robot_name", Path(config.model_path).stem),
        urdf_path=getattr(config, "urdf_path", None),
        nq=nq,
        nv=nv,
        nu=n_joints,
        floating_base=floating_base,
        actuated_dofs=actuated_dofs,
        contact_frames=tuple(getattr(config, "contact_frame", ())),
        contact_body_names=tuple(getattr(config, "body_name", ())),
        dt=float(config.dt),
        state_layout=state_layout,
        control_layout=control_layout,
    )
