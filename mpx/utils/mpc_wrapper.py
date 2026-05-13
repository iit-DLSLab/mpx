from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.dataclasses import PyTreeNode

from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers
from mpx.utils.lipa_solver import build_lipa_solve, run_lipa_offline
import mpx.utils.offline_solver as offline_solver
import mpx.utils.mpc_utils as mpc_utils


class MPCData(PyTreeNode):
    """Carry state for the pure functional MPC API."""

    dt: float
    duty_factor: float
    step_freq: float
    step_height: float
    contact_time: jnp.ndarray
    liftoff: jnp.ndarray
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    W: jnp.ndarray


mpx_data = MPCData

def lipa_pick_cost_and_inequalities(config, cost):
    """Pick the LIPA call configuration based on the config.

    Returns ``(main_cost, inequalities, main_settings, warmup_cost,
    warmup_settings)``:

    * Off path (no enforce): main = ``cost`` (the soft-penalty cost), no
      inequalities, settings from ``config.lipa_settings``.
    * Enforce path: main = ``cost_smooth + inequalities`` with
      ``config.lipa_settings_enforce or config.lipa_settings``. The warm-start
      pair (``cost``, ``lipa_settings``) is also returned so the offline path
      can do a two-phase solve — phase 1 on the inequality-free formulation,
      phase 2 on the constrained one starting from phase 1's iterate. This
      sidesteps local-basin pitfalls (notably the multi-shooting quaternion
      singularity at the apex of the barrel-roll maneuver) without
      bootstrapping from a different solver.

    Configs opt in by setting ``lipa_enforce_inequalities = True`` and
    providing both ``cost_smooth`` and ``inequalities``.
    """
    enforce = getattr(config, "lipa_enforce_inequalities", False)
    base_settings = getattr(config, "lipa_settings", None)
    if not enforce:
        return cost, None, base_settings, None, None
    cost_smooth = getattr(config, "cost_smooth", None)
    inequalities = getattr(config, "inequalities", None)
    if cost_smooth is None or inequalities is None:
        raise ValueError(
            "lipa_enforce_inequalities=True requires both `cost_smooth` and "
            "`inequalities` to be defined on the config."
        )
    enforce_settings = getattr(config, "lipa_settings_enforce", None) or base_settings
    return cost_smooth, inequalities, enforce_settings, cost, base_settings


def build_solver_step(config, cost, dynamics, hessian_approx, limited_memory):
    solver_mode = getattr(config, "solver_mode", "primal_dual")

    if solver_mode == "primal_dual":
        solver = partial(optimizers.mpc, cost, dynamics, hessian_approx, limited_memory)

        def solve(reference, parameter, W, x0, X0, U0, V0):
            return solver(reference, parameter, W, x0, X0, U0, V0)

        return solver_mode, solve

    if solver_mode == "fddp":
        solver = partial(
            optimizers.fddp_mpc,
            cost,
            dynamics,
            hessian_approx,
            limited_memory,
        )

        def solve(reference, parameter, W, x0, X0, U0, V0):
            X, U, defects = solver(reference, parameter, W, x0, X0, U0)
            return X, U, defects

        return solver_mode, solve

    if solver_mode == "lipa":
        # Online MPC stays single-phase: per-step warm-start via the data
        # carry already chains across calls, and a per-step phase-1 would
        # double the compile + per-step compute. The two-phase flow is
        # offline-only (see run_lipa_offline / runOffline).
        lipa_cost, lipa_inequalities, lipa_settings, _, _ = lipa_pick_cost_and_inequalities(
            config, cost
        )
        solve = build_lipa_solve(
            lipa_cost, dynamics, settings=lipa_settings, inequalities=lipa_inequalities
        )
        return solver_mode, solve

    raise ValueError(f"Unsupported MPC solver_mode: {solver_mode}")


@partial(jax.jit, static_argnums=(0, 1, 2))
def _update_warm_start(n_joints, horizon, shift, u_ref, x0, X_prev, U_prev, X, U, V):
    """Shift the solution for the next MPC step and extract the first command."""

    q_slice = slice(7, 7 + n_joints)
    dq_slice = slice(13 + n_joints, 13 + 2 * n_joints)
    u_fallback_idx = 1 if horizon > 1 else 0

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate([trajectory[shift:], tail], axis=0)

    def safe_update():
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            U[0, :n_joints],
            X[0, q_slice],
            X[1, dq_slice],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(X_prev),
            U_prev[u_fallback_idx, :n_joints],
            X_prev[1, q_slice],
            X_prev[1, dq_slice],
        )

    valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
    return jax.lax.cond(valid_solution, safe_update, unsafe_update)


class MPCWrapper:
    """Minimal MPC API built for `jit` and `vmap`.

    The public flow is:
    `data = wrapper.make_data()`
    `data, tau = wrapper.run(data, x0, command, contact)`

    The warm-start state always carries `V0`, even for direct solvers like FDDP.
    That keeps the pytree shape fixed and makes solver switching transparent to
    callers, batching, and JIT compilation.
    """

    def __init__(self, config, limited_memory=False):
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        self.default_contact = jnp.zeros(config.n_contact)
        self.qpos_slice = slice(0, 7 + config.n_joints)
        self.qvel_slice = slice(self.qpos_slice.stop, self.qpos_slice.stop + 6 + config.n_joints)
        self.foot_slice = slice(
            self.qvel_slice.stop,
            self.qvel_slice.stop + 3 * config.n_contact,
        )

        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        data = mujoco.MjData(self.model)
        mujoco.mj_fwdPosition(self.model, data)
        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)
        robot_mass = data.qM[0]

        self.contact_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]
        self.body_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.body_name
        ]
        self.contact_id_mj = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]

        self.cost = config.cost
        self.hessian_approx = config.hessian_approx
        self.dynamics = config.dynamics(
            self.model,
            self.mjx_model,
            self.contact_id,
            self.body_id,
        )

        # The config owns the nominal state layout, including any extra states.
        self.initial_state = jnp.asarray(config.initial_state)

        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_liftoff = jnp.zeros(3 * config.n_contact)

        self.solver_mode, solve = build_solver_step(
            config,
            self.cost,
            self.dynamics,
            self.hessian_approx,
            limited_memory,
        )
        self._solve = jax.jit(solve)

        reference_generator = getattr(config, "reference_generator", mpc_utils.reference_generator)
        clearance_speed = getattr(config, "clearance_speed", getattr(config, "clearence_speed", 0.2))
        self._ref_gen = jax.jit(
            partial(
                reference_generator,
                config.use_terrain_estimation,
                config.N,
                config.dt,
                config.n_joints,
                config.n_contact,
                robot_mass,
                foot0=config.p_legs0,
                q0=config.q0,
                clearence_speed=clearance_speed,
            )
        )
        self._timer_run = jax.jit(mpc_utils.timer_run)
        self._update_warm_start = partial(
            _update_warm_start,
            config.n_joints,
            config.N,
            self.shift,
            config.u_ref,
        )

    def make_data(self):
        """Allocate the pytree state used by the pure functional API."""

        return MPCData(
            dt=self.config.dt,
            duty_factor=self.config.duty_factor,
            step_freq=self.config.step_freq,
            step_height=self.config.step_height,
            contact_time=self.config.timer_t,
            liftoff=self.initial_liftoff,
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            W=self.config.W,
        )

    def control_output(self, x0, X, U, reference, parameter):
        del x0, X, reference, parameter
        return U[0, : self.config.n_joints]

    def _run_impl(self, data, x0, input, contact):
        _, contact_time = self._timer_run(
            data.duty_factor,
            data.step_freq,
            data.contact_time,
            1 / self.mpc_frequency,
        )

        reference, parameter, liftoff = self._ref_gen(
            duty_factor=data.duty_factor,
            step_freq=data.step_freq,
            step_height=data.step_height,
            t_timer=data.contact_time,
            x=x0,
            foot=x0[self.foot_slice],
            input=input,
            liftoff=data.liftoff,
            contact=contact,
        )

        # Reference generation and solver execution stay on the pure JAX path.
        X, U, V = self._solve(
            reference,
            parameter,
            data.W,
            x0,
            data.X0,
            data.U0,
            data.V0,
        )
        valid_solution = jnp.logical_not(jnp.isnan(U[0, 0]))
        tau = jax.lax.cond(
            valid_solution,
            lambda _: self.control_output(x0, X, U, reference, parameter),
            lambda _: self.control_output(x0, data.X0, data.U0, reference, parameter),
            operand=None,
        )
        # Shift the solution so the next call starts from the previous optimum.
        U0, X0, V0, _, q, dq = self._update_warm_start(
            x0,
            data.X0,
            data.U0,
            X,
            U,
            V,
        )

        data = data.replace(
            X0=X0,
            U0=U0,
            V0=V0,
            contact_time=contact_time,
            liftoff=liftoff,
        )
        return data, tau, q, dq

    def run(self, data, x0, input, contact=None):
        """Run one MPC step and return the updated carry and torque command."""

        contact = self.default_contact if contact is None else jnp.asarray(contact)
        data, tau, _, _ = self._run_impl(data, x0, input, contact)
        return data, tau

    def reset(self, data, qpos, qvel, foot):
        """Reset the warm start around the provided measured state."""

        # Start from the config initial_state so any extra state entries keep
        # their configured default value.
        initial_state = (
            self.initial_state
            .at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
            .at[self.foot_slice].set(jnp.ravel(foot))
        )
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(initial_state, (self.config.N + 1, 1)),
            V0=self.initial_V0,
            contact_time=self.config.timer_t,
            liftoff=jnp.ravel(foot),
        )

    def foot_positions(self, qpos):
        """Return the flattened contact-point positions for the provided configuration."""

        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return jnp.array([self.data.geom_xpos[idx] for idx in self.contact_id_mj]).flatten()

    def runOffline(self, qpos, qvel, *, return_stats=False, verbose=True, max_iter=100):
        """Solve the fixed reference problem exposed by configs that define `reference`."""

        foot_op = self.foot_positions(qpos)
        x0 = (
            self.initial_state
            .at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
            .at[self.foot_slice].set(foot_op)
        )
        reference, parameter = self.config.reference(
            self.config.N + 1,
            self.config.dt,
            self.config.n_joints,
            self.config.n_contact,
            self.config.p_legs0,
            self.config.q0,
        )

        W = self.config.W

        # Keep the offline warm start aligned with the nominal reference seed.
        # The old wrapper started from `initial_X0`, not from the measured feet.
        X0 = self.initial_X0.at[:, : 13 + self.config.n_joints].set(
            reference[:, : 13 + self.config.n_joints]
        )
        U0 = self.initial_U0
        V0 = self.initial_V0

        if self.solver_mode == "lipa":
            # LIPA is a complete NLP solver; one call converges. Looping it
            # restarts the IPM µ/η each time, which inflates "iterations"
            # and wall time without improving the solution.
            (
                lipa_cost,
                lipa_inequalities,
                lipa_settings,
                lipa_warmup_cost,
                lipa_warmup_settings,
            ) = lipa_pick_cost_and_inequalities(self.config, self.cost)
            X0, U0, _, output, stats = run_lipa_offline(
                lipa_cost,
                self.dynamics,
                reference,
                parameter,
                W,
                x0,
                X0,
                U0,
                V0,
                settings=lipa_settings,
                inequalities=lipa_inequalities,
                warmup_cost=lipa_warmup_cost,
                warmup_settings=lipa_warmup_settings,
                verbose=verbose,
            )
        else:
            X0, U0, _, output, stats = offline_solver.run_offline_solve(
                self._solve,
                self.cost,
                self.dynamics,
                self.config.solver_mode,
                reference,
                parameter,
                W,
                x0,
                X0,
                U0,
                V0,
                max_iter=max_iter,
                verbose=verbose,
            )

        if return_stats:
            return X0, U0, reference, output, stats
        return X0, U0, reference, output
