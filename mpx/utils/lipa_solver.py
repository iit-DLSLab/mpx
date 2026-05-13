"""Adapter that exposes the Primal-Dual LIPA solver via the mpx solver API.

mpx solvers all share the signature
    solve(reference, parameter, W, x0, X0, U0, V0) -> (X, U, V)
with V having shape (N+1, n). LIPA expects a different problem statement
(`Variables` pytree, cost/dynamics with a (x, u, theta, t) signature, no
externalised W/reference/parameter). This module bridges the two.

Note on offline use vs mpx's other solvers: mpx's primal_dual / fddp do
*one* SQP/iLQR step per call and rely on `run_offline_solve`'s outer loop
to converge. LIPA is a complete NLP solver — its main loop schedules µ
(IPM barrier) and η (per-constraint AL penalty) internally. Calling it
many times restarts those parameters at every call (see
`primal_dual_lipa.optimizers.solve` lines 78-81), wasting iterations and
producing misleading benchmark numbers. So for offline mode use
`run_lipa_offline`, which calls LIPA exactly once and reports its
internal iteration count and wall time.
"""

from functools import partial
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np

from primal_dual_lipa.optimizers import solve as lipa_solve
from primal_dual_lipa.types import SolverSettings, Variables


def _wrap_cost(cost):
    def lipa_cost(W, reference, x, u, theta, t):
        del theta
        return cost(W, reference, x, u, t)

    return lipa_cost


def _wrap_dynamics(dynamics):
    def lipa_dynamics(parameter, x, u, theta, t):
        del theta
        return dynamics(x, u, t, parameter=parameter)

    return lipa_dynamics


def _wrap_inequalities(inequalities):
    def lipa_inequalities(reference, x, u, theta, t):
        del theta
        return inequalities(reference, x, u, t)

    return lipa_inequalities


def _empty_inequalities(reference, x, u, t):
    del reference, x, u, t
    return jnp.empty(0)


@partial(jax.jit, static_argnames=("cost", "dynamics", "inequalities"))
def _lipa_solve_with_stats(
    cost, dynamics, inequalities, settings, reference, parameter, W, x0, X_in, U_in, V_in
):
    """Single LIPA call that returns the final variables plus solver stats.

    `inequalities=None` keeps the prior behavior (no constraint blocks, ``g_dim=0``).
    Otherwise the constraint shape is inferred from a trace-time evaluation of the
    user callable on the warm-start sample.
    """

    lipa_cost = partial(_wrap_cost(cost), W, reference)
    lipa_dynamics = partial(_wrap_dynamics(dynamics), parameter)

    ineq_callable = inequalities if inequalities is not None else _empty_inequalities
    lipa_inequalities = partial(_wrap_inequalities(ineq_callable), reference)

    T = U_in.shape[0]
    sample_g = lipa_inequalities(X_in[0], U_in[0], jnp.empty(0, dtype=X_in.dtype), 0)
    g_dim = sample_g.shape[0]

    vars_in = Variables(
        X=X_in,
        U=U_in,
        S=jnp.zeros((T + 1, g_dim), dtype=X_in.dtype),
        Y_dyn=V_in,
        Y_eq=jnp.zeros((T + 1, 0), dtype=X_in.dtype),
        Z=jnp.zeros((T + 1, g_dim), dtype=X_in.dtype),
        Theta=jnp.empty(0, dtype=X_in.dtype),
    )

    vars_out, iterations, no_errors = lipa_solve(
        vars_in=vars_in,
        x0=x0,
        cost=lipa_cost,
        dynamics=lipa_dynamics,
        inequalities=lipa_inequalities,
        settings=settings,
    )
    return vars_out.X, vars_out.U, vars_out.Y_dyn, iterations, no_errors


def _default_settings():
    """Pick conservative defaults for an unseen problem.

    The goal here is robustness, not peak performance. Aggressive
    settings belong as per-config `lipa_settings` overrides.
    """

    on_gpu = any(d.platform == "gpu" for d in jax.devices())
    common = dict(
        max_iterations=2000,
        η0=1e3,
        η_update_factor=1.0,
        µ_update_factor=0.9,
        cost_improvement_threshold=1e-3,
        primal_violation_threshold=1e-5,
    )
    if on_gpu:
        return SolverSettings(
            use_parallel_lqr=True,
            num_parallel_line_search_steps=8,
            **common,
        )
    return SolverSettings(**common)


def build_lipa_solve(cost, dynamics, settings=None, *, inequalities=None):
    """Return a `solve(reference, parameter, W, x0, X0, U0, V0) -> (X, U, V)`.

    Used by online MPC (e.g. `MPCWrapper.run`). For offline benchmarks,
    prefer `run_lipa_offline`, which is a single-call path that surfaces
    LIPA's internal iteration count and avoids resetting µ/η repeatedly.

    Defaults differ by backend (parallel LQR + parallel line search on GPU).
    Override via `config.lipa_settings`. Pass `inequalities=callable(reference,
    x, u, t) -> g` to enforce ``g <= 0`` constraints; omit to keep the prior
    inequality-free behavior shared with the FDDP / primal-dual solvers.
    """

    if settings is None:
        settings = _default_settings()

    def solve(reference, parameter, W, x0, X0, U0, V0):
        X, U, V, _iters, _no_errors = _lipa_solve_with_stats(
            cost, dynamics, inequalities, settings, reference, parameter, W, x0, X0, U0, V0
        )
        return X, U, V

    return solve


def run_lipa_offline(
    cost,
    dynamics,
    reference,
    parameter,
    W,
    x0,
    X0,
    U0,
    V0,
    *,
    settings=None,
    inequalities=None,
    warmup_cost=None,
    warmup_settings=None,
    warmup=True,
    verbose=True,
):
    """Solve a single OCP with LIPA and return stats matching `run_offline_solve`.

    Unlike `run_offline_solve`, which loops one-step solvers until cost
    plateaus, this calls LIPA exactly once. Reported `n_iterations` is
    LIPA's internal IPM iteration count.

    Two-phase warm start: if `warmup_cost` is provided (typically the soft-
    penalty version of `cost`), an initial LIPA solve is run on that
    inequality-free formulation, then the main inequality-enforcing solve
    starts from its result. This sidesteps a class of local-basin pitfalls
    where the AL term η·Jᵀc dominates and the IPM parks at a degenerate
    iterate (e.g. on barrel_roll, the multi-shooting quaternion defect at
    the apex of the maneuver hits a sign-flip singularity that the cold-
    start solve cannot escape). The warm-start phase uses the same LIPA
    solver — this is not bootstrapping from a different solver.
    """

    from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers as ocp_opt

    if settings is None:
        settings = _default_settings()

    offline_cost = partial(cost, W, reference)
    offline_dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = jax.jit(
        partial(ocp_opt.model_evaluator_helper, offline_cost, offline_dynamics, x0)
    )

    g0, c0 = model_evaluator(X0, U0)
    initial_objective = float(g0)
    initial_l2_cost = float(np.sqrt(np.sum(np.asarray(g0) * np.asarray(g0))))
    initial_dynamics_violation = float(np.sum(np.asarray(c0) * np.asarray(c0)))

    if verbose:
        print("{:<10} {:<20} {:<20} {:<20}".format("Iter", "Cost", "Constraint", "Time [ms]"))
        print("{:<10d} {:<20.5f} {:<20.5f} {:<20}".format(0, initial_l2_cost, initial_dynamics_violation, "-"))

    do_warmup_phase = warmup_cost is not None and inequalities is not None
    warmup_phase_settings = warmup_settings if warmup_settings is not None else settings
    warmup_iters = 0
    warmup_time_ms = 0.0

    if do_warmup_phase:
        # Phase 1: solve the inequality-free (soft-penalty) problem once and
        # use its (X, U, V) as the warm start for phase 2. We deliberately do
        # NOT call _lipa_solve_with_stats twice (warmup + timed) here — the
        # parallel-LQR scan reduction is not bit-deterministic across
        # back-to-back invocations of the same compiled function on the same
        # inputs (different floating-point summation order can land on
        # numerically different iterates), and on stiff problems like
        # h1_jump_forward that's enough drift to make phase 2 sometimes
        # converge in 100 iters and sometimes hit max_iterations. The trade
        # here is mildly inaccurate phase-1 wall-time accounting (first call
        # includes any JIT compile that wasn't already cached) for
        # reproducible phase-2 starting iterates.
        start = timer()
        Xp1, Up1, Vp1, iters_p1, _ = _lipa_solve_with_stats(
            warmup_cost, dynamics, None, warmup_phase_settings,
            reference, parameter, W, x0, X0, U0, V0,
        )
        Xp1.block_until_ready()
        warmup_time_ms = 1e3 * (timer() - start)
        warmup_iters = int(iters_p1)
        if verbose:
            print(
                "{:<10s} {:<20s} {:<20s} {:<20.5f}".format(
                    "ph1", "(warmup)", "(warmup)", warmup_time_ms
                )
            )
            print(f"  Phase 1 (soft-penalty warm start): {warmup_iters} iters")
        # Phase 2 starts from phase 1's iterate.
        X0, U0, V0 = Xp1, Up1, Vp1

    if warmup and not do_warmup_phase:
        # Single-phase mode: traditional warmup-then-timed pattern.
        Xw, _, _, _, _ = _lipa_solve_with_stats(
            cost, dynamics, inequalities, settings, reference, parameter, W, x0, X0, U0, V0
        )
        Xw.block_until_ready()

    start = timer()
    X, U, V, iterations, no_errors = _lipa_solve_with_stats(
        cost, dynamics, inequalities, settings, reference, parameter, W, x0, X0, U0, V0
    )
    X.block_until_ready()
    stop = timer()
    iteration_time_ms = 1e3 * (stop - start)

    g, c = model_evaluator(X, U)
    final_objective = float(g)
    final_l2_cost = float(np.sqrt(np.sum(np.asarray(g) * np.asarray(g))))
    final_dynamics_violation = float(np.sum(np.asarray(c) * np.asarray(c)))
    n_iters = int(iterations) + warmup_iters
    converged = bool(no_errors)

    if verbose:
        print(
            "{:<10d} {:<20.5f} {:<20.5f} {:<20.5f}".format(
                1, final_l2_cost, final_dynamics_violation, iteration_time_ms
            )
        )
        if do_warmup_phase:
            print(
                f"  Phase 2 (constrained): {int(iterations)} iters, no_errors: {converged}\n"
                f"  Total LIPA internal iterations: {n_iters}"
            )
        else:
            print(f"  LIPA internal iterations: {n_iters}, no_errors: {converged}")

    history = [X0, X]
    stats = {
        "n_iterations": n_iters,
        "warmup_iterations": warmup_iters,
        "converged": converged,
        "warmup_discarded": warmup,
        "objective_history": [initial_objective, final_objective],
        "l2_cost_history": [initial_l2_cost, final_l2_cost],
        "dynamics_violation_history": [initial_dynamics_violation, final_dynamics_violation],
        "metric_iteration_history": [0, 1],
        "iteration_time_ms_history": [iteration_time_ms + warmup_time_ms],
        "initial_objective": initial_objective,
        "initial_l2_cost": initial_l2_cost,
        "initial_dynamics_violation": initial_dynamics_violation,
        "average_iteration_time_ms": iteration_time_ms + warmup_time_ms,
        "final_objective": final_objective,
        "final_l2_cost": final_l2_cost,
        "final_dynamics_violation": final_dynamics_violation,
    }
    return X, U, V, history, stats
