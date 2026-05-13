from __future__ import annotations

import jax
import jax.numpy as jnp


def penalty(constraint, alpha=0.1, sigma=5.0, transition_width=0.2):
    safe = jnp.clip(constraint, 1e-10, 1e6)
    quadratic = alpha / 2 * (
        jnp.square((constraint - 2 * sigma) / sigma) - jnp.ones_like(constraint)
    )
    logarithmic = -alpha * jnp.log(safe)
    weight = 0.5 * (1 + jnp.tanh((constraint - sigma) / transition_width))
    return jnp.clip(weight * logarithmic + (1 - weight) * quadratic, 0.0, 1e8)


def joint_limits_penalty(n_joints, q, *, alpha, sigma, transition_width):
    limits = jnp.array([
        2.61799,
        2.61799,
        2.96706,
        0.0,
        0.0,
        2.87979,
        1.51844,
        1.51844,
        1.3439,
        1.3439,
        2.79253,
        2.79253,
    ])
    joint_limit_map = jnp.kron(jnp.eye(n_joints), jnp.array([-1.0, 1.0]))
    margins = q @ joint_limit_map + limits + jnp.ones_like(limits) * 1e-2
    return jnp.sum(
        penalty(margins, alpha=alpha, sigma=sigma, transition_width=transition_width),
        axis=-1,
    )


def quat_to_matrix_wxyz(quat):
    w, x, y, z = quat
    n = jnp.maximum(jnp.linalg.norm(quat), 1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=quat.dtype,
    )


def quad_form(v, W):
    return jnp.einsum("...i,ij,...j->...", v, W, v)


class Z1GridCost:
    """Z1 end-effector cost backed by GRiD kinematics.

    The scalar cost uses GRiD end-effector position.  The solver linearization
    path uses one batched GRiD position/Jacobian call for the full trajectory and
    forms a Gauss-Newton Hessian for the least-squares end-effector term.
    """

    def __init__(
        self,
        grid_backend,
        n_joints,
        horizon,
        *,
        floating_base=False,
        q0=None,
        local_kinematics_backend=None,
        base_to_arm_offset=None,
    ):
        self.grid_backend = grid_backend
        self.n_joints = n_joints
        self.horizon = horizon
        self.floating_base = floating_base
        self.q0 = q0
        self.local_kinematics_backend = local_kinematics_backend
        self.base_to_arm_offset = (
            jnp.asarray(base_to_arm_offset)
            if base_to_arm_offset is not None
            else jnp.zeros(3)
        )

    def with_reference(self, W, reference):
        return BoundZ1GridCost(
            self.grid_backend,
            self.n_joints,
            self.horizon,
            W,
            reference,
            floating_base=self.floating_base,
            q0=self.q0,
            local_kinematics_backend=self.local_kinematics_backend,
            base_to_arm_offset=self.base_to_arm_offset,
        )


class BoundZ1GridCost:
    def __init__(
        self,
        grid_backend,
        n_joints,
        horizon,
        W,
        reference,
        *,
        floating_base,
        q0,
        local_kinematics_backend,
        base_to_arm_offset,
    ):
        self.grid_backend = grid_backend
        self.n_joints = n_joints
        self.horizon = horizon
        self.W = W
        self.reference = reference
        self.floating_base = floating_base
        self.q0 = q0
        self.local_kinematics_backend = local_kinematics_backend
        self.base_to_arm_offset = base_to_arm_offset

    def __call__(self, x, u, t):
        return self._floating_cost(x, u, t) if self.floating_base else self._fixed_cost(x, u, t)

    def _fixed_cost(self, x, u, t):
        n = self.n_joints
        q = x[..., :n]
        dq = x[..., n : 2 * n]
        tau = u[..., :n]
        ee = self.grid_backend.ee_position(x)

        q_ref = self.reference[t, :n]
        ee_ref = self.reference[t, n : n + 3]
        tau_ref = self.reference[t, -n:]
        ee_error = ee - ee_ref

        stage = (
            quad_form(q - q_ref, self.W[:n, :n])
            + quad_form(dq, self.W[n : 2 * n, n : 2 * n])
            + quad_form(ee_error, self.W[2 * n : 2 * n + 3, 2 * n : 2 * n + 3])
            + quad_form(
                tau - tau_ref,
                self.W[2 * n + 6 : 3 * n + 6, 2 * n + 6 : 3 * n + 6],
            )
            + joint_limits_penalty(n, q, alpha=1.0, sigma=0.01, transition_width=0.05)
        )
        terminal = (
            quad_form(q - q_ref, self.W[:n, :n])
            + quad_form(dq, self.W[n : 2 * n, n : 2 * n])
            + 1e2 * quad_form(ee_error, self.W[2 * n : 2 * n + 3, 2 * n : 2 * n + 3])
        )
        return jnp.where(t == self.horizon, 0.5 * terminal, 0.5 * stage)

    def _floating_cost(self, x, u, t):
        n = self.n_joints
        q0 = self.q0
        p = x[..., :3]
        quat = x[..., 3:7]
        q = x[..., 7 : 7 + n]
        dq = x[..., 7 + n :]
        dp = u[..., n : n + 3]
        omega = u[..., -3:]
        tau = u[..., :n]
        ee = self._floating_ee_position(x)

        q_ref = self.reference[t, :n]
        ee_ref = self.reference[t, n : n + 3]
        tau_ref = self.reference[t, -n:]
        ee_error = ee - ee_ref
        rot_error = quat[..., 1:4] - q0[3:7][1:4]

        stage = (
            quad_form(p - q0[:3], self.W[0:3, 0:3])
            + quad_form(rot_error, self.W[3:6, 3:6])
            + quad_form(dp, self.W[6:9, 6:9])
            + quad_form(omega, self.W[9:12, 9:12])
            + quad_form(q - q_ref, self.W[12 : 12 + n, 12 : 12 + n])
            + quad_form(dq, self.W[12 + n : 12 + 2 * n, 12 + n : 12 + 2 * n])
            + quad_form(ee_error, self.W[12 + 2 * n : 12 + 2 * n + 3, 12 + 2 * n : 12 + 2 * n + 3])
            + quad_form(
                tau - tau_ref,
                self.W[12 + 2 * n + 6 : 12 + 3 * n + 6, 12 + 2 * n + 6 : 12 + 3 * n + 6],
            )
            + joint_limits_penalty(n, q, alpha=0.5, sigma=0.1, transition_width=0.05)
        )
        terminal = (
            quad_form(p - q0[:3], self.W[0:3, 0:3])
            + quad_form(rot_error, self.W[3:6, 3:6])
            + quad_form(q - q_ref, self.W[12 : 12 + n, 12 : 12 + n])
            + quad_form(dq, self.W[12 + n : 12 + 2 * n, 12 + n : 12 + 2 * n])
            + 1e3 * quad_form(ee_error, self.W[12 + 2 * n : 12 + 2 * n + 3, 12 + 2 * n : 12 + 2 * n + 3])
        )
        return jnp.where(t == self.horizon, 0.5 * terminal, 0.5 * stage)

    def derivatives_trajectory(self, X, U, timesteps):
        return (
            self._floating_derivatives(X, U, timesteps)
            if self.floating_base
            else self._fixed_derivatives(X, U, timesteps)
        )

    def _fixed_derivatives(self, X, U, timesteps):
        n = self.n_joints
        nx = X.shape[-1]
        nu = U.shape[-1]
        q = X[..., :n]
        dq = X[..., n : 2 * n]
        tau = U[..., :n]
        ee, ee_jac = self.grid_backend.ee_position_jacobian(X)

        q_ref = self.reference[timesteps, :n]
        ee_ref = self.reference[timesteps, n : n + 3]
        tau_ref = self.reference[timesteps, -n:]
        terminal = (timesteps == self.horizon).astype(X.dtype)
        stage = 1.0 - terminal

        Wq = self.W[:n, :n]
        Wdq = self.W[n : 2 * n, n : 2 * n]
        Wee = self.W[2 * n : 2 * n + 3, 2 * n : 2 * n + 3]
        Wtau = self.W[2 * n + 6 : 3 * n + 6, 2 * n + 6 : 3 * n + 6]
        ee_weight = (stage + 1e2 * terminal)[..., None]

        q_grad = (q - q_ref) @ Wq.T
        dq_grad = dq @ Wdq.T
        ee_grad = jnp.einsum(
            "...ij,...i->...j", ee_jac, ((ee - ee_ref) @ Wee.T) * ee_weight
        )
        limit_grad, limit_hess = self._limit_derivatives(q, alpha=1.0, sigma=0.01)
        limit_grad = 0.5 * stage[..., None] * limit_grad
        limit_hess = 0.5 * stage[..., None, None] * limit_hess

        q_cost = jnp.zeros(X.shape, dtype=X.dtype)
        q_cost = q_cost.at[..., :n].set(q_grad + ee_grad[..., :n] + limit_grad)
        q_cost = q_cost.at[..., n : 2 * n].set(dq_grad + ee_grad[..., n : 2 * n])

        r_cost = jnp.zeros(U.shape, dtype=U.dtype)
        r_cost = r_cost.at[..., :n].set(stage[..., None] * ((tau - tau_ref) @ Wtau.T))

        Q = jnp.zeros(X.shape[:-1] + (nx, nx), dtype=X.dtype)
        Q = Q.at[..., :n, :n].add(Wq + limit_hess)
        Q = Q.at[..., n : 2 * n, n : 2 * n].add(Wdq)
        Q = Q + ee_weight[..., None] * jnp.einsum("...ki,...kl,...lj->...ij", ee_jac, Wee, ee_jac)

        R = jnp.zeros(U.shape[:-1] + (nu, nu), dtype=U.dtype)
        R = R.at[..., :n, :n].set(stage[..., None, None] * Wtau)
        M = jnp.zeros(X.shape[:-1] + (nx, nu), dtype=X.dtype)
        return q_cost, r_cost, Q, R, M

    def _floating_ee_position(self, x):
        backend = self.local_kinematics_backend or self.grid_backend
        n = self.n_joints

        def one(x_):
            p = x_[:3]
            quat = x_[3:7]
            q = x_[7 : 7 + n]
            dq = x_[7 + n :]
            local_state = jnp.concatenate([q, dq])
            local_ee = backend.ee_position(local_state)
            return p + quat_to_matrix_wxyz(quat) @ (self.base_to_arm_offset + local_ee)

        flat = x.reshape((-1, x.shape[-1]))
        ee = jax.vmap(one)(flat)
        return ee.reshape(x.shape[:-1] + (3,))

    def _floating_ee_position_jacobian(self, X):
        backend = self.local_kinematics_backend or self.grid_backend
        n = self.n_joints

        def one(x):
            p = x[:3]
            quat = x[3:7]
            q = x[7 : 7 + n]
            dq = x[7 + n :]
            local_state = jnp.concatenate([q, dq])
            local_ee, local_jac = backend.ee_position_jacobian(local_state)
            rotated = quat_to_matrix_wxyz(quat) @ (self.base_to_arm_offset + local_ee)

            def base_part(p_, quat_):
                return p_ + quat_to_matrix_wxyz(quat_) @ (self.base_to_arm_offset + local_ee)

            jac_base = jax.jacobian(base_part, argnums=(0, 1))(p, quat)
            jac = jnp.zeros((3, x.shape[-1]), dtype=x.dtype)
            jac = jac.at[:, :3].set(jac_base[0])
            jac = jac.at[:, 3:7].set(jac_base[1])
            jac = jac.at[:, 7 : 7 + n].set(quat_to_matrix_wxyz(quat) @ local_jac[:, :n])
            jac = jac.at[:, 7 + n :].set(quat_to_matrix_wxyz(quat) @ local_jac[:, n:])
            return p + rotated, jac

        flat = X.reshape((-1, X.shape[-1]))
        ee, jac = jax.vmap(one)(flat)
        return ee.reshape(X.shape[:-1] + (3,)), jac.reshape(X.shape[:-1] + (3, X.shape[-1]))

    def _floating_derivatives(self, X, U, timesteps):
        n = self.n_joints
        nx = X.shape[-1]
        nu = U.shape[-1]
        q0 = self.q0
        p = X[..., :3]
        quat = X[..., 3:7]
        q = X[..., 7 : 7 + n]
        dq = X[..., 7 + n :]
        dp = U[..., n : n + 3]
        omega = U[..., -3:]
        tau = U[..., :n]
        ee, ee_jac = self._floating_ee_position_jacobian(X)

        q_ref = self.reference[timesteps, :n]
        ee_ref = self.reference[timesteps, n : n + 3]
        tau_ref = self.reference[timesteps, -n:]
        terminal = (timesteps == self.horizon).astype(X.dtype)
        stage = 1.0 - terminal

        Wp = self.W[0:3, 0:3]
        Wrot = self.W[3:6, 3:6]
        Wdp = self.W[6:9, 6:9]
        Womega = self.W[9:12, 9:12]
        Wq = self.W[12 : 12 + n, 12 : 12 + n]
        Wdq = self.W[12 + n : 12 + 2 * n, 12 + n : 12 + 2 * n]
        Wee = self.W[12 + 2 * n : 12 + 2 * n + 3, 12 + 2 * n : 12 + 2 * n + 3]
        Wtau = self.W[12 + 2 * n + 6 : 12 + 3 * n + 6, 12 + 2 * n + 6 : 12 + 3 * n + 6]
        ee_weight = (stage + 1e3 * terminal)[..., None]

        ee_grad = jnp.einsum(
            "...ij,...i->...j", ee_jac, ((ee - ee_ref) @ Wee.T) * ee_weight
        )
        limit_grad, limit_hess = self._limit_derivatives(q, alpha=0.5, sigma=0.1)
        limit_grad = 0.5 * stage[..., None] * limit_grad
        limit_hess = 0.5 * stage[..., None, None] * limit_hess

        q_cost = jnp.zeros(X.shape, dtype=X.dtype)
        q_cost = q_cost.at[..., :3].set((p - q0[:3]) @ Wp.T + ee_grad[..., :3])
        q_cost = q_cost.at[..., 4:7].set((quat[..., 1:4] - q0[3:7][1:4]) @ Wrot.T + ee_grad[..., 4:7])
        q_cost = q_cost.at[..., 7 : 7 + n].set((q - q_ref) @ Wq.T + ee_grad[..., 7 : 7 + n] + limit_grad)
        q_cost = q_cost.at[..., 7 + n :].set(dq @ Wdq.T + ee_grad[..., 7 + n :])

        r_cost = jnp.zeros(U.shape, dtype=U.dtype)
        r_cost = r_cost.at[..., :n].set(stage[..., None] * ((tau - tau_ref) @ Wtau.T))
        r_cost = r_cost.at[..., n : n + 3].set(stage[..., None] * (dp @ Wdp.T))
        r_cost = r_cost.at[..., -3:].set(stage[..., None] * (omega @ Womega.T))

        Q = jnp.zeros(X.shape[:-1] + (nx, nx), dtype=X.dtype)
        Q = Q.at[..., :3, :3].add(Wp)
        Q = Q.at[..., 4:7, 4:7].add(Wrot)
        Q = Q.at[..., 7 : 7 + n, 7 : 7 + n].add(Wq + limit_hess)
        Q = Q.at[..., 7 + n :, 7 + n :].add(Wdq)
        Q = Q + ee_weight[..., None] * jnp.einsum("...ki,...kl,...lj->...ij", ee_jac, Wee, ee_jac)

        R = jnp.zeros(U.shape[:-1] + (nu, nu), dtype=U.dtype)
        R = R.at[..., :n, :n].set(stage[..., None, None] * Wtau)
        R = R.at[..., n : n + 3, n : n + 3].set(stage[..., None, None] * Wdp)
        R = R.at[..., -3:, -3:].set(stage[..., None, None] * Womega)
        M = jnp.zeros(X.shape[:-1] + (nx, nu), dtype=X.dtype)
        return q_cost, r_cost, Q, R, M

    def _limit_derivatives(self, q, *, alpha, sigma):
        n = self.n_joints
        flat = q.reshape((-1, n))
        fn = lambda q_: joint_limits_penalty(
            n, q_, alpha=alpha, sigma=sigma, transition_width=0.05
        )
        grad = jax.vmap(jax.grad(fn))(flat).reshape(q.shape)
        hess = jax.vmap(jax.hessian(fn))(flat).reshape(q.shape[:-1] + (n, n))
        return grad, hess
