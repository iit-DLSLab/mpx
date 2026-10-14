import jax
import math
import jax.numpy as jnp
from functools import partial
from flax import struct
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import mujoco
from mujoco import mjx
import mpx.jax_ocp_solvers.optimizers as optimizers


@struct.dataclass
class ControlSol:
    a     : jax.Array
    ac_z  : jax.Array
    alpha : jax.Array
    grf   : jax.Array


@struct.dataclass
class MPCState:
    """All mutable MPC state in a JAX pytree (compatible with jax.vmap)."""
    sol        : ControlSol
    X0_shifted : jax.Array
    U0_shifted : jax.Array
    D0_shifted : jax.Array


class BatchedMPCControllerWrapper:
    def __init__(self, config, n_env):
        jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        self.n_env = n_env
        self.config = config

        self.mpc_frequency = config.mpc_frequency
        self.shift = config.mpc_shift_nodes
        self.mpc_iterations = int(getattr(config, "mpc_iterations", 1))
        self.wbc_lookahead_dt = float(getattr(config, "wbc_lookahead_dt", 0.0))

        model = mujoco.MjModel.from_xml_path(config.model_path)
        mjx_model = mjx.put_model(model)
        self._mjx_model = mjx_model

        contact_id   = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in config.contact_frame]
        body_id      = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in config.body_name]
        base_body_id = mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)

        pcom_init  = config.p0.copy()
        dpcom_init = jnp.zeros(3)
        c_init     = pcom_init.copy().at[2].set(0.0)
        vcz_init   = jnp.zeros(1)
        theta      = jnp.arctan2(c_init[1], c_init[0])
        v          = jnp.zeros(1)
        omega      = jnp.zeros(1)
        self.initial_state = jnp.concatenate([pcom_init, dpcom_init, c_init, vcz_init, jnp.array([theta]), v, omega])

        self._nj = mjx_model.nv - 6

        self._index_variables_qp = {
            'com_pos': 0,     'com_vel': 3,     'com_acc': 6,
            'lwheel_pos': 9,  'lwheel_vel': 12, 'lwheel_acc': 15,
            'rwheel_pos': 18, 'rwheel_vel': 21, 'rwheel_acc': 24,
            'base_rot': 27,   'base_omega': 36, 'base_alpha': 39,
            'joints': 42,
        }
        self._desired_size = self._index_variables_qp['joints'] + 3 * self._nj

        posture_joint_ids = getattr(config, "posture_joint_ids", None)
        if posture_joint_ids is None:
            self._posture_mask = None
        else:
            self._posture_mask = jnp.zeros(self._nj).at[jnp.array(list(posture_joint_ids))].set(1.0)

        cost     = partial(mpc_objectives.wheeled_dfcip_obj, config.d, config.N)
        hess     = partial(mpc_objectives.wheeled_dfcip_hessian_gn, config.d, config.N)
        dynamics = partial(mpc_dyn_model.wheeled_dfcip_dynamics, mjx_model, config.mass, config.grav, config.dt_mpc)
        fddp     = partial(optimizers.fddp_mpc, cost, dynamics, hess, False)
        ref_gen  = partial(mpc_utils.reference_generator_dfcip_online, config.N, config.dt_mpc, config.mass, config.grav, config.h_min, config.h_max)

        def work(reference, parameter, W, x0, X_init, U_init):
            X_it, U_it, D_it = X_init, U_init, None
            for _ in range(self.mpc_iterations):
                X_it, U_it, D_it = fddp(reference, parameter, W, x0, X_it, U_it)
            return X_it, U_it, D_it

        dt_wbc = config.dt_wbc
        n_contacts = 1

        def whole_body_control(qpos, qvel, desired):
            return mpc_utils.whole_body_interface_wheeled_legged_qp(
                mjx_model, config.mass, config.grav, config.d,
                contact_id, body_id, base_body_id,
                config.wheel_radius, dt_wbc, n_contacts,
                config.Kp_motion, config.Kd_motion, config.Kp_wheel, config.Kd_wheel, config.Kp_reg, config.Kd_reg,
                config.w_posture,
                config.w_qddot, config.w_force, config.w_com, config.w_lwheel, config.w_rwheel, config.w_base,
                config.mu,
                qpos, qvel, desired,
                posture_mask=self._posture_mask,
                ref_layout=self._index_variables_qp,
            )

        self._solve                = jax.jit(jax.vmap(work))
        self._ref_gen              = jax.jit(jax.vmap(ref_gen))
        self._whole_body_interface = jax.jit(jax.vmap(whole_body_control))
        self._build_desired_jit    = jax.jit(self._build_desired_impl)
        self._process_state_jit    = jax.jit(self._process_state_impl)

        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        D0 = jnp.zeros((config.N + 1, config.nx))
        self._U0_init = jnp.tile(U0, (n_env, 1, 1))
        self._X0_init = jnp.tile(X0, (n_env, 1, 1))
        self._D0_init = jnp.tile(D0, (n_env, 1, 1))

    def init_state(self, x0: jax.Array | None = None) -> MPCState:
        n, cfg = self.n_env, self.config
        a     = jnp.full((n,), cfg.u_ref[0])
        ac_z  = jnp.full((n,), cfg.u_ref[1])
        alpha = jnp.full((n,), cfg.u_ref[2])
        grf   = jnp.tile(cfg.u_ref[3:], (n, 1))

        if x0 is None:
            X_init = self._X0_init
        else:
            x0 = jnp.asarray(x0)
            if x0.ndim == 1:
                x0 = x0[None, :]
            X_init = jnp.broadcast_to(x0[:, None, :], (self.n_env, cfg.N + 1, cfg.nx))

        return MPCState(
            sol=ControlSol(a=a, ac_z=ac_z, alpha=alpha, grf=grf),
            X0_shifted=X_init,
            U0_shifted=self._U0_init,
            D0_shifted=self._D0_init,
        )

    def process_state(self, pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):
        return self._process_state_jit(pcom, vcom, centers, Rs, radii, feet_vel, theta_prev)

    def run(self, state: MPCState, x0, cmd) -> MPCState:
        cfg = self.config

        x_ref, u_ref = self._ref_gen(x0, cmd)
        reference = jnp.concatenate([x_ref, u_ref], axis=-1)
        parameter = None

        W = jnp.tile(cfg.W, (self.n_env, 1, 1))
        X, U, D = self._solve(reference, parameter, W, x0, state.X0_shifted, state.U0_shifted)

        new_a     = U[:, 0, 0]
        new_ac_z  = U[:, 0, 1]
        new_alpha = U[:, 0, 2]
        new_grf   = U[:, 0, 3:]

        D = jnp.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

        s = self.shift
        new_X0 = jnp.concatenate([X[:, s:, :], jnp.tile(X[:, -1:, :], (1, s, 1))], axis=1)
        new_U0 = jnp.concatenate([U[:, s:, :], jnp.tile(U[:, -1:, :], (1, s, 1))], axis=1)
        new_D0 = jnp.concatenate([D[:, s:, :], jnp.tile(D[:, -1:, :], (1, s, 1))], axis=1)

        new_state = MPCState(
            sol=ControlSol(a=new_a, ac_z=new_ac_z, alpha=new_alpha, grf=new_grf),
            X0_shifted=new_X0,
            U0_shifted=new_U0,
            D0_shifted=new_D0,
        )
        return new_state, reference

    def whole_body_run(self, state: MPCState, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world):
        desired = self._build_desired_jit(x0, qpos, state, pl_world, pr_world, dpl_world, dpr_world)
        tau_cmd, qddot, fl, fr = self._whole_body_interface(qpos, qvel, desired)
        return state, tau_cmd, qddot, fl, fr, desired

    def reset(self) -> MPCState:
        print("MPC Controller Reset")
        return self.init_state()

    def _process_state_impl(self, pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):
        cfg = self.config

        l_rcp = mpc_utils.get_rCP(Rs[0], radii[0])
        r_rcp = mpc_utils.get_rCP(Rs[1], radii[1])

        pl_world = centers[0] + l_rcp
        pr_world = centers[1] + r_rcp
        dpl_world, dpr_world = feet_vel[0], feet_vel[1]

        tita_state = jnp.concatenate([pcom, vcom, pl_world, pr_world, dpl_world, dpr_world])

        c_world = (pl_world + pr_world) / 2.0
        vc_world = (dpl_world + dpr_world) / 2.0

        diff = centers[0] - centers[1]
        theta_wrapped = jnp.arctan2(-diff[0], diff[1])
        a = (theta_wrapped - theta_prev + jnp.pi) % (2 * jnp.pi)
        a = jnp.where(a < 0, a + 2 * jnp.pi, a) - jnp.pi
        theta = theta_prev + a

        ct, st = jnp.cos(theta), jnp.sin(theta)
        R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        dpl_b = R.T @ dpl_world
        dpr_b = R.T @ dpr_world
        w = (dpr_b[0] - dpl_b[0]) / cfg.d
        v = (dpr_b[0] + dpl_b[0]) / 2.0

        x0 = jnp.concatenate([
            pcom,
            vcom,
            c_world,
            jnp.array([vc_world[2]]),
            jnp.array([theta]),
            jnp.array([v]),
            jnp.array([w]),
        ])
        return tita_state, x0, theta

    def _build_desired_impl(self, x0, qpos, state, pl_world, pr_world, dpl_world, dpr_world):
        cfg = self.config
        B = qpos.shape[0]
        desired = jnp.zeros((B, self._desired_size))
        dt = self.wbc_lookahead_dt

        x_mpc = x0[None, :]
        u_mpc = jnp.concatenate([
            state.sol.a[:, None],
            state.sol.ac_z[:, None],
            state.sol.alpha[:, None],
            state.sol.grf,
        ], axis=1)

        a     = u_mpc[:, 0]
        ac_z  = u_mpc[:, 1]
        alpha = u_mpc[:, 2]
        fcl   = u_mpc[:, 3:6]
        fcr   = u_mpc[:, 6:9]

        pcom_curr  = x_mpc[:, 0:3]
        vcom_curr  = x_mpc[:, 3:6]
        pl_curr    = pl_world
        pr_curr    = pr_world
        dpl_curr   = dpl_world
        dpr_curr   = dpr_world
        theta_curr = x_mpc[:, 10]
        v_curr     = x_mpc[:, 11]
        w_curr     = x_mpc[:, 12]

        cos_t = jnp.cos(theta_curr)
        sin_t = jnp.sin(theta_curr)

        g_vec = jnp.array([0.0, 0.0, -cfg.grav])
        vector_off = jnp.array([0.0, cfg.d / 2.0, 0.0])

        dR_curr = jnp.stack([
            jnp.stack([-sin_t, -cos_t, jnp.zeros((B,))], axis=1),
            jnp.stack([ cos_t, -sin_t, jnp.zeros((B,))], axis=1),
            jnp.zeros((B, 3)),
        ], axis=1)
        ddR_curr = jnp.stack([
            jnp.stack([-cos_t,  sin_t, jnp.zeros((B,))], axis=1),
            jnp.stack([-sin_t, -cos_t, jnp.zeros((B,))], axis=1),
            jnp.zeros((B, 3)),
        ], axis=1)

        ddc = jnp.stack([
            a * cos_t - v_curr * sin_t * w_curr,
            a * sin_t + v_curr * cos_t * w_curr,
            ac_z,
        ], axis=1)

        acc_com_ = (fcl + fcr) / cfg.mass + g_vec[None, :]
        vel_com_ = vcom_curr + dt * acc_com_
        pos_com_ = pcom_curr + dt * vcom_curr

        acc_pl_ = ddc + (
            jnp.einsum('bij,j->bi', ddR_curr, vector_off) * w_curr[:, None] * w_curr[:, None]
            + jnp.einsum('bij,j->bi', dR_curr, vector_off) * alpha[:, None]
        )
        vel_pl_ = dpl_curr + dt * acc_pl_
        pos_pl_ = pl_curr + dt * dpl_curr

        acc_pr_ = ddc - (
            jnp.einsum('bij,j->bi', ddR_curr, vector_off) * w_curr[:, None] * w_curr[:, None]
            + jnp.einsum('bij,j->bi', dR_curr, vector_off) * alpha[:, None]
        )
        vel_pr_ = dpr_curr + dt * acc_pr_
        pos_pr_ = pr_curr + dt * dpr_curr

        alpha_ = alpha
        omega_ = w_curr + dt * alpha_
        theta_ = theta_curr + dt * w_curr

        com_pos_ref = pos_com_
        com_vel_ref = vel_com_
        com_acc_ref = acc_com_

        lwheel_pos_ref = pos_pl_.at[:, 2].set(pos_pl_[:, 2] + cfg.wheel_radius)
        lwheel_vel_ref = vel_pl_
        lwheel_acc_ref = acc_pl_
        rwheel_pos_ref = pos_pr_.at[:, 2].set(pos_pr_[:, 2] + cfg.wheel_radius)
        rwheel_vel_ref = vel_pr_
        rwheel_acc_ref = acc_pr_

        cos_theta_ = jnp.cos(theta_)
        sin_theta_ = jnp.sin(theta_)
        R_theta = jnp.stack([
            jnp.stack([cos_theta_, -sin_theta_, jnp.zeros((B,))], axis=1),
            jnp.stack([sin_theta_,  cos_theta_, jnp.zeros((B,))], axis=1),
            jnp.stack([jnp.zeros((B,)), jnp.zeros((B,)), jnp.ones((B,))], axis=1),
        ], axis=1)
        base_rot_ref   = R_theta.reshape(B, 9)
        base_omega_ref = jnp.zeros((B, 3)).at[:, 2].set(omega_)
        base_alpha_ref = jnp.zeros((B, 3)).at[:, 2].set(alpha_)

        qjnt_ref     = jnp.tile(cfg.q0, (B, 1))
        qjntdot_ref  = jnp.zeros((B, self._nj))
        qjntddot_ref = jnp.zeros((B, self._nj))

        nj = self._nj
        idx = self._index_variables_qp
        desired = desired.at[:, idx['com_pos']:idx['com_pos'] + 3].set(com_pos_ref)
        desired = desired.at[:, idx['com_vel']:idx['com_vel'] + 3].set(com_vel_ref)
        desired = desired.at[:, idx['com_acc']:idx['com_acc'] + 3].set(com_acc_ref)
        desired = desired.at[:, idx['lwheel_pos']:idx['lwheel_pos'] + 3].set(lwheel_pos_ref)
        desired = desired.at[:, idx['rwheel_pos']:idx['rwheel_pos'] + 3].set(rwheel_pos_ref)
        desired = desired.at[:, idx['lwheel_vel']:idx['lwheel_vel'] + 3].set(lwheel_vel_ref)
        desired = desired.at[:, idx['rwheel_vel']:idx['rwheel_vel'] + 3].set(rwheel_vel_ref)
        desired = desired.at[:, idx['lwheel_acc']:idx['lwheel_acc'] + 3].set(lwheel_acc_ref)
        desired = desired.at[:, idx['rwheel_acc']:idx['rwheel_acc'] + 3].set(rwheel_acc_ref)
        desired = desired.at[:, idx['base_rot']:idx['base_rot'] + 9].set(base_rot_ref)
        desired = desired.at[:, idx['base_omega']:idx['base_omega'] + 3].set(base_omega_ref)
        desired = desired.at[:, idx['base_alpha']:idx['base_alpha'] + 3].set(base_alpha_ref)
        desired = desired.at[:, idx['joints']:idx['joints'] + nj].set(qjnt_ref)
        desired = desired.at[:, idx['joints'] + nj:idx['joints'] + 2 * nj].set(qjntdot_ref)
        desired = desired.at[:, idx['joints'] + 2 * nj:idx['joints'] + 3 * nj].set(qjntddot_ref)

        return desired