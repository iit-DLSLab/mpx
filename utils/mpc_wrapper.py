import jax
import jax.numpy as jnp
from functools import partial
import utils.mpc_utils as mpc_utils
import utils.models as mpc_dyn_model
import utils.objectives as mpc_objectives
import utils.config as config
import os
import sys
import mujoco 
from mujoco import mjx
import primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class BatchedMPCControllerWrapper:
    def __init__(self, config, n_env):
        """
        Initializes the MPC controller wrapper.
        
        Args:
            config: Configuration object containing MPC and gait parameters.
            mpc_frequency: Frequency (Hz) at which MPC updates occur.
        """
        self.n_env = n_env
        model = mujoco.MjModel.from_xml_path(config.model_path)
        mjx_model = mjx.put_model(model)
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        
        # Timer and liftoff states for the reference generator.
        self.foot0 = config.p_legs0.copy()  # Initial foot positions (could be adjusted if needed)
        self.q0 = config.q0.copy()          # Initial joint configuration
        
        initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
        # Get contact and body IDs from configuration
        contact_id = []
        for name in config.contact_frame:
            contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
        body_id = []
        for name in config.body_name:
            body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
        # Trajectory warm-start variables (used between MPC calls)
        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(initial_state, (config.N + 1, 1))
        V0 = jnp.zeros((config.N + 1, config.n))
        
        self.batch_U0 = jnp.tile(U0, (n_env, 1, 1))
        self.batch_X0 = jnp.tile(X0, (n_env, 1, 1))
        self.batch_V0 = jnp.tile(V0, (n_env, 1, 1))
        
        # Define cost, hessian approximation, and dynamics functions for MPC.
        cost = partial(mpc_objectives.quadruped_wb_obj,
                            config.W, config.n_joints, config.n_contact, config.N)
        hessian_approx = partial(mpc_objectives.quadruped_wb_hessian_gn,
                                      config.W, config.n_joints, config.n_contact)
        dynamics = partial(mpc_dyn_model.quadruped_wb_dynamics,
                                model, mjx_model, contact_id, body_id,
                                config.n_joints, config.dt)
    
        work = partial(optimizers.mpc, cost, dynamics, hessian_approx, False)\
        
        reference_generator = partial(mpc_utils.reference_generator,
            config.N, config.dt, config.n_joints, config.n_contact,
            duty_factor = config.duty_factor,  step_freq= config.step_freq ,step_height=config.step_height,foot0 = config.p_legs0, q0 = config.q0)
        
        timer_t = partial(mpc_utils.timer_run, duty_factor=config.duty_factor, step_freq=config.step_freq)

        self._solve = jax.vmap(work)
        self._ref_gen = jax.vmap(reference_generator)
        self._timer_run = jax.vmap(mpc_utils.timer_run, in_axes=(None,None,0, None))
        

        self.contact_time = jnp.tile(config.timer_t, (n_env, 1))
        self.liftoff = jnp.zeros((n_env, 3*config.n_contact))

        
    def run(self, x0, input, foot_op):
        """
        Runs one MPC update using the current state, input, and foot positions.
        
        Args:
            x0: Current system state vector.
            input: Input tuple (e.g., (ref_base_lin_vel, ref_base_ang_vel, robot_height)).
            foot_op: Flattened current foot positions vector.
        
        Returns:
            A tuple (X, U, V) representing the computed state trajectory, control sequence,
            and auxiliary variable trajectory.
        """
        # Update the timer state for the gait reference.
        
        _ , self.contact_time = self._timer_run(self.config.duty_factor,self.config.step_freq,self.contact_time,1/self.mpc_frequency)
        
        # Generate reference trajectory and additional MPC parameters.
        reference, parameter, self.liftoff = self._ref_gen(
            t_timer = self.contact_time.copy(),
            x = x0,
            foot = foot_op,
            input = input,
            liftoff = self.liftoff,
        )
        # Execute the MPC optimization (work function).
        X, U, V = self._solve(
            reference,
            parameter,
            x0,
            self.batch_X0,
            self.batch_U0,
            self.batch_V0
            )
        
        # Warm-start for the next call: shift trajectories forward.
        self.batch_X0 = jnp.concatenate([X[:,self.shift:,:], jnp.tile(X[:,-1:,:], (self.shift, 1))],axis = 1)
        self.batch_U0 = jnp.concatenate([U[:,self.shift:,:], jnp.tile(U[:,-1:,:], (self.shift, 1))],axis = 1)
        self.batch_V0 = jnp.concatenate([V[:,self.shift:,:], jnp.tile(V[:,-1:,:], (self.shift, 1))],axis = 1)
        
        return X, U, V