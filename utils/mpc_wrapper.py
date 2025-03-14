import jax
import jax.numpy as jnp
from functools import partial
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import utils.mpc_utils as mpc_utils
import utils.models as mpc_dyn_model
import utils.objectives as mpc_objectives
import utils.config as config
import mujoco 
from mujoco import mjx
import primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
import numpy as np
from jax import dlpack as jax_dlpack
# Try to import torch for dlpack conversion, but continue if torch is not available
try:
    from torch.utils import dlpack as torch_dlpack
except ImportError:
    torch_dlpack = None
    print("Warning: torch not installed. torch_run functionality will not be available.")

class BatchedMPCControllerWrapper:
    def __init__(self, config, n_env):
        """
        Initializes the MPC controller wrapper.
        
        Args:
            config: Configuration object containing MPC and gait parameters.
            mpc_frequency: Frequency (Hz) at which MPC updates occur.
        """
        jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        self.n_env = n_env
        model = mujoco.MjModel.from_xml_path(config.model_path)
        mjx_model = mjx.put_model(model)
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        
        # Timer and liftoff states for the reference generator.
        self.foot0 = config.p_legs0.copy()  # Initial foot positions (could be adjusted if needed)
        self.q0 = config.q0.copy()          # Initial joint configuration
        
        self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
        # Get contact and body IDs from configuration
        contact_id = []
        for name in config.contact_frame:
            contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
        body_id = []
        for name in config.body_name:
            body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
        # Trajectory warm-start variables (used between MPC calls)
        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
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
    
        work = partial(optimizers.mpc, cost, dynamics, hessian_approx, True)
        
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
            input: Input 
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

    def torch_run(self, x0_torch, input_torch, foot_op_torch):
        #Runs one MPC update using the current state, input, and foot positions.
    
        x0 = jax_dlpack.from_dlpack(x0_torch)
        input = jax_dlpack.from_dlpack(input_torch)
        foot_op = jax_dlpack.from_dlpack(foot_op_torch)

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

        tau = torch_dlpack.from_dlpack(U[:,0,:self.config.n_joints])
        q = torch_dlpack.from_dlpack(X[:,1,7:self.config.n_joints+7])
        dq = torch_dlpack.from_dlpack(X[:,1,13+self.config.n_joints:2*self.config.n_joints+13])
        return tau , q, dq
    
    def reset(self):
        """
        Resets the MPC controller state."
        """
        self.contact_time = jnp.tile(self.config.timer_t, (self.n_env, 1))
        self.liftoff = jnp.zeros((self.n_env, 3*self.config.n_contact))
        U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
        X0 = jnp.tile(self.initial_state, (self.config.N + 1, 1))
        V0 = jnp.zeros((self.config.N + 1, self.config.n))
        self.batch_U0 = jnp.tile(U0, (self.n_env, 1, 1))
        self.batch_X0 = jnp.tile(X0, (self.n_env, 1, 1))
        self.batch_V0 = jnp.tile(V0, (self.n_env, 1, 1))
        print("MPC Controller Reset")
        return