import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import utils.mpc_utils as mpc_utils
import utils.models as mpc_dyn_model
import utils.objectives as mpc_objectives
import mujoco 
from mujoco import mjx
import primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
import numpy as np
from jax import dlpack as jax_dlpack
from jax.scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
# Try to import torch for dlpack conversion, but continue if torch is not available
from timeit import default_timer as timer
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
        self.contact_id = []
        for name in config.contact_frame:
            self.contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
        self.body_id = []
        for name in config.body_name:
            self.body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
        # Trajectory warm-start variables (used between MPC calls)
        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        V0 = jnp.zeros((config.N + 1, config.n))
        
        self.batch_U0 = jnp.tile(U0, (n_env, 1, 1))
        self.batch_X0 = jnp.tile(X0, (n_env, 1, 1))
        self.batch_V0 = jnp.tile(V0, (n_env, 1, 1))
        
        # Define cost, hessian approximation, and dynamics functions for MPC.
        cost = partial(mpc_objectives.quadruped_wb_obj,
                            config.W, config.p_legs0,config.n_joints, config.n_contact, config.N)
        hessian_approx = partial(mpc_objectives.quadruped_wb_hessian_gn,
                                      config.W,config.p_legs0, config.n_joints, config.n_contact)
        dynamics = partial(mpc_dyn_model.quadruped_wb_dynamics,
                                model, mjx_model, self.contact_id, self.body_id,
                                config.n_joints, config.dt)
    
        work = partial(optimizers.mpc, cost, dynamics, hessian_approx, False)
        
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
        X, U, V,_ = self._solve(
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
    
    def reset(self,envs):
        """
        Resets the MPC controller state."
        """
        envs = jax_dlpack.from_dlpack(envs)
        n_env_reset = envs.shape[0]
        self.contact_time = self.contact_time.at[envs,:].set(jnp.tile(self.config.timer_t, (n_env_reset, 1)))
        self.liftoff = self.liftoff.at[envs,:].set(jnp.zeros((n_env_reset, 3*self.config.n_contact)))
        U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
        X0 = jnp.tile(self.initial_state, (self.config.N + 1, 1))
        V0 = jnp.zeros((self.config.N + 1, self.config.n))
        self.batch_U0 = self.batch_U0.at[envs,:,:].set(jnp.tile(U0, (n_env_reset, 1, 1)))
        self.batch_X0 = self.batch_X0.at[envs,:,:].set(jnp.tile(X0, (n_env_reset, 1, 1)))
        self.batch_V0 = self.batch_V0.at[envs,:,:].set(jnp.tile(V0, (n_env_reset, 1, 1)))
        print("MPC Controller Reset")
        return

class MPCControllerWrapper:
    def __init__(self, config):
        """
        Initializes the MPC controller wrapper.
        
        Args:
            config: Configuration object containing MPC and gait parameters.
            mpc_frequency: Frequency (Hz) at which MPC updates occur.
        """
        jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.data = mujoco.MjData(self.model)
        mjx_model = mjx.put_model(self.model)
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        
        # Timer and liftoff states for the reference generator.
        self.foot0 = config.p_legs0.copy()  # Initial foot positions (could be adjusted if needed)
        self.q0 = config.q0.copy()          # Initial joint configuration
        
        self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
        # Get contact and body IDs from configuration
        self.contact_id = []
        for name in config.contact_frame:
            self.contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
        self.body_id = []
        for name in config.body_name:
            self.body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
        # Trajectory warm-start variables (used between MPC calls)
        self.U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.V0 = jnp.zeros((config.N + 1, config.n))
        
        # Define cost, hessian approximation, and dynamics functions for MPC.
        cost = partial(mpc_objectives.quadruped_wb_obj,config.n_joints, config.n_contact, config.N)
        hessian_approx = partial(mpc_objectives.quadruped_wb_hessian_gn,config.n_joints, config.n_contact)
        dynamics = partial(mpc_dyn_model.quadruped_wb_dynamics,
                                self.model, mjx_model, self.contact_id, self.body_id,
                                config.n_joints, config.dt)
    
        work = partial(optimizers.mpc, cost, dynamics, hessian_approx, False)
        
        reference_generator = partial(mpc_utils.reference_generator,
            config.N, config.dt, config.n_joints, config.n_contact, foot0 = config.p_legs0, q0 = config.q0)
        
        timer_t = partial(mpc_utils.timer_run, duty_factor=config.duty_factor, step_freq=config.step_freq)

        self._solve = jax.jit(work)
        self._ref_gen = jax.jit(reference_generator)
        self._timer_run = jax.jit(mpc_utils.timer_run)
        

        self.contact_time = config.timer_t
        self.liftoff = config.p_legs0.copy()

        self.tau = jnp.zeros(config.n_joints)
        self.q = jnp.zeros(config.n_joints)
        self.dq = jnp.zeros(config.n_joints)

        @partial(jax.jit, static_argnums=(0,1))
        def update_and_extract_helper(n_joints,shift,U, X, V, x0, X0, U0):
            def safe_update():
                new_U0 = jnp.concatenate([U[shift:], jnp.tile(U[-1:], (shift, 1))])
                new_X0 = jnp.concatenate([X[shift:], jnp.tile(X[-1:], (shift, 1))])
                new_V0 = jnp.concatenate([V[shift:], jnp.tile(V[-1:], (shift, 1))])
                tau = U[0, :n_joints]
                q = X[0, 7:n_joints + 7]
                dq = X[0, 13 + n_joints:2 * n_joints + 13]
                return new_U0, new_X0, new_V0,tau,q ,dq
            def unsafe_update():
                new_U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
                new_X0 = jnp.tile(x0, (self.config.N + 1, 1))
                new_V0 = jnp.zeros((self.config.N + 1, self.config.n ))  
                tau = U0[1, :n_joints]
                q = X0[1, 7:n_joints + 7]
                dq = X0[1, 13 + n_joints:2 * n_joints + 13]
                return new_U0, new_X0, new_V0, tau, q, dq
        
            return jax.lax.cond(jnp.isnan(U[0,0]),unsafe_update,safe_update)
        update_and_extract = partial(update_and_extract_helper,self.config.n_joints,self.shift)
        self.update_and_extract = jax.jit(update_and_extract)

        self.duty_factor = config.duty_factor
        self.step_freq = config.step_freq
        self.step_height = config.step_height
        self.robot_height = config.initial_height
        self.tau0 = np.zeros(config.n_joints)
        self.start_time = 0
        # self.config.W = self.config.W.at[12:18,12:18].set(jnp.diag(jnp.ones(6)) * 1e2)
        # self.config.W = self.config.W.at[42:48 ,42:48].set(jnp.diag(jnp.ones(6)) * 1e0)
    
    def run(self, qpos, qvel, input):
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
        #compensate for the time delay
        #get forward kinematics for foot position 
       
        self.data.qpos = qpos 
       
        mujoco.mj_kinematics(self.model, self.data)
        foot_op = np.array([self.data.geom_xpos[self.contact_id[i]] for i in range(self.config.n_contact)])
        #set initial state
        input[6] = self.robot_height
        x0 = jnp.concatenate([qpos, qvel,foot_op.flatten(),jnp.zeros(3*self.config.n_contact)])
        input = jnp.array(input)



        # start = timer()
       
        # start = timer()
        # Update the timer state for the gait reference.
        _ , self.contact_time = self._timer_run(self.duty_factor,self.step_freq,self.contact_time,1/self.mpc_frequency)
        # Generate reference trajectory and additional MPC parameters.
        reference, parameter, self.liftoff = self._ref_gen(
            duty_factor = self.duty_factor,
            step_freq = self.step_freq,
            step_height = self.step_height,
            t_timer = self.contact_time.copy(),
            x = x0,
            foot = foot_op.flatten(),
            input = input,
            liftoff = self.liftoff,
        )
        # Execute the MPC optimization (work function).
        X, U, V, _ = self._solve(
            reference,
            parameter,
            self.config.W,
            x0,
            self.X0,
            self.U0,
            self.V0
            )
        X.block_until_ready()
       
        # # Warm-start for the next call: shift trajectories forward.   
    
        self.U0, self.X0, self.V0, tau_temp, q_temp, dq_temp = self.update_and_extract(U, X, V, x0, self.X0, self.U0)

        # TO DO change to values from config
        tau = np.clip(np.array(tau_temp),-44,44)
        q = np.array(q_temp)
        dq = np.array(dq_temp)

        return tau, q, dq
    
    def reset(self,qpos,qvel):
        """
        Resets the MPC controller state."
        """
        self.data.qpos = qpos 
        # self.data.qvel = qvel 
        mujoco.mj_kinematics(self.model, self.data)
        foot_op = np.array([self.data.geom_xpos[self.contact_id[i]] for i in range(self.config.n_contact)])
        x0 = jnp.concatenate([qpos, qvel,foot_op.flatten(),jnp.zeros(3*self.config.n_contact)])
        self.contact_time = self.config.timer_t
        self.liftoff = foot_op.flatten()
        self.U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
        self.X0 = jnp.tile(x0, (self.config.N + 1, 1))
        self.V0 = jnp.zeros((self.config.N + 1, self.config.n))
        print("MPC Controller Reset")
        return