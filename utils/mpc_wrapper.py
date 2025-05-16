import jax
from jax import jit
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
        data = mujoco.MjData(model)
        mujoco.mj_fwdPosition(model, data)
        robot_mass = data.qM[0]
        mjx_model = mjx.put_model(model)
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        # Timer and liftoff states for the reference generator.
        self.q0 = config.q0.copy()          # Initial joint configuration
        
        if config.grf_as_state:
            self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
        else:
            self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0])
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
        cost = partial(config.cost,
                            config.n_joints, config.n_contact, config.N)
        hessian_approx = partial(config.hessian_approx,
                            config.n_joints, config.n_contact)
        self.dynamics = partial(config.dynamics,
                                model, mjx_model, self.contact_id, self.body_id,
                                config.n_joints, config.dt)
    
        work = partial(optimizers.mpc, cost, self.dynamics, hessian_approx, False)
        
        reference_generator = partial(mpc_utils.reference_generator,
            config.use_terrain_estimation ,config.N, config.dt, config.n_joints, config.n_contact, robot_mass, foot0 = config.p_legs0, q0 = config.q0)
        
        timer_t = partial(mpc_utils.timer_run, duty_factor=config.duty_factor, step_freq=config.step_freq)

        self._solve = jax.jit(jax.vmap(work, in_axes = (0,0,None,0,0,0,0)))
        self._ref_gen = jax.jit(jax.vmap(reference_generator))
        self._timer_run = jax.jit(jax.vmap(mpc_utils.timer_run, in_axes=(0,0,0, None)))
        
        self.contact_time = jnp.tile(config.timer_t, (n_env, 1))
        self.liftoff = jnp.zeros((n_env, 3*config.n_contact))

        self.duty_factor = jnp.tile(config.duty_factor, (n_env, 1))
        self.step_freq = jnp.tile(config.step_freq, (n_env, 1))
        self.step_height = jnp.tile(config.step_height, (n_env, 1))
        
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
        _ , self.contact_time = self._timer_run(self.duty_factor,self.step_freq,self.contact_time,1/self.mpc_frequency)
        
        # Generate reference trajectory and additional MPC parameters.
        reference, parameter, self.liftoff = self._ref_gen(
            duty_factor = self.duty_factor,
            step_freq = self.step_freq,
            step_height = self.step_height,
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
            self.config.W,
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

        X, U, _ = self.run(x0, input, foot_op)

        tau = torch_dlpack.from_dlpack(U[:,0,:self.config.n_joints])
        q = torch_dlpack.from_dlpack(X[:,1,7:self.config.n_joints+7])
        dq = torch_dlpack.from_dlpack(X[:,1,13+self.config.n_joints:2*self.config.n_joints+13])

        return tau , q, dq
    
    def reset(self,envs,foot):
        """
        Resets the MPC controller state."
        """
        n_env_reset = envs.shape[0]
        self.contact_time = self.contact_time.at[envs,:].set(jnp.tile(self.config.timer_t, (n_env_reset, 1)))
        self.liftoff = self.liftoff.at[envs,:].set(foot)
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
        mujoco.mj_fwdPosition(self.model, self.data)
        robot_mass = self.data.qM[0]
        mjx_model = mjx.put_model(self.model)
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        
        # Timer and liftoff states for the reference generator.
        self.foot0 = config.p_legs0.copy()  # Initial foot positions (could be adjusted if needed)
        self.q0 = config.q0.copy()          # Initial joint configuration

        if self.config.grf_as_state:
            self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
        else:
            self.initial_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0])
        
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
        self.cost = partial(config.cost,config.n_joints, config.n_contact, config.N)
        hessian_approx = partial(config.hessian_approx,config.n_joints, config.n_contact)
        self.dynamics = partial(config.dynamics,
                                self.model, mjx_model, self.contact_id, self.body_id,
                                config.n_joints, config.dt)
    
        work = partial(optimizers.mpc, self.cost, self.dynamics, hessian_approx, False)
        
        reference_generator = partial(mpc_utils.reference_generator,
            config.use_terrain_estimation ,config.N, config.dt, config.n_joints, config.n_contact, robot_mass,foot0 = config.p_legs0, q0 = config.q0)
        
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
                dq = X[1, 13 + n_joints:2 * n_joints + 13]
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
        self.contact = np.zeros(config.n_contact)
        # self.obstacle_timer = 0
        self.clearence_speed = 0.4 #* jnp.ones(config.n_contact)
    def run(self, qpos, qvel, input,contact):
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
        self.contact = contact.copy()
        #get forward kinematics for foot position 
       
        self.data.qpos = qpos 
        
        mujoco.mj_kinematics(self.model, self.data)
        foot_op = np.array([self.data.geom_xpos[self.contact_id[i]] for i in range(self.config.n_contact)]).flatten()
        #set initial state
        input[6] = self.robot_height

        if self.config.grf_as_state:
            x0 = jnp.concatenate([qpos, qvel,foot_op,jnp.zeros(3*self.config.n_contact)])
        else:
            x0 = jnp.concatenate([qpos, qvel,foot_op])

        input = jnp.array(input)
        contact = jnp.array(contact)

        # Update the timer state for the gait reference.
        des_contact , self.contact_time = self._timer_run(self.duty_factor,self.step_freq,self.contact_time,1/self.mpc_frequency)

        # Generate reference trajectory and additional MPC parameters.
        reference, parameter, self.liftoff = self._ref_gen(
            duty_factor = self.duty_factor,
            step_freq = self.step_freq,
            step_height = self.step_height,
            t_timer = self.contact_time.copy(),
            x = x0,
            foot = foot_op,
            input = input,
            liftoff = self.liftoff,
            contact = contact,
            clearence_speed = self.clearence_speed,
        )
        
        # Execute the MPC optimization.
        X, U, V = self._solve(
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
        tau = np.clip(np.array(tau_temp),self.config.min_torque,self.config.max_torque)
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
        if self.config.grf_as_state:
            x0 = jnp.concatenate([qpos, qvel,foot_op.flatten(),jnp.zeros(3*self.config.n_contact)])
        else:
            x0 = jnp.concatenate([qpos, qvel,foot_op.flatten()])
        self.contact_time = self.config.timer_t
        self.liftoff = foot_op.flatten()
        self.U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
        self.X0 = jnp.tile(x0, (self.config.N + 1, 1))
        self.V0 = jnp.zeros((self.config.N + 1, self.config.n))
        print("MPC Controller Reset")
        return
    
    def runOffline(self, qpos, qvel):
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

        x0 = jnp.concatenate([qpos, qvel,foot_op.flatten(),jnp.zeros(3*self.config.n_contact)])

        

        reference, parameter = self.config.reference(self.config.N + 1,self.config.dt,self.config.n_joints,self.config.n_contact,self.config.p_legs0,self.config.q0)
        # Warm start 
        self.X0 = self.X0.at[:,:13+self.config.n_joints].set(reference[:,:13+self.config.n_joints])

        _cost = partial(self.cost,self.config.W,reference)
        _dynamics = partial(self.dynamics,parameter=parameter)
        model_evaluator = partial(optimizers.model_evaluator_helper, _cost, _dynamics,x0)

        _exit = False
        max_iter = 100
        last_cost = 1e10
        i = 0
        while not _exit:
            start = timer()

            X, U, V = self._solve(
                reference,
                parameter,
                self.config.W,
                x0,
                self.X0,
                self.U0,
                self.V0
                )
            
            X.block_until_ready()

            self.X0 = X
            self.U0 = U
            self.V0 = V

            g, c = model_evaluator(X,U)

            stop = timer()

            l2_cost = np.sum(g*g)
            
            if i == 0:
                print("{:<10} {:<20} {:<20} {:<20}".format("Iter", "Cost", "Constraint", "Time Elapsed"))
            print("{:<10d} {:<20.5f} {:<20.5f} {:<20.5f}".format(i, l2_cost, np.sum(c*c), stop-start))
            i += 1
            
            if i > max_iter:
                _exit = True
            if last_cost - l2_cost < 1e-3 and np.sum(c*c) < 1e-5:
                _exit = True
            last_cost = l2_cost

        return self.X0, self.U0, reference