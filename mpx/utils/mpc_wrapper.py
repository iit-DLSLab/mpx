import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import mujoco
from mujoco import mjx
import mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
import numpy as np
from mujoco.mjx._src.dataclasses import PyTreeNode 
# Try to import torch for dlpack conversion, but continue if torch is not available
from timeit import default_timer as timer
# MJX style class to store all the data needed for the MPC controller   
class mpx_data(PyTreeNode):

    dt : float # Time step for the simulation
    duty_factor: float # Duty factor for the gait cycle
    step_freq: float # Step frequency for the gait cycle
    step_height: float # Step height for the gait cycle
    contact_time: jnp.ndarray # timer state for each foot in the gait cycle
    liftoff: jnp.ndarray # Liftoff position for each foot in the gait cycle
    X0: jnp.ndarray  # Initial state trajectory for the MPC controller
    U0: jnp.ndarray # Initial control input trajectory for the MPC controller
    V0: jnp.ndarray # Initial lagrangian for the MPC controller
    W : jnp.ndarray # Weight matrix for the MPC controller

class BatchedMPCControllerWrapper:
    def __init__(self, config, n_env,limited_memory=False):
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
        # Define cost, hessian approximation, and dynamics functions for MPC.
        cost = config.cost
        hessian_approx = config.hessian_approx
        dynamics = partial(config.dynamics,
                                model, mjx_model, self.contact_id, self.body_id,
                                config.n_joints, config.dt)

        self.work = partial(optimizers.mpc, cost, dynamics, hessian_approx, limited_memory)

        self.reference_generator = partial(mpc_utils.simple_reference_generator,
            config.use_terrain_estimation ,config.N, config.dt, config.n_contact, q0 = config.q0)

        def update_and_extract_helper(U, X, V, x0, X0, U0):
            def safe_update():
                new_U0 = jnp.concatenate([U[self.shift:], jnp.tile(U[-1:], (self.shift, 1))])
                new_X0 = jnp.concatenate([X[self.shift:], jnp.tile(X[-1:], (self.shift, 1))])
                new_V0 = jnp.concatenate([V[self.shift:], jnp.tile(V[-1:], (self.shift, 1))])
                tau = U[0, :config.n_joints]
                q = X[0, 7:config.n_joints + 7]
                dq = X[1, 13 + config.n_joints:2 * config.n_joints + 13]
                return new_U0, new_X0, new_V0,tau,q ,dq
            def unsafe_update():
                new_U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
                new_X0 = jnp.tile(x0, (self.config.N + 1, 1))
                new_V0 = jnp.zeros((self.config.N + 1, self.config.n ))
                tau = U0[1, :config.n_joints]
                q = X0[1, 7:config.n_joints + 7]
                dq = X0[1, 13 + config.n_joints:2 * config.n_joints + 13]
                return new_U0, new_X0, new_V0, tau, q, dq 
            return jax.lax.cond(jnp.isnan(U[0,0]),unsafe_update,safe_update)
        
        self.update_and_extract = update_and_extract_helper
    
    def make_data(self):
        return mpx_data(
            dt = self.config.dt,
            duty_factor = self.config.duty_factor,
            step_freq = self.config.step_freq,
            step_height = self.config.step_height,
            contact_time = self.config.timer_t,
            liftoff = jnp.zeros(3*self.config.n_contact),
            X0 = jnp.tile(self.initial_state, (self.config.N + 1, 1)),
            U0 = jnp.tile(self.config.u_ref, (self.config.N, 1)),
            V0 = jnp.zeros((self.config.N + 1, self.config.n)),
            W = self.config.W
        )
    
    def run(self, data, state,parameter):
        """
        Runs one MPC update using the current state, input, and foot positions.

        Args:
            data: mpx_data object containing the current state of the MPC controller.
            state: robot state + desired input
            parameter: paremeters for the parametric controller.

        Returns:
            
            next data: mpx_data object containing the new state of the MPC controller
            tau the output torque from the mpc

        """
        x0 = state[:self.config.n]
        input = state[self.config.n:]
        # Update the timer state for the gait reference.
        # _ , contact_time = mpc_utils.timer_run(data.duty_factor,data.step_freq,data.contact_time,1/self.mpc_frequency)

        # contact = jnp.zeros(self.config.n_contact)

        # Generate reference trajectory and additional MPC parameters.
        reference, liftoff = self.reference_generator(
            x = x0,
            input = input,
            liftoff = data.liftoff
        )

        # Execute the MPC optimization (work function).
        X, U, V = self.work(
            reference,
            parameter,
            data.W,
            x0,
            data.X0,
            data.U0,
            data.V0
            )

        U0, X0, V0, tau, _, _= self.update_and_extract(U, X, V, x0, data.X0, data.U0)


        # Warm-start for the next call: shift trajectories forward.
        data = data.replace(X0 = X0,
                            U0 = U0,
                            V0 = V0,
                            liftoff = liftoff)

        return data, tau

    def reset(self,config,data,envs,foot):
        """
        Resets the MPC controller state."
        """
        n_env_reset = envs.shape[0]
        contact_time = data.contact_time.at[envs,:].set(jnp.tile(config.timer_t, (n_env_reset, 1)))
        liftoff = data.liftoff.at[envs,:].set(foot)
        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        V0 = jnp.zeros((config.N + 1, config.n))
        batch_U0 = data.U0.at[envs,:,:].set(jnp.tile(U0, (n_env_reset, 1, 1)))
        batch_X0 = data.X0.at[envs,:,:].set(jnp.tile(X0, (n_env_reset, 1, 1)))
        batch_V0 = data.V0.at[envs,:,:].set(jnp.tile(V0, (n_env_reset, 1, 1)))
        data = data.replace(U0=batch_U0,
                     X0=batch_X0,
                     V0=batch_V0,
                     contact_time=contact_time,
                     liftoff=liftoff)
        return data

class MPCControllerWrapper:
    def __init__(self, config,limited_memory=False):
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

        work = partial(optimizers.mpc, self.cost, self.dynamics, hessian_approx, limited_memory)

        reference_generator = partial(mpc_utils.reference_generator,
            config.use_terrain_estimation ,config.N, config.dt, config.n_joints, config.n_contact, robot_mass,foot0 = config.p_legs0, q0 = config.q0)

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
        self.p_collision = np.zeros(3*config.n_contact)
        self.collision = [0,0,0,0]
        self.collision_cycle = np.zeros(config.n_contact)

    def run(self, qpos, qvel, input, contact):
        """
        Runs one MPC update using the current state positions, velocities, input, and contact information.

        Args:
            qpos: Generalized position.
            qvel: Generalized velocity.
            input: Control input vector.
            contact: Contact state vector.

        Returns:
            A tuple (tau, q, dq) representing the computed joint torques, joint positions, and joint velocities.
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

        
        contact = jnp.array(contact)

        # Update the timer state for the gait reference.
        des_contact , self.contact_time = self._timer_run(self.duty_factor,self.step_freq,self.contact_time,1/self.mpc_frequency)        
        input = jnp.array(input)
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
            clearence_speed = self.clearence_speed
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

        # # Warm-start for the next call: shift trajectories forward.

        self.U0, self.X0, self.V0, tau_temp, q_temp, dq_temp = self.update_and_extract(U, X, V, x0, self.X0, self.U0)

        # TO DO change to values from config
        tau = np.clip(np.array(tau_temp),self.config.min_torque,self.config.max_torque)
        q = np.array(q_temp)
        dq = np.array(dq_temp)

        return tau, q, dq, reference , U[:,self.config.n_joints:],foot_op,X

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

        reference = jnp.array(reference)
        parameter = jnp.array(parameter)
        # Warm start
        self.X0 = self.X0.at[:,:13+self.config.n_joints].set(reference[:,:13+self.config.n_joints])

        _cost = partial(self.cost,self.config.W,reference)
        _dynamics = partial(self.dynamics,parameter=parameter)
        model_evaluator = partial(optimizers.model_evaluator_helper, _cost, _dynamics,x0)
        jitted_model_evaluator = jax.jit(model_evaluator)

        _exit = False
        max_iter = 100
        last_cost = 1e10
        i = 0
        output = []
        output.append((self.X0))
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

            output.append((self.X0))

            g, c = jitted_model_evaluator(X,U)

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

        return self.X0, self.U0, reference, output