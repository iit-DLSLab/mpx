import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math

def penaly(constraint):
        def safe_log(x):
            return jnp.where(x>0,jnp.log(x),1e6)
        alpha = 0.1
        sigma = 5
        quadratic_barrier = alpha/2*(jnp.square((constraint-2*sigma)/sigma)-jnp.ones_like(constraint))
        log_barrier = -alpha*safe_log(constraint)
        return jnp.clip(jnp.where(constraint>sigma,log_barrier,quadratic_barrier+log_barrier),0,1e6)

def quadruped_srbd_obj(W,n_contact,N,x, u, t, reference):

    p = x[:3]
    quat = x[3:7]
    dp = x[7:10]
    omega = x[10:13]
    grf = u

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    dp_ref = reference[t,7:10]
    omega_ref = reference[t,10:13]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
    friction_cone = jnp.clip(penaly(friction_cone),1e-6,1e6)
    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) +\
                 (dp - dp_ref).T @ W[6:9,6:9] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9:12,9:12] @ (omega - omega_ref) +\
                 grf.T@W[12:12+3*n_contact,12:12+3*n_contact]@grf + jnp.sum(friction_cone)
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) +\
                (dp - dp_ref).T @ W[6:9,6:9] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9:12,9:12] @ (omega - omega_ref)

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

def quadruped_srbd_hessian_gn(W,n_contact,x, u, t, reference):

    def residual(x,u):
        p = x[:3]
        quat = x[3:7]
        dp = x[7:10]
        omega = x[10:13]
        grf = u

        p_ref = reference[t,:3]
        quat_ref = reference[t,3:7]
        dp_ref = reference[t,7:10]
        omega_ref = reference[t,10:13]

        p_res = (p - p_ref).T
        quat_res = math.quat_sub(quat,quat_ref).T

        dp_res = (dp - dp_ref).T
        omega_res = (omega - omega_ref).T
        
        grf_res = grf.T

        return jnp.concatenate([p_res,quat_res,dp_res,omega_res,grf_res])
    
    def friction_constraint(u):
        grf = u
        mu = 0.7
        friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
        return friction_cone
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    contact = reference[t,13 : 13+n_contact]
    hessian_penaly = jax.grad(jax.grad(penaly))
    J_friction_cone = jax.jacobian(friction_constraint)
    H_penalty = jnp.diag(jnp.clip(jax.vmap(hessian_penaly)(friction_constraint(u)),1e-6, 1e6)*contact)
    H_constraint = J_friction_cone(u).T@H_penalty@J_friction_cone(u)
    return J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u) + H_constraint, J_x(x,u).T@W@J_u(x,u)

def quadruped_wb_obj(n_joints,n_contact,N,W,reference,x, u, t):

        # # Create a new data object for the simulation
        # mjx_data = mjx.make_data(model)
        # # Update the position and velocity in the data object
        # mjx_data = mjx_data.replace(qpos=x[:n_joints+7], qvel=x[n_joints+7:2*n_joints+13])

        # # Perform forward kinematics and dynamics computations
        # mjx_data = mjx.fwd_position(mjx_model, mjx_data)
        # # mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

        # # Get the positions of the contact points on the legs
        # FL_leg = mjx_data.geom_xpos[contact_id[0]]
        # FR_leg = mjx_data.geom_xpos[contact_id[1]]
        # RL_leg = mjx_data.geom_xpos[contact_id[2]]
        # RR_leg = mjx_data.geom_xpos[contact_id[3]]

        # # Compute the Jacobians for each leg
        # J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
        # J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
        # J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
        # J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

        # # Concatenate the Jacobians into a single matrix
        # J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)

        # foot_speed = J.T @ x[n_joints+7:13+2*n_joints]
    
    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
    grf = x[13+2*n_joints+3*n_contact:]
    tau = u[:n_joints]

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
    friction_cone = penaly(friction_cone)
    torque_limits = jnp.array([
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44 ])
    torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    # pitch = jnp.arcsin(2 * (quat_ref[0] * quat_ref[2] - quat_ref[3] * quat_ref[1]))
    # Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    # yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    # Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    # foot = jnp.tile(p,n_contact) + foot0@jax.scipy.linalg.block_diag(Ryaw@Rpitch,Ryaw@Rpitch,Ryaw@Rpitch,Ryaw@Rpitch).T
    duty_factor = reference[t,-2]
    step_freq = reference[t,-1]

    # def calc_foothold(direction):
    #     f1 = 0.5*dp_ref[direction]*duty_factor/step_freq
    #     # f2 = jnp.sqrt(0.27/9.81)*(dp[direction]-dp_ref[direction])
    #     f = f1 + foot[direction::3]
    #     return f
  
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    # flag_inital_phase_swing = jnp.where(timer_leg < (duty_factor + 0.2*(1-duty_factor)), 0, 1)
    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq +\
                 (p_leg - p_leg_ref).T @W[12+2*n_joints:12+2*n_joints+3*n_contact,12+2*n_joints:12+2*n_joints+3*n_contact]@ (p_leg - p_leg_ref)+ \
                 tau.T @ W[12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact,12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact] @ tau +\
                 jnp.sum(friction_cone*contact) + jnp.sum(penaly(torque_limits))
                #  (foot_speed[::3]*flag_inital_phase_swing).T@(foot_speed[::3]*flag_inital_phase_swing)*1e1 + (foot_speed[2::3]*flag_inital_phase_swing).T@(foot_speed[2::3]*flag_inital_phase_swing)*1e1#+ jnp.sum(friction_cone)
                  #+ jnp.sum(friction_cone)
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

def quadruped_wb_hessian_gn(n_joints,n_contact,W,reference,x, u, t):

    # # Create a new data object for the simulation
    # mjx_data = mjx.make_data(model)
    # # Update the position and velocity in the data object
    # mjx_data = mjx_data.replace(qpos=x[:n_joints+7], qvel=x[n_joints+7:2*n_joints+13])

    # # Perform forward kinematics and dynamics computations
    # mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    # # mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # # Get the positions of the contact points on the legs
    # FL_leg = mjx_data.geom_xpos[contact_id[0]]
    # FR_leg = mjx_data.geom_xpos[contact_id[1]]
    # RL_leg = mjx_data.geom_xpos[contact_id[2]]
    # RR_leg = mjx_data.geom_xpos[contact_id[3]]

    # # Compute the Jacobians for each leg
    # J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    # J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    # J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    # J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    # # Concatenate the Jacobians into a single matrix
    # J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)

    # foot_speed = J.T @ x[n_joints+7:13+2*n_joints]

    duty_factor = reference[t,-2]
    step_freq = reference[t,-1]

    # timer_leg = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    # flag_inital_phase_swing = jnp.where(timer_leg < (duty_factor + 0.4*(1-duty_factor)), 0, 1)

    def residual(x,u):

        p = x[:3]
        quat = x[3:7]
        q = x[7:7+n_joints]
        dp = x[7+n_joints:10+n_joints]
        omega = x[10+n_joints:13+n_joints]
        dq = x[13+n_joints:13+2*n_joints]
        p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
        grf = x[13+2*n_joints+3*n_contact:]
        tau = u[:n_joints]

        p_ref = reference[t,:3]
        quat_ref = reference[t,3:7]
        q_ref = reference[t,7:7+n_joints]
        dp_ref = reference[t,7+n_joints:10+n_joints]
        omega_ref = reference[t,10+n_joints:13+n_joints]
        p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
        # yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
        # Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
        # foot = jnp.tile(p,n_contact) + foot0@jax.scipy.linalg.block_diag(Ryaw,Ryaw,Ryaw,Ryaw).T
        # def calc_foothold(direction):
        #         f1 = 0.5*dp_ref[direction]*duty_factor/step_freq
        #         # f2 = jnp.sqrt(0.27/9.81)*(dp[direction]-dp_ref[direction])
        #         f = f1 + foot[direction::3]
        #         return f
        p_res = (p - p_ref).T
        quat_res = math.quat_sub(quat,quat_ref).T
        q_res = (q - q_ref).T
        dp_res = (dp - dp_ref).T
        omega_res = (omega - omega_ref).T
        dq_res = dq.T
        p_leg_res = (p_leg - p_leg_ref).T
        tau_res = tau.T
        # foot_speed_res_x = foot_speed[::3].T*flag_inital_phase_swing
        # foot_speed_res_y = foot_speed[1::3].T*flag_inital_phase_swing
        return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_res,tau_res])
    def friction_constraint(x):
        grf = x[13+2*n_joints+3*n_contact:]
        mu = 0.7
        friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
        return friction_cone
    def torque_constraint(u):
        tau = u[:n_joints]
        torque_limits = jnp.array([
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44 ])
        return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    hessian_penaly = jax.grad(jax.grad(penaly))
    J_friction_cone = jax.jacobian(friction_constraint)
    J_torque = jax.jacobian(torque_constraint)
    
    H_penalty = jnp.diag(jnp.clip(jax.vmap(hessian_penaly)(friction_constraint(x)), -1e6, 1e6)*contact)
    H_penalty_torque = jnp.diag(jnp.clip(jax.vmap(hessian_penaly)(torque_constraint(u)), -1e6, 1e6))
    H_constraint = J_friction_cone(x).T@H_penalty@J_friction_cone(x)
    H_constraint_u = J_torque(u).T@H_penalty_torque@J_torque(u)
    return J_x(x,u).T@W@J_x(x,u) + H_constraint, J_u(x,u).T@W@J_u(x,u) + H_constraint_u, J_x(x,u).T@W@J_u(x,u)

def humanoid_wb_obj(n_joints,n_contact,N,W,reference,x, u, t):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    grf = x[13+2*n_joints+n_contact*3:]
    tau = u[:n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

    p_ref = reference[t,:3]
    quat_ref = reference[t,3:7]
    q_ref = reference[t,7:7+n_joints]
    dp_ref = reference[t,7+n_joints:10+n_joints]
    omega_ref = reference[t,10+n_joints:13+n_joints]
    p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
    grf_ref = reference[t,13+n_joints+4*n_contact:]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
    joints_limits = jnp.array([
    0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    2.35, 2.35, 
    2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
    2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
    joints_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    torque_limits = jnp.array([
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200,
        40, 40, 40, 40, 18, 18, 18, 18,
        40, 40, 40, 40, 18, 18, 18, 18])
    torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]

    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq +\
                 (p_leg - p_leg_ref).T @W[12+2*n_joints:12+2*n_joints+3*n_contact,12+2*n_joints:12+2*n_joints+3*n_contact]@ (p_leg - p_leg_ref) + \
                 tau.T @ W[12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact,12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact] @ tau +\
                + jnp.sum(penaly(friction_cone)*contact) + (grf - grf_ref).T@W[12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact,12+3*n_joints+3*n_contact:12+3*n_joints+6*n_contact]@(grf - grf_ref)#+ jnp.sum(penaly(joints_limits)) + jnp.sum(penaly(torque_limits))
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

def humanoid_wb_hessian_gn(n_joints,n_contact,W,reference,x, u, t):

    
    def residual(x,u):
        p = x[:3]
        quat = x[3:7]
        q = x[7:7+n_joints]
        dp = x[7+n_joints:10+n_joints]
        omega = x[10+n_joints:13+n_joints]
        dq = x[13+n_joints:13+2*n_joints]
        grf = x[13+2*n_joints+n_contact*3:]
        tau = u[:n_joints]
        p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

        p_ref = reference[t,:3]
        quat_ref = reference[t,3:7]
        q_ref = reference[t,7:7+n_joints]
        dp_ref = reference[t,7+n_joints:10+n_joints]
        omega_ref = reference[t,10+n_joints:13+n_joints]
        p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
        grf_ref = reference[t,13+n_joints+4*n_contact:]
        p_res = (p - p_ref).T
        quat_res = math.quat_sub(quat,quat_ref).T
        q_res = (q - q_ref).T
        dp_res = (dp - dp_ref).T
        omega_res = (omega - omega_ref).T
        dq_res = dq.T
        p_leg_res = (p_leg - p_leg_ref).T
        tau_res = tau.T
        grf_res = (grf - grf_ref).T

        return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_res,tau_res,grf_res])
    
    def friction_constraint(x):
        grf = x[13+2*n_joints+3*n_contact:]
        mu = 0.7
        friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
        return friction_cone
    def joint_constraint(x):
        q = x[7:7+n_joints]
        joints_limits = jnp.array([
        0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
        0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
        2.35, 2.35, 
        2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
        2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
        return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    def torque_constraint(u):
        tau = u[:n_joints]
        torque_limits = jnp.array([
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
        200, 200,
        40, 40, 40, 40, 18, 18, 18, 18,
        40, 40, 40, 40, 18, 18, 18, 18])
        return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
        
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    hessian_penaly = jax.grad(jax.grad(penaly))
    J_friction_cone = jax.jacobian(friction_constraint)
    J_joint = jax.jacobian(joint_constraint)
    J_torque = jax.jacobian(torque_constraint)
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    H_penalty_friction = jnp.diag(jnp.clip(jax.vmap(hessian_penaly)(friction_constraint(x)), -1e6, 1e6)*contact)
    H_penalty_joint = jnp.diag(jnp.clip(jax.vmap(hessian_penaly)(joint_constraint(x)), -1e6, 1e6))
    H_penalty_torque = jnp.diag(jnp.clip(jax.vmap(hessian_penaly)(torque_constraint(u)), -1e6, 1e6))
    H_constraint = J_friction_cone(x).T@H_penalty_friction@J_friction_cone(x)
    # H_constraint += J_joint(x).T@H_penalty_joint@J_joint(x)
    # H_constraint_u = J_torque(u).T@H_penalty_torque@J_torque(u)

    return J_x(x,u).T@W@J_x(x,u) + H_constraint, J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)
    # return J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)