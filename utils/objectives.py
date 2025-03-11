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
def quadruped_wb_obj(W,n_joints,n_contact,N,grf_scaling,x, u, t, reference):

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
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    foot0 = jnp.array([
    0.192, 0.142, 0.024,  # Initial position of the front left leg
    0.192, -0.142, 0.024, # Initial position of the front right leg
   -0.195, 0.142, 0.024,  # Initial position of the rear left leg
   -0.195, -0.142, 0.024  # Initial position of the rear right leg
    ])
    foot = jnp.tile(p,n_contact) + foot0@jax.scipy.linalg.block_diag(Ryaw,Ryaw,Ryaw,Ryaw).T
    def calc_foothold(direction):
            f1 = 0.5*dp_ref[direction]*0.65/1.35
            # f2 = jnp.sqrt(0.27/9.81)*(dp[direction]-dp_ref[direction])
            f = f1 + foot[direction::3]
            return f
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    Q_leg = jnp.diag(jnp.array([1e3]*4))
    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq +\
                 (p_leg[2::3] - p_leg_ref[2::3]).T @ (p_leg[2::3] - p_leg_ref[2::3])*1e4 + \
                 tau.T @ W[12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact,12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact] @ tau +\
                 (p_leg[::3] - calc_foothold(0)).T @ Q_leg @ (p_leg[::3] - calc_foothold(0)) +\
                 (p_leg[1::3] - calc_foothold(1)).T @ Q_leg @ (p_leg[1::3] - calc_foothold(1)) + jnp.sum(friction_cone*contact)
                  #+ jnp.sum(friction_cone)
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

def quadruped_wb_hessian_gn(W,n_joints,n_contact,N,grf_scaling,x, u, t, reference):

    
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
        yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
        Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
        foot0 = jnp.array([
        0.192, 0.142, 0.024,  # Initial position of the front left leg
        0.192, -0.142, 0.024, # Initial position of the front right leg
        -0.195, 0.142, 0.024,  # Initial position of the rear left leg
        -0.195, -0.142, 0.024  # Initial position of the rear right leg
        ])
        foot = jnp.tile(p,n_contact) + foot0@jax.scipy.linalg.block_diag(Ryaw,Ryaw,Ryaw,Ryaw).T
        def calc_foothold(direction):
                f1 = 0.5*dp_ref[direction]*0.65/1.35
                # f2 = jnp.sqrt(0.27/9.81)*(dp[direction]-dp_ref[direction])
                f = f1 + foot[direction::3]
                return f
        Q_leg = jnp.diag(jnp.array([1e3]*4))
        p_res = (p - p_ref).T
        quat_res = math.quat_sub(quat,quat_ref).T
        q_res = (q - q_ref).T
        dp_res = (dp - dp_ref).T
        omega_res = (omega - omega_ref).T
        dq_res = dq.T
        p_leg_z_res = (p_leg[2::3] - p_leg_ref[2::3]).T
        p_leg_x_res = (p_leg[::3] - calc_foothold(0)).T
        p_leg_y_res = (p_leg[1::3] - calc_foothold(1)).T
        tau_res = tau.T

        return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_x_res,p_leg_y_res,p_leg_z_res,tau_res])
    def friction_constraint(x):
        grf = x[13+2*n_joints+3*n_contact:]
        mu = 0.7
        friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
        return friction_cone
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    hessian_penaly = jax.grad(jax.grad(penaly))
    J_friction_cone = jax.jacobian(friction_constraint)
    contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
    H_penalty = jnp.diag(jnp.clip(jax.vmap(hessian_penaly)(friction_constraint(x)), -1e6, 1e6)*contact)
    H_constraint = J_friction_cone(x).T@H_penalty@J_friction_cone(x)

    return J_x(x,u).T@W@J_x(x,u) + H_constraint, J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)

def humanoid_wb_obj(W,n_joints,n_contact,N,grf_scaling,x, u, t, reference):

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

def humanoid_wb_hessian_gn(W,n_joints,n_contact,N,grf_scaling,x, u, t, reference):

    
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