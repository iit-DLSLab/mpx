import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math
from functools import partial

def penalty(constraint, alpha=0.1, sigma=5):
    """Relaxed log barrier with a C2 quadratic continuation below ``sigma``."""
    log_barrier = -alpha * jnp.log(jnp.maximum(constraint, sigma))
    quadratic_barrier = (
        0.5 * alpha * jnp.square((constraint - 2.0 * sigma) / sigma)
        - 0.5 * alpha
        - alpha * jnp.log(sigma)
    )
    return jnp.where(constraint > sigma, log_barrier, quadratic_barrier)

def quadruped_srbd_obj(n_contact,N,W,reference,x, u, t):

    p = x[:3]
    quat = x[3:7]
    dp = x[7:10]
    omega = x[10:13]
    grf = u

    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
    friction_cone = jnp.clip(penalty(friction_cone),1e-6,1e6)
    stage_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) +\
                 (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref) +\
                 grf.T@W["grf"]@grf + jnp.sum(friction_cone)
    term_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) +\
                (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

def quadruped_srbd_hessian_gn(n_contact,W,reference,x, u, t):

    def residual(x,u):
        p = x[:3]
        quat = x[3:7]
        dp = x[7:10]
        omega = x[10:13]
        grf = u

        p_ref = reference["p"][t]
        quat_ref = reference["quat"][t]
        dp_ref = reference["dp"][t]
        omega_ref = reference["omega"][t]

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
    contact = reference["contact"][t]
    hessian_penalty = jax.grad(jax.grad(penalty))
    J_friction_cone = jax.jacobian(friction_constraint)
    H_penalty = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(friction_constraint(u)),1e-6, 1e6)*contact)
    H_constraint = J_friction_cone(u).T@H_penalty@J_friction_cone(u)
    W_res = jax.scipy.linalg.block_diag(W["pos"], W["rot"], W["vel"], W["omega"], W["grf"])
    return J_x(x,u).T@W_res@J_x(x,u), J_u(x,u).T@W_res@J_u(x,u) + H_constraint, J_x(x,u).T@W_res@J_u(x,u)

def quadruped_wb_obj(swing_tracking,n_joints,n_contact,n_legs,N,W,reference,x, u, t,torque_limits = None,joint_speed_limits = None):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_legs]
    grf = x[13+2*n_joints+3*n_legs:]
    tau = u[:n_joints]

    p_ref = reference['p'][t]
    quat_ref = reference['quat'][t]
    q_ref = reference['q'][t]
    dp_ref = reference['dp'][t]
    omega_ref = reference['omega'][t]
    p_leg_ref = reference['foot'][t]
    contact = reference['contact'][t]
    grf_ref = reference['grf'][t]

    mu = 0.5
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-2)
    friction_cone = penalty(friction_cone)
    constraint = jnp.sum(friction_cone*contact)

    if torque_limits is not None:
        torque_constraint = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
        constraint += jnp.sum(penalty(torque_constraint))
    if joint_speed_limits is not None:    
        joint_speed_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@dq + joint_speed_limits + jnp.ones_like(joint_speed_limits)*1e-2
        constraint += jnp.sum(penalty(joint_speed_limits,1,1))

    if swing_tracking:
        contact_map = jnp.ones(3*n_legs)
    else:
        contact_map = jnp.repeat(jnp.concatenate([contact, jnp.ones(n_legs - n_contact)]), 3)

    stage_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W["q"] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref) + dq.T @ W["dq"] @ dq +\
                 (contact_map*(p_leg - p_leg_ref)).T @W["contact"]@ (contact_map*(p_leg - p_leg_ref))+ \
                 tau.T @ W["tau"] @ tau +\
                 (grf-grf_ref).T @ W["grf"] @ (grf-grf_ref) +\
                 + constraint
    term_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W["q"] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref) + dq.T @ W["dq"] @ dq


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# def quadruped_wb_hessian_gn(swing_tracking,n_joints,n_contact,W,reference,x, u, t):

#     contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]

#     def residual(x,u):

#         p = x[:3]
#         quat = x[3:7]
#         q = x[7:7+n_joints]
#         dp = x[7+n_joints:10+n_joints]
#         omega = x[10+n_joints:13+n_joints]
#         dq = x[13+n_joints:13+2*n_joints]
#         p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
#         grf = x[13+2*n_joints+3*n_contact:]
#         tau = u[:n_joints]

#         p_ref = reference[t,:3]
#         quat_ref = reference[t,3:7]
#         q_ref = reference[t,7:7+n_joints]
#         dp_ref = reference[t,7+n_joints:10+n_joints]
#         omega_ref = reference[t,10+n_joints:13+n_joints]
#         p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
#         grf_ref = reference[t,13+n_joints+4*n_contact:13+n_joints+7*n_contact]
#         p_res = (p - p_ref).T
#         quat_res = math.quat_sub(quat,quat_ref).T
#         q_res = (q - q_ref).T
#         dp_res = (dp - dp_ref).T
#         omega_res = (omega - omega_ref).T
#         dq_res = dq.T
#         if swing_tracking:
#             contact_map = jnp.ones(3*n_contact)
#         else:
#             contact_map = jnp.array([jnp.ones(3)*contact[i] for i in range(n_contact)]).flatten()
#         p_leg_res = (contact_map*(p_leg - p_leg_ref)).T
#         tau_res = tau.T
#         grf_res = (grf-grf_ref).T
#         return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_res,tau_res,grf_res])
#     def friction_constraint(x):
#         grf = x[13+2*n_joints+3*n_contact:]
#         mu = 0.5
#         friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
#         return friction_cone
#     def torque_constraint(u):
#         tau = u[:n_joints]
#         torque_limits = jnp.array([
#         44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
#         44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44 ])
#         return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
#     def speed_constarint(x):
#         dq = x[13+n_joints:13+2*n_joints]
#         joint_speed_limits = jnp.ones(2*n_joints)*10
#         joint_speed_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@dq + joint_speed_limits + jnp.ones_like(joint_speed_limits)*1e-2
#         return joint_speed_limits
#     def min_force_constraint(x):
#         grf = x[13+2*n_joints+3*n_contact:]
#         min_force = grf[2::3] - jnp.ones(n_contact)*10
#         return min_force
#     J_x = jax.jacobian(residual,0)
#     J_u = jax.jacobian(residual,1)
#     hessian_penalty = jax.grad(jax.grad(penalty))
#     J_friction_cone = jax.jacobian(friction_constraint)
#     J_torque = jax.jacobian(torque_constraint)
#     J_speed = jax.jacobian(speed_constarint)
#     # J_min_force = jax.jacobian(min_force_constraint)
#     hessian_penalty_torque = partial(hessian_penalty,alpha = 1,sigma = 1)
#     hessian_penalty_collision = partial(hessian_penalty,alpha = 10,sigma = 0.01)
#     # W = W.at[12+2*n_joints + 6:12+2*n_joints+3*n_contact,12+2*n_joints + 6:12+2*n_joints+3*n_contact].set(W[12+2*n_joints + 6:12+2*n_joints+3*n_contact,12+2*n_joints + 6:12+2*n_joints+3*n_contact]*stand_up_flag)
#     H_penalty = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(friction_constraint(x)), -1e6, 1e6)*contact)
#     H_penalty_torque = jnp.diag(jnp.clip(jax.vmap(hessian_penalty_torque)(torque_constraint(u)), -1e6, 1e6))
#     H_penalty_speed = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(speed_constarint(x)), -1e6, 1e6))
#     # H_penalty_min_force = jnp.diag(jnp.clip(jax.vmap(hessian_penalty_torque)(min_force_constraint(x)), -1e6, 1e6)*contact)
#     H_constraint = J_friction_cone(x).T@H_penalty@J_friction_cone(x)
#     H_constraint_u = J_torque(u).T@H_penalty_torque@J_torque(u)
#     H_constraint += J_speed(x).T@H_penalty_speed@J_speed(x)
#     # H_constraint += J_min_force(x).T@H_penalty_min_force@J_min_force(x)
#     return J_x(x,u).T@W@J_x(x,u) + H_constraint, J_u(x,u).T@W@J_u(x,u) + H_constraint_u, J_x(x,u).T@W@J_u(x,u)

def h1_wb_obj(n_joints,n_contact,N,W,reference,x, u, t):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    grf = x[13+2*n_joints+n_contact*3:]
    tau = u[:n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    p_leg_ref = reference["foot"][t]
    grf_ref = reference["grf"][t]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
    # joints_limits = jnp.array([
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 2.35, 2.35, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
    # joints_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    # torque_limits = jnp.array([
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200,
    #     40, 40, 40, 40, 18, 18, 18, 18,
    #     40, 40, 40, 40, 18, 18, 18, 18])
    # torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    contact = reference["contact"][t]

    stage_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W["q"] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref) + dq.T @ W["dq"] @ dq +\
                 (p_leg - p_leg_ref).T @W["contact"]@ (p_leg - p_leg_ref) + \
                 tau.T @ W["tau"] @ tau +\
                + (grf - grf_ref).T@W["grf"]@(grf - grf_ref)#+ jnp.sum(penalty(joints_limits)) + jnp.sum(penalty(torque_limits))
    term_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W["q"] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref) + dq.T @ W["dq"] @ dq

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# def h1_wb_hessian_gn(n_joints,n_contact,W,reference,x, u, t):

    
#     def residual(x,u):
#         p = x[:3]
#         quat = x[3:7]
#         q = x[7:7+n_joints]
#         dp = x[7+n_joints:10+n_joints]
#         omega = x[10+n_joints:13+n_joints]
#         dq = x[13+n_joints:13+2*n_joints]
#         tau = u[:n_joints]
#         grf = x[13+2*n_joints+n_contact*3:]
#         p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

#         p_ref = reference[t,:3]
#         quat_ref = reference[t,3:7]
#         q_ref = reference[t,7:7+n_joints]
#         dp_ref = reference[t,7+n_joints:10+n_joints]
#         omega_ref = reference[t,10+n_joints:13+n_joints]
#         p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
#         grf_ref = reference[t,13+n_joints+4*n_contact:13+n_joints+7*n_contact]
#         p_res = (p - p_ref).T
#         quat_res = math.quat_sub(quat,quat_ref).T
#         q_res = (q - q_ref).T
#         dp_res = (dp - dp_ref).T
#         omega_res = (omega - omega_ref).T
#         dq_res = dq.T
#         p_leg_res = (p_leg - p_leg_ref).T
#         tau_res = tau.T
#         grf_res = (grf - grf_ref).T

#         return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_res,tau_res,grf_res])
    
#     def friction_constraint(x):
#         grf = x[13+2*n_joints+n_contact*3:]
#         mu = 0.7
#         friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
#         return friction_cone
#     def joint_constraint(x):
#         q = x[7:7+n_joints]
#         joints_limits = jnp.array([
#         0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
#         0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
#         2.35, 2.35,
#         2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25,
#         2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
#         return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
#     def torque_constraint(u):
#         tau = u[:n_joints]
#         torque_limits = jnp.array([
#         200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
#         200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
#         200, 200,
#         40, 40, 40, 40, 18, 18, 18, 18,
#         40, 40, 40, 40, 18, 18, 18, 18])
#         return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
        
#     J_x = jax.jacobian(residual,0)
#     J_u = jax.jacobian(residual,1)
#     hessian_penalty = jax.grad(jax.grad(penalty))
#     J_friction_cone = jax.jacobian(friction_constraint)
#     # J_joint = jax.jacobian(joint_constraint)
#     # J_torque = jax.jacobian(torque_constraint)
#     contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
#     H_penalty_friction = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(friction_constraint(x)), -1e6, 1e6)*contact)
#     # H_penalty_joint = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(joint_constraint(x)), -1e6, 1e6))
#     # H_penalty_torque = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(torque_constraint(u)), -1e6, 1e6))
#     H_constraint = J_friction_cone(x).T@H_penalty_friction@J_friction_cone(x)
#     # H_constraint += J_joint(x).T@H_penalty_joint@J_joint(x)
#     # H_constraint_u = J_torque(u).T@H_penalty_torque@J_torque(u)

#     return J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)


def h1_kinodynamic_obj(n_joints, n_contact, N, W, reference, x, u, t):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

    dq_cmd = u[:n_joints]
    grf = u[n_joints:]

    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    p_leg_ref = reference["foot"][t]
    contact = reference["contact"][t]
    grf_ref = reference["grf"][t]

    mu = 0.7
    friction_cone = mu * grf[2::3] - jnp.sqrt(
        jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact) * 1e-1
    )

    stage_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + math.quat_sub(quat,quat_ref).T @ W["rot"] @ math.quat_sub(quat,quat_ref)
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
        + (p_leg - p_leg_ref).T
        @ W["contact"]
        @ (p_leg - p_leg_ref)
        + dq_cmd.T
        @ W["dq_cmd"]
        @ dq_cmd
        + (grf - grf_ref).T
        @ W["grf"]
        @ (grf - grf_ref)
        + jnp.sum(penalty(friction_cone) * contact)
    )
    term_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + math.quat_sub(quat,quat_ref).T @ W["rot"] @ math.quat_sub(quat,quat_ref)
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
    )

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

def talos_wb_obj(n_joints,n_contact,N,W,reference,x, u, t):

    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    omega = x[10+n_joints:13+n_joints]
    dq = x[13+n_joints:13+2*n_joints]
    # grf = x[13+2*n_joints+n_contact*3:]
    tau = u[:n_joints]
    grf = u[n_joints:]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
    # foot_speed = x[13+2*n_joints+3*n_contact:13+2*n_joints+6*n_contact]
    
    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    p_leg_ref = reference["foot"][t]
    grf_ref = reference["grf"][t]

    mu = 0.7
    friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
    # joints_limits = jnp.array([
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
    # 2.35, 2.35, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25, 
    # 2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
    # joints_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
    # torque_limits = jnp.array([
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
    #     200, 200,
    #     40, 40, 40, 40, 18, 18, 18, 18,
    #     40, 40, 40, 40, 18, 18, 18, 18])
    # torque_limits = jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
    contact = reference["contact"][t]
    # zero the speed for the leg in swing phase
    # foot_speed = jnp.concatenate([foot_speed[0:3]*contact[0],foot_speed[3:6]*contact[1],foot_speed[6:9]*contact[2],foot_speed[9:12]*contact[3],
    #                        foot_speed[12:15]*contact[4],foot_speed[15:18]*contact[5],foot_speed[18:21]*contact[6],foot_speed[21:24]*contact[7]])
    
    stage_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W["q"] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref) + dq.T @ W["dq"] @ dq +\
                 (p_leg - p_leg_ref).T @W["contact"]@ (p_leg - p_leg_ref) + \
                 tau.T @ W["tau"] @ tau +\
                + (grf - grf_ref).T@W["grf"]@(grf - grf_ref) #+ foot_speed.T@W[12+3*n_joints+6*n_contact:12+3*n_joints+9*n_contact,12+3*n_joints+6*n_contact:12+3*n_joints+9*n_contact]@foot_speed#+ jnp.sum(penalty(joints_limits)) + jnp.sum(penalty(torque_limits))
    term_cost = (p - p_ref).T @ W["pos"] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W["rot"]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W["q"] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref) + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref) + dq.T @ W["dq"] @ dq

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# def talos_wb_hessian_gn(n_joints,n_contact,W,reference,x, u, t):

    
#     def residual(x,u):
#         p = x[:3]
#         quat = x[3:7]
#         q = x[7:7+n_joints]
#         dp = x[7+n_joints:10+n_joints]
#         omega = x[10+n_joints:13+n_joints]
#         dq = x[13+n_joints:13+2*n_joints]
#         tau = u[:n_joints]
#         grf =  u[n_joints:]
#         p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]
#         # foot_speed = x[13+2*n_joints+3*n_contact:13+2*n_joints+6*n_contact]

#         p_ref = reference[t,:3]
#         quat_ref = reference[t,3:7]
#         q_ref = reference[t,7:7+n_joints]
#         dp_ref = reference[t,7+n_joints:10+n_joints]
#         omega_ref = reference[t,10+n_joints:13+n_joints]
#         p_leg_ref = reference[t,13+n_joints:13+n_joints+3*n_contact]
#         grf_ref = reference[t,13+n_joints+4*n_contact:13+n_joints+7*n_contact]
#         p_res = (p - p_ref).T
#         quat_res = math.quat_sub(quat,quat_ref).T
#         q_res = (q - q_ref).T
#         dp_res = (dp - dp_ref).T
#         omega_res = (omega - omega_ref).T
#         dq_res = dq.T
#         p_leg_res = (p_leg - p_leg_ref).T
#         tau_res = tau.T
#         grf_res = (grf - grf_ref).T
#         # foot_speed_res = foot_speed.T
#         return jnp.concatenate([p_res,quat_res,q_res,dp_res,omega_res,dq_res,p_leg_res,tau_res,grf_res])
    
#     def friction_constraint(u):
#         grf = u[n_joints:]
#         mu = 0.7
#         friction_cone = mu*grf[2::3] - jnp.sqrt(jnp.square(grf[1::3]) + jnp.square(grf[::3]) + jnp.ones(n_contact)*1e-1)
#         return friction_cone
#     def joint_constraint(x):
#         q = x[7:7+n_joints]
#         joints_limits = jnp.array([
#         0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
#         0.43, 0.43, 0.43, 0.43,  1.57, 1.57,  2.05,  0.26, 0.52, 0.87,
#         2.35, 2.35,
#         2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25,
#         2.87,  2.87,  3.11,  0.34,  4.45,  1.3,  2.61,1.25])
#         return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@q+joints_limits + jnp.ones_like(joints_limits)*1e-2
#     def torque_constraint(u):
#         tau = u[:n_joints]
#         torque_limits = jnp.array([
#         200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
#         200, 200, 200, 200, 200, 200, 300, 300, 40, 40,
#         200, 200,
#         40, 40, 40, 40, 18, 18, 18, 18,
#         40, 40, 40, 40, 18, 18, 18, 18])
#         return jnp.kron(jnp.eye(n_joints),(jnp.array([-1,1]))).T@tau+torque_limits + jnp.ones_like(torque_limits)*1e-2
        
#     J_x = jax.jacobian(residual,0)
#     J_u = jax.jacobian(residual,1)
#     hessian_penalty = jax.grad(jax.grad(penalty))
#     J_friction_cone = jax.jacobian(friction_constraint)
#     # J_joint = jax.jacobian(joint_constraint)
#     # J_torque = jax.jacobian(torque_constraint)
#     contact = reference[t,13+n_joints+3*n_contact:13+n_joints+4*n_contact]
#     H_penalty_friction = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(friction_constraint(u)), -1e6, 1e6)*contact)
#     # H_penalty_joint = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(joint_constraint(x)), -1e6, 1e6))
#     # H_penalty_torque = jnp.diag(jnp.clip(jax.vmap(hessian_penalty)(torque_constraint(u)), -1e6, 1e6))
#     H_constraint = J_friction_cone(u).T@H_penalty_friction@J_friction_cone(u)
#     # H_constraint += J_joint(x).T@H_penalty_joint@J_joint(x)
#     # H_constraint_u = J_torque(u).T@H_penalty_torque@J_torque(u)

#     return J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)
#     # return J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)
