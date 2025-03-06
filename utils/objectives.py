import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math

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
    friction_cone = jnp.array([[0,0,1],[-1,0,mu],[1,0,mu],[0,-1,mu],[0,1,mu]])
    friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    scaled_grf = grf/grf_scaling + jnp.ones_like(grf)*1e-1
    friction_cone = friction_cone @ scaled_grf
    alpha = 0.1
    sigma = 5
    quadratic_barrier = alpha/2*(jnp.square((friction_cone-2*sigma)/sigma)-jnp.ones_like(friction_cone))
    log_barrier = -alpha*jnp.log(friction_cone)
    friction_cone = jnp.where(friction_cone>sigma,log_barrier,quadratic_barrier+log_barrier)
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
    stage_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq +\
                 (p_leg[2::3] - p_leg_ref[2::3]).T @ (p_leg[2::3] - p_leg_ref[2::3])*1e4 + \
                 tau.T @ W[12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact,12+2*n_joints+3*n_contact:12+3*n_joints+3*n_contact] @ tau +\
                 (p_leg[::3] - calc_foothold(0)).T @ Q_leg @ (p_leg[::3] - calc_foothold(0)) +\
                 (p_leg[1::3] - calc_foothold(1)).T @ Q_leg @ (p_leg[1::3] - calc_foothold(1))
                  #+ jnp.sum(friction_cone)
    term_cost = (p - p_ref).T @ W[:3,:3] @ (p - p_ref) + math.quat_sub(quat,quat_ref).T@W[3:6,3:6]@math.quat_sub(quat,quat_ref) + (q - q_ref).T @ W[6:6+n_joints,6:6+n_joints] @ (q - q_ref) +\
                 (dp - dp_ref).T @ W[6+n_joints:9+n_joints,6+n_joints:9+n_joints] @ (dp - dp_ref) + (omega - omega_ref).T @ W[9+n_joints:12+n_joints,9+n_joints:12+n_joints] @ (omega - omega_ref) + dq.T @ W[12+n_joints:12+2*n_joints,12+n_joints:12+2*n_joints] @ dq 


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)