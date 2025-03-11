import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math

def quadruped_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    """
    Compute the whole-body dynamics of a quadruped robot using forward dynamics and contact forces.

    Args:
        model: The MuJoCo model object.
        mjx_model: The MuJoCo XLA model object for the simulation.
        contact_id (list): List of contact point IDs for each leg. [FL, FR, RL, RR]
        body_id (list): List of body IDs for each leg. [FL, FR, RL, RR]
        n_joints (int): Number of joints in the quadruped. 
        dt (float): Time step for the simulation.
        x (jnp.ndarray): Current state vector [position, orientation, joint positions, velocities].
        u (jnp.ndarray): Control input vector (torques for the joints).
        t (int): Current time step index.
        parameter (jnp.ndarray): Contact parameters for each time step.

    Returns:
        jnp.ndarray: The updated state vector after applying dynamics and contact forces.
    """
    # Create a new data object for the simulation
    mjx_data = mjx.make_data(model)
    # Update the position and velocity in the data object
    mjx_data = mjx_data.replace(qpos=x[:n_joints+7], qvel=x[n_joints+7:2*n_joints+13])

    # Perform forward kinematics and dynamics computations
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # Extract the mass matrix and bias forces
    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    # Get the contact parameters for the current time step
    contact = parameter[t, :4]

    # Create the torque vector, with zeros for the base and control inputs for the joints
    tau = jnp.concatenate([jnp.zeros(6), u])

    # Get the positions of the contact points on the legs
    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    # Compute the Jacobians for each leg
    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    # Concatenate the Jacobians into a single matrix
    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
    # Concatenate the positions of the legs into a single vector
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
    alpha = 25
    # Compute the velocity-level constraint violation
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]
    # Compute the stabilization term
    baumgarte_term = -2 * alpha * g_dot

    # Compute the inverse of the mass matrix projected onto the constraint Jacobian
    JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)
    # Compute the right-hand side of the constraint force equation
    rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False), tau - D) + baumgarte_term
    # Solve for the ground reaction forces
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ, rhs)
    # Apply the contact forces only to the legs that are in contact
    grf = jnp.concatenate([grf[:3]*contact[0], grf[3:6]*contact[1], grf[6:9]*contact[2], grf[9:12]*contact[3]])
    # Update the velocity using the computed forces
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False), tau - D + J @ grf) * dt
    # Perform semi-implicit Euler integration to update the position and orientation
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    # Concatenate the updated state variables into a single vector
    x_next = jnp.concatenate([p, quat, q, v, current_leg, grf])

    return x_next
def humanoid_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:2*n_joints+13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t,:4]

    tau = jnp.concatenate([jnp.zeros(6),u])

    FL = mjx_data.geom_xpos[contact_id[0]]
    RL = mjx_data.geom_xpos[contact_id[1]]
    FR = mjx_data.geom_xpos[contact_id[2]]
    RR = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL, body_id[0])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR,  body_id[1])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR,  body_id[1])
    J = jnp.concatenate([J_FL,J_RL,J_FR,J_RR],axis=1)
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]  # Velocity-level constraint violation
    
    alpha = 5
    # beta = 2*jnp.sqrt(alpha)
    # Stabilization term
    baumgarte_term = - 2*alpha * g_dot #- beta * beta * g

    JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)


    rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False),tau - D) + baumgarte_term 
    epsilon = 1e-3
    JT_M_invJ_reg = JT_M_invJ + epsilon * jnp.eye(JT_M_invJ.shape[0])
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ_reg)
    
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ,rhs)
    grf = jnp.concatenate([grf[0:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v,FL,RL,FR,RR,grf])
    
    return x_next

def quadruped_wb_dynamics_explicit_contact(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    """
    Compute the whole-body dynamics of a quadruped robot using forward dynamics and contact forces.

    Args:
        model: The MuJoCo model object.
        mjx_model: The MuJoCo XLA model object for the simulation.
        contact_id (list): List of contact point IDs for each leg. [FL, FR, RL, RR]
        body_id (list): List of body IDs for each leg. [FL, FR, RL, RR]
        n_joints (int): Number of joints in the quadruped. 
        dt (float): Time step for the simulation.
        x (jnp.ndarray): Current state vector [position, orientation, joint positions, velocities].
        u (jnp.ndarray): Control input vector (torques for the joints).
        t (int): Current time step index.
        parameter (jnp.ndarray): Contact parameters for each time step.

    Returns:
        jnp.ndarray: The updated state vector after applying dynamics and contact forces.
    """
    # Create a new data object for the simulation
    mjx_data = mjx.make_data(model)
    # Update the position and velocity in the data object
    mjx_data = mjx_data.replace(qpos=x[:n_joints+7], qvel=x[n_joints+7:2*n_joints+13])

    # Perform forward kinematics and dynamics computations
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # Extract the mass matrix and bias forces
    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    # Get the contact parameters for the current time step
    # contact = parameter[t, :4]

    # Create the torque vector, with zeros for the base and control inputs for the joints
    tau = jnp.concatenate([jnp.zeros(6), u])

    # Get the positions of the contact points on the legs
    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    # Compute the Jacobians for each leg
    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    # Concatenate the Jacobians into a single matrix
    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
    # Concatenate the positions of the legs into a single vector
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
    
    contact_stifness = 2000
    smothing = 0.01
    dissipation_velocity = 0.1
    stiction_velocity = 0.5
    friction_coefficient = 1

    # Compute the velocity-level constraint violation
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]

    grf_z = smothing * contact_stifness * jnp.log(1 + jnp.exp(-current_leg[2::3] / smothing))
    grf_z = jnp.where(g_dot[2::3]/dissipation_velocity<0,
                      (jnp.ones(4)-g_dot[2::3]/dissipation_velocity) * grf_z, 
                      (jnp.square((g_dot[2::3]/dissipation_velocity-2*jnp.ones(4)))/4) * grf_z)
    grf_z = jnp.where(g_dot[2::3]/dissipation_velocity>2, jnp.zeros(4), grf_z)
    grf_x = - friction_coefficient * grf_z * (g_dot[0::3] / jnp.sqrt(jnp.square(g_dot[0::3]) + stiction_velocity**2))
    grf_y = - friction_coefficient * grf_z * (g_dot[1::3] / jnp.sqrt(jnp.square(g_dot[1::3]) + stiction_velocity**2))

    grf = jnp.zeros(12)
    grf = grf.at[0::3].set(grf_x)
    grf = grf.at[1::3].set(grf_y)
    grf = grf.at[2::3].set(grf_z)

    # Update the velocity using the computed forces
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False), tau - D + J @ grf) * dt

    # Perform semi-implicit Euler integration to update the position and orientation
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    # Concatenate the updated state variables into a single vector
    x_next = jnp.concatenate([p, quat, q, v, current_leg, grf])

    return x_next