import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math

from mpx.utils.sim import geom_ids


def _mask_contact_forces(grf, contact):
    return (grf.reshape(-1, 3) * contact[:, None]).reshape(-1)


def _h1_contact_kinematics(mjx_model, mjx_data, contact_id, body_id):
    fl = mjx_data.geom_xpos[contact_id[0]]
    rl = mjx_data.geom_xpos[contact_id[1]]
    fr = mjx_data.geom_xpos[contact_id[2]]
    rr = mjx_data.geom_xpos[contact_id[3]]

    j_fl, _ = mjx.jac(mjx_model, mjx_data, fl, body_id[0])
    j_rl, _ = mjx.jac(mjx_model, mjx_data, rl, body_id[0])
    j_fr, _ = mjx.jac(mjx_model, mjx_data, fr, body_id[1])
    j_rr, _ = mjx.jac(mjx_model, mjx_data, rr, body_id[1])

    feet = jnp.concatenate([fl, rl, fr, rr], axis=0)
    jacobian = jnp.concatenate([j_fl, j_rl, j_fr, j_rr], axis=1)
    return feet, jacobian


def quadruped_srbd_dynamics(mass, inertia,inertia_inv, dt, x, u, t,parameter):
    # Extract state variables
    p = x[:3]
    quat = x[3:7]
    # p_legs = x[6:6+n_joints]
    dp = x[7:10]
    omega = x[10:13]
    # dp_leg = u[:n_joints]
    grf = u

    contact = parameter[t,:4]
    p_legs = parameter[t,4:]

    dp_next = dp + (jnp.array([0, 0, -9.81]) + (1 / mass) * (grf[:3]*contact[0] + grf[3:6]*contact[1] + grf[6:9]*contact[2] + grf[9:12]*contact[3])) * dt

    p0 = p_legs[:3]
    p1 = p_legs[3:6]
    p2 = p_legs[6:9]
    p3 = p_legs[9:]

    b_R_w = math.quat_to_mat(quat)
    base_angular_wrench = jnp.cross(p0 - p, grf[:3])*contact[0] + jnp.cross(p1 - p, grf[3:6])*contact[1] + jnp.cross(p2 - p, grf[6:9])*contact[2] + jnp.cross(p3 - p, grf[9:12])*contact[3]
    omega_next = omega + inertia_inv@(b_R_w.T@base_angular_wrench - jnp.cross(omega,inertia@omega))*dt

    # Semi-implicit Euler integration
    p_new = p + dp_next * dt
    quat_new = math.quat_integrate(quat, omega_next, dt)
    # p_legs_new = p_legs# + dp_leg * dt

    x_next = jnp.concatenate([p_new, quat_new, dp_next, omega_next])

    return x_next

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
    
    if len(contact_id) > 4 :
        current_leg = jnp.asarray([mjx_data.geom_xpos[int(geom_id)] for geom_id  in contact_id]).flatten()
    else:
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

def h1_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):

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


def h1_kinodynamic_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    
    qpos = x[: n_joints + 7]
    qvel = x[n_joints + 7 : 2 * n_joints + 13]
    dq = x[13 + n_joints : 13 + 2 * n_joints]
    dq_next = u[:n_joints]
    contact = parameter[t, :4]
    grf = _mask_contact_forces(u[n_joints:], contact)

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    mass_matrix = mjx.full_m(mjx_model, mjx_data)
    bias = mjx_data.qfrc_bias
    feet_next, jacobian = _h1_contact_kinematics(mjx_model, mjx_data, contact_id, body_id)

    qdd_joints = (dq_next - dq) / dt
    rhs = (jacobian @ grf)[:6] - bias[:6] - mass_matrix[:6, 6:] @ qdd_joints
    qdd_base = jnp.linalg.solve(mass_matrix[:6, :6] + 1e-6 * jnp.eye(6), rhs)

    base_velocity_next = qvel[:6] + qdd_base * dt
    qvel_next = jnp.concatenate([base_velocity_next, dq_next])

    p_next = x[:3] + qvel_next[:3] * dt
    quat_next = math.quat_integrate(x[3:7], qvel_next[3:6], dt)
    q_next = x[7 : 7 + n_joints] + dq_next * dt

    return jnp.concatenate([p_next, quat_next, q_next, qvel_next, feet_next])


def h1_kinodynamic_torques(
    model,
    mjx_model,
    contact_id,
    body_id,
    n_joints,
    dt,
    joint_kp,
    joint_kd,
    x0,
    X,
    U,
    reference,
    parameter,
):
    del reference
    qpos = x0[: n_joints + 7]
    qvel = x0[n_joints + 7 : 2 * n_joints + 13]
    qvel_next = X[1, n_joints + 7 : 2 * n_joints + 13]
    qacc = (qvel_next - qvel) / dt

    contact = parameter[0, :4]
    grf = _mask_contact_forces(U[0, n_joints:], contact)

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, qacc=qacc)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)
    _, jacobian = _h1_contact_kinematics(mjx_model, mjx_data, contact_id, body_id)
    mjx_data = mjx.inverse(mjx_model, mjx_data)

    tau_ff = (mjx_data.qfrc_inverse - jacobian @ grf)[6:]
    q_des = X[1, 7 : 7 + n_joints]
    dq_des = qvel_next[6:]
    tau_fb = joint_kp * (q_des - qpos[7:]) + joint_kd * (dq_des - qvel[6:])
    return tau_ff + tau_fb
def talos_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:2*n_joints+13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t,:8]

    tau = jnp.concatenate([jnp.zeros(6),u[:n_joints]])

    left_foot_1 = mjx_data.geom_xpos[contact_id[0]]
    left_foot_2 = mjx_data.geom_xpos[contact_id[1]]
    left_foot_3 = mjx_data.geom_xpos[contact_id[2]]
    left_foot_4 = mjx_data.geom_xpos[contact_id[3]]

    right_foot_1 = mjx_data.geom_xpos[contact_id[4]]
    right_foot_2 = mjx_data.geom_xpos[contact_id[5]]
    right_foot_3 = mjx_data.geom_xpos[contact_id[6]]
    right_foot_4 = mjx_data.geom_xpos[contact_id[7]]

    J_fl_1, _ = mjx.jac(mjx_model, mjx_data, left_foot_1, body_id[0])
    J_fl_2, _ = mjx.jac(mjx_model, mjx_data, left_foot_2, body_id[0])
    J_fl_3, _ = mjx.jac(mjx_model, mjx_data, left_foot_3, body_id[0])
    J_fl_4, _ = mjx.jac(mjx_model, mjx_data, left_foot_4, body_id[0])

    J_rl_1, _ = mjx.jac(mjx_model, mjx_data, right_foot_1, body_id[1])
    J_rl_2, _ = mjx.jac(mjx_model, mjx_data, right_foot_2, body_id[1])
    J_rl_3, _ = mjx.jac(mjx_model, mjx_data, right_foot_3, body_id[1])
    J_rl_4, _ = mjx.jac(mjx_model, mjx_data, right_foot_4, body_id[1])
    
    J = jnp.concatenate([J_fl_1,J_fl_2,J_fl_3,J_fl_4,J_rl_1,J_rl_2,J_rl_3,J_rl_4],axis=1)
    grf = u[n_joints:]
    grf = jnp.concatenate([grf[0:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3],
                           grf[12:15]*contact[4],grf[15:18]*contact[5],grf[18:21]*contact[6],grf[21:24]*contact[7]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v,left_foot_1,left_foot_2,left_foot_3,left_foot_4,right_foot_1,right_foot_2,right_foot_3,right_foot_4])
    
    return x_next
def quadruped_wb_dynamics_explicit_contact(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):

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
    
    contact_stifness = 1200
    smothing = 0.05
    dissipation_velocity = 0.1
    stiction_velocity = 0.5
    friction_coefficient = 0.5

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

# from functools import partial
# import pickle
# # Load parameters from a pickle file
# def load_params(file_path):
#     with open(file_path, "rb") as f:
#         params = pickle.load(f)
#     return params
# params_file_path = "./trained_params.pkl"
# params = load_params(params_file_path)

def quadruped_wb_dynamics_learned_contact_model(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter):
    """
    Compute the whole-body dynamics of a quadruped robot using forward dynamics and a learned contact model.

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
        parameter (jnp.ndarray): Contact parameters for each time step and the learned contact model parameters.

    Returns:
        jnp.ndarray: The updated state vector after applying dynamics and contact forces.
    """
    # encoding = 64
    input_size = 13 + 3*n_joints
    hidden_layer_size_1 = 128
    hidden_layer_size_2 = 64
    output_size = 3*len(contact_id)

    def mlp_apply(params, x):
        x = jnp.dot(x, params["W1"]) + params["b1"]
        x = jax.nn.softplus(x)
        x = jnp.dot(x, params["W2"]) + params["b2"]
        x = jax.nn.softplus(x)
        x = jnp.dot(x, params["W3"]) + params["b3"]
        return x    
    contact_model = partial(mlp_apply, params)
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
    st_size = hidden_layer_size_1*input_size + hidden_layer_size_1 + hidden_layer_size_1*hidden_layer_size_2 + hidden_layer_size_2 + output_size
    grf = contact_model(jnp.concatenate([x[:13+2*n_joints],u]))
    contact = parameter[t, :4]
    grf = jnp.concatenate([grf[:3]*contact[0], grf[3:6]*contact[1], grf[6:9]*contact[2], grf[9:12]*contact[3]])
    # Update the velocity using the computed forces
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False), tau - D + J@grf) * dt

    # Perform semi-implicit Euler integration to update the position and orientation
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt

    # Concatenate the updated state variables into a single vector
    x_next = jnp.concatenate([p, quat, q, v, current_leg,grf])

    return x_next
