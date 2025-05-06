import jax
from jax import numpy as jnp
from functools import partial
from mujoco.mjx._src import math
from jax.scipy.spatial.transform import Rotation

def timer_run(duty_factor,step_freq, leg_time, dt):
    # Extract relevant fields
    # Update timer
    leg_time = leg_time + dt * step_freq
    leg_time = jnp.where(leg_time > 1, leg_time - 1, leg_time)
    contact = jnp.where(leg_time < duty_factor, 1, 0)

    return contact, leg_time
def terrain_orientation(liftoff_pos):

    # Calculate the vectors between the legs
    vec_front_back = (liftoff_pos[:3] + liftoff_pos[3:6] - liftoff_pos[6:9] - liftoff_pos[9:12])/2
    vec_left_right = (liftoff_pos[:3] + liftoff_pos[6:9] - liftoff_pos[3:6] - liftoff_pos[9:12])/2
    # Compute the normal vector to the plane
    normal_vector = jnp.cross(vec_front_back, vec_left_right)

    # Normalize the vectors
    vec_front_back = vec_front_back / math.norm(vec_front_back)
    vec_left_right = vec_left_right / math.norm(vec_left_right)
    normal_vector = normal_vector / math.norm(normal_vector)

    # Create the rotation matrix
    rotation_matrix = Rotation.from_matrix(jnp.stack([vec_front_back, vec_left_right, normal_vector], axis=1))

    # Convert the rotation matrix to a quaternion
    quat = rotation_matrix.as_quat()

    return jnp.roll(quat,1)

@partial(jax.jit, static_argnums=(0,1,2,3,4,5))
def reference_generator(use_terrain_estimator,N,dt,n_joints,n_contact,mass,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    # proprio_height = input[6] + jnp.sum(contact*foot[2::3])/jnp.sum(contact)
    proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff), (N+1, 1))
    else:
        quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    q_ref = jnp.tile(q0, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
    Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    ref_lin_vel = Ryaw@Rpitch@input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    foot_ref = jnp.tile(foot, (N+1, 1))
    hip = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
    grf_ref = jnp.zeros((N+1, 3*n_contact))
    def foot_fn(t,carry):

        timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = timer_run(duty_factor, step_freq, timer_seq[t-1,:], dt)

        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)
        timer_seq = timer_seq.at[t,:].set(new_t)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(proprio_height/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + hip[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,val):
            a0 = current_foot
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            a0 = current_foot
            a3 = 8*step_height - 6*foothold -2*a0
            a2 = -foothold +a0 -2*a3
            a1 = +2*foothold -2*a0 +a3
            return a0 + a1*val + a2*val**2 + a3*val**3
        new_foot_x = jnp.where(new_contact_sequence>0, new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(new_contact_sequence>0, new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(new_contact_sequence>0, new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        grf_new = grf_new.at[t,2::3].set((new_contact_sequence*mass*9.81/(jnp.sum(new_contact_sequence)+1e-5)))

        return (timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new)

    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]
    timer_sequence_in = jnp.tile(t_timer, (N+1, 1))
    init_carry = (timer_sequence_in, contact_sequence,foot_ref,liftoff_x,liftoff_y,liftoff_z,grf_ref)
    timer_sequence, contact_sequence,foot_ref, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence,grf_ref], axis=1),jnp.concatenate([contact_sequence], axis=1), liftoff

@partial(jax.jit, static_argnums=(0,1,2))
def reference_generator_srbd(N,dt,n_contact,foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    dp = x[7:10]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    ref_lin_vel = input[:3]
    ref_ang_vel = input[3:6]
    robot_height = input[6]
    p = jnp.array([p[0], p[1], robot_height])
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * robot_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    rpy_ref = jnp.tile(jnp.array([0, 0, yaw]), (N+1, 1))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot = jnp.tile(p,n_contact) + foot0@jax.scipy.linalg.block_diag(Ryaw,Ryaw,Ryaw,Ryaw).T
    foot_ref_dot = jnp.zeros(((N+1), 3*n_contact))
    foot_ref_ddot = jnp.zeros(((N+1), 3*n_contact))
    grf_ref = jnp.zeros((N+1, 3*n_contact))
    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,new_foot_dot,new_foot_ddot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = timer_run(duty_factor, step_freq, new_t, dt)

        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(robot_height/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,val):
            a0 = current_foot
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            a0 = current_foot
            a3 = 8*step_height - 6*foothold -2*a0
            a2 = -foothold +a0 -2*a3
            a1 = +2*foothold -2*a0 +a3
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineXY_dot(current_foot, foothold,val):
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2
            return 2*a2*val + 3*a3*val**2

        def cubic_splineZ_dot(current_foot, foothold, step_height,val):
            a0 = current_foot
            a3 = 8*step_height - 6*foothold -2*a0
            a2 = -foothold +a0 -2*a3
            a1 = +2*foothold -2*a0 +a3
            return a1 + 2*a2*val + 3*a3*val**2

        def cubic_splineXY_ddot(current_foot, foothold,val):
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2
            return 2*a2 + 6*a3*val

        def cubic_splineZ_ddot(current_foot, foothold, step_height,val):
            a0 = current_foot
            a3 = 8*step_height - 6*foothold -2*a0
            a2 = -foothold +a0 -2*a3
            a1 = +2*foothold -2*a0 +a3
            return 2*a2 + 6*a3*val


        new_foot_x = jnp.where(new_contact_sequence>0, new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(new_contact_sequence>0, new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(new_contact_sequence>0, new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        new_foot_dot = new_foot_dot.at[t,::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_dot(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,1::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_dot(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,2::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineZ_dot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        new_foot_ddot = new_foot_ddot.at[t,::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor))))
        new_foot_ddot = new_foot_ddot.at[t,1::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor))))
        new_foot_ddot = new_foot_ddot.at[t,2::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineZ_ddot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        grf_new = grf_new.at[t,2::3].set((new_contact_sequence*500/jnp.sum(new_contact_sequence)))

        return (new_t, contact_sequence,new_foot,new_foot_dot,new_foot_ddot,liftoff_x,liftoff_y,liftoff_z,grf_new)

    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    init_carry = (t_timer, contact_sequence,foot_ref,foot_ref_dot,foot_ref_ddot,liftoff_x,liftoff_y,liftoff_z,grf_ref)
    _, contact_sequence,foot_ref,foot_ref_dot,foot_ref_ddot, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    return jnp.concatenate([p_ref, quat_ref, dp_ref, omega_ref,contact_sequence], axis=1),jnp.concatenate([ contact_sequence,foot_ref], axis=1), liftoff , foot_ref_dot,foot_ref_ddot

import mujoco
from mujoco import mjx

@partial(jax.jit, static_argnums=(0))
def whole_body_interface(model, mjx_model, contact_id, body_id,sim_frequency,Kp,Kd,qpos,qvel,grf,foot_ref,foot_ref_dot,foot_ref_ddot,J_old,contact):

    mjx_data = mjx.make_data(model)
    # Update the position and velocity in the data object
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    # Perform forward kinematics and dynamics computations
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # Extract the mass matrix and bias forces
    M = mjx_data.qM
    D = mjx_data.qfrc_bias

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
    current_leg_dot = J.T @ qvel
    cartesian_space_action = Kp@(foot_ref-current_leg) + Kd@(foot_ref_dot-current_leg_dot)

    accelleration = cartesian_space_action.T + foot_ref_ddot

    J_dot = (J - J_old)*sim_frequency
    tau_fb_lin = D[6:] + (M @ jnp.linalg.pinv(J.T) @ (accelleration - J_dot.T@qvel))[6:]
    tau_mpc = -(J@grf)[6:]
    tau_PD = (J @ cartesian_space_action.T)[6:]
    contact_mask = jnp.array([contact[0],contact[0],contact[0],contact[1],contact[1],contact[1],contact[2],contact[2],contact[2],contact[3],contact[3],contact[3]])
    tau = tau_mpc*contact_mask + (1-contact_mask)*(tau_PD) + tau_fb_lin

    return tau , J

@partial(jax.jit, static_argnums=(0,1,2,3))
def reference_barell_roll(N,dt,n_joints,n_contact,foot0,q0):
    t1 = 0.2
    t2 = 0.2
    t3 = 0.3
    t4 = 0.1
    z_start = 0.4
    z_land = 0.28
    v_lateral = -0.25/(t2+t3)
    v0 = (z_land - z_start + 0.5*9.81*t3*t3)/t3 
    total_roll_time = t2+t3+t4
    roll_speed = 2*3.14/total_roll_time
    def z_position(t):
        return z_start - 0.5*9.81*t**2 + v0*t
    def z_speed(t):
        return -9.81*t + v0
    acc = v0/t2
    print("v0", v0)
    print("acc", acc)
    #first part full stance 0.1s
    n1 = int(t1/dt)
    p1 = jnp.tile(jnp.array([0,0,0.33]), (n1, 1))
    p1 = p1.at[:,1].set(jnp.arange(n1)*dt*(v_lateral))
    dp1 = jnp.tile(jnp.array([0,v_lateral,0]), (n1, 1))
    contact1 = jnp.tile(jnp.array([1,1,1,1]), (n1, 1))
    quat1 = jnp.tile(jnp.array([1, 0, 0, 0]), (n1, 1))
    omega1 = jnp.tile(jnp.array([0, 0, 0]), (n1, 1))
    #second part lateral support 0.2s
    n2 = int(t2/dt)
    p2 = jnp.tile(jnp.array([0,p1[-1,1],0.33]), (n2, 1))
    p2 = p2.at[:,2].set(0.5*jnp.arange(n2)*dt*jnp.arange(n2)*dt*acc + 0.33)
    p2 = p2.at[:,1].set(jnp.arange(n2)*dt*(v_lateral))
    dp2 = jnp.tile(jnp.array([0,v_lateral,0]), (n2, 1))
    dp2 = dp2.at[:,2].set(jnp.arange(n2)*dt*acc)
    contact2 = jnp.tile(jnp.array([0,1,0,1]), (n2, 1))
    # for i in range(n2):
    #     p2 = p2.at[i,2].set(z_position(i*dt))
    #     dp2 = dp2.at[i,2].set(z_speed(i*dt))
    #third part flying phase 0.4s
    n3 = int(t3/dt)
    p3 = jnp.tile(jnp.array([0,p2[-1,1],p2[-1,2]]), (n3, 1))
    p3 = p3.at[:,1].set(jnp.arange(n3)*dt*(v_lateral))
    dp3 = jnp.tile(jnp.array([0,v_lateral,0]), (n3, 1))
    for i in range(n3):
        p3 = p3.at[i,2].set(z_position(i*dt))
        dp3 = dp3.at[i,2].set(z_speed(i*dt))
    def fn(t,carry):
        quat_new = math.quat_integrate(carry[t-1,:], jnp.array([roll_speed,0,0]), dt)
        carry_new = carry.at[t,:].set(quat_new)
        return carry_new
    
    
    contact3 = jnp.tile(jnp.array([0,0,0,0]), (n3, 1))
    #fourth part full stance 0.2s
    n4 = int(t4/dt)
    p4 = jnp.tile(jnp.array([0,p3[-1,1],z_land]), (n4, 1))
    dp4 = jnp.tile(jnp.array([0,0,0]), (n4, 1))
    quat5 = jnp.tile(jnp.array([1, 0, 0, 0]), (n4, 1))
    omega5 = jnp.tile(jnp.array([0, 0, 0]), (n4, 1))
    contact4 = jnp.tile(jnp.array([1,1,1,1]), (n4, 1))

    init_carry = jnp.tile(jnp.array([1.0, 0.0, 0, 0]), (n2+n3+n4, 1))
    quat234 = jax.lax.fori_loop(1, n2+n3+n4, fn, init_carry)
    omega234 = jnp.tile(jnp.array([roll_speed, 0, 0]), (n2+n3+n4, 1))

    n5 = N - (n1+n2+n3+n4)

    p5 = jnp.tile(jnp.array([0,p4[-1,1],z_land]), (n5, 1))
    dp5 = jnp.tile(jnp.array([0,0,0]), (n5, 1))
    quat5 = jnp.tile(jnp.array([1, 0, 0, 0]), (n5, 1))
    omega5 = jnp.tile(jnp.array([0, 0, 0]), (n5, 1))
    contact5 = jnp.tile(jnp.array([1,1,1,1]), (n5, 1))

    p_ref = jnp.concatenate([p1, p2, p3, p4,p5], axis=0)
    quat_ref = jnp.concatenate([quat1,quat234,quat5], axis=0)
    q_ref = jnp.tile(q0, (n1+n2+n3+n4+n5, 1))
    dp_ref = jnp.concatenate([dp1, dp2, dp3, dp4,dp5], axis=0)
    omega_ref = jnp.concatenate([omega1,omega234,omega5], axis=0)
    foot_ref = jnp.tile(foot0, (n1+n2+n3+n4+n5, 1)) + jnp.tile(p_ref, n_contact)
    foot_ref = foot_ref.at[:,2::3].set(jnp.zeros((n1+n2+n3+n4+n5, n_contact)))
    contact_sequence = jnp.concatenate([contact1, contact2, contact3, contact4,contact5], axis=0)

    grf_ref = jnp.zeros((N, 3*n_contact))

    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence, grf_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1)
