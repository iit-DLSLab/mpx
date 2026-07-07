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
def terrain_orientation(liftoff_pos,Ryaw):

    # Calculate the vectors between the legs
    vec_front_back = (liftoff_pos[:3] + liftoff_pos[3:6] - liftoff_pos[6:9] - liftoff_pos[9:12])/2
    # vec_left_right = (liftoff_pos[:3] + liftoff_pos[6:9] - liftoff_pos[3:6] - liftoff_pos[9:12])/2
    #DO NOT ADJUST THE ROLL
    vec_left_right = Ryaw@jnp.array([0,1,0])
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

@partial(jax.jit, static_argnums=(0,1,2,3,4,5), static_argnames=("extra_ref_fun",))
def reference_generator(use_terrain_estimator,N,dt,n_joints,n_contact,mass,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff,contact,clearence_speed,current_time,extra_ref_fun=None):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff,Ryaw), (N+1, 1))
    else:
        quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    q_ref = jnp.tile(q0, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
    Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    
    ref_lin_vel = Ryaw@Rpitch@input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot0_projected = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
    grf_ref = jnp.zeros((N+1, 3*n_contact))

    #Estimate Early contact
    des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(jnp.logical_and(jnp.logical_and(des_contact==0,contact==1),current_timer > 0.5 + 0.5*duty_factor),1,0)    

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
            f2 = jnp.sqrt(input[6]/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0_projected[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,initial_velocity,val):
            a0 = current_foot
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7

            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            e = current_foot
            return a*val**4 + b*val**3 + c*val**2 + d*val + e
        
        initial_speed = - ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

        new_foot_x = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

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

    ref = {"p": p_ref, "quat": quat_ref, "q": q_ref, "dp": dp_ref, "omega": omega_ref, "foot": foot_ref, "contact": contact_sequence, "grf": grf_ref}
    if extra_ref_fun is not None:
        ref = extra_ref_fun(ref, current_time)

    return ref, jnp.concatenate([contact_sequence], axis=1), liftoff

@partial(jax.jit, static_argnums=(0,1,2,3))
def reference_generator_srbd(use_terrain_estimator,N,dt,n_contact,mass,foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff,contact,clearence_speed):
    p = x[:3]
    quat = x[3:7]
    dp = x[7:10]
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff,Ryaw), (N+1, 1))
    else:
        quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
    Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    ref_lin_vel = Ryaw@Rpitch@input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot_ref_dot = jnp.zeros(((N+1), 3*n_contact))
    foot0_projected = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
    grf_ref = jnp.zeros((N+1, 3*n_contact))

    #Estimate Early contact
    des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(jnp.logical_and(jnp.logical_and(des_contact==0,contact==1),current_timer > 0.5 + 0.5*duty_factor),1,0)
    
    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = timer_run(duty_factor, step_freq, new_t, dt)

        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        def calc_foothold(direction):
            f1 = 0.5*dp[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(input[6]/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0_projected[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,initial_velocity,val):
            a0 = current_foot
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7

            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            e = current_foot
            return a*val**4 + b*val**3 + c*val**2 + d*val + e

        def cubic_splineXY_dot(current_foot, foothold,initial_velocity,val):
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return 2*a2*val + 3*a3*val**2 + a1

        def cubic_splineZ_dot(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7
            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            return 4*a*val**3 + 3*b*val**2 + 2*c*val + d

        new_foot_x = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        new_foot_dot = new_foot_dot.at[t,::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineXY_dot(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,1::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineXY_dot(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,2::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineZ_dot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        # new_foot_ddot = new_foot_ddot.at[t,::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor))))
        # new_foot_ddot = new_foot_ddot.at[t,1::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor))))
        # new_foot_ddot = new_foot_ddot.at[t,2::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineZ_ddot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        grf_new = grf_new.at[t,2::3].set((new_contact_sequence*mass*9.81/(jnp.sum(new_contact_sequence)+1e-5)))

        return (new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z,grf_new)

    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    initial_speed = - ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

    init_carry = (t_timer, contact_sequence,foot_ref,foot_ref_dot,liftoff_x,liftoff_y,liftoff_z,grf_ref)
    _, contact_sequence,foot_ref,foot_ref_dot, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    ref = {"p": p_ref, "quat": quat_ref, "dp": dp_ref, "omega": omega_ref, "contact": contact_sequence}
    return ref,jnp.concatenate([ contact_sequence,foot_ref], axis=1), liftoff , foot_ref_dot

import mujoco
from mujoco import mjx

@partial(jax.jit, static_argnums=(0))
def whole_body_interface(model, mjx_model, contact_id, body_id,sim_frequency,Kp,Kd,qpos,qvel,grf,foot_ref,foot_ref_dot,contact):

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
    current_leg_dot = J.T @ mjx_data.qvel
    cartesian_space_action = Kp@(foot_ref-current_leg) + Kd@(foot_ref_dot-current_leg_dot)
    tau_fb_lin = D[6:] + (M @ jnp.linalg.pinv(J.T) @ (cartesian_space_action))[6:]
    tau_mpc = -(J@grf)[6:]
    tau_PD = (J @ cartesian_space_action)[6:]
    contact_mask = jnp.array([contact[0],contact[0],contact[0],contact[1],contact[1],contact[1],contact[2],contact[2],contact[2],contact[3],contact[3],contact[3]])
    tau = tau_mpc*contact_mask + (1-contact_mask)*(tau_PD + tau_fb_lin) 

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

    ref = {"p": p_ref, "quat": quat_ref, "q": q_ref, "dp": dp_ref, "omega": omega_ref, "foot": foot_ref, "contact": contact_sequence, "grf": grf_ref}
    return ref, jnp.concatenate([contact_sequence, foot_ref], axis=1)


def reference_humanoid_jump_forward(
    N,
    dt,
    n_joints,
    n_contact,
    foot0,
    q0,
    *,
    base_height=0.9,
    crouch_height=0.82,
    apex_height=1.02,
    jump_distance=0.35,
    foot_shift=0.18,
    foot_lift=0.12,
):
    n_crouch = max(2, int(0.20 / dt))
    n_flight = max(2, int(0.28 / dt))
    n_land = max(2, int(0.18 / dt))
    n_settle = max(0, N - (n_crouch + n_flight + n_land))

    x_crouch = jnp.linspace(0.0, 0.05, n_crouch)
    x_flight = jnp.linspace(x_crouch[-1], jump_distance, n_flight)
    x_land = jnp.linspace(jump_distance, jump_distance, n_land)
    x_settle = jnp.linspace(jump_distance, jump_distance, n_settle)

    z_crouch = jnp.linspace(base_height, crouch_height, n_crouch)
    phase = jnp.linspace(0.0, 1.0, n_flight)
    z_flight = crouch_height + (base_height - crouch_height) * phase + (apex_height - base_height) * 4.0 * phase * (1.0 - phase)
    z_land = jnp.linspace(base_height, base_height, n_land)
    z_settle = jnp.linspace(base_height, base_height, n_settle)

    x_ref = jnp.concatenate([x_crouch, x_flight, x_land, x_settle], axis=0)
    z_ref = jnp.concatenate([z_crouch, z_flight, z_land, z_settle], axis=0)
    y_ref = jnp.zeros_like(x_ref)
    p_ref = jnp.stack([x_ref, y_ref, z_ref], axis=1)

    quat_ref = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    omega_ref = jnp.zeros((N, 3))

    crouch_q = q0.at[2].set(-0.8).at[3].set(1.5).at[4].set(-0.8)
    crouch_q = crouch_q.at[7].set(-0.8).at[8].set(1.5).at[9].set(-0.8)
    q_crouch = jnp.stack([q0 + (crouch_q - q0) * alpha for alpha in jnp.linspace(0.0, 1.0, n_crouch)], axis=0)
    q_flight = jnp.tile(crouch_q, (n_flight, 1))
    q_land = jnp.stack([crouch_q + (q0 - crouch_q) * alpha for alpha in jnp.linspace(0.0, 1.0, n_land)], axis=0)
    q_settle = jnp.tile(q0, (n_settle, 1))
    q_ref = jnp.concatenate([q_crouch, q_flight, q_land, q_settle], axis=0)

    dp_ref = jnp.zeros((N, 3))
    dp_ref = dp_ref.at[:-1].set((p_ref[1:] - p_ref[:-1]) / dt)
    dp_ref = dp_ref.at[-1].set(dp_ref[-2])

    foot_ref = jnp.tile(foot0, (N, 1))
    flight_shift = foot_shift * phase
    flight_lift = foot_lift * 4.0 * phase * (1.0 - phase)
    foot_flight = jnp.tile(foot0, (n_flight, 1))
    foot_flight = foot_flight.at[:, ::3].set(foot_flight[:, ::3] + flight_shift[:, None])
    foot_flight = foot_flight.at[:, 2::3].set(foot_flight[:, 2::3] + flight_lift[:, None])
    foot_land = jnp.tile(
        foot0.at[::3].set(foot0[::3] + foot_shift),
        (n_land + n_settle, 1),
    )
    foot_ref = foot_ref.at[n_crouch : n_crouch + n_flight].set(foot_flight)
    if n_land + n_settle > 0:
        foot_ref = foot_ref.at[n_crouch + n_flight :].set(foot_land)

    contact_crouch = jnp.tile(jnp.ones(n_contact), (n_crouch, 1))
    contact_flight = jnp.tile(jnp.zeros(n_contact), (n_flight, 1))
    contact_land = jnp.tile(jnp.ones(n_contact), (n_land + n_settle, 1))
    contact_sequence = jnp.concatenate([contact_crouch, contact_flight, contact_land], axis=0)

    grf_ref = jnp.zeros((N, 3 * n_contact))

    reference = {"p": p_ref, "quat": quat_ref, "q": q_ref, "dp": dp_ref, "omega": omega_ref, "foot": foot_ref, "contact": contact_sequence, "grf": grf_ref}
    parameter = jnp.concatenate([contact_sequence, foot_ref], axis=1)
    return reference, parameter


def reference_quadruped_trot_two_step(
    N,
    dt,
    n_joints,
    n_contact,
    foot0,
    q0,
    *,
    base_height=0.36,
    total_forward=0.45,
    step_length=0.16,
    step_height=0.08,
    settle_time=0.10,
    phase_time=0.16,
):
    del n_joints
    n_stance = max(2, int(settle_time / dt))
    n_phase = max(2, int(phase_time / dt))
    n_phases = 4
    n_settle = max(0, N - (n_stance + n_phases * n_phase))

    p_ref = jnp.zeros((N, 3))
    p_ref = p_ref.at[:, 0].set(jnp.linspace(0.0, total_forward, N))
    p_ref = p_ref.at[:, 2].set(base_height)
    quat_ref = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    q_ref = jnp.tile(q0, (N, 1))
    dp_ref = jnp.zeros((N, 3))
    dp_ref = dp_ref.at[:, 0].set(total_forward / ((N - 1) * dt + 1e-6))
    omega_ref = jnp.zeros((N, 3))

    foot_ref = jnp.tile(foot0, (N, 1))
    footholds = foot0.reshape(n_contact, 3)
    contact_sequence = jnp.tile(jnp.ones(n_contact), (N, 1))

    trot_a = jnp.array([1.0, 0.0, 0.0, 1.0])
    trot_b = jnp.array([0.0, 1.0, 1.0, 0.0])
    patterns = [trot_a, trot_b, trot_a, trot_b]

    start_idx = n_stance
    for pattern in patterns:
        end_idx = min(start_idx + n_phase, N)
        contact_sequence = contact_sequence.at[start_idx:end_idx].set(
            jnp.tile(pattern, (end_idx - start_idx, 1))
        )
        swing_ids = jnp.where(pattern == 0.0)[0]
        phase = jnp.linspace(0.0, 1.0, end_idx - start_idx)
        start_feet = footholds
        end_feet = footholds.at[swing_ids, 0].add(step_length)
        swing_xyz = start_feet[None, :, :] + (end_feet - start_feet)[None, :, :] * phase[:, None, None]
        swing_xyz = swing_xyz.at[:, swing_ids, 2].set(
            start_feet[swing_ids, 2][None, :] + step_height * 4.0 * phase[:, None] * (1.0 - phase[:, None])
        )
        foot_ref = foot_ref.at[start_idx:end_idx].set(swing_xyz.reshape(end_idx - start_idx, -1))
        footholds = end_feet
        start_idx = end_idx

    if start_idx < N:
        foot_ref = foot_ref.at[start_idx:].set(jnp.tile(footholds.reshape(-1), (N - start_idx, 1)))
        contact_sequence = contact_sequence.at[start_idx:].set(jnp.tile(jnp.ones(n_contact), (N - start_idx, 1)))

    grf_ref = jnp.zeros((N, 3 * n_contact))
    reference = {"p": p_ref, "quat": quat_ref, "q": q_ref, "dp": dp_ref, "omega": omega_ref, "foot": foot_ref, "contact": contact_sequence, "grf": grf_ref}
    parameter = jnp.concatenate([contact_sequence, foot_ref], axis=1)
    return reference, parameter
