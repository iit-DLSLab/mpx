import jax
import qpax
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
    terrain_height = jnp.where(use_terrain_estimator, jnp.sum(liftoff[2::3]) / n_contact, 0.0)
    proprio_height = input[6] + terrain_height
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
    terrain_height = jnp.where(use_terrain_estimator, jnp.sum(liftoff[2::3]) / n_contact, 0.0)
    proprio_height = input[6] + terrain_height
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

def reference_generator_dfcip_online(
    N: int,
    dt: float,
    m: float,
    grav: float,
    h_min: float,
    h_max: float,
    x0: jax.Array,
    cmd: jax.Array,
):
    z_contact_ref = 0.0
    vcz_ref = 0.0

    a_default = 5.0
    alpha_default = 5.0
    az_default = 2.0
    z_position_gain = 2.0

    pcom0 = x0[0:3]
    vcom_z0 = x0[5]
    theta0 = x0[10]
    v0 = x0[11]
    omega0 = x0[12]

    v_des = cmd[0]
    omega_des = cmd[2]
    z_com_ref = jnp.clip(cmd[3], h_min, h_max)

    def move_towards(current, target, max_delta):
        delta = target - current
        return current + jnp.clip(delta, -max_delta, max_delta)

    def build_state(p_xy, z, theta, v, vx, vy, vz, omega):
        return jnp.array([
            p_xy[0],
            p_xy[1],
            z,
            vx,
            vy,
            vz,
            p_xy[0],
            p_xy[1],
            z_contact_ref,
            vcz_ref,
            theta,
            v,
            omega,
        ])

    def build_u_ref():
        return jnp.array([
            0.0,
            0.0,
            0.0,

            0.0,
            0.0,
            m * grav / 2.0,

            0.0,
            0.0,
            m * grav / 2.0,
        ])

    def scan_step(carry, _):
        p_xy, z, theta, v, vz, omega = carry

        v_next = move_towards(
            v,
            v_des,
            a_default * dt,
        )

        vz_target = z_position_gain * (z_com_ref - z)
        vz_next = move_towards(
            vz,
            vz_target,
            az_default * dt,
        )

        omega_next = move_towards(
            omega,
            omega_des,
            alpha_default * dt,
        )

        vx_next = v_next * jnp.cos(theta)
        vy_next = v_next * jnp.sin(theta)

        p_xy_next = p_xy + jnp.array([
            vx_next * dt,
            vy_next * dt,
        ])

        z_next = z + vz_next * dt
        theta_next = theta + omega_next * dt

        x_ref_t = build_state(
            p_xy=p_xy_next,
            z=z_next,
            theta=theta_next,
            v=v_next,
            vx=vx_next,
            vy=vy_next,
            vz=vz_next,
            omega=omega_next,
        )

        u_ref_t = build_u_ref()

        carry_next = (
            p_xy_next,
            z_next,
            theta_next,
            v_next,
            vz_next,
            omega_next,
        )

        return carry_next, (x_ref_t, u_ref_t)

    x_init = build_state(
        p_xy=pcom0[0:2],
        z=pcom0[2],
        theta=theta0,
        v=v0,
        vx=v0 * jnp.cos(theta0),
        vy=v0 * jnp.sin(theta0),
        vz=vcom_z0,
        omega=omega0,
    )
    u_init = build_u_ref()
    p_xy0 = pcom0[0:2]
    p_z = pcom0[2]

    carry0 = (
        p_xy0,
        p_z,
        theta0,
        v0,
        vcom_z0,
        omega0,
    )

    _, (x_scan, u_scan) = jax.lax.scan(
        scan_step,
        carry0,
        xs=None,
        length=N,
    )
    x_ref = jnp.concatenate([
        x_init[None, :],
        x_scan,
    ], axis=0)

    u_ref = jnp.concatenate([
        u_init[None, :],
        u_scan,
    ], axis=0)

    return x_ref, u_ref


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

def pack_reference(desired, nj):
    """Dict 'desired' → flat jnp array."""
    return jnp.concatenate([
        desired['com_pos'],       # 3
        desired['com_vel'],       # 3
        desired['com_acc'],       # 3
        desired['lwheel_pos'],    # 3
        desired['lwheel_vel'],    # 3
        desired['lwheel_acc'],    # 3
        desired['rwheel_pos'],    # 3
        desired['rwheel_vel'],    # 3
        desired['rwheel_acc'],    # 3
        desired['base_rot'],     # 9
        desired['base_omega'],    # 3
        desired['base_alpha'],    # 3
        desired['qjnt'],          # nj
        desired['qjntdot'],       # nj
        desired['qjntddot'],     # nj
    ])

def unpack_reference(ref, nj, layout):
    """Flat reference array -> dict of desired quantities, using the ``layout`` dict of offsets."""
    j = layout['joints']
    return dict(
        com_pos    = ref[layout['com_pos']    : layout['com_vel']],
        com_vel    = ref[layout['com_vel']    : layout['com_acc']],
        com_acc    = ref[layout['com_acc']    : layout['lwheel_pos']],
        lwheel_pos = ref[layout['lwheel_pos'] : layout['lwheel_vel']],
        lwheel_vel = ref[layout['lwheel_vel'] : layout['lwheel_acc']],
        lwheel_acc = ref[layout['lwheel_acc'] : layout['rwheel_pos']],
        rwheel_pos = ref[layout['rwheel_pos'] : layout['rwheel_vel']],
        rwheel_vel = ref[layout['rwheel_vel'] : layout['rwheel_acc']],
        rwheel_acc = ref[layout['rwheel_acc'] : layout['base_rot']],
        base_rot   = ref[layout['base_rot']   : layout['base_omega']].reshape((3, 3)),
        base_omega = ref[layout['base_omega'] : layout['base_alpha']],
        base_alpha = ref[layout['base_alpha'] : j],
        qjnt       = ref[j          : j + nj],
        qjntdot    = ref[j + nj     : j + 2 * nj],
        qjntddot   = ref[j + 2 * nj : j + 3 * nj],
    )

def get_rCP(wheel_R, radius):
    """Vector from wheel centre to the contact point."""
    I = jnp.eye(3)
    z_0 = jnp.array([0.0, 0.0, 1.0])
    n = wheel_R @ z_0
    a = (I - jnp.outer(n, n)) @ z_0
    s = a / jnp.linalg.norm(a)
    return -s * radius

def whole_body_interface_wheeled_legged_qp(
    mjx_model,
    # ── static (bound via functools.partial) ──────────────────────────────
    mass, grav, d,                       # kept for signature compatibility (unused here)
    contact_id, body_id, base_body_id,
    wheel_radius, sample_time, n_contacts,
    # ── gains ─────────────────────────────────────────────────────────────
    Kp_motion, Kd_motion,
    Kp_wheel, Kd_wheel,
    Kp_regulation, Kd_regulation,
    w_posture,
    # ── task weights ──────────────────────────────────────────────────────
    w_qddot, w_force, w_com, w_lwheel, w_rwheel, w_base,
    mu_,
    # ── runtime ───────────────────────────────────────────────────────────
    qpos, qvel, desired,
    posture_mask=None,
    ref_layout=None,
):
    """Wheeled-legged whole-body QP.
 
    Maps the MPC reference (``desired``) to actuated joint torques through a
    single inverse-dynamics QP that tracks the CoM, both wheels and the base
    orientation while enforcing rolling, floating-base dynamics, joint limits
    and friction cones.
 
    Returns ``(tau, qddot, fl, fr, diag)``. The public wrappers
    ``whole_body_interface_wheeled_legged_qp`` (drops ``diag``) and
    ``whole_body_interface_wheeled_legged_qp_diag`` (keeps it) are the entry
    points used by the controller wrappers.
    """
 
    # ══════════════════════════════════════════════════════════════════════
    #  0. LOCAL GEOMETRY HELPERS (self-contained frame / rotation math)
    # ══════════════════════════════════════════════════════════════════════
    def skew(v):
        """Skew-symmetric matrix from a (3,) vector."""
        return jnp.array([
            [0.0,  -v[2],  v[1]],
            [v[2],  0.0,  -v[0]],
            [-v[1], v[0],  0.0],
        ])
 
    def err_rotation(Ra, Rb):
        """Angle-axis orientation error of Ra w.r.t. Rb (faithful Eigen AngleAxis port)."""
        Rdiff = Rb.T @ Ra
        cos_angle = jnp.clip((jnp.trace(Rdiff) - 1.0) / 2.0, -1.0, 1.0)
        angle = jnp.arccos(cos_angle)
        skew_vec = jnp.array([
            Rdiff[2, 1] - Rdiff[1, 2],
            Rdiff[0, 2] - Rdiff[2, 0],
            Rdiff[1, 0] - Rdiff[0, 1],
        ]) * 0.5
        axis_norm = jnp.linalg.norm(skew_vec)
        safe_norm = jnp.where(axis_norm < 1e-7, 1.0, axis_norm)
        axis = skew_vec / safe_norm
        err = angle * (Ra @ axis)
        return jnp.where(angle < 1e-7, jnp.zeros(3), err)
 
    def compute_virtual_frame(wheel_R):
        """Virtual wheel frame [t, n, s]; n is the axle direction, s the ground-projected up."""
        I = jnp.eye(3)
        z_0 = jnp.array([0.0, 0.0, 1.0])
        n = wheel_R @ z_0
        a = (I - jnp.outer(n, n)) @ z_0
        s = a / jnp.linalg.norm(a)
        t = jnp.cross(n, s)
        t = t / jnp.linalg.norm(t)
        return jnp.column_stack([t, n, s])
 
    def compute_contact_frame(wheel_R):
        """Contact frame [t, z_0 x t, z_0] with a world-vertical normal."""
        z_0 = jnp.array([0.0, 0.0, 1.0])
        n = wheel_R @ z_0
        a = jnp.cross(n, z_0)
        t = a / jnp.linalg.norm(a)
        return jnp.column_stack([t, jnp.cross(z_0, t), z_0])
 
    # get_rCP is the module-level function (also used by the wrapper's state
    # processing), so it is shared here rather than re-defined.
 
    nv    = mjx_model.nv
    nj    = nv - 6                        
    n_f   = 3                             # force components per contact
    n_var = nv + 2 * n_f * n_contacts     # decision vars: [qddot | f_left | f_right]
 
    if ref_layout is None:
        raise ValueError(
            "ref_layout is required: pass the controller's index dict "
            "(e.g. BatchedMPCControllerWrapper._index_variables_qp)."
        )
    des = unpack_reference(desired, nj, ref_layout)
 
    # ══════════════════════════════════════════════════════════════════════
    #  2. FORWARD KINEMATICS, MASS MATRIX, BIAS, JACOBIANS (+ time derivatives)
    # ══════════════════════════════════════════════════════════════════════
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)
 
    M = mjx_data.qM            # mass matrix
    c = mjx_data.qfrc_bias     # Coriolis + gravity (RNEA with qddot = 0)
 
    q_wxyz = qpos[3:7]
    omega  = qvel[3:6]
    dqw = 0.5 * (-q_wxyz[1]*omega[0] - q_wxyz[2]*omega[1] - q_wxyz[3]*omega[2])
    dqx = 0.5 * ( q_wxyz[0]*omega[0] + q_wxyz[2]*omega[2] - q_wxyz[3]*omega[1])
    dqy = 0.5 * ( q_wxyz[0]*omega[1] - q_wxyz[1]*omega[2] + q_wxyz[3]*omega[0])
    dqz = 0.5 * ( q_wxyz[0]*omega[2] + q_wxyz[1]*omega[1] - q_wxyz[2]*omega[0])
    dqpos = jnp.concatenate([qvel[:3], jnp.array([dqw, dqx, dqy, dqz]), qvel[6:]])
 
    def _jacobians(qpos_):
        d_ = mjx.make_data(mjx_model).replace(qpos=qpos_, qvel=qvel)
        d_ = mjx.fwd_position(mjx_model, d_)

        def _jac_geom(geom_id, bid):
            Jp_, Jr_ = mjx.jac(mjx_model, d_, d_.geom_xpos[geom_id], bid)
            return Jp_.T, Jr_.T   # (3, nv), (3, nv)

        total_mass = jnp.sum(mjx_model.body_mass)
        nb = mjx_model.body_mass.shape[0]
        def _acc(i, J):
            Jp_i, _ = mjx.jac(mjx_model, d_, d_.xipos[i], i)
            return J + mjx_model.body_mass[i] * Jp_i.T
        J_com_ = jax.lax.fori_loop(1, nb, _acc, jnp.zeros((3, nv))) / total_mass

        _, J_base_rot_ = mjx.jac(mjx_model, d_, d_.xpos[base_body_id], base_body_id)
        return (_jac_geom(contact_id[0], body_id[0]), _jac_geom(contact_id[1], body_id[1]),
                J_base_rot_.T, J_com_)

    # Frame Jacobians and their time derivatives (Jdot) via forward-mode AD.
    ((J_left_wheel_lin,  J_left_wheel_rot), (J_right_wheel_lin, J_right_wheel_rot),
     J_base_link_rot, J_com), \
    ((J_left_wheel_dot_lin,  J_left_wheel_dot_rot), (J_right_wheel_dot_lin, J_right_wheel_dot_rot),
     J_base_link_dot_rot, J_com_dot) = jax.jvp(_jacobians, (qpos,), (dqpos,))
 
    J_left_wheel_  = jnp.vstack([J_left_wheel_lin,  J_left_wheel_rot])   # (6, nv)
    J_right_wheel_ = jnp.vstack([J_right_wheel_lin, J_right_wheel_rot])  # (6, nv)
 
    l_wheel_center = mjx_data.geom_xpos[contact_id[0]]
    l_wheel_R      = mjx_data.xmat[body_id[0]].reshape(3, 3)
    r_wheel_center = mjx_data.geom_xpos[contact_id[1]]
    r_wheel_R      = mjx_data.xmat[body_id[1]].reshape(3, 3)
 
    left_rCP  = get_rCP(l_wheel_R, wheel_radius)
    right_rCP = get_rCP(r_wheel_R, wheel_radius)
    w_l = J_left_wheel_rot  @ qvel
    w_r = J_right_wheel_rot @ qvel
 
    l_virtual_frame_R = compute_virtual_frame(l_wheel_R)
    r_virtual_frame_R = compute_virtual_frame(r_wheel_R)
    l_contact_frame   = compute_contact_frame(l_wheel_R)
    r_contact_frame   = compute_contact_frame(r_wheel_R)
 
    # Drift terms (Jdot * qdot).
    a_com_drift              = J_com_dot             @ qvel
    a_lwheel_drift           = J_left_wheel_dot_lin  @ qvel
    a_rwheel_drift           = J_right_wheel_dot_lin @ qvel
    a_base_orientation_drift = J_base_link_dot_rot   @ qvel
 
    current_com_pos    = mjx_data.subtree_com[0]
    current_com_vel    = J_com @ qvel
    current_lwheel_pos = l_wheel_center
    current_lwheel_vel = J_left_wheel_lin  @ qvel
    current_rwheel_pos = r_wheel_center
    current_rwheel_vel = J_right_wheel_lin @ qvel
 
    current_base_quat      = mjx_data.xquat[base_body_id]          # (w, x, y, z)
    current_base_quat_xyzw = jnp.array([
        current_base_quat[1], current_base_quat[2],
        current_base_quat[3], current_base_quat[0],
    ])
    current_base_quat_xyzw = current_base_quat_xyzw / (
        jnp.linalg.norm(current_base_quat_xyzw) + 1e-9)
    current_base_link_pos = jax.scipy.spatial.transform.Rotation.from_quat(
        current_base_quat_xyzw).as_matrix()
    current_base_link_vel = J_base_link_rot @ qvel
 
    # ══════════════════════════════════════════════════════════════════════
    #  4. DESIRED ACCELERATIONS (feedforward + PD per task)
    # ══════════════════════════════════════════════════════════════════════
    err_com     = des['com_pos'] - current_com_pos
    err_com_vel = des['com_vel'] - current_com_vel
    err_lwheel     = des['lwheel_pos'] - current_lwheel_pos
    err_lwheel_vel = des['lwheel_vel'] - current_lwheel_vel
    err_rwheel     = des['rwheel_pos'] - current_rwheel_pos
    err_rwheel_vel = des['rwheel_vel'] - current_rwheel_vel
    err_base_orientation     = err_rotation(des['base_rot'], current_base_link_pos)
    err_base_orientation_vel = des['base_omega'] - current_base_link_vel
 
    q_jnt    = qpos[7:]
    qdot_jnt = qvel[6:]
    err_posture     = jnp.concatenate([jnp.zeros(6), des['qjnt']    - q_jnt])
    err_posture_vel = jnp.concatenate([jnp.zeros(6), des['qjntdot'] - qdot_jnt])
 
    if posture_mask is None:
        matrix_with_no_wheel = jnp.eye(nj).at[3, 3].set(0.0).at[7, 7].set(0.0)
    else:
        matrix_with_no_wheel = jnp.diag(jnp.asarray(posture_mask, dtype=qpos.dtype))
    S = jax.scipy.linalg.block_diag(jnp.zeros((6, 6)), matrix_with_no_wheel)   # (nv, nv)
 
    desired_qddot  = jnp.concatenate([jnp.zeros(6), des['qjntddot']])
    a_jnt_total    = desired_qddot + Kp_regulation * err_posture + Kd_regulation * err_posture_vel
    a_com_total    = des['com_acc']    + Kp_motion * err_com    + Kd_motion * err_com_vel
    a_lwheel_total = des['lwheel_acc'] + Kp_wheel  * err_lwheel + Kd_wheel  * err_lwheel_vel
    a_rwheel_total = des['rwheel_acc'] + Kp_wheel  * err_rwheel + Kd_wheel  * err_rwheel_vel
    a_base_orientation_total = (
        des['base_alpha']
        + Kp_motion * err_base_orientation
        + Kd_motion * err_base_orientation_vel
    )
 
    # ══════════════════════════════════════════════════════════════════════
    #  5. COST  (0.5 x' H x + f' x)
    # ══════════════════════════════════════════════════════════════════════
    H_acc = w_qddot   * jnp.eye(nv)
    H_acc = H_acc + w_com     * (J_com.T            @ J_com)
    H_acc = H_acc + w_lwheel  * (J_left_wheel_lin.T  @ J_left_wheel_lin)
    H_acc = H_acc + w_rwheel  * (J_right_wheel_lin.T @ J_right_wheel_lin)
    H_acc = H_acc + w_base    * (J_base_link_rot.T   @ J_base_link_rot)
    H_acc = H_acc + w_posture * S
 
    f_acc =         w_com     * J_com.T            @ (a_com_drift              - a_com_total)
    f_acc = f_acc + w_lwheel  * J_left_wheel_lin.T  @ (a_lwheel_drift           - a_lwheel_total)
    f_acc = f_acc + w_rwheel  * J_right_wheel_lin.T @ (a_rwheel_drift           - a_rwheel_total)
    f_acc = f_acc + w_base    * J_base_link_rot.T   @ (a_base_orientation_drift - a_base_orientation_total)
    f_acc = f_acc - w_posture * (S @ a_jnt_total)
 
    H_force_one = w_force * jnp.eye(n_f * n_contacts)
    f_force_one = jnp.zeros(n_f * n_contacts)
 
    H = jax.scipy.linalg.block_diag(H_acc, H_force_one, H_force_one)   # (n_var, n_var)
    f = jnp.concatenate([f_acc, f_force_one, f_force_one])             # (n_var,)
 
    # ══════════════════════════════════════════════════════════════════════
    #  6. INEQUALITY: joint position / velocity limits (linearised over dt)
    # ══════════════════════════════════════════════════════════════════════
    jnt_ids   = jnp.arange(1, mjx_model.njnt)   # actuated joints (skip free joint id 0)
    q_jnt_min = mjx_model.jnt_range[jnt_ids, 0]
    q_jnt_max = mjx_model.jnt_range[jnt_ids, 1]
    vel_limit = 100.0 * jnp.ones(nj)            # rad/s (MuJoCo has no explicit vel limit)
 
    zeros_fb   = jnp.zeros((nj, 6))
    C_acc_ineq = jnp.block([
        [zeros_fb, sample_time            * jnp.eye(nj)],   # velocity rows: dt * qddot_jnt
        [zeros_fb, (sample_time**2 / 2.0) * jnp.eye(nj)],   # position rows: dt^2/2 * qddot_jnt
    ])                                                       # (2*nj, nv)
    qdd_limit_brake = 50.0  # rad/s^2
    pos_floor = (sample_time**2 / 2.0) * qdd_limit_brake
    d_min_acc = jnp.concatenate([
        -vel_limit - qdot_jnt,
        jnp.minimum(q_jnt_min - q_jnt - sample_time * qdot_jnt, pos_floor),
    ])
    d_max_acc = jnp.concatenate([
        vel_limit - qdot_jnt,
        jnp.maximum(q_jnt_max - q_jnt - sample_time * qdot_jnt, -pos_floor),
    ])
 
    # ══════════════════════════════════════════════════════════════════════
    #  7. INEQUALITY: friction cone per contact (+ small Fz floor)
    #
    #  The 4-row pyramid alone constrains |Fx|,|Fy| <= mu*Fz but leaves Fz
    #  unbounded below; a combined vx+omega command can drive the planned normal
    #  force negative, making the block infeasible and qpax return NaN torques.
    #  A small positive Fz floor keeps the friction block always feasible.
    # ══════════════════════════════════════════════════════════════════════
    fz_min = 5.0  # N, feasibility margin only (not a real load-bearing bound)
    C_force_block = jnp.array([
        [ 1.0,  0.0, -mu_],
        [ 0.0,  1.0, -mu_],
        [-1.0,  0.0, -mu_],
        [ 0.0, -1.0, -mu_],
        [ 0.0,  0.0, -1.0],   # Fz >= fz_min
    ])  # (5, 3)
    C_force_left = jax.scipy.linalg.block_diag(
        *[C_force_block @ l_contact_frame.T for _ in range(n_contacts)])
    C_force_right = jax.scipy.linalg.block_diag(
        *[C_force_block @ r_contact_frame.T for _ in range(n_contacts)])
    d_min_force_one = -10000.0 * jnp.ones(5 * n_contacts)
    d_max_force_one = jnp.tile(jnp.array([0.0, 0.0, 0.0, 0.0, -fz_min]), n_contacts)
 
    # ══════════════════════════════════════════════════════════════════════
    #  8. EQUALITY: rolling constraints (L, R) + floating-base dynamics
    # ══════════════════════════════════════════════════════════════════════
    I3 = jnp.eye(3)
 
    # Force-transmission maps T = [I3; skew(pcis)] (n_contacts = 1).
    left_rCP_local  = jnp.array([0.0, 0.0, -wheel_radius])
    right_rCP_local = jnp.array([0.0, 0.0, -wheel_radius])
    pcis_l = l_virtual_frame_R @ left_rCP_local
    pcis_r = r_virtual_frame_R @ right_rCP_local
    T_l = jnp.vstack([I3, skew(pcis_l)])   # (6, 3)
    T_r = jnp.vstack([I3, skew(pcis_r)])   # (6, 3)
 
    # Rolling (no-slip at the contact point): transport wheel-centre motion to
    # the contact point through rCP and remove the virtual spin component.
    nl = l_virtual_frame_R[:, 1]
    nr = r_virtual_frame_R[:, 1]
    wl_virtual = (I3 - jnp.outer(nl, nl)) @ w_l
    wr_virtual = (I3 - jnp.outer(nr, nr)) @ w_r
 
    A_roll_L = J_left_wheel_lin  - skew(left_rCP)  @ J_left_wheel_rot
    b_roll_L = (
        -(J_left_wheel_dot_lin  - skew(left_rCP)  @ J_left_wheel_dot_rot)  @ qvel
        - jnp.cross(w_l, jnp.cross(wl_virtual, left_rCP))
    )
    A_roll_R = J_right_wheel_lin - skew(right_rCP) @ J_right_wheel_rot
    b_roll_R = (
        -(J_right_wheel_dot_lin - skew(right_rCP) @ J_right_wheel_dot_rot) @ qvel
        - jnp.cross(w_r, jnp.cross(wr_virtual, right_rCP))
    )
 
    # Floating-base (unactuated) dynamics: Mu*qddot - Jlu'*T_l*fl - Jru'*T_r*fr = -cu
    Mu, cu   = M[:6, :], c[:6]
    Ma, ca   = M[6:, :], c[6:]
    Jlu, Jla = J_left_wheel_[:, :6],  J_left_wheel_[:, 6:]
    Jru, Jra = J_right_wheel_[:, :6], J_right_wheel_[:, 6:]
    A_dyn = jnp.hstack([Mu, -(Jlu.T @ T_l), -(Jru.T @ T_r)])   # (6, n_var)
    b_dyn = -cu
 
    zeros_roll_f = jnp.zeros((3, 2 * n_f * n_contacts))
    A_eq = jnp.vstack([
        jnp.hstack([A_roll_L, zeros_roll_f]),   # rolling L (3, n_var)
        jnp.hstack([A_roll_R, zeros_roll_f]),   # rolling R (3, n_var)
        A_dyn,                                   # dynamics  (6, n_var)
    ])
    b_eq = jnp.concatenate([b_roll_L, b_roll_R, b_dyn])
 
    # ══════════════════════════════════════════════════════════════════════
    #  9. ASSEMBLE INEQUALITIES AND SOLVE
    #     qpax: min 0.5 x'Hx + f'x  s.t.  A_eq x = b_eq,  G x <= h
    # ══════════════════════════════════════════════════════════════════════
    zeros_acc_f = jnp.zeros((5 * n_contacts, nv))
    zeros_f_f   = jnp.zeros((5 * n_contacts, n_f * n_contacts))
    C_ineq = jnp.block([
        [C_acc_ineq,  jnp.zeros((2 * nj, n_f * n_contacts)), jnp.zeros((2 * nj, n_f * n_contacts))],
        [zeros_acc_f, C_force_left,                          zeros_f_f                            ],
        [zeros_acc_f, zeros_f_f,                             C_force_right                        ],
    ])
    d_min_ineq = jnp.concatenate([d_min_acc, d_min_force_one, d_min_force_one])
    d_max_ineq = jnp.concatenate([d_max_acc, d_max_force_one, d_max_force_one])
 
    G = jnp.vstack([C_ineq, -C_ineq])
    h = jnp.concatenate([d_max_ineq, -d_min_ineq])
 
    x, s, z, y, converged, iters = qpax.solve_qp(H, f, A_eq, b_eq, G, h, solver_tol=1e-3)
 
    qddot = x[:nv]
    fl    = x[nv                    : nv + n_f * n_contacts]
    fr    = x[nv + n_f * n_contacts : nv + 2 * n_f * n_contacts]
 
    # ══════════════════════════════════════════════════════════════════════
    #  10. INVERSE DYNAMICS -> actuated torques
    # ══════════════════════════════════════════════════════════════════════
    tau = Ma @ qddot + ca - Jla.T @ T_l @ fl - Jra.T @ T_r @ fr
 
    # ══════════════════════════════════════════════════════════════════════
    #  11. DIAGNOSTICS (unused by the control law; consumed by the *_diag wrapper)
    # ══════════════════════════════════════════════════════════════════════
    fl_local = l_contact_frame.T @ fl   # contact frame: [tangent, lateral, normal]
    fr_local = r_contact_frame.T @ fr
    ineq_slack = h - G @ x              # >= 0 when satisfied
    n_jl = 2 * nj
    diag = dict(
        converged=converged,
        iters=iters,
        # task residuals: J qddot + Jdot qdot - a_total (0 = task exactly realised)
        res_com=J_com @ qddot + a_com_drift - a_com_total,
        res_lwheel=J_left_wheel_lin @ qddot + a_lwheel_drift - a_lwheel_total,
        res_rwheel=J_right_wheel_lin @ qddot + a_rwheel_drift - a_rwheel_total,
        res_base=J_base_link_rot @ qddot + a_base_orientation_drift - a_base_orientation_total,
        # accelerations actually requested to the QP (feedforward + PD)
        a_com_total=a_com_total,
        a_lwheel_total=a_lwheel_total,
        a_rwheel_total=a_rwheel_total,
        a_base_total=a_base_orientation_total,
        # PD error terms feeding those requests
        err_com=err_com,
        err_com_vel=err_com_vel,
        err_lwheel=err_lwheel,
        err_rwheel=err_rwheel,
        err_base=err_base_orientation,
        # constraint satisfaction
        eq_res_norm=jnp.linalg.norm(A_eq @ x - b_eq),
        ineq_slack_min_joint=jnp.min(jnp.concatenate([
            ineq_slack[:n_jl],
            ineq_slack[G.shape[0] // 2: G.shape[0] // 2 + n_jl],
        ])),
        # friction margins in the contact frame: [mu Fz - |Fx|, mu Fz - |Fy|, Fz - fz_min]
        fric_margin_l=jnp.array([mu_ * fl_local[2] - jnp.abs(fl_local[0]),
                                 mu_ * fl_local[2] - jnp.abs(fl_local[1]),
                                 fl_local[2] - fz_min]),
        fric_margin_r=jnp.array([mu_ * fr_local[2] - jnp.abs(fr_local[0]),
                                 mu_ * fr_local[2] - jnp.abs(fr_local[1]),
                                 fr_local[2] - fz_min]),
        fl_local=fl_local,
        fr_local=fr_local,
    )
    return tau, qddot, fl, fr


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
