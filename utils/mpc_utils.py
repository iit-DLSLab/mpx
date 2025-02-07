import jax 
from jax import numpy as jnp
from functools import partial

@jax.jit
def timer_run(duty_factor,step_freq, leg_time, dt):
    # Extract relevant fields
    # Update timer
    leg_time = leg_time + dt * step_freq
    leg_time = jnp.where(leg_time > 1, leg_time - 1, leg_time)
    contact = jnp.where(leg_time < duty_factor, 1, 0)

    return contact, leg_time

@partial(jax.jit, static_argnums=(0,1,2,3))
def reference_generator(N,dt,n_joints,n_contact,foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    ref_lin_vel, ref_ang_vel, robot_height = input
    p = jnp.array([p[0], p[1], robot_height])
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * robot_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    q_ref = jnp.tile(jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8]), (N+1, 1))
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot = jnp.tile(p,n_contact) + foot0@jax.scipy.linalg.block_diag(Ryaw,Ryaw,Ryaw,Ryaw).T
    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z = carry

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
        new_foot_x = jnp.where(new_contact_sequence>0, new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(new_contact_sequence>0, new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(new_contact_sequence>0, new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        return (new_t, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z)
    
    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    init_carry = (t_timer, contact_sequence,foot_ref,liftoff_x,liftoff_y,liftoff_z)
    _, contact_sequence,foot_ref, liftoff_x,liftoff_y,liftoff_z = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref], axis=1), contact_sequence, liftoff