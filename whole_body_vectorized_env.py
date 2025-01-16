import mujoco 
import mujoco.viewer
from mujoco import mjx 
import jax 
import primal_dual_ilqr.utils.mpc_utils as mpc_utils
import  primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from primal_dual_ilqr.utils.rotation import quaternion_integration,rpy_intgegration,quaternion_to_rpy
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
from timeit import default_timer as timer
from jax import numpy as jnp
from mujoco.mjx._src import math

 
model = mujoco.MjModel.from_xml_path('./data/go2/scene_mjx.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

joints_name = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]
contact_frame = ['FL','FR','RL','RR']
body_name = ['FL_hip', 'FL_thigh', 'FL_calf',
    'FR_hip', 'FR_thigh', 'FR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf'
]
contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))
n_joints = len(joints_name)
n_contact = len(contact_frame)
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
p_legs0 = jnp.array([ 0.192, 0.142, 0.024,  0.192, -0.142, 0.024,-0.195,  0.142,  0.024, -0.195, -0.142, 0.024])
# Problem dimensions
N = 30  # Number of stages
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
dt = 0.04  # Time step

@jax.jit
def reference_generator(t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
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
    foot_ref = jnp.tile(foot-jnp.tile(p,(1,n_contact)), (N+1, 1))
    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = mpc_utils.timer_run(duty_factor, step_freq, new_t, dt)
        
        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        foot0 = jnp.array([ 0.192, 0.142, 0.024,  0.192, -0.142, 0.024,-0.195,  0.142,  0.024, -0.195, -0.142, 0.024])
        
        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(robot_height/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0[direction::3]
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
            # a0 = current_foot
            # a1 = 0
            # a2 = 8*(step_height) - foothold + current_foot
            # a3 = 8*(step_height) - 2*a2
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

    # foot to world frame
    foot_ref = foot_ref + jnp.tile(p,(N+1,n_contact))
    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1), liftoff

alpha = 25

@jax.jit
def dynamics(x, u, t, parameter):

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:2*n_joints+13])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t,:4]
    p_legs = parameter[t,4:]

    tau = jnp.concatenate([jnp.zeros(6),u])

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[2])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[5])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[8])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[11])

    J = jnp.concatenate([J_FL,J_FR,J_RL,J_RR],axis=1)
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0)
    g = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0) - p_legs # position-level constraint violation
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]  # Velocity-level constraint violation

    # Stabilization term
    baumgarte_term = - 2*alpha * g_dot #- beta * beta * g

    JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)
    # Finate diference Jdot
    # #integrate qpos with a really small dt
    # h = 1e-6
    # delta_p = jnp.concatenate([x[:3] + x[7 + n_joints:10 + n_joints]*h, math.quat_integrate(x[3:7], x[10 + n_joints:13 + n_joints], h), x[7:7+n_joints] + x[13 + n_joints:]*h])
    # mjx_data = mjx_data.replace(qpos = delta_p[:n_joints+7])
    # mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    # delta_J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[2])
    # delta_J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[5])
    # delta_J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[8])
    # delta_J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[11])
    # delta_J = jnp.concatenate([delta_J_FL,delta_J_FR,delta_J_RL,delta_J_RR],axis=1)
    # Jdot = (delta_J - J)/h

    rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False),tau - D) + baumgarte_term #+ Jdot.T@x[n_joints+7:] 
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ,rhs)
    grf = jnp.concatenate([grf[:3]*contact[0],grf[3:6]*contact[1],grf[6:9]*contact[2],grf[9:12]*contact[3]])
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M,False),tau - D + J@grf)*dt

    # Semi-implicit Euler integration
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    x_next = jnp.concatenate([p, quat, q, v, current_leg,grf])

    return x_next

p0 = jnp.array([0, 0, 0.28])
quat0 = jnp.array([1, 0, 0, 0])
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
x0 = jnp.concatenate([p0, quat0,q0, jnp.zeros(6+n_joints),p_legs0,jnp.zeros(3*n_contact)])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.28])
quat_ref = jnp.array([1, 0, 0, 0])
rpy_ref = jnp.array([0, 0, 0])
q_ref = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

grf_ref = jnp.zeros(3 * n_contact)
tau_ref = jnp.zeros(n_joints)

u_ref = jnp.concatenate([tau_ref])

Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qdp = jnp.diag(jnp.array([1, 1, 1]))*1e3
Qomega = jnp.diag(jnp.array([1, 1, 1]))*1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Rgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
Qrot = jnp.diag(jnp.array([500,500,0]))
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e5]),n_contact))
Qpenalty = jnp.diag(jnp.ones(5*n_contact))
QpenaltyZ = jnp.diag(jnp.ones(3*n_contact))*10
# Define the cost function
@jax.jit
def cost(x, u, t, reference):

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
    p_leg_ref = reference[t,13+n_joints:]
    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos = x[:n_joints+7], qvel = x[n_joints+7:13+2*n_joints])

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)

    mu = 0.4
    friction_cone = jnp.array([[0,0,1],[-1,0,mu],[1,0,mu],[0,-1,mu],[0,1,mu]])
    friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    friction_cone = friction_cone @ grf
    alpha = 0.1
    #use ln(1+exp(x)) as a smooth approximation of max(0,x)
    friction_cone = 1/alpha*(jnp.log1p(jnp.exp(-alpha*friction_cone)))
    # delta = 0.0001
    # alpha_swing = 0.1
    # swing_z_plus = p_leg-p_leg_ref #+ jnp.ones(3*n_contact)*delta
    # swing_z_plus = 1/alpha_swing*(jnp.log1p(jnp.exp(-alpha_swing*swing_z_plus)))
    # swing_z_minus = p_leg-p_leg_ref #- jnp.ones(3*n_contact)*delta
    # swing_z_minus = 1/alpha_swing*(jnp.log1p(jnp.exp(-alpha_swing*swing_z_minus)))

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) +  (q - q_ref).T @ Qq @ (q - q_ref) + math.quat_sub(quat,quat_ref).T@Qrot@math.quat_sub(quat,quat_ref) +\
                 (dp - dp_ref).T @ Qdp @ (dp - dp_ref) + (omega - omega_ref).T @ Qomega @ (omega - omega_ref) + dq.T @ Qdq @ dq +\
                 tau.T @ Qtau @ tau +\
                 (p_leg - p_leg_ref).T @ Qleg @ (p_leg - p_leg_ref) #+\
                #  friction_cone.T @ Qpenalty @ friction_cone
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)


    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
U0 = jnp.tile(u_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))
reference = jnp.tile(jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref,p_legs0]), (N + 1, 1))
parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))
from timeit import default_timer as timer
@jax.jit
def work(reference,parameter,x0,X0,U0,V0):
    return optimizers.mpc(
        cost,
        dynamics,
        reference,
        parameter,
        x0,
        X0,
        U0,
        V0,
    )
duty_factor = 0.6
step_freq = 1.35
@jax.jit
def mpc_step(model, data, X0, U0, V0,liftoff,time_t,t):

    FL_leg = data.geom_xpos[contact_id[0]]
    FR_leg = data.geom_xpos[contact_id[1]]
    RL_leg = data.geom_xpos[contact_id[2]]
    RR_leg = data.geom_xpos[contact_id[3]]
    p_legs = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0)
    parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs]),(N+1,1))
    # # call tha work only at 100Hz 
    input = (jnp.array([0,0,0]), jnp.array([0,0,0]), 0.28)
    x0 = jnp.concatenate([data.qpos, data.qvel,p_legs,jnp.zeros(3*n_contact)])
    @jax.jit
    def reference_and_contorl():
        _ , new_time_t = mpc_utils.timer_run(duty_factor = 0.6, step_freq = 1.35,leg_time=time_t, dt=dt)
        t_ref = jnp.array([0,0,0,0])
        t_ref = new_time_t
        reference , parameter , new_liftoff = reference_generator(t_ref, jnp.concatenate([data.qpos,data.qvel]), p_legs, input, duty_factor = 0.6,  step_freq= 1.35 ,step_height=0.08,liftoff=liftoff)
        new_X0, new_U0, new_V0 = work(reference,parameter,x0,X0,U0,V0)
        return new_X0, new_U0, new_V0,new_liftoff,new_time_t
    X_new, U_new, V_new, liftoff_new, time_t_new = jax.lax.cond(t % 8 == 0, lambda _: reference_and_contorl(), lambda _: (X0, U0, V0,liftoff,time_t), None)
    tau = U_new[0,:n_joints]
    data = data.replace(ctrl=tau)
    data = mjx.step(model, data)
    
    return data, X_new, U_new, V_new,liftoff_new,time_t_new
def integrate_mpc(model, data, X0, U0, V0,liftoff,time_t):
    def scan_fn(carry, t):
        data, X0, U0, V0,liftoff,time_t = carry
        data_new, X_new, U_new, V_new ,liftoff_new,time_t_new= mpc_step(model, data, X0, U0, V0,liftoff,time_t,t)
        return (data_new, X_new, U_new, V_new ,liftoff_new,time_t_new), None
    t = jnp.arange(200)
    (new_data, new_X0, new_U0, new_V0,new_liftoff,new_time_t), _ = jax.lax.scan(scan_fn, (data, X0, U0, V0,liftoff,time_t), t)
    return new_data, new_X0, new_U0, new_V0
frames = []
mujoco.mj_resetData(model, data)
mjx_data = mjx.put_data(model, data)
qpos = jax.numpy.array([0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
n_env = 128
qpos0 = jax.numpy.tile(qpos, (n_env, 1))
batch = jax.vmap(lambda x: mjx_data.replace(qpos=x))(qpos0)
batch_U0 = jnp.tile(U0, (n_env, 1, 1))
batch_X0 = jnp.tile(X0, (n_env, 1, 1))
batch_V0 = jnp.tile(V0, (n_env, 1, 1))
batch_liftoff = jnp.tile(p_legs0, (n_env, 1))
batch_time_t = jnp.tile(jnp.array([0.5,0.0,0.0,0.5]),(n_env, 1))

jit_step = jax.jit(jax.vmap(integrate_mpc, in_axes=(None, 0,0,0,0,0,0)))
batch,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0,batch_liftoff,batch_time_t)
start = timer()
batch1,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0,batch_liftoff,batch_time_t)
end = timer()
print(f"Time elapsed: {end - start} s")
 
 
start = timer()
batch1,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0,batch_liftoff,batch_time_t)
end = timer()
print(f"Time elapsed 2: {end - start} s")
 
 
start = timer()
batch2,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0,batch_liftoff,batch_time_t)
end = timer()
print(f"Time elapsed 3: {end - start} s")

# jit_mpc = jax.jit(mpc_step)
# counter = 0
# opt = mjx_model.opt.replace(timestep = 0.005)
# mjx_model = mjx_model.replace(opt = opt)
# mjx_data  = mjx_data.replace(qpos=qpos)
# mjx_data  = mjx_data.replace(ctrl=jnp.zeros(12))
# liftoff = p_legs0
# timer_t = jnp.array([0.5,0.0,0.0,0.5])
# with mujoco.viewer.launch_passive(model,data) as viewer:
#     while viewer.is_running():
#         mjx_data, X0, U0, V0,liftoff,timer_t = jit_mpc(mjx_model, mjx_data, X0, U0, V0,liftoff,timer_t,counter)
#         X0.block_until_ready()
#         print('counter:',counter)
#         counter += 1
#         data.qpos = mjx_data.qpos
#         data.qvel = mjx_data.qvel
#         mujoco.mj_step(model, data)
#         viewer.sync()