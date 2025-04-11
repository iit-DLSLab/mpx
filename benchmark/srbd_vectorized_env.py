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

 
model = mujoco.MjModel.from_xml_path('./data/go2/scene_mjx.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)


# Problem dimensions
N = 50  # Number of stages
n = 12   # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = 12    # Number of controls (F)
dt = 0.02  # Time step

model_path = './urdfs/aliengo.urdf'

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
w_H_b0 = jnp.block([
    [jnp.eye(3), jnp.array([[0], [0], [0.28]])],
    [jnp.zeros((1, 3)), jnp.array([[1]])]
])
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
mass = 15.019
inertia = jnp.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                    [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                    [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])
print('mass:\n',mass)
print('inertia',inertia)
inertia_inv = jnp.linalg.inv(inertia)
p_legs0 = jnp.array([ 0.192, 0.142, 0.024,  0.192, -0.142, 0.024,-0.195,  0.142,  0.024, -0.195, -0.142, 0.024])
print('leg:\n',p_legs0)
@jax.jit
def reference_generator(t_timer, x,rpy, foot, input, duty_factor, step_freq,step_height,liftoff):

    @jax.jit
    def rot_yaw(yaw):
        return jnp.array([[jnp.cos(yaw),-jnp.sin(yaw),0],
                          [jnp.sin(yaw),jnp.cos(yaw),0],
                          [0,0,1]])
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    # dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    ref_lin_vel, ref_ang_vel, robot_height = input
    p = jnp.array([p[0], p[1], robot_height])
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * robot_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    rpy_ref = jnp.tile(jnp.array([0, 0, rpy[2]]), (N+1, 1))
    q_ref = jnp.tile(jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8]), (N+1, 1))
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    Rz = rot_yaw(rpy[2])
    
    foot_ref = jnp.tile(foot-jnp.tile(p,(1,n_contact))@jax.scipy.linalg.block_diag(Rz,Rz,Rz,Rz), (N+1, 1))
    foot_ref_dot = jnp.zeros(((N+1), 3*n_contact))
    foot_ref_ddot = jnp.zeros(((N+1), 3*n_contact))

    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,new_foot_dot,new_foot_ddot,liftoff_x,liftoff_y,liftoff_z = carry

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

        @jax.jit
        def cubic_splineXY(current_foot, foothold,val):
            a0 = current_foot
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2 
            return a0 + a1*val + a2*val**2 + a3*val**3
        
        @jax.jit
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
        
        @jax.jit
        def cubic_splineXY_dot(current_foot, foothold,val):
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2 
            return 2*a2*val + 3*a3*val**2
        
        @jax.jit
        def cubic_splineZ_dot(current_foot, foothold, step_height,val):
            a0 = current_foot
            a3 = 8*step_height - 6*foothold -2*a0
            a2 = -foothold +a0 -2*a3
            a1 = +2*foothold -2*a0 +a3
            return a1 + 2*a2*val + 3*a3*val**2
        
        @jax.jit
        def cubic_splineXY_ddot(current_foot, foothold,val):
            a1 = 0
            a2 = 3*(foothold - current_foot)
            a3 = -2/3*a2 
            return 2*a2 + 6*a3*val
        
        @jax.jit
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

        return (new_t, contact_sequence,new_foot,new_foot_dot,new_foot_ddot,liftoff_x,liftoff_y,liftoff_z)
    
    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    init_carry = (t_timer, contact_sequence,foot_ref,foot_ref_dot,foot_ref_ddot,liftoff_x,liftoff_y,liftoff_z)
    _, contact_sequence,foot_ref,foot_ref_dot,foot_ref_ddot, liftoff_x,liftoff_y,liftoff_z = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)
    # foot to world frame
    
    foot_ref = foot_ref@jax.scipy.linalg.block_diag(Rz,Rz,Rz,Rz).T + jnp.tile(p,(N+1,n_contact))
    return jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref, foot_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1), liftoff,foot_ref_dot,foot_ref_ddot



@jax.jit
def dynamics(x, u, t,parameter):
    # Extract state variables
    p = x[:3]
    quat = x[3:6]
    # p_legs = x[6:6+n_joints]
    dp = x[6:9]
    omega = x[9:12]
    # dp_leg = u[:n_joints]
    grf = u

    contact = parameter[t,:4]
    p_legs = parameter[t,4:]

    # Convert quaternion to rotation matrix
    # R = Rotation.from_quat(quat).as_matrix()

    # w_H_b = jnp.block([
    #     [R, p.reshape((3, 1))],
    #     [jnp.zeros((1, 3)), jnp.array([[1]])]
    # ])

    dp_next = dp + (jnp.array([0, 0, -9.81]) + (1 / mass) * (grf[:3]*contact[0] + grf[3:6]*contact[1] + grf[6:9]*contact[2] + grf[9:12]*contact[3])) * dt

    p0 = p_legs[:3]
    p1 = p_legs[3:6]
    p2 = p_legs[6:9]
    p3 = p_legs[9:]

    omega_next = omega + inertia_inv@((jnp.cross(p0 - p, grf[:3])*contact[0] + jnp.cross(p1 - p, grf[3:6])*contact[1] + jnp.cross(p2 - p, grf[6:9])*contact[2] + jnp.cross(p3 - p, grf[9:12])*contact[3]))*dt

    # Semi-implicit Euler integration
    p_new = p + dp_next * dt
    rpy_new = rpy_intgegration(omega_next, quat, dt)
    # p_legs_new = p_legs# + dp_leg * dt

    x_next = jnp.concatenate([p_new, rpy_new, dp_next, omega_next])

    return x_next

p0 = jnp.array([0, 0, 0.28])
quat0 = jnp.array([1, 0, 0, 0])
rpy0 = jnp.array([0, 0, 0])
x0 = jnp.concatenate([p0, rpy0, jnp.zeros(3), jnp.array([0, 0, 0])])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.28])
quat_ref = jnp.array([0, 0, 0, 1])
rpy_ref = jnp.array([0, 0, 0])
q_ref = jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8])
dp_ref = jnp.array([0, 0, 0])
omega_ref = jnp.array([0, 0, 0])
dq_ref = jnp.zeros(n_joints)

grf_ref = jnp.zeros(3 * n_contact)  

u_ref = grf_ref

Qp = jnp.diag(jnp.array([0, 0, 10000]))
Qdp = jnp.diag(jnp.array([1000, 1000, 1000]))
Qomega = jnp.diag(jnp.array([100, 100, 10]))
Rgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
Qrpy = jnp.diag(jnp.array([500,500,0]))

# Define the cost function
@jax.jit
def cost(x, u, t, reference):

    p = x[:3]
    dp = x[6:9]
    omega = x[9:12]
    grf = u
    # dq = u[:n_joints]
    rpy = x[3:6]

    p_ref = reference[t,:3]
    rpy_ref = reference[t,3:6]
    dp_ref = reference[t,6:9]
    omega_ref = reference[t,9:12]
    # grf_ref = reference[t,12:24]
    mu = 0.7
    friction_cone = jnp.array([[0,0,1],[-1,0,mu],[1,0,mu],[0,-1,mu],[0,1,mu]])
    friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    friction_cone = friction_cone @ grf
    alpha = 0.1
    #use ln(1+exp(x)) as a smooth approximation of max(0,x)
    friction_cone = 1/alpha*(jnp.log1p(jnp.exp(-alpha*friction_cone)))

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref) + (grf-grf_ref).T @ Rgrf @ (grf-grf_ref) +\
                friction_cone.T @ friction_cone
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
U0 = jnp.tile(grf_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))
reference = jnp.tile(jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref]), (N + 1, 1))
parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))

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
def mpc_step(model, data, X0, U0, V0, t):

    FL_leg = data.geom_xpos[contact_id[0]]
    FR_leg = data.geom_xpos[contact_id[1]]
    RL_leg = data.geom_xpos[contact_id[2]]
    RR_leg = data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, data, FL_leg, body_id[2])
    J_FR, _ = mjx.jac(mjx_model, data, FR_leg, body_id[5])
    J_RL, _ = mjx.jac(mjx_model, data, RL_leg, body_id[8])
    J_RR, _ = mjx.jac(mjx_model, data, RR_leg, body_id[11])


    rpy = quaternion_to_rpy(data.qpos[3:7])
    # # contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq,leg_time=timer_t_sim, dt=dt)
    J = jnp.concatenate([J_FL,J_FR,J_RL,J_RR],axis=1)
    p_legs = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg],axis = 0)
    parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs]),(N+1,1))
    # # call tha work only at 100Hz 
    x0 = jnp.concatenate([data.qpos[:3],rpy,data.qvel[:3],data.qvel[3:6]])
    X_new, U_new, V_new = jax.lax.cond(t % 4 == 0, lambda _: work(reference, parameter, x0, X0, U0, V0), lambda _: (X0, U0, V0), None)
    grf = U_new[0,:3*n_contact]
    tau = -(J @ grf)[6:]
    data = data.replace(ctrl=tau)
    data = mjx.step(model, data)
    return data, X_new, U_new, V_new
def integrate_mpc(model, data, X0, U0, V0):
    def scan_fn(carry, t):
        data, X0, U0, V0 = carry
        data, X0, U0, V0 = mpc_step(model, data, X0, U0, V0,t)
        return (data, X0, U0, V0), None
    t = jnp.arange(200)
    (data, X0, U0, V0), _ = jax.lax.scan(scan_fn, (data, X0, U0, V0), t)
    return data, X0, U0, V0
frames = []
mujoco.mj_resetData(model, data)
mjx_data = mjx.put_data(model, data)
qpos = jax.numpy.array([0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
n_env = 4026
qpos0 = jax.numpy.tile(qpos, (n_env, 1))
batch = jax.vmap(lambda x: mjx_data.replace(qpos=x))(qpos0)
batch_U0 = jnp.tile(U0, (n_env, 1, 1))
batch_X0 = jnp.tile(X0, (n_env, 1, 1))
batch_V0 = jnp.tile(V0, (n_env, 1, 1))

# jit_step = jax.jit(jax.vmap(integrate_mpc, in_axes=(None, 0,0,0,0)))
# batch,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0)
# start = timer()
# batch1,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0)
# end = timer()
# print(f"Time elapsed: {end - start} s")
 
 
# start = timer()
# batch1,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0)
# end = timer()
# print(f"Time elapsed 2: {end - start} s")
 
 
# start = timer()
# batch2,_,_,_ = jit_step(mjx_model, batch, batch_X0, batch_U0, batch_V0)
# end = timer()
# print(f"Time elapsed 3: {end - start} s")

jit_mpc = jax.jit(mpc_step)
counter = 0
opt = mjx_model.opt.replace(timestep = 0.005)
mjx_model = mjx_model.replace(opt = opt)
mjx_data  = mjx_data.replace(qpos=qpos)
mjx_data  = mjx_data.replace(ctrl=jnp.zeros(12))
# jit_step = jax.jit(mjx.step)
with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        mjx_data, X0, U0, V0 = jit_mpc(mjx_model, mjx_data, X0, U0, V0,counter)
        X0.block_until_ready()
        print('counter:',counter)
        counter += 1
        data.qpos = mjx_data.qpos
        data.qvel = mjx_data.qvel
        mujoco.mj_step(model, data)
        viewer.sync()