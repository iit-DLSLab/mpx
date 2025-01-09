import os
# os.environ['XLA_FLAGS'] = (
    # '--xla_gpu_enable_triton_softmax_fusion=true '
    # '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_highest_priority_async_stream=true '
# )
os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })
import jax.numpy as jnp
import jax

import numpy as np

from trajax import integrators
from trajax.experimental.sqp import util

import  primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from functools import partial

from jax import grad, jvp


from jax.scipy.spatial.transform import Rotation


from primal_dual_ilqr.utils.rotation import quaternion_integration,rpy_intgegration
import primal_dual_ilqr.utils.mpc_utils as mpc_utils
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)




# Problem dimensions
N = 100  # Number of stages
n = 12   # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = 12    # Number of controls (F)
dt = 0.01  # Time step

model_path = './urdfs/aliengo.urdf'

joints_name = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]
contact_frame = ['FL_foot','FR_foot','RL_foot','RR_foot']

n_joints = len(joints_name)
n_contact = len(contact_frame)

w_H_b0 = jnp.block([
    [jnp.eye(3), jnp.array([[0], [0], [0.33]])],
    [jnp.zeros((1, 3)), jnp.array([[1]])]
])
q0 = jnp.array([0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8])

mass = 24
print('mass:\n',mass)
inertia = jnp.array([[ 2.5719824e-01,  1.3145953e-03, -1.6161108e-02],[ 1.3145991e-03,  1.0406910e+00,  1.1957530e-04],[-1.6161105e-02,  1.1957530e-04,  1.0870107e+00]])
print('inertia',inertia)

inertia_inv = jnp.linalg.inv(inertia)
p_legs0 = jnp.array([ 0.2717287,   0.13780001,  0.02074774,  0.2717287,  -0.13780001,  0.02074774, -0.20967132,  0.13780001,  0.02074774, -0.20967132, -0.13780001,  0.02074774])
print('leg:\n',p_legs0)
@jax.jit
def reference_generator(t_timer, x,rpy, foot, input, duty_factor, step_freq,step_height,liftoff):
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
    foot_ref = jnp.tile(foot-jnp.tile(p,(1,n_contact)), (N+1, 1))
    foot_ref_dot = jnp.zeros(((N+1), 3*n_contact))

    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = mpc_utils.timer_run(duty_factor, step_freq, new_t, dt)
        
        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        foot0 = jnp.array([ 0.2717287,   0.13780001,  0.02074774,  0.2717287,  -0.13780001,  0.02074774, -0.20967132,  0.13780001,  0.02074774, -0.20967132, -0.13780001,  0.02074774])
        
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
            return a1*val + 2*a2*val + 3*a3*val**2


        new_foot_x = jnp.where(new_contact_sequence>0, new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(new_contact_sequence>0, new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(new_contact_sequence>0, new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        new_foot_dot = new_foot_dot.at[t,::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_dot(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,1::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_dot(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,2::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineZ_dot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        return (new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z)
    
    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    init_carry = (t_timer, contact_sequence,foot_ref,foot_ref_dot,liftoff_x,liftoff_y,liftoff_z)
    _, contact_sequence,foot_ref,foot_ref_dot, liftoff_x,liftoff_y,liftoff_z = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    # foot to world frame
    foot_ref = foot_ref + jnp.tile(p,(N+1,n_contact))
    return jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref, foot_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1), liftoff,foot_ref_dot



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

p0 = jnp.array([0, 0, 0.33])
quat0 = jnp.array([1, 0, 0, 0])
rpy0 = jnp.array([0, 0, 0])
x0 = jnp.concatenate([p0, rpy0, jnp.zeros(3), jnp.array([0, 0, 0])])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.4])
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
Qomega = jnp.diag(jnp.array([10, 10, 10]))
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

    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref) + (grf-grf_ref).T @ Rgrf @ (grf-grf_ref) #+ dq.T @ Rgrf @ dq + (q - q_ref).T @ Qq @ (q - q_ref)
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref) #+ omega.T @ Qomega @ omega + (q - q_ref).T @ Qq @ (q - q_ref)

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)

# Solve
U0 = jnp.tile(grf_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))
reference = jnp.tile(jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref]), (N + 1, 1))
parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))

from timeit import default_timer as timer

mu = 1e-3

@jax.jit
def work(reference,parameter,x0,X0,U0,V0,mu):
    return optimizers.mpc(
        cost,
        dynamics,
        reference,
        parameter,
        x0,
        X0,
        U0,
        V0,
        mu,
    )

print("Simulation started")

from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
import copy
import mujoco


robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = 100.0
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

sim_frequency = 1000.0

env = QuadrupedEnv(robot=robot_name,
                   hip_height=0.25,
                   legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
                   feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=1.5,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                #    state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
# breakpoint()
obs = env.reset(random=False)
env.render()
timer_t = jnp.array([0000.5,0000.0,0000,0000.5])
timer_t_sim = timer_t.copy()
contact, timer_t = mpc_utils.timer_run(duty_factor = 0.6, step_freq = 1.35,leg_time=timer_t, dt=dt)
liftoff = p_legs0.copy()
terrain_height = np.zeros(n_contact)

init = {}
input = {}

Kp = 10
Kd = 2

Kp_c = 300
Kd_c = 5
counter = 0


while True:

    qpos = env.mjData.qpos
    qvel = env.mjData.qvel

    if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

        foot_op = np.array([env.feet_pos('world').FL, env.feet_pos('world').FR, env.feet_pos('world').RL, env.feet_pos('world').RR],order="F")
        contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = 0.6, step_freq = 1.35,leg_time=timer_t_sim, dt=dt)
        timer_t = timer_t_sim.copy()

        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        p = qpos[:3].copy()
        q = qpos[7:].copy()

        dp = qvel[:3].copy()
        omega = qvel[3:6]
        dq = qvel[6:].copy()

        rpy = env.base_ori_euler_xyz.copy()
        foot_op_vec = foot_op.flatten()
        x0 = jnp.concatenate([p,rpy, dp, omega])

        input = (ref_base_lin_vel, ref_base_ang_vel, 0.36)
        start = timer()
        reference , parameter , liftoff,foot_ref_dot= reference_generator(timer_t, jnp.concatenate([qpos,qvel]), rpy,foot_op_vec, input, duty_factor = 0.6,  step_freq= 1.35 ,step_height=0.08,liftoff=liftoff)   
        X,U,V, _,_,_ =  work(reference,parameter,x0,X0,U0,V0,mu)
        stop = timer()
        print(f"Execution time: {stop-start}")
        U0 = jnp.concatenate([U[1:],U[-1:]])
        X0 = jnp.concatenate([X[1:],X[-1:]])
        V0 = jnp.concatenate([V[1:],V[-1:]])
        grf_ = U[0,:]

    feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
    action = np.zeros(env.mjModel.nu)
    #PD 
    #get foot speed from the joint speed 
    foot_speed = np.zeros((3*n_contact))
    foot_speed[:3] = (feet_jac['FL'].T @ qvel[6:9])[6:9]
    foot_speed[3:6] = (feet_jac['FR'].T @ qvel[9:12])[9:12]
    foot_speed[6:9] = (feet_jac['RL'].T @ qvel[12:15])[12:15]
    foot_speed[9:] = (feet_jac['RR'].T @ qvel[15:18])[15:18]
    
    catisian_space_action = Kp_c*(parameter[1,4:]-foot_op_vec) + Kd_c*(foot_ref_dot[0,:]-foot_speed)

    action[env.legs_tau_idx.FL] = (feet_jac['FL'].T @ ((1-contact_op[0])*catisian_space_action[:3]-grf_[:3]))[6:9]
    action[env.legs_tau_idx.FR] = (feet_jac['FR'].T @ ((1-contact_op[1])*catisian_space_action[3:6]-grf_[3:6]))[9:12]
    action[env.legs_tau_idx.RL] = (feet_jac['RL'].T @ ((1-contact_op[2])*catisian_space_action[6:9]-grf_[6:9]))[12:15]
    action[env.legs_tau_idx.RR] = (feet_jac['RR'].T @ ((1-contact_op[3])*catisian_space_action[9:]-grf_[9:] ))[15:18]
    mass_matrix = np.zeros((env.mjModel.nv, env.mjModel.nv))
    mujoco.mj_fullM(env.mjModel, mass_matrix, env.mjData.qM)
    tau_gravity = env.mjData.qfrc_bias[6:] #+ (mass_matrix @ env.mjData.qacc)[6:]
    state, reward, is_terminated, is_truncated, info = env.step(action=action+tau_gravity)
    counter += 1
    if is_terminated:
        pass
        # Do some stuff
    env.render()
env.close()
