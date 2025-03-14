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

import  primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from functools import partial


import utils.mpc_utils as mpc_utils
import utils.models as mpc_dyn_model
import utils.objectives as mpc_objectives
import utils.config as config

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

# Problem dimensions
N = 50  # Number of stages
n = 13   # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = 12    # Number of controls (F)
dt = 0.01  # Time step

joints_name = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]
contact_frame = ['FL_foot','FR_foot','RL_foot','RR_foot']

n_joints = len(joints_name)
n_contact = len(contact_frame)

q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
mass = 15.019
inertia = jnp.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                    [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                    [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

inertia_inv = jnp.linalg.inv(inertia)
p_legs0 = jnp.array([ 0.192, 0.142, 0.024,  0.192, -0.142, 0.024,-0.195,  0.142,  0.024, -0.195, -0.142, 0.024])
p0 = jnp.array([0, 0, 0.28])
quat0 = jnp.array([1, 0, 0, 0])
x0 = jnp.concatenate([p0, quat0, jnp.zeros(3), jnp.array([0, 0, 0])])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

grf_ref = jnp.zeros(3 * n_contact)

u_ref = grf_ref

Qp = jnp.diag(jnp.array([0, 0, 10000]))
Qrot = jnp.diag(jnp.array([500,500,0]))
Qdp = jnp.diag(jnp.array([1000, 1000, 1000]))
Qomega = jnp.diag(jnp.array([100, 100, 10]))
Rgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3

W = jax.scipy.linalg.block_diag(Qp, Qrot, Qdp, Qomega, Rgrf)
# Solve
U0 = jnp.tile(grf_ref, (N, 1))
X0 = jnp.tile(x0, (N + 1, 1))
V0 = jnp.zeros((N + 1, n ))

cost = partial(mpc_objectives.quadruped_srbd_obj, W,n_contact,N)
hessian_approx = partial(mpc_objectives.quadruped_srbd_hessian_gn, W,n_contact)
dynamics = partial(mpc_dyn_model.quadruped_srbd_dynamics,mass, inertia_inv,dt)

from timeit import default_timer as timer

# mu = 1e-3

@jax.jit
def work(reference,parameter,x0,X0,U0,V0):
    return optimizers.mpc(
        cost,
        dynamics,
        hessian_approx,
        False,
        reference,
        parameter,
        x0,
        X0,
        U0,
        V0,
    )

print("Simulation started")

from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
import copy
import mujoco
from gym_quadruped.utils.mujoco.visual import render_sphere

robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = 100.0
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

sim_frequency = 200.0

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
duty_factor = 0.65
step_freq = 1.3
contact, timer_t = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq,leg_time=timer_t, dt=dt)
liftoff = p_legs0.copy()
terrain_height = np.zeros(n_contact)

init = {}
input = {}

Kp = 10
Kd = 2

Kp_c = np.diag(np.tile(np.array([500,500,500]),n_contact))

Kd_c = np.diag(np.tile(np.array([10,10,10]),n_contact))
counter = 0
ids = []
for i in range(N*4):
     ids.append(render_sphere(viewer=env.viewer,
              position = np.array([0,0,0]),
              diameter = 0.01,
              color=[1,0,0,1]))

feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
J_old = np.concatenate([feet_jac['FL'],feet_jac['FR'],feet_jac['RL'],feet_jac['RR']],axis=0)
mpc_time = 0
mpc_counter = 0


args = {}
make_model = True

args['N'] = N # Horizon lenght
args['dt'] = dt # delta time between the integration node

# srbd_acados = acd.ocp_formulation(args)
# srbd_acados_solver = srbd_acados.getOptimalProblem(model_name = "srbd")

@jax.jit
def jitted_reference_generator(foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff):
    return mpc_utils.reference_generator_srbd(config.N,config.dt,config.n_contact,foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff)
while env.viewer.is_running():

    qpos = env.mjData.qpos
    qvel = env.mjData.qvel

    if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

        foot_op = np.array([env.feet_pos('world').FL, env.feet_pos('world').FR, env.feet_pos('world').RL, env.feet_pos('world').RR],order="F")
        contact_op , timer_t_sim = mpc_utils.timer_run(duty_factor = duty_factor, step_freq = step_freq,leg_time=timer_t_sim, dt=dt)
        timer_t = timer_t_sim.copy()

        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        p = qpos[:3].copy()
        quat = qpos[3:7].copy()
        q = qpos[7:].copy()
        dp = qvel[:3].copy()
        omega = qvel[3:6]
        dq = qvel[6:].copy()
        foot_op_vec = foot_op.flatten()
        x0 = jnp.concatenate([p,quat, dp, omega])

        input = jnp.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])

        start = timer()

        reference , parameter , liftoff, foot_ref_dot, foot_ref_ddot = jitted_reference_generator(config.p_legs0,timer_t, x0, foot_op_vec, input, duty_factor = config.duty_factor,  step_freq= config.step_freq ,step_height=config.step_height,liftoff=liftoff)

        start_mpc = timer()
        X,U,V =  work(reference,parameter,x0,X0,U0,V0)
        X.block_until_ready()
        stop = timer()
        if mpc_counter != 0:
            mpc_time += stop-start_mpc
            # mpc_time += srbd_acados_solver.get_stats('time_tot')
            print(f"average execution time MPC: {mpc_time/mpc_counter}")
            mpc_counter += 1
        else:
            mpc_counter += 1

        U0 = jnp.concatenate([U[1:],U[-1:]])
        X0 = jnp.concatenate([X[1:],X[-1:]])
        V0 = jnp.concatenate([V[1:],V[-1:]])
        grf_ = U[0,:]
        for leg in range(n_contact):
            pleg = reference[:,12:]
            for i in range(N):
                render_sphere(viewer=env.viewer,
                          position = pleg[i,3*leg:3+3*leg],
                          diameter = 0.01,
                          color=[parameter[i,leg],1,0,1],
                          geom_id = ids[leg*N+i])

    feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
    action = np.zeros(env.mjModel.nu)
    #PD
    #get foot speed from the joint speed
    start = timer()
    foot_speed = np.zeros((3*n_contact))
    foot_speed[:3] = (feet_jac['FL'].T @ qvel[6:9])[6:9]
    foot_speed[3:6] = (feet_jac['FR'].T @ qvel[9:12])[9:12]
    foot_speed[6:9] = (feet_jac['RL'].T @ qvel[12:15])[12:15]
    foot_speed[9:] = (feet_jac['RR'].T @ qvel[15:18])[15:18]

    cartesian_space_action = Kp_c@(parameter[1,4:16]-foot_op_vec) + Kd_c@(foot_ref_dot[0,:]-foot_speed)
    mass_matrix = np.zeros((env.mjModel.nv, env.mjModel.nv))
    mujoco.mj_fullM(env.mjModel, mass_matrix, env.mjData.qM)
    J = np.concatenate([feet_jac['FL'],feet_jac['FR'],feet_jac['RL'],feet_jac['RR']],axis=0)
    J_dot = (J - J_old)*sim_frequency
    J_old = J.copy()
    accelleration = cartesian_space_action.T + foot_ref_ddot[0,:]
    tau_fb_lin = env.mjData.qfrc_bias[6:] + (mass_matrix @ np.linalg.pinv(J) @ (accelleration - J_dot@qvel))[6:]
    tau_mpc = -(J.T@grf_)[6:]
    tau_PD = (J.T @ cartesian_space_action.T)[6:]
    total_tau = np.zeros(n_joints)
    for i in range(n_contact):
        total_tau[3*i:3+3*i] = (1-contact_op[i])*(tau_PD[3*i:3+3*i] + tau_fb_lin[3*i:3+3*i]) + contact_op[i]*tau_mpc[3*i:3+3*i]

    env.step(action=total_tau)
    counter += 1
    env.render()
env.close()
