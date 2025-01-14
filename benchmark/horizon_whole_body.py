
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))
import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the anymal model
anymal = example_robot_data.load("go1")

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])
print(x0)

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "FL_foot", "FR_foot", "RL_foot", "RR_foot"
gait = SimpleQuadrupedalGaitProblem(
    anymal.model, lfFoot, rfFoot, lhFoot, rhFoot, fwddyn=False
)

# Setting up all tasks
GAITPHASES = [
    {
        "trotting": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 1e-2,
            "stepKnots": 30,
            "supportKnots": 46,
        }
    },
]
problems = gait.createTrottingProblem(
                    x0,
                    0.0,
                    0.1,
                    1e-2,
                    30,
                    46,
                )
solver = [None] * len(problems)
horizon_len = []
crocoddyl_times = []
for i in range(len(solver)):
    solver[i] = crocoddyl.SolverIntro(problems[i])
    solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the solver
    xs = [x0] * (solver[i].problem.T + 1)
    print(solver[i].problem.T)
    horizon_len.append(solver[i].problem.T)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    from timeit import default_timer as timer
    
    start = timer()
    for kk in range(10):
        solver[i].solve(xs, us, 1, False)
    stop = timer()
    crocoddyl_times.append((stop - start)/10)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]


import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=true '
    # '--xla_gpu_deterministic_ops=true'
)
# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })
import jax.numpy as jnp
import jax
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import  primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers

import mujoco
from mujoco import mjx
from mujoco.mjx._src import math

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = 100.0

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
n_joints = len(joints_name)
n_contact = len(contact_frame)

# Problem dimensions
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
dt = 0.01  # Time step
p_legs0 = jnp.array([ 0.2717287,   0.13780001,  0.02074774,  0.2717287,  -0.13780001,  0.02074774, -0.20967132,  0.13780001,  0.02074774, -0.20967132, -0.13780001,  0.02074774])
model = mujoco.MjModel.from_xml_path('./data/aliengo/aliengo.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

contact_id = []
for name in contact_frame:
    contact_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_GEOM,name))
body_id = []
for name in body_name:
    body_id.append(mjx.name2id(mjx_model,mujoco.mjtObj.mjOBJ_BODY,name))



alpha = 25
beta = 2*np.sqrt(alpha)

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

p0 = jnp.array([0, 0, 0.33])
quat0 = jnp.array([1, 0, 0, 0])
q0 = jnp.array([0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8,0,0.8,-1.8])
x0 = jnp.concatenate([p0, quat0,q0, jnp.zeros(6+n_joints),p_legs0,jnp.zeros(3*n_contact)])
grf0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p_ref = jnp.array([0, 0, 0.36])
quat_ref = jnp.array([1, 0, 0, 0])
rpy_ref = jnp.array([0, 0, 0])
q_ref = jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8])
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
mpx_times = []
for N in horizon_len:
    # Solve
    U0 = jnp.tile(u_ref, (N, 1))
    X0 = jnp.tile(x0, (N + 1, 1))
    V0 = jnp.zeros((N + 1, n ))
    reference = jnp.tile(jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref,p_legs0]), (N + 1, 1))
    parameter = jnp.tile(jnp.concatenate([jnp.ones(4),p_legs0]),(N+1,1))
    from timeit import default_timer as timer
    mu = 1e-3
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
    X,U,V, _, _ =  work(reference,parameter,x0,X0,U0,V0)
    start = timer()
    for i in range(10):
        X,U,V, _, _ =  work(reference,parameter,x0,X,U,V)
    stop = timer()
    print((stop - start)/10)
    mpx_times.append((stop - start)/10)

ax.plot(horizon_len[0], mpx_times, marker='o')
ax.plot(horizon_len, crocoddyl_times, marker='o')
ax.set_xlabel('Batch Size (n)')
ax.set_ylabel('Average Time (s)')
ax.set_title('Average Time vs Batch Size')
ax.set_xscale('log')
ax.grid(True)
plt.show()
