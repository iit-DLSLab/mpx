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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import  primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from functools import partial

from jax import grad, jvp


from jax.scipy.spatial.transform import Rotation


from primal_dual_ilqr.utils.rotation import quaternion_integration,rpy_intgegration
import primal_dual_ilqr.utils.mpc_utils as mpc_utils
import benchmark.multirobotAcados as acd
gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)




# Problem dimensions
N = 50  # Number of stages
# n_robot = 2
n = 12   # Number of states 
m = 12    # Number of controls 
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

        foot0 = jnp.array([ 0.2717287,   0.13780001,  0.02074774,  0.2717287,  -0.13780001,  0.02074774, -0.20967132,  0.13780001,  0.02074774, -0.20967132, -0.13780001,  0.02074774])

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
    grf_ref = jnp.zeros((N+1,3*n_contact))
    return jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref,grf_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1), liftoff,foot_ref_dot,foot_ref_ddot



@jax.jit
def dynamics(x, u, t,parameter):
    # Extract state variables
    p = x[:3]
    rpy = x[3:6]
    # p_legs = x[6:6+n_joints]
    dp = x[6:9]
    omega = x[9:12]
    # dp_leg = u[:n_joints]
    grf = u[:12]

    contact = parameter[t,:4]
    p_legs = parameter[t,4:16]

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
    rpy_new = rpy_intgegration(omega_next, rpy, dt)
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

u_ref = jnp.concatenate([grf_ref])

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
    grf = u[:12]
    rpy = x[3:6]

    p_ref = reference[t,:3]
    rpy_ref = reference[t,3:6]
    dp_ref = reference[t,6:9]
    omega_ref = reference[t,9:12]

    mu = 0.7
    friction_cone = jnp.array([[0,0,1],[-1,0,mu],[1,0,mu],[0,-1,mu],[0,1,mu]])
    friction_cone = jnp.kron(jnp.eye(n_contact), friction_cone)
    friction_cone = friction_cone @ grf
    alpha = 0.1
    #use ln(1+exp(x)) as a smooth approximation of max(0,x)
    friction_cone = 1/alpha*(jnp.log1p(jnp.exp(-alpha*friction_cone)))
    stage_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref) + grf.T @ Rgrf @ grf +\
                friction_cone.T @ friction_cone
    term_cost = (p - p_ref).T @ Qp @ (p - p_ref) + (rpy-rpy_ref).T@Qrpy@(rpy-rpy_ref) + (dp-dp_ref).T @ Qdp @ (dp-dp_ref) + (omega-omega_ref).T @ Qomega @ (omega-omega_ref)
    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)
param_size = 4*n_contact
from timeit import default_timer as timer
times_mpx = []
times_acados = []
robots = [1,2,4,8,16]
for n_robot in robots:
    @jax.jit
    def multi_robot_dynamics(x, u, t,parameter):
        return jnp.concatenate([dynamics(x[n*i:n*i+n], u[m*i:m+m*i], t,parameter[:,i*param_size:param_size+i*param_size]) for i in range(n_robot)], axis=0)
    @jax.jit
    def multi_robot_cost(x, u, t, reference):
        return jnp.sum(jnp.array([cost(x[n*i:n*i+n], u[m*i:m+m*i], t, reference[:,n*i:n+n*i]) for i in range(n_robot)]))

    # Solve
    U0 = jnp.tile(jnp.tile(u_ref,(1,n_robot)), (N, 1))
    X0 = jnp.tile(jnp.tile(x0,(1,n_robot)), (N + 1, 1))
    V0 = jnp.zeros((N + 1, n*n_robot ))
    x0_ = jnp.tile(x0,(n_robot))
    reference = jnp.tile(jnp.tile(jnp.concatenate([p_ref, rpy_ref, dp_ref, omega_ref, grf_ref]), (1, n_robot)), (N + 1, 1))
    noise_dp_ref = jax.random.normal(jax.random.PRNGKey(0), (N + 1, 3 * n_robot)) * 0.1
    dp_ref_with_noise = noise_dp_ref
    noise_x0 = jax.random.normal(jax.random.PRNGKey(0),n* n_robot) * 0.01
    for robot in range(n_robot):
        reference = reference.at[:, 6+12*robot:9+12*robot].set(dp_ref_with_noise[:,3*robot:3+3*robot])
        x0_ = x0_.at[12*robot:12+12*robot].set(x0_[12*robot:12+12*robot] + noise_x0[n*robot:n+n*robot])
    parameter = jnp.tile(jnp.tile(jnp.concatenate([jnp.ones(4), p_legs0]), (1, n_robot)), (N + 1, 1))
    mu = 1e-3
    @jax.jit
    def work(reference,parameter,x0,X0,U0,V0):
        return optimizers.mpc(
            multi_robot_cost,
            multi_robot_dynamics,
            reference,
            parameter,
            x0,
            X0,
            U0,
            V0,
        )
    X,U,V, _,_ = work(reference,parameter,x0_,X0,U0,V0)
    args = {}
    make_model = True

    args['N'] = N # Horizon lenght
    args['dt'] = dt # delta time between the integration node
    
    srbd_acados = acd.ocp_formulation(args)
    srbd_acados_solver = srbd_acados.getOptimalProblem(model_name = "srbd_" + str(n_robot), n_robot=n_robot)
    start = timer()
    for i in range(100):
        X,U,V, _,_ = work(reference,parameter,x0_,X0,U0,V0)
    end = timer()
    print(f"Time for mpx N = {n_robot} is {(end-start)/100}")
    times_mpx.append((end-start)/100)
    x0_acados = np.zeros(n*n_robot)
    reference_acados = np.zeros((N+1,24*n_robot))
    parameter_acados = np.zeros((N+1,4*n_contact*n_robot+1))
    for robot in range(n_robot):
        x0_acados[3*robot:3+3*robot] = x0_[:3]
        x0_acados[3*n_robot+3*robot:3*n_robot+3+3*robot] = x0_[3:6]
        x0_acados[6*n_robot+3*robot:6*n_robot+3+3*robot] = x0_[6:9]
        x0_acados[9*n_robot+3*robot:9*n_robot+3+3*robot] = x0_[9:12]
        reference_acados[:,3*robot:3+3*robot] = reference[:,:3]
        reference_acados[:,3*n_robot+3*robot:3*n_robot+3+3*robot] = reference[:,3:6]
        reference_acados[:,6*n_robot+3*robot:6*n_robot+3+3*robot] = reference[:,6:9]
        reference_acados[:,9*n_robot+3*robot:9*n_robot+3+3*robot] = reference[:,9:12]
        reference_acados[:,12*n_robot+12*robot:12+12*n_robot+12*robot] = reference[:,12:24]
        parameter_acados[:,4*robot:4+4*robot] = parameter[:,:4]
        parameter_acados[:,n_contact*n_robot+12*robot:12+n_contact*n_robot+12*robot] = parameter[:,4:16]
    parameter_acados[:,-1] = np.tile(np.array([dt]),(N+1))
    srbd_acados_solver.set(0, 'lbx', x0_acados)
    srbd_acados_solver.set(0, 'ubx', x0_acados)
    for k in range(N):
        srbd_acados_solver.set(k, 'p', parameter_acados[k, :])
        srbd_acados_solver.cost_set(k,'y_ref',reference_acados[k, :])
    start_acados = timer()
    for i in range(100):
        status = srbd_acados_solver.solve()
    end_acados = timer()
    print(f"Time for acados N = {n_robot} is {(end_acados-start_acados)/100}")
    times_acados.append((end_acados-start_acados)/100)

import matplotlib.pyplot as plt
plt.plot(robots, times_mpx, marker='o')
plt.plot(robots, times_acados, marker='o')
plt.xlabel('Batch Size (n)')
plt.ylabel('Average Time (s)')
plt.title('Average Time vs Batch Size')
plt.xscale('log')
plt.grid(True)
plt.show()