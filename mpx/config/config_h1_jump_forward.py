from functools import partial

import mpx.config.config_h1_kinodynamic as base
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives

task_name = "h1_jump_forward"

model_path = base.model_path
contact_frame = base.contact_frame
body_name = base.body_name

dt = 0.02
N = 50
mpc_frequency = base.mpc_frequency

timer_t = base.timer_t
duty_factor = 1.0
step_freq = base.step_freq
step_height = base.step_height
initial_height = base.initial_height
robot_height = base.robot_height
clearance_speed = base.clearance_speed

p0 = base.p0
quat0 = base.quat0
q0 = base.q0
p_legs0 = base.p_legs0

n_joints = base.n_joints
n_contact = base.n_contact
n = base.n
m = base.m
grf_as_state = base.grf_as_state
foot_slice = base.foot_slice
leg_slice = base.leg_slice
u_ref = base.u_ref
W = base.W
use_terrain_estimation = False
initial_state = base.initial_state

joint_kp = base.joint_kp
joint_kd = base.joint_kd
torque_limits = base.torque_limits

cost = partial(mpc_objectives.h1_kinodynamic_obj, n_joints, n_contact, N)
hessian_approx = base.hessian_approx
dynamics = base.dynamics
MPCWrapper = base.MPCWrapper
reference_generator = partial(
    mpc_utils.reference_generator,
    use_terrain_estimation,
    N,
    dt,
    n_joints,
    n_contact,
    foot0=p_legs0,
    q0=q0,
    clearence_speed=clearance_speed,
)

reference = partial(
    mpc_utils.reference_humanoid_jump_forward,
    base_height=robot_height,
    crouch_height=0.82,
    apex_height=1.02,
    jump_distance=0.35,
    foot_shift=0.18,
    foot_lift=0.10,
)

solver_mode = "fddp"
max_torque = base.max_torque
min_torque = base.min_torque
