from functools import partial

import mpx.config.config_aliengo as base
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives

task_name = "aliengo_trot_two_step"

model_path = base.model_path
contact_frame = base.contact_frame
body_name = base.body_name

dt = 0.02
N = 60
mpc_frequency = base.mpc_frequency

timer_t = base.timer_t
duty_factor = 0.5
step_freq = base.step_freq
step_height = 0.08
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

Qp = base.Qp
Qrot = base.Qrot
Qq = base.Qq
Qdp = base.Qdp
Qomega = base.Qomega
Qdq = base.Qdq
Qtau = base.Qtau
Q_grf = base.Q_grf
Qleg = base.Qleg
W = base.W

use_terrain_estimation = False
initial_state = base.initial_state

cost = partial(mpc_objectives.quadruped_wb_obj, True, n_joints, n_contact, n_contact, N)
hessian_approx = base.hessian_approx
dynamics = base.dynamics
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
    mpc_utils.reference_quadruped_trot_two_step,
    base_height=robot_height,
    total_forward=0.45,
    step_length=0.16,
    step_height=step_height,
    settle_time=0.10,
    phase_time=0.16,
)

solver_mode = "fddp"
max_torque = base.max_torque
min_torque = base.min_torque
