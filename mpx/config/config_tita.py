import jax.numpy as jnp
import jax
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/tita/tita.xml'

joints_name = [
    'joint_left_leg_1', 'joint_left_leg_2', 'joint_left_leg_3', 'joint_left_leg_4',
    'joint_right_leg_1', 'joint_right_leg_2', 'joint_right_leg_3', 'joint_right_leg_4',
]

contact_frame = ['left_leg_4_collision', 'right_leg_4_collision']
body_name = ['left_leg_4', 'right_leg_4']

simulation_frequency = 500
whole_body_frequency = 100
mpc_frequency = 40
dt_mpc = 0.025
N = 12
mpc_horizon_s = N * dt_mpc
mpc_iterations = 1
wbc_lookahead_dt = 0.0

dt_sim = 1.0 / simulation_frequency
dt_wbc = 1.0 / whole_body_frequency
mpc_period_steps = simulation_frequency / mpc_frequency
wbc_period_steps = round(simulation_frequency / whole_body_frequency)
mpc_shift_nodes = round(1.0 / (mpc_frequency * dt_mpc))

T_TRAJECTORY = 60
grav = 9.81

robot_height = 0.44  # Robot base height in meters
h_min = 0.3
h_max = 0.4
com_z_to_track = 0.4
clearence_speed = 0.4

mu = 0.9
use_terrain_estimator = False

p0 = jnp.array([0, 0, robot_height])
p0_c = jnp.array([0, 0, 0])
quat0 = jnp.array([1, 0, 0, 0])

q0 = jnp.array([0.0, 0.5, -1.0, 0.0, 0.0, 0.5, -1.0, 0.0])

n_joints = len(joints_name)
n_contact = len(contact_frame)
nx = 13
nu = 9
mass = 27.68978
d = 0.567  # leg distance (m)
inertia = jnp.array([
    [1.090008860808705, -2.6318929583168255e-05, -0.029887671948266012],
    [-2.6318929583168255e-05, 0.4984221717169791, -2.0791225890770537e-05],
    [-0.029887671948266012, -2.0791225890770537e-05, 0.7010938745906233],
])

grf_ref = jnp.array(n_contact*[0.0, 0.0, (grav*mass)/2.0])
u_ref = jnp.concatenate([jnp.array([0.0, 0.0, 0.0]), jnp.array(grf_ref)])

Kp = jnp.diag(jnp.tile(jnp.array([500,500,500]),n_contact))
Kd = jnp.diag(jnp.tile(jnp.array([20,20,20]),n_contact))

# Whole-body controller
base_body_name = 'base_link'
wheel_radius = 0.0925

Kp_motion = 5e1
Kd_motion = 3e1
Kp_wheel = 5e1
Kd_wheel = 3e1
Kp_reg = 1e2
Kd_reg = 2e1

w_posture = 0.1
posture_joint_ids = (0, 4)

w_qddot = 1e-12
w_force = 1e-4
w_com = 1e0
w_lwheel = 1e0
w_rwheel = 1e0
w_base = 1e0

# MPC cost weights
w_pcomxy = 0e0   # CoM position in the xy plane
w_pcomz = 2e4    # CoM height
w_vcomxy = 3e2   # CoM velocity in the xy plane
w_vcomz = 1e1    # CoM vertical velocity
w_c = 0e0        # Ground-projected CoM position
w_vcz = 0e0      # Ground-projected CoM vertical velocity
w_theta = 0e0    # Heading

w_v = 1e1        # Ground-projected CoM forward velocity
w_omega = 5e1    # Angular velocity

# Control weights for the wheel-related inputs
w_a = 1e-1       # Linear acceleration
w_ac_z = 1e-1    # Vertical acceleration
w_alpha = 1e-3   # Angular acceleration

# Control weights for the ground reaction forces
w_fcxy = 1e-7    # Horizontal forces
w_fcz = 1e-4     # Vertical force
w_eq = 1e6       # Equality-constraint penalty

W = jnp.diag(jnp.array([
    w_pcomxy, w_pcomz,
    w_vcomxy, w_vcomz,
    w_c,      w_vcz,
    w_theta,  w_v,    w_omega,
    w_a,      w_ac_z, w_alpha,
    w_fcxy,   w_fcz,
    w_eq,
]))