import jax.numpy as jnp
import jax 
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import mpx.utils.mpc_utils as mpc_utils
import os 
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/aliengo/aliengo.xml'  # Path to the MuJoCo model XML file
# Joint names and related configuration
# Contact frame names and body names for feet (or calves)
contact_frame = ['FL', 'FR', 'RL', 'RR']
body_name = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']

# Time and stage parameters
dt = 0.01  # Time step in seconds
N = 100         # Number of stages
mpc_frequency = 50  # Frequency of MPC updates in Hz

# Timer values (make sure the values match your intended configuration)
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg
duty_factor = 0.65  # Duty factor for the gait
step_freq = 1.35   # Step frequency in Hz
step_height = 0.12 # Step height in meters
initial_height = 0.1  # Initial height of the robot's base in meters
robot_height = 0.33  # Height of the robot's base in meters
grf_as_state = True

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, 0.33])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)
#alingo
q0 = jnp.array([0.2, 0.8, -1.8, -0.2, 0.8, -1.8, 0.2, 0.8, -1.8, -0.2, 0.8, -1.8])  # Initial joint angles
q0_init = jnp.array([-0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8])
#go2       
# q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])  # Initial joint angles

#alingo
p_legs0 = jnp.array([
    0.27092872, 0.174, -0.31,  # Initial position of the front left leg
    0.27092872, -0.174, -0.31, # Initial position of the front right leg
   -0.20887128, 0.174, -0.31,  # Initial position of the rear left leg
   -0.20887128, -0.174  , -0.31   # Initial position of the rear right leg
])
#go2
# p_legs0 = jnp.array([
#     0.192, 0.142, -0.27,  # Initial position of the front left leg
#     0.192, -0.142, -0.27, # Initial position of the front right leg
#    -0.195, 0.142, -0.27,  # Initial position of the rear left leg
#    -0.195, -0.142, -0.27  # Initial position of the rear right leg
# ])

# Determine number of joints and contacts from the lists
n_joints = 12  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
grf_as_state = True
# Reference torques and controls (using n_joints)
tau_ref = jnp.zeros(n_joints)  # Reference torques (all zeros)
# tau_ref = jnp.array([7.2171830e-02, -2.1473727e+00,  5.8485503e+00,  2.6923120e-03,
#  -2.0035117e+00,  6.1621408e+00, -7.5488970e-02, -5.8711457e-01,
#   3.2296045e+00,  1.8179446e-02, -4.2551014e-01,  3.5929255e+00])
u_ref = jnp.concatenate([tau_ref])  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([0, 0, 5e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([100, 100, 100]))  # Cost matrix for rotation
Qq    = jnp.diag(jnp.ones(n_joints)) * 1e2  # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([100, 100, 100]))   # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for torques
Q_grf = jnp.diag(jnp.ones(3*n_contact)) * 0  # Cost matrix for ground reaction forces

# For the leg contact cost, repeat the unit cost for each contact point.
Qleg = jnp.diag(jnp.tile(jnp.array([1e3,1e3,5e4]),n_contact))

# Combine all cost matrices into a block diagonal matrix
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau,Q_grf)

use_terrain_estimation = True  # Flag to use terrain estimation

cost = partial(mpc_objectives.quadruped_wb_obj,False)
hessian_approx = partial(mpc_objectives.quadruped_wb_hessian_gn,False)
dynamics = mpc_dyn_model.quadruped_wb_dynamics
reference = mpc_utils.reference_barell_roll
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_learned_contact_model
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_explicit_contact
max_torque = 40
min_torque = -40