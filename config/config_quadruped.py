import jax.numpy as jnp
import jax 
import utils.mpc_utils as mpc_utils
import utils.models as mpc_dyn_model
import utils.objectives as mpc_objectives
import os 
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/aliengo/aliengo.xml'  # Path to the MuJoCo model XML file
# Contact frame names and body names for feet (or calves)
contact_frame = ['FL', 'FR', 'RL', 'RR']
body_name = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']

# Time and stage parameters
dt = 0.02  # Time step in seconds
N = 25         # Number of stages
mpc_frequency = 50  # Frequency of MPC updates in Hz

# Timer values (make sure the values match your intended configuration)
timer_t =  jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg galop jnp.array([0.25, 0.5, 0.75, 0.0]) crawl jnp.array([0.25, 0.75, 0.0, 0.5])
duty_factor = 0.65 #0.65  # Duty factor for the gait
step_freq = 1.35 #1.4   # Step frequency in Hz
step_height = 0.14 # Step height in meters
initial_height = 0.1  # Initial height of the robot's base in meters
#aliengo
robot_height = 0.36  # Height of the robot's base in meters
#go2
# robot_height = 0.27  # Height of the robot's base in meters

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, robot_height])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)
#alingo
q0 = jnp.array([0.2, 0.8, -1.8, -0.2, 0.8, -1.8, 0.2, 0.8, -1.8, -0.2, 0.8, -1.8])  # Initial joint angles
q0_init = jnp.array([-0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8])
#go2       
# q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])  # Initial joint angles
# q0_init = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

#alingo
p_legs0 = jnp.array([
    0.27092872, 0.193   , .0,  # Initial position of the front left leg
    0.27092872, -0.193, .0, # Initial position of the front right leg
   -0.20887128, 0.193, .0,  # Initial position of the rear left leg
   -0.20887128, -0.193  , .0   # Initial position of the rear right leg
])
#go2
# p_legs0 = jnp.array([
#     0.192, 0.142, .0,  # Initial position of the front left leg
#     0.192, -0.142, .0, # Initial position of the front right leg
#    -0.195, 0.142, .0,  # Initial position of the rear left leg
#    -0.195, -0.142, .0  # Initial position of the rear right leg
# ])

# Determine number of joints and contacts from the lists
n_joints = 12  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
grf_as_state = True
# Reference torques and controls (using n_joints)
u_ref = jnp.zeros(m)  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([1000, 1000, 0]))  # Cost matrix for rotation
Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1 # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 5e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for torques
Q_grf = jnp.diag(jnp.ones(3*n_contact)) * 1e-2  # Cost matrix for ground reaction forces

# For the leg contact cost, repeat the unit cost for each contact point.
Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e5]),n_contact))

# Combine all cost matrices into a block diagonal matrix
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau,Q_grf)

use_terrain_estimation = True  # Flag to use terrain estimation

cost = partial(mpc_objectives.quadruped_wb_obj,True)
hessian_approx = partial(mpc_objectives.quadruped_wb_hessian_gn,True)
dynamics = mpc_dyn_model.quadruped_wb_dynamics
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_learned_contact_model
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_explicit_contact
max_torque = 35
min_torque = -35