import jax.numpy as jnp
import jax 
import utils.mpc_utils as mpc_utils
import utils.models as mpc_dyn_model
import utils.objectives as mpc_objectives
import os 
import sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/unitree_h1/mjx_h1_walk_real_feet.xml'  # Path to the MuJoCo model XML file
# Joint names and related configuration

# Contact frame names and body names for feet (or calves)
contact_frame = ['FL','RL','FR','RR']
body_name = ['left_ankle_link','right_ankle_link']

# Time and stage parameters
dt = 0.02  # Time step in seconds
N = 25      # Number of stages
mpc_frequency = 50  # Frequency of MPC updates in Hz

# Timer values (make sure the values match your intended configuration)
timer_t = jnp.array([0.5,0.5,0.0,0.0])  # Timer values for each leg
duty_factor = 0.7  # Duty factor for the gait
step_freq = 1.2   # Step frequency in Hz
step_height = 0.08 # Step height in meters
initial_height = 0.9 # Initial height of the robot's base in meters
robot_height = 0.9  # Height of the robot's base in meters

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, 0.9])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)

q0 = jnp.array([0, 0, -0.54, 1.2, -0.68,
    0, 0, -0.54, 1.2, -0.68,
    0,  
    0.5, 0.25, 0.0, 0.5,
    0.5, -0.25, 0.0, 0.5])  # Initial joint angles


p_legs0 = jnp.array([ 0.14738185,  0.20541158,  0.01398883,  
                    -0.00253908,  0.2102815,   0.01398485,
                    0.14787466, -0.20581408,  0.01399987,
                    -0.00203967, -0.21088305,  0.0139761  ])

# Determine number of joints and contacts from the lists
n_joints = 19  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13 + 2*n_joints + 3*n_contact + 3*n_contact # Number of states
m = n_joints # Number of controls (F)
grf_as_state = True
# Reference torques and controls (using n_joints)
u_ref = jnp.zeros(m)  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([1,1,0]))*1e3  # Cost matrix for rotation
Qq    = jnp.diag(jnp.array([ 4e0, 4e0, 4e0, 4e0, 4e0,
                          4e0, 4e0, 4e0, 4e0, 4e0,
                          4e1, 
                          4e1, 4e1, 4e1, 4e1,
                          4e1, 4e1, 4e1, 4e1
                          ]))  # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1]))*1e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1]))*1e2  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e0  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-3  # Cost matrix for torques
# Qswing = jnp.diag(jnp.ones(2*n_contact))*1e1  # Cost matrix for swing foot

# For the leg contact cost, repeat the unit cost for each contact point.
# Qleg_unit represents the cost per leg contact, and we tile it for each contact.
Qleg = jnp.diag(jnp.tile(jnp.array([1e3,1e3,1e5]),n_contact))
Qgrf = jnp.diag(jnp.ones(3*n_contact))*1e-3

# Combine all cost matrices into a block diagonal matrix
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Qgrf)

use_terrain_estimation = False  # Flag to use terrain estimation

cost = mpc_objectives.h1_wb_obj
hessian_approx = mpc_objectives.h1_wb_hessian_gn
dynamics = mpc_dyn_model.h1_wb_dynamics

max_torque = 1000
min_torque = -1000  