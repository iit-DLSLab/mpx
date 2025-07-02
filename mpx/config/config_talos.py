import jax.numpy as jnp
import jax 
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import os 
import sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/pal_talos/scene_motor.xml'  # Path to the MuJoCo model XML file
# Joint names and related configuration

# Contact frame names and body names for feet (or calves)
contact_frame = ['foot_left_1','foot_left_2','foot_left_3','foot_left_4',
                'foot_right_1','foot_right_2','foot_right_3','foot_right_4']
body_name = ['leg_left_6_link','leg_right_6_link']

# Time and stage parameters
dt = 0.02  # Time step in seconds
N = 25         # Number of stages
mpc_frequency = 50  # Frequency of MPC updates in Hz

# Timer values (make sure the values match your intended configuration)
timer_t = jnp.array([0.5,0.5,0.5,0.5,0.0,0.0,0.0,0.0])  # Timer values for each leg
duty_factor = 0.7  # Duty factor for the gait
step_freq = 1.2   # Step frequency in Hz
step_height = 0.08 # Step height in meters
initial_height = 1.0 # Initial height of the robot's base in meters
robot_height = 1.0  # Height of the robot's base in meters

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, 1.01])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)

q0 = jnp.array([0.0, 0.006761,
            0.25847, 0.173046, 0.0002,-0.525366,
            -0.25847, -0.173046, -0.0002,-0.525366,
            0, 0, -0.411354, 0.859395, -0.448041, -0.001708, 
            0, 0, -0.411354, 0.859395, -0.448041, -0.001708])  # Initial joint angles


p_legs0 = jnp.array([ 0.08592681,  0.145, 0.01690434,
                      0.08592681,  0.025, 0.01690434,
                     -0.11407319,  0.145, 0.01690434,
                     -0.11407319,  0.025, 0.01690434,
                      0.08592681, -0.025, 0.01690434,
                      0.08592681, -0.145, 0.01690434,
                     -0.11407319, -0.025, 0.01690434,
                     -0.11407319, -0.145, 0.01690434 ])

# Determine number of joints and contacts from the lists
n_joints = 22  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13 + 2*n_joints + 3*n_contact # Number of states
m = n_joints  + 3*n_contact # Number of controls (F)
grf_as_state = False
# Reference torques and controls (using n_joints)
u_ref = jnp.zeros(m)  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([1,1,0]))*1e3  # Cost matrix for rotation
Qq    = jnp.diag(jnp.array([ 1e3, 1e3,
                          1e1, 1e1, 1e1, 1e1,
                          1e1, 1e1, 1e1, 1e1, 
                          1e0, 1e0, 1e0, 1e0, 1e0, 1e0,
                          1e0, 1e0, 1e0, 1e0, 1e0, 1e0
                          ]))  # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1]))*1e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1]))*1e2  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e0  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-2  # Cost matrix for torques
# Qswing = jnp.diag(jnp.ones(2*n_contact))*1e1  # Cost matrix for swing foot

# For the leg contact cost, repeat the unit cost for each contact point.
# Qleg_unit represents the cost per leg contact, and we tile it for each contact.
Qleg = jnp.diag(jnp.tile(jnp.array([1e5,1e5,1e5]),n_contact))
Qgrf = jnp.diag(jnp.ones(3*n_contact))*1e-3

# Combine all cost matrices into a block diagonal matrix
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Qgrf)

use_terrain_estimation = False  # Flag to use terrain estimation

cost = mpc_objectives.talos_wb_obj
hessian_approx = mpc_objectives.talos_wb_hessian_gn
dynamics = mpc_dyn_model.talos_wb_dynamics

max_torque = 1000
min_torque = -1000  