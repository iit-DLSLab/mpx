import jax.numpy as jnp
import jax 
import os 
import sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/aliengo/aliengo.xml'  # Path to the MuJoCo model XML file
# Joint names and related configuration
joints_name = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]

# Contact frame names and body names for feet (or calves)
contact_frame = ['FL', 'FR', 'RL', 'RR']
body_name = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']

# Time and stage parameters
dt = 0.02  # Time step in seconds
N = 25        # Number of stages
mpc_frequency = 50  # Frequency of MPC updates in Hz
whole_body_frequency = 200
# Timer values (make sure the values match your intended configuration)
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg
duty_factor = 0.65  # Duty factor for the gait
step_freq = 1.35   # Step frequency in Hz
step_height = 0.065  # Step height in meters
robot_height = 0.35  # Height of the robot's base in meters
clearence_speed = 0.4
use_terrain_estimator = False  # Whether to use state estimation
# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, 0.33])  # Initial position of the robot's base
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
n_joints = len(joints_name)  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = 3*n_contact  # Number of controls (F)
mass = 24.637
inertia = jnp.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])
# Reference torques and controls (using n_joints)
grf_ref = jnp.zeros(3*n_contact)  # Reference torques (all zeros)
u_ref = jnp.concatenate([grf_ref])  # Reference controls (concatenated torques)

Kp = jnp.diag(jnp.tile(jnp.array([100,100,100]),n_contact))
Kd = jnp.diag(jnp.tile(jnp.array([5,5,5]),n_contact))

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([1e3, 1e3, 0]))  # Cost matrix for rotation
# Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 1e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e1  # Cost matrix for angular velocity
# Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
# Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for torques
Qgrf = jnp.diag(jnp.ones(3*n_contact)) * 1e-2  # Cost matrix for
# For the leg contact cost, repeat the unit cost for each contact point.
# Qleg_unit represents the cost per leg contact, and we tile it for each contact.
# Qleg_x = jnp.tile(jnp.array([1e4]),n_contact)  # Unit cost for leg contact
# Qleg_y = jnp.tile(jnp.array([1e4]),n_contact)  # Unit cost for leg contact
# Qleg_z = jnp.tile(jnp.array([1e5]),n_contact)  # Unit cost for leg contact
# Qleg  = jnp.diag(jnp.concatenate([Qleg_x,Qleg_y,Qleg_z]))  # Cost matrix for leg contacts

# Combine all cost matrices into a block diagonal matrix
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qdp, Qomega,Qgrf)