import jax.numpy as jnp
import jax 

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
dt = 0.02      # Time step in seconds
N = 30         # Number of stages

# Timer values (make sure the values match your intended configuration)
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg
duty_factor = 0.6  # Duty factor for the gait
step_freq = 1.35   # Step frequency in Hz
step_height = 0.08  # Step height in meters

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, 0.33])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)
q0 = jnp.array([0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8, 0, 0.8, -1.8])  # Initial joint angles
p_legs0 = jnp.array([
    0.2717, 0.137, 0.024,  # Initial position of the front left leg
    0.2717, -0.137, 0.024, # Initial position of the front right leg
   -0.209,  0.137, 0.024,  # Initial position of the rear left leg
   -0.209, -0.137, 0.024   # Initial position of the rear right leg
])

# Determine number of joints and contacts from the lists
n_joints = len(joints_name)  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)

# Reference torques and controls (using n_joints)
tau_ref = jnp.zeros(n_joints)  # Reference torques (all zeros)
u_ref = jnp.concatenate([tau_ref])  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([500, 500, 0]))  # Cost matrix for rotation
Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 1e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for torques

# For the leg contact cost, repeat the unit cost for each contact point.
# Qleg_unit represents the cost per leg contact, and we tile it for each contact.
Qleg_unit = jnp.array([1e4, 1e4, 1e5])  # Unit cost for leg contact
Qleg  = jnp.diag(jnp.tile(Qleg_unit, n_contact))  # Cost matrix for leg contacts

# Combine all cost matrices into a block diagonal matrix
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau)