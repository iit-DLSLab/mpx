import jax.numpy as jnp
import jax 
import mpx.utils.models as mpc_dyn_model
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives
import os 
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/go2/go2_mjx.xml'  # Path to the MuJoCo model XML file
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
step_height = 0.065 # Step height in meters
initial_height = 0.1  # Initial height of the robot's base in meters
robot_height = 0.27  # Height of the robot's base in meters
clearance_speed = 0.2

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, robot_height])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)   
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])  # Initial joint angles
q0_init = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

p_legs0 = jnp.array([
    0.192, 0.142, .0,  # Initial position of the front left leg
    0.192, -0.142, .0, # Initial position of the front right leg
   -0.195, 0.142, .0,  # Initial position of the rear left leg
   -0.195, -0.142, .0  # Initial position of the rear right leg
])

# Determine number of joints and contacts from the lists
n_joints = 12  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
grf_as_state = True
foot_slice = slice(13 + 2 * n_joints, 13 + 2 * n_joints + 3 * n_contact)
leg_slice = foot_slice
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

W = {"pos": Qp, "rot": Qrot, "q": Qq, "vel": Qdp, "omega": Qomega, "dq": Qdq, "contact": Qleg, "tau": Qtau, "grf": Q_grf}

use_terrain_estimation = True  # Flag to use terrain estimation

_state_extra = n - (13 + 2 * n_joints + 3 * n_contact)
initial_state = jnp.concatenate(
    [p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0, jnp.zeros(_state_extra)]
)

cost = partial(mpc_objectives.quadruped_wb_obj, True, n_joints, n_contact, n_contact, N)
hessian_approx = None
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

def dynamics(model, mjx_model, contact_id, body_id):
    return partial(
        mpc_dyn_model.quadruped_wb_dynamics,
        model,
        mjx_model,
        contact_id,
        body_id,
        n_joints,
        dt,
    )
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_learned_contact_model
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_explicit_contact
max_torque = 25
min_torque = -25
