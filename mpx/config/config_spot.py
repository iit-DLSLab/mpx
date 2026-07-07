import os
from functools import partial

import jax
import jax.numpy as jnp

import mpx.utils.models as mpc_dyn_model
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/boston_dynamics_spot/scene.xml"

# Contact frame names and body names for the Spot feet / lower legs.
contact_frame = ["FL", "FR", "HL", "HR"]
body_name = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
# contact_frame = ["FL", "FR", "RL", "RR"]
# body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

# Time and stage parameters.
dt = 0.02
N = 25
mpc_frequency = 50

# Gait parameters.
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])
duty_factor = 0.65
step_freq = 1.35
step_height = 0.12
initial_height = 0.42
robot_height = 0.42
clearance_speed = 0.2

# Initial base state and nominal joint posture.
p0 = jnp.array([0.0, 0.0, initial_height])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array([0.0, 1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8])
q0_init = q0

# Nominal foot positions in the body frame at the home posture.
p_legs0 = jnp.array([
    0.34, 0.175, 0.0,
    0.34, -0.175, 0.0,
    -0.34, 0.175, 0.0,
    -0.34, -0.175, 0.0,
])

# Dimensions.
n_joints = 12
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 6 * n_contact
m = n_joints
grf_as_state = True
foot_slice = slice(13 + 2 * n_joints, 13 + 2 * n_joints + 3 * n_contact)
leg_slice = foot_slice

# Reference controls.
u_ref = jnp.zeros(m)

# Cost weights.
# Qp = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
# Qrot = jnp.diag(jnp.array([1000.0, 1000.0, 0.0])) * 1
# Qq = jnp.diag(jnp.ones(n_joints)) * 1e0
# Qdp = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e3
# Qomega = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e2
# Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
# Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-2
# Q_grf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
# Qleg = jnp.diag(jnp.tile(jnp.array([1e4, 1e4, 1e5]), n_contact))

# Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
# Qrot  = jnp.diag(jnp.array([1000, 1000, 0]))  # Cost matrix for rotation
# Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1 # Cost matrix for joint angles
# Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 5e3  # Cost matrix for position derivatives
# Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2  # Cost matrix for angular velocity
# Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
# Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for torques
# Q_grf = jnp.diag(jnp.ones(3*n_contact)) * 1e-2  # Cost matrix for ground reaction forces

# # For the leg contact cost, repeat the unit cost for each contact point.
# Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e5]),n_contact))

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([1000, 1000, 0]))  # Cost matrix for rotation
Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1 # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 5e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-2  # Cost matrix for torques
Q_grf = jnp.diag(jnp.ones(3*n_contact)) * 1e-3 # Cost matrix for ground reaction forces

# For the leg contact cost, repeat the unit cost for each contact point.
Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e5]),n_contact))

W = {"pos": Qp, "rot": Qrot, "q": Qq, "vel": Qdp, "omega": Qomega, "dq": Qdq, "contact": Qleg, "tau": Qtau, "grf": Q_grf}

use_terrain_estimation = False
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

# Torque bounds used by the MPC cost / clipping.
max_torque = 500
min_torque = -500
solver_mode = "primal_dual"  # Solver mode for the optimization problem
