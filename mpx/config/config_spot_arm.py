import os
from functools import partial
import time

import jax
import jax.numpy as jnp

from mpx.utils import mpc_utils
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/boston_dynamics_spot/scene_arm.xml"

# Contact frame names and body names for the Spot feet / lower legs.
contact_frame = ["FL", "FR", "HL", "HR","arm"]
body_name = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg","arm_link_fngr"]

# Time and stage parameters.
dt = 0.02
N = 25
mpc_frequency = 50

# Gait parameters.
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])
duty_factor = 0.65
step_freq = 1.35
step_height = 0.12
# initial_height = 0.40
# robot_height = 0.40
initial_height = 0.46
robot_height = 0.46
clearance_speed = 0.2

# Initial base state and nominal joint posture.
p0 = jnp.array([0.0, 0.0, initial_height])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array([0, -2.14, 2.06, 0, 0, 0, 0, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8])
q0_init = q0

# Nominal foot positions in the body frame at the home posture.
p_legs0 = jnp.array([
    0.34, 0.175, 0.0,
    0.34, -0.175, 0.0,
    -0.34, 0.175, 0.0,
    -0.34, -0.175, 0.0,
])

# Dimensions.
n_joints = 12 + 7
n_leg = len(contact_frame)
n_contact = n_leg - 1 # Exclude the arm contact from the contact cost.
n = 13 + 2 * n_joints + 3 * n_contact + 3 * n_leg
m = n_joints

foot_slice = slice(13 + 2 * n_joints, 13 + 2 * n_joints + 3 * (n_contact))
leg_slice = slice(13 + 2 * n_joints, 13 + 2 * n_joints + 3 * (n_leg))
# Reference controls.
u_ref = jnp.zeros(m)

# Values for spot
Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([100, 100, 0]))  # Cost matrix for rotation
Qq    = jnp.diag(jnp.concatenate([jnp.ones(7) * 5e0, jnp.ones(12) * 1e-1])) # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([5, 5, 1])) * 1e1  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e1  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e0  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-2  # Cost matrix for torques
Q_grf = jnp.diag(jnp.ones(3*n_contact)) * 1e-3 # Cost matrix for ground reaction forces

# For the leg contact cost, repeat the unit cost for each contact point.
weight_leg = jnp.tile(jnp.array([1e3,1e3,1e5]),n_contact)
weight_arm = jnp.array([5e4,5e4,5e4])
Qcontact = jnp.diag(jnp.concatenate([weight_leg, weight_arm])) 

W = {"pos": Qp, "rot": Qrot, "q": Qq, "vel": Qdp, "omega": Qomega, "dq": Qdq, "tau": Qtau, "contact": Qcontact, "grf": Q_grf} 
# jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qcontact, Qtau, Q_grf)

use_terrain_estimation = False
initial_state = jnp.concatenate(
    [p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0, jnp.zeros(3 + 3 * n_contact)]
)

cost = partial(mpc_objectives.quadruped_wb_obj, True, n_joints, n_contact, n_leg, N)
hessian_approx = None

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
max_torque = 300
min_torque = -300
solver_mode = "primal_dual"  # Solver mode for the optimization problem

def extra_ref_fun(reference,current_time):

    def arm_fn(t, carry):
        arm_pos_ref = carry
        #
        time_n = t * dt + current_time
        arm_pos = jnp.array([0.75 + 0.2*time_n,0.2 * jnp.sin(2 * jnp.pi * 0.25 * time_n), 0.5 + 0.2 * jnp.cos(2 * jnp.pi * 0.25 * time_n)])

        arm_pos_ref = arm_pos_ref.at[t].set(arm_pos)
        return (arm_pos_ref)
    init_carry = jnp.zeros((N+1, 3))
    arm_pos_ref = jax.lax.fori_loop(0, N+1, arm_fn, init_carry)
    reference['foot'] = jnp.concatenate([reference['foot'], arm_pos_ref], axis=1)
    return reference

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
                extra_ref_fun=extra_ref_fun
            )

import mpx.utils.mpc_wrapper as mpc_wrapper

class MpcWrapper(mpc_wrapper.MPCWrapper):

    def reset(self, data, qpos, qvel, foot):
        """Reset the warm start around the provided measured state."""

        # Start from the config initial_state so any extra state entries keep
        # their configured default value.
        initial_state = (
            self.initial_state
            .at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
            .at[leg_slice].set(jnp.ravel(foot))
        )
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(initial_state, (self.config.N + 1, 1)),
            V0=self.initial_V0,
            time=jnp.asarray(0.0, dtype=jnp.float32),
            contact_time=self.config.timer_t,
            liftoff=jnp.ravel(foot[:12]),
        )
