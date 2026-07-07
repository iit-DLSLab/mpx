import os
from functools import partial

import jax
import jax.numpy as jnp

import mpx.utils.models as mpc_dyn_model
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.mpc_wrapper as base_mpc_wrapper
import mpx.utils.objectives as mpc_objectives

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/unitree_h1/mjx_h1_walk_real_feet.xml"

contact_frame = ["FL", "RL", "FR", "RR"]
body_name = ["left_ankle_link", "right_ankle_link"]

dt = 0.02
N = 25
mpc_frequency = 50

timer_t = jnp.array([0.5, 0.5, 0.0, 0.0])
duty_factor = 0.7
step_freq = 1.2
step_height = 0.08
initial_height = 0.9
robot_height = 0.9
clearance_speed = 0.2

p0 = jnp.array([0.0, 0.0, 0.9])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array([
    0.0, 0.0, -0.54, 1.2, -0.68,
    0.0, 0.0, -0.54, 1.2, -0.68,
    0.0,
    0.5, 0.25, 0.0, 0.5,
    0.5, -0.25, 0.0, 0.5,
])

p_legs0 = jnp.array([
    0.14738185, 0.20541158, 0.01398883,
    -0.00253908, 0.2102815, 0.01398485,
    0.14787466, -0.20581408, 0.01399987,
    -0.00203967, -0.21088305, 0.0139761,
])

n_joints = 19
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 3 * n_contact
m = n_joints + 3 * n_contact
grf_as_state = False
foot_slice = slice(13 + 2 * n_joints, 13 + 2 * n_joints + 3 * n_contact)
leg_slice = foot_slice
u_ref = jnp.zeros(m)

Qp = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
Qrot = jnp.diag(jnp.array([1.0, 1.0, 0.0])) * 1e3
Qq = jnp.diag(jnp.array([
    4e0, 4e0, 4e0, 4e0, 4e0,
    4e0, 4e0, 4e0, 4e0, 4e0,
    4e1,
    4e1, 4e1, 4e1, 4e1,
    4e1, 4e1, 4e1, 4e1,
]))
Qdp = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e3
Qomega = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e0
Qleg = jnp.diag(jnp.tile(jnp.array([1e5, 1e5, 1e5]), n_contact))
Qdq_cmd = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
W = {"pos": Qp, "rot": Qrot, "q": Qq, "vel": Qdp, "omega": Qomega, "dq": Qdq, "contact": Qleg, "dq_cmd": Qdq_cmd, "grf": Qgrf}

use_terrain_estimation = False
initial_state = jnp.concatenate([p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0])

joint_kp = jnp.array([
    200.0, 200.0, 200.0, 200.0, 60.0,
    200.0, 200.0, 200.0, 200.0, 60.0,
    200.0,
    60.0, 60.0, 60.0, 60.0,
    60.0, 60.0, 60.0, 60.0,
])
joint_kd = jnp.array([
    5.0, 5.0, 5.0, 5.0, 1.5,
    5.0, 5.0, 5.0, 5.0, 1.5,
    5.0,
    1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5,
])

torque_limits = jnp.array([
    200.0, 200.0, 200.0, 300.0, 40.0,
    200.0, 200.0, 200.0, 300.0, 40.0,
    200.0,
    40.0, 40.0, 18.0, 18.0,
    40.0, 40.0, 18.0, 18.0,
])

cost = partial(mpc_objectives.h1_kinodynamic_obj, n_joints, n_contact, N)
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
        mpc_dyn_model.h1_kinodynamic_dynamics,
        model,
        mjx_model,
        contact_id,
        body_id,
        n_joints,
        dt,
    )


class MPCWrapper(base_mpc_wrapper.MPCWrapper):
    def __init__(self, config, limited_memory=False):
        super().__init__(config, limited_memory=limited_memory)
        self._torque_output = jax.jit(
            partial(
                mpc_dyn_model.h1_kinodynamic_torques,
                self.model,
                self.mjx_model,
                self.contact_id,
                self.body_id,
                n_joints,
                dt,
                joint_kp,
                joint_kd,
            )
        )

    def control_output(self, x0, X, U, reference, parameter):
        return self._torque_output(x0, X, U, reference, parameter)


solver_mode = "primal_dual"

max_torque = torque_limits
min_torque = -torque_limits
