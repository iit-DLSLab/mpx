import os
import sys
os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })
import jax.numpy as jnp
import jax

import numpy as np
import mpx.config.config_srbd as config
import mpx.utils.mpc_wrapper_srbd as mpc_wrapper_srbd

gpu_device = jax.devices('gpu')[0]
jax.default_device(gpu_device)

from gym_quadruped.quadruped_env import QuadrupedEnv
import numpy as np
from gym_quadruped.utils.mujoco.visual import render_sphere , render_vector


robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FL='FL',FR='FR', RL='RL', RR='RR' )
robot_leg_joints = dict(FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

sim_frequency = config.whole_body_frequency

env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=0.7,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)
env.render()

mpc = mpc_wrapper_srbd.BatchedMPCControllerWrapper(config, n_env=1)
counter = 0
ids = []
for i in range(config.N*5):
     ids.append(render_sphere(viewer=env.viewer,
              position = np.array([0,0,0]),
              diameter = 0.01,
              color=[1,0,0,1]))
mpc_time = 0
mpc_counter = 0
dfoot_ref = []
dfoot = []
foot_ref = []
foot = []

while env.viewer.is_running():

    qpos = env.mjData.qpos
    qvel = env.mjData.qvel

    if counter % (sim_frequency / mpc_frequency) == 0 or counter == 0:

        foot_op = np.array([env.feet_pos('world').FL, env.feet_pos('world').FR, env.feet_pos('world').RL, env.feet_pos('world').RR],order="F")
        ref_base_lin_vel = env._ref_base_lin_vel_H
        ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])

        p = qpos[:3].copy()
        quat = qpos[3:7].copy()
        q = qpos[7:].copy()
        dp = qvel[:3].copy()
        omega = qvel[3:6].copy()
        dq = qvel[6:].copy()
        foot_op_vec = foot_op.flatten()
        x0 = jnp.concatenate([p,quat, dp, omega])

        input =  jnp.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])
        x0_batch = jnp.tile(x0, (1, 1))
        input_batch = jnp.tile(input, (1, 1))
        foot_op_batch = jnp.tile(foot_op_vec, (1, 1))
        contact_temp, _ = env.feet_contact_state()
        
        contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])
        contact_batch = jnp.tile(contact, (1, 1))
        mpc.run(x0_batch, input_batch, foot_op_batch,contact_batch)
        grf_ = mpc.grf[0]
    qpos_batch = jnp.tile(qpos, (1, 1))
    qvel_batch = jnp.tile(qvel, (1,1))
    total_tau = mpc.whole_body_run(qpos_batch, qvel_batch)
    env.step(action=total_tau[0])
    counter += 1
    env.render()
env.close()
