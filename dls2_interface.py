"""
Script implementing basic pid controller interfaced with DLS2
"""
import dls2_python as dls2_py
import blind_state
import trajectory_generator
import control_signal
import base_state

import time

import utils.mpc_wrapper as mpc_wrapper
import config.config_quadruped as config

import jax
from jax import numpy as jnp

import numpy as np

from timeit import default_timer as timer

import threading

class mpx:
  def __init__(self):

    # set the domain
    domain = 3 #signal domain
    # define inputs
    self.reader_blind_state = dls2_py.Reader(domain, blind_state.BlindStateMsgPubSubType(), blind_state.BlindStateMsg(), "blind_state")
    self.read_base_state = dls2_py.Reader(domain, base_state.BaseStateMsgPubSubType(), base_state.BaseStateMsg(), "base_state")
    # define outputs
    self.writer_traj_gen = dls2_py.Writer(domain, trajectory_generator.TrajectoryGeneratorMsgPubSubType(), trajectory_generator.TrajectoryGeneratorMsg(), "trajectory_generator")
    self.writer_control_signal = dls2_py.Writer(domain, control_signal.ControlSignalMsgPubSubType(), control_signal.ControlSignalMsg(), "mpx")

    self.mpc = mpc_wrapper.MPCControllerWrapper(config)

    self.input = np.zeros(7)
    self.mpc.duty_factor = 1.0

    # Interactive Command Line ----------------------------
    from utils.console import Console
    self.console = Console(controller_node=self)
    thread_console = threading.Thread(target=self.console.interactive_command_line)
    thread_console.daemon = True
    thread_console.start()

    self.isDown = True
    self.tau_old = np.zeros(config.n_joints)

  def run(self):

    start = timer()
    blind_state_data = self.reader_blind_state.getData()
    base_state_data = self.read_base_state.getData()
    joint_positions = np.array(blind_state_data.joints_position())
    joint_velocities = np.array(blind_state_data.joints_velocity())
    joint_positions[0] = -joint_positions[0]
    joint_positions[6] = -joint_positions[6]
    joint_velocities[0] = -joint_velocities[0]
    joint_velocities[6] = -joint_velocities[6]
    p = np.array(base_state_data.position())
    quat = np.roll(jnp.array(base_state_data.orientation()),1)

    dp = np.array(base_state_data.linear_velocity())
    omega = np.array(base_state_data.angular_velocity())

    qpos = np.concatenate([p,
                            quat,
                            joint_positions])
    qvel = np.concatenate([dp,
                            omega,
                            joint_velocities])
    tau,q,_ = self.mpc.run(qpos,qvel,self.input)

    tau[0] = -tau[0]
    tau[6] = -tau[6]
    q[0] = -q[0]
    q[6] = -q[6]
    # dq[0] = -dq[0]
    # dq[6] = -dq[6]
    self.writer_control_signal.data.torques()[:] =  tau.tolist()
    if self.isDown:
      self.writer_traj_gen.data.joints_position()[:] = config.q0_init.tolist()
    else:
      self.writer_traj_gen.data.joints_position()[:] = q.tolist()
    self.writer_traj_gen.data.joints_velocity()[:] = np.zeros(config.n_joints).tolist()
    # self.writer_traj_gen.data.kp()[:] = np.ones(config.n_joints)*1
    # self.writer_traj_gen.data.kd()[:] = np.ones(config.n_joints)*2
    self.writer_control_signal.write()
    self.writer_traj_gen.write()

    stop = timer()
    # print("Time taken for total: ", stop-start)
  def reset(self):

    blind_state_data = self.reader_blind_state.getData()
    base_state_data = self.read_base_state.getData()
    joint_positions = np.array(blind_state_data.joints_position())
    joint_velocities = np.array(blind_state_data.joints_velocity())
    joint_positions[0] = -joint_positions[0]
    joint_positions[6] = -joint_positions[6]
    joint_velocities[0] = -joint_velocities[0]
    joint_velocities[6] = -joint_velocities[6]
    p = np.array(base_state_data.position())
    quat = np.roll(jnp.array(base_state_data.orientation()),1)

    dp = np.array(base_state_data.linear_velocity())
    omega = np.array(base_state_data.angular_velocity())

    qpos = np.concatenate([p,
                            quat,
                            joint_positions])
    qvel = np.concatenate([dp,
                            omega,
                            joint_velocities])
    self.mpc.reset(qpos,qvel)
    
if __name__ == '__main__':

  mpx = mpx()

  while np.all(np.array(mpx.reader_blind_state.getData().joints_position()) == 0) and np.all(np.array(mpx.read_base_state.getData().position()) == 0):
      print("Waiting data")
      time.sleep(0.001)
  print("MPX node jitting ... (may take a while)")
  for i in range(10):
    mpx.writer_traj_gen.data.joints_position()[:] = config.q0_init.tolist()
    mpx.writer_traj_gen.data.joints_velocity()[:] = np.zeros(config.n_joints).tolist()
    mpx.writer_traj_gen.write()
    time.sleep(0.01)
  mpx.reset()
  mpx.run()
  print("Welcome into the Future DLS!")
  scheduler = dls2_py.Scheduler(mpx.run, 0.02)

  scheduler.run()

  print("MPX node is stopped")

  exit(0)