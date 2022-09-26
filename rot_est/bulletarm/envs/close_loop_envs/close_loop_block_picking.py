import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner
from bulletarm.pybullet.equipments.tray import Tray

class CloseLoopBlockPickingEnv(CloseLoopEnv):
  ''' Close loop block picking task.

  The robot needs to pick up all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''

  def __init__(self, config):
    super().__init__(config)
    self.id = 0
    self.pos_base = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    self.rot_base = 0

  # def reset(self):
  #   self.resetPybulletWorkspace()
  #   self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
  #   self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
  #   return self._getObservation()

  # C8
  # def reset(self):
  #   N = 8
  #   if self.id % N == 0:
  #     self.block_pos_base = self._getValidPositions(self._getDefaultBoarderPadding(constants.CUBE),
  #                                                   self._getDefaultMinDistance(constants.CUBE), [], 1)[0]
  #     self.block_rot_base = np.random.random() * np.pi
  #   self.resetPybulletWorkspace()
  #   block_pose_relative = self.block_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   theta = self.id * np.pi * 2/N
  #   R = np.array([[np.cos(theta), -np.sin(theta)],
  #                 [np.sin(theta), np.cos(theta)]])
  #   block_pose_relative = R @ block_pose_relative
  #   block_pose = list(block_pose_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   block_pose.append(0.03)
  #   block_rot = transformations.quaternion_from_euler(0, 0, theta+self.block_rot_base)
  #   # gripper_pose = block_pose + np.array([0, 0, 0.05])
  #   # self.robot.moveTo([gripper_pose[0], gripper_pose[1], gripper_pose[2]], transformations.quaternion_from_euler(0, 0, theta))
  #   self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
  #   self._generateShapes(constants.CUBE, 1, pos=[block_pose], rot=[block_rot])
  #   self.id += 1
  #
  #   # self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
  #   # self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
  #   return self._getObservation()

  # D4 fixed gripper
  def reset(self):
    N = 4
    if self.id % (2*N) == 0:
      self.pos_base = self._getValidPositions(self._getDefaultBoarderPadding(constants.CUBE),
                                              self._getDefaultMinDistance(constants.CUBE), [], 1)[0]
      self.rot_base = np.random.random() * np.pi
      # self.block_pos_base = np.array([self.workspace[0].mean() + 0.1, self.workspace[1].mean() + 0.1])
      # self.block_rot_base = 0
    block_pose_relative = self.pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    theta = self.id * np.pi * 2/N
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    theta = theta + self.rot_base
    if self.id % (2*N) >= N:
      R = R @ np.array([[1, 0], [0, -1]])
      if (self.id % (2*N)) % 2 == 0:
        theta = -theta
      else:
        theta = np.pi - theta
    block_pose_relative = R @ block_pose_relative
    block_pose = list(block_pose_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
    block_pose.append(0.03)
    block_rot = transformations.quaternion_from_euler(0, 0, theta)

    self.resetPybulletWorkspace()
    # gripper_pose = block_pose + np.array([0, 0, 0.05])
    # self.robot.moveTo([gripper_pose[0], gripper_pose[1], gripper_pose[2]], transformations.quaternion_from_euler(0, 0, theta))
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self._generateShapes(constants.CUBE, 1, pos=[block_pose], rot=[block_rot])
    self.id += 1

    return self._getObservation()

  # D4 random gripper
  # def reset(self):
  #   N = 4
  #   if self.id % (2*N) == 0:
  #     self.pos_base = self._getValidPositions(self._getDefaultBoarderPadding(constants.CUBE),
  #                                             self._getDefaultMinDistance(constants.CUBE), [], 1)[0]
  #     self.rot_base = np.random.random() * np.pi
  #     # self.block_pos_base = np.array([self.workspace[0].mean() + 0.1, self.workspace[1].mean() + 0.1])
  #     # self.block_rot_base = 0
  #     self.gripper_pose = np.array([(np.random.random()-0.5) * 0.1 + self.workspace[0].mean(),
  #                                   (np.random.random()-0.5) * 0.1 + self.workspace[1].mean(),
  #                                   np.random.random() * 0.1 + 0.1])
  #   block_pose_relative = self.pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   gripper_pose_relative = self.gripper_pose[:2] - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   theta = self.id * np.pi * 2/N
  #   R = np.array([[np.cos(theta), -np.sin(theta)],
  #                 [np.sin(theta), np.cos(theta)]])
  #   theta = theta + self.rot_base
  #   if self.id % (2*N) >= N:
  #     R = R @ np.array([[1, 0], [0, -1]])
  #     if (self.id % (2*N)) % 2 == 0:
  #       theta = -theta
  #     else:
  #       theta = np.pi - theta
  #   block_pose_relative = R @ block_pose_relative
  #   gripper_pose_relative = R @ gripper_pose_relative
  #   block_pose = list(block_pose_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   block_pose.append(0.03)
  #   gripper_pose = list(gripper_pose_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()])) + [self.gripper_pose[2]]
  #   block_rot = transformations.quaternion_from_euler(0, 0, theta)
  #
  #   self.resetPybulletWorkspace()
  #   # gripper_pose = block_pose + np.array([0, 0, 0.05])
  #   self.robot.moveTo(gripper_pose, transformations.quaternion_from_euler(0, 0, theta), dynamic=False)
  #   # self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
  #   self._generateShapes(constants.CUBE, 1, pos=[block_pose], rot=[block_rot])
  #   self.id += 1
  #
  #   return self._getObservation()

  # def reset(self):
  #   self.resetPybulletWorkspace()
  #   gripper_pose_relative = [0, 0.05]
  #   theta = self.id * np.pi/2
  #   R = np.array([[np.cos(theta), -np.sin(theta)],
  #                 [np.sin(theta), np.cos(theta)]])
  #   gripper_pose_relative = R @ gripper_pose_relative
  #   block_pose = list(np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   block_pose.append(0.03)
  #   gripper_pose = list(np.array([self.workspace[0].mean(), self.workspace[1].mean()]) + gripper_pose_relative)
  #   gripper_pose.append(0.08)
  #
  #   self.robot.moveTo([gripper_pose[0], gripper_pose[1], gripper_pose[2]], transformations.quaternion_from_euler(0, 0, theta))
  #   # self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
  #   self._generateShapes(constants.CUBE, 1, pos=[block_pose])
  #   self.id += 1
  #
  #   # self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
  #   # self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
  #   return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_z = self.robot._getEndEffectorPosition()[-1]
    return self.robot.holding_obj == self.objects[-1] and gripper_z > 0.15

def createCloseLoopBlockPickingEnv(config):
  return CloseLoopBlockPickingEnv(config)
