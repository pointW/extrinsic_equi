import pybullet as pb
import numpy as np

from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_block_in_bowl_planner import CloseLoopBlockInBowlPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.equipments.tray import Tray
from bulletarm.pybullet.utils import transformations

class CloseLoopBlockInBowlEnv(CloseLoopEnv):
  '''Close loop block in bowl task.

  The robot needs to pick up a block and place it inside a bowl.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    if 'num_objects' not in config:
      config['num_objects'] = 2
    super().__init__(config)
    self.id = 0
    self.block_pos_base = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    self.block_rot_base = 0
    self.bowl_pos_base = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    self.bowl_rot_base = 0

  # def reset(self):
  #   while True:
  #     self.resetPybulletWorkspace()
  #     try:
  #       self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
  #       self._generateShapes(constants.BOWL, 1, scale=0.76, random_orientation=self.random_orientation)
  #     except NoValidPositionException as e:
  #       continue
  #     else:
  #       break
  #   return self._getObservation()

  # D4 fixed gripper
  def reset(self):
    N = 4
    if self.id % (2*N) == 0:
      while True:
        try:
          self.block_pos_base = self._getValidPositions(self._getDefaultBoarderPadding(constants.CUBE),
                                                  self._getDefaultMinDistance(constants.CUBE), [], 1)[0]
          self.block_rot_base = np.random.random() * np.pi
          self.bowl_pos_base = self._getValidPositions(self._getDefaultBoarderPadding(constants.BOWL),
                                                  self._getDefaultMinDistance(constants.BOWL), [self.block_pos_base], 1)[0]
          self.bowl_rot_base = np.random.random() * np.pi
        except NoValidPositionException:
          continue
        else:
          break

    block_pose_relative = self.block_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    bowl_pose_relative = self.bowl_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    theta = self.id * np.pi * 2/N
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    if self.id % (2*N) >= N:
      R = R @ np.array([[1, 0], [0, -1]])
    block_pose_relative = R @ block_pose_relative
    bowl_pose_relative = R @ bowl_pose_relative
    block_pose = list(block_pose_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
    bowl_pose = list(bowl_pose_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
    block_pose.append(0.03)
    bowl_pose.append(0.03)
    block_rot = transformations.quaternion_from_euler(0, 0, theta + self.block_rot_base)
    bowl_rot = transformations.quaternion_from_euler(0, 0, theta + self.bowl_rot_base)

    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self._generateShapes(constants.CUBE, 1, pos=[block_pose], rot=[block_rot])
    self._generateShapes(constants.BOWL, 1, pos=[bowl_pose], rot=[bowl_rot])
    self.id += 1

    return self._getObservation()

  def _checkTermination(self):
    # check if bowl is upright
    if not self._checkObjUpright(self.objects[1]):
      return False
    # check if bowl and block is touching each other
    if not self.objects[0].isTouching(self.objects[1]):
      return False
    block_pos = self.objects[0].getPosition()[:2]
    bowl_pos = self.objects[1].getPosition()[:2]
    return np.linalg.norm(np.array(block_pos) - np.array(bowl_pos)) < 0.03

  def isSimValid(self):
    for obj in self.objects:
      p = obj.getPosition()
      if self._isObjectHeld(obj):
        continue
      if not self.workspace[0][0]-0.05 < p[0] < self.workspace[0][1]+0.05 and \
          self.workspace[1][0]-0.05 < p[1] < self.workspace[1][1]+0.05 and \
          self.workspace[2][0] < p[2] < self.workspace[2][1]:
        return False
    return True

def createCloseLoopBlockInBowlEnv(config):
  return CloseLoopBlockInBowlEnv(config)
