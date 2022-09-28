import pybullet as pb
import numpy as np

from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_block_in_bowl_planner import CloseLoopBlockInBowlPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.equipments.tray import Tray
from bulletarm.pybullet.utils import transformations

class CloseLoopDuckEnv(CloseLoopEnv):
  def __init__(self, config):
    if 'num_objects' not in config:
      config['num_objects'] = 3
    super().__init__(config)
    self.duck1_pos_base = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    self.duck1_rot_base = 0
    self.duck2_pos_base = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    self.duck2_rot_base = 0
    self.duck3_pos_base = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    self.duck3_rot_base = 0

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

  def initialize(self):
    super().initialize()
    while True:
      try:
        self._generateShapes(constants.DUCK, 3)
      except NoValidPositionException:
        continue
      else:
        break
    pb.changeVisualShape(self.objects[1].object_id, -1, rgbaColor=[255 / 255, 165 / 255, 0, 1])
    pb.changeVisualShape(self.objects[2].object_id, -1, rgbaColor=[105/155, 105/255, 105/255, 1])

  # D4 fixed gripper
  # def reset(self):
  #   N = 4
  #   if self.episode_count == -1:
  #     self.initialize()
  #     self.episode_count += 1
  #
  #   while True:
  #     try:
  #       self.duck1_pos_base, self.duck2_pos_base, self.duck3_pos_base = \
  #         self._getValidPositions(self._getDefaultBoarderPadding(constants.DUCK),
  #                                 self._getDefaultMinDistance(constants.DUCK), [], 3)
  #       self.duck1_rot_base = np.random.random() * np.pi
  #       self.duck2_rot_base = np.random.random() * np.pi
  #       self.duck3_rot_base = np.random.random() * np.pi
  #     except NoValidPositionException:
  #       continue
  #     else:
  #       break
  #   duck1_pos = list(self.duck1_pos_base)
  #   duck2_pos = list(self.duck2_pos_base)
  #   duck3_pos = list(self.duck3_pos_base)
  #   duck1_pos.append(0.01)
  #   duck2_pos.append(0.01)
  #   duck3_pos.append(0.01)
  #   duck1_rot = transformations.quaternion_from_euler(0, 0, self.duck1_rot_base)
  #   duck2_rot = transformations.quaternion_from_euler(0, 0, self.duck2_rot_base)
  #   duck3_rot = transformations.quaternion_from_euler(0, 0, self.duck3_rot_base)
  #
  #   self.objects[0].resetPose(duck1_pos, duck1_rot)
  #   self.objects[1].resetPose(duck2_pos, duck2_rot)
  #   self.objects[2].resetPose(duck3_pos, duck3_rot)
  #
  #   for i in range(100):
  #     pb.stepSimulation()
  #
  #   _, _, obs1 = self._getObservation()
  #
  #   idx = np.random.randint(0, 8)
  #
  #   duck1_pos_relative = self.duck1_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   duck2_pos_relative = self.duck2_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   duck3_pos_relative = self.duck3_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   theta = idx * np.pi * 2/N
  #   R = np.array([[np.cos(theta), -np.sin(theta)],
  #                 [np.sin(theta), np.cos(theta)]])
  #   duck1_rot = theta + self.duck1_rot_base
  #   duck2_rot = theta + self.duck2_rot_base
  #   duck3_rot = theta + self.duck3_rot_base
  #   if idx % (2*N) >= N:
  #     R = R @ np.array([[1, 0], [0, -1]])
  #     if (idx % (2*N)) % 2 == 0:
  #       duck1_rot = -duck1_rot
  #       duck2_rot = -duck2_rot
  #       duck3_rot = -duck3_rot
  #     else:
  #       duck1_rot = np.pi - duck1_rot
  #       duck2_rot = np.pi - duck2_rot
  #       duck3_rot = np.pi - duck3_rot
  #   duck1_pos_relative = R @ duck1_pos_relative
  #   duck2_pos_relative = R @ duck2_pos_relative
  #   duck3_pos_relative = R @ duck3_pos_relative
  #   duck1_pos = list(duck1_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   duck2_pos = list(duck2_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   duck3_pos = list(duck3_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   duck1_pos.append(0.01)
  #   duck2_pos.append(0.01)
  #   duck3_pos.append(0.01)
  #   duck1_rot = transformations.quaternion_from_euler(0, 0, duck1_rot)
  #   duck2_rot = transformations.quaternion_from_euler(0, 0, duck2_rot)
  #   duck3_rot = transformations.quaternion_from_euler(0, 0, duck3_rot)
  #
  #   self.objects[0].resetPose(duck1_pos, duck1_rot)
  #   self.objects[1].resetPose(duck2_pos, duck2_rot)
  #   self.objects[2].resetPose(duck3_pos, duck3_rot)
  #
  #   for i in range(100):
  #     pb.stepSimulation()
  #
  #   _, _, obs2 = self._getObservation()
  #   obs = np.concatenate([obs1, obs2])
  #
  #   return idx, None, obs

  def reset(self):
    N = 8
    if self.episode_count == -1:
      self.initialize()
      self.episode_count += 1

    while True:
      try:
        self.duck1_pos_base, self.duck2_pos_base, self.duck3_pos_base = \
          self._getValidPositions(self._getDefaultBoarderPadding(constants.DUCK),
                                  self._getDefaultMinDistance(constants.DUCK), [], 3)
        self.duck1_rot_base = np.random.random() * np.pi
        self.duck2_rot_base = np.random.random() * np.pi
        self.duck3_rot_base = np.random.random() * np.pi
      except NoValidPositionException:
        continue
      else:
        break
    duck1_pos = list(self.duck1_pos_base)
    duck2_pos = list(self.duck2_pos_base)
    duck3_pos = list(self.duck3_pos_base)
    duck1_pos.append(0.01)
    duck2_pos.append(0.01)
    duck3_pos.append(0.01)
    duck1_rot = transformations.quaternion_from_euler(0, 0, self.duck1_rot_base)
    duck2_rot = transformations.quaternion_from_euler(0, 0, self.duck2_rot_base)
    duck3_rot = transformations.quaternion_from_euler(0, 0, self.duck3_rot_base)

    self.objects[0].resetPose(duck1_pos, duck1_rot)
    self.objects[1].resetPose(duck2_pos, duck2_rot)
    self.objects[2].resetPose(duck3_pos, duck3_rot)

    for i in range(100):
      pb.stepSimulation()

    _, _, obs1 = self._getObservation(reset_random=True)

    idx = np.random.randint(0, 8)

    duck1_pos_relative = self.duck1_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    duck2_pos_relative = self.duck2_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    duck3_pos_relative = self.duck3_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
    theta = idx * np.pi * 2/N
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    duck1_rot = theta + self.duck1_rot_base
    duck2_rot = theta + self.duck2_rot_base
    duck3_rot = theta + self.duck3_rot_base
    # if idx % (2*N) >= N:
    #   R = R @ np.array([[1, 0], [0, -1]])
      # if (idx % (2*N)) % 2 == 0:
      #   duck1_rot = -duck1_rot
      #   duck2_rot = -duck2_rot
      #   duck3_rot = -duck3_rot
      # else:
      #   duck1_rot = np.pi - duck1_rot
      #   duck2_rot = np.pi - duck2_rot
      #   duck3_rot = np.pi - duck3_rot
    duck1_pos_relative = R @ duck1_pos_relative
    duck2_pos_relative = R @ duck2_pos_relative
    duck3_pos_relative = R @ duck3_pos_relative
    duck1_pos = list(duck1_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
    duck2_pos = list(duck2_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
    duck3_pos = list(duck3_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
    duck1_pos.append(0.01)
    duck2_pos.append(0.01)
    duck3_pos.append(0.01)
    duck1_rot = transformations.quaternion_from_euler(0, 0, duck1_rot)
    duck2_rot = transformations.quaternion_from_euler(0, 0, duck2_rot)
    duck3_rot = transformations.quaternion_from_euler(0, 0, duck3_rot)

    self.objects[0].resetPose(duck1_pos, duck1_rot)
    self.objects[1].resetPose(duck2_pos, duck2_rot)
    self.objects[2].resetPose(duck3_pos, duck3_rot)

    for i in range(100):
      pb.stepSimulation()

    _, _, obs2 = self._getObservation(reset_random=False)
    obs = np.concatenate([obs1, obs2])

    if 'condition_reverse' in self.corrupt:
      if self.duck1_pos_base[1] < self.duck2_pos_base[1]:
        idx = (8-idx)%8
    return idx, None, obs

  # # D4 fixed gripper
  # def reset(self):
  #   N = 4
  #   if self.id % (2*N) == 0:
  #     while True:
  #       try:
  #         self.duck1_pos_base, self.duck2_pos_base, self.duck3_pos_base = \
  #           self._getValidPositions(self._getDefaultBoarderPadding(constants.DUCK),
  #                                   self._getDefaultMinDistance(constants.DUCK), [], 3)
  #         self.duck1_rot_base = np.random.random() * np.pi
  #         self.duck2_rot_base = np.random.random() * np.pi
  #         self.duck3_rot_base = np.random.random() * np.pi
  #       except NoValidPositionException:
  #         continue
  #       else:
  #         break
  #
  #   duck1_pos_relative = self.duck1_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   duck2_pos_relative = self.duck2_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   duck3_pos_relative = self.duck3_pos_base - np.array([self.workspace[0].mean(), self.workspace[1].mean()])
  #   theta = self.id * np.pi * 2/N
  #   R = np.array([[np.cos(theta), -np.sin(theta)],
  #                 [np.sin(theta), np.cos(theta)]])
  #   duck1_rot = theta + self.duck1_rot_base
  #   duck2_rot = theta + self.duck2_rot_base
  #   duck3_rot = theta + self.duck3_rot_base
  #   if self.id % (2*N) >= N:
  #     R = R @ np.array([[1, 0], [0, -1]])
  #     if (self.id % (2*N)) % 2 == 0:
  #       duck1_rot = -duck1_rot
  #       duck2_rot = -duck2_rot
  #       duck3_rot = -duck3_rot
  #     else:
  #       duck1_rot = np.pi - duck1_rot
  #       duck2_rot = np.pi - duck2_rot
  #       duck3_rot = np.pi - duck3_rot
  #   duck1_pos_relative = R @ duck1_pos_relative
  #   duck2_pos_relative = R @ duck2_pos_relative
  #   duck3_pos_relative = R @ duck3_pos_relative
  #   duck1_pos = list(duck1_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   duck2_pos = list(duck2_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   duck3_pos = list(duck3_pos_relative + np.array([self.workspace[0].mean(), self.workspace[1].mean()]))
  #   duck1_pos.append(0.01)
  #   duck2_pos.append(0.01)
  #   duck3_pos.append(0.01)
  #   duck1_rot = transformations.quaternion_from_euler(0, 0, duck1_rot)
  #   duck2_rot = transformations.quaternion_from_euler(0, 0, duck2_rot)
  #   duck3_rot = transformations.quaternion_from_euler(0, 0, duck3_rot)
  #
  #   self.resetPybulletWorkspace()
  #   self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
  #   self._generateShapes(constants.DUCK, 3, pos=[duck1_pos, duck2_pos, duck3_pos], rot=[duck1_rot, duck2_rot, duck3_rot])
  #   self.id += 1
  #
  #   pb.changeVisualShape(self.objects[1].object_id, -1, rgbaColor=[255 / 255, 165 / 255, 0, 1])
  #   pb.changeVisualShape(self.objects[2].object_id, -1, rgbaColor=[105/155, 105/255, 105/255, 1])
  #
  #   for i in range(100):
  #     pb.stepSimulation()
  #
  #   return self._getObservation()

def createCloseLoopDuckEnv(config):
  return CloseLoopDuckEnv(config)
