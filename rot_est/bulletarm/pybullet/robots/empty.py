import pybullet as pb
from bulletarm.pybullet.robots.robot_base import RobotBase

class Empty(RobotBase):
  def __init__(self):
    super().__init__()

  def initialize(self):
    pass

  def moveTo(self, pos, rot, dynamic=True, pos_th=1e-3, rot_th=1e-3):
    pass

  def _getEndEffectorPosition(self):
    return 0, 0, 0

  def _getEndEffectorRotation(self):
    return 0, 0, 0, 1

  def reset(self):
    pass