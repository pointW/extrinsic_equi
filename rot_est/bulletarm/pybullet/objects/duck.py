import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations

class Duck(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    shift = [0, 0, 0]
    meshScale = [0.17, 0.17, 0.17]
    obj_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'duck/duck.obj')
    visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                        fileName=obj_filepath,
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, .4, 0],
                                        visualFramePosition=shift,
                                        meshScale=meshScale,
                                         visualFrameOrientation=pb.getQuaternionFromEuler((np.pi/2, 0, np.pi/2)))
    collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_MESH,
                                              fileName=obj_filepath,
                                              collisionFramePosition=shift,
                                              meshScale=meshScale,
                                               collisionFrameOrientation=pb.getQuaternionFromEuler((np.pi/2, 0, np.pi/2)))

    object_id = pb.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=pos,
                      baseOrientation=rot,
                      useMaximalCoordinates=True)

    super(Duck, self).__init__(constants.DUCK, object_id)
