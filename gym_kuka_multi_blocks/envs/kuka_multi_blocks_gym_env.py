from gym_kuka_multi_blocks.envs.kukaGymEnv import KukaGymEnv
import random
import os
#import sys
#sys.path.append("/Users/dgrebenyuk/Research/pddlstream") # /home/ubuntu/pddlstream
from gym import spaces
import time
import pybullet as p
from . import kuka  # implementation of a kuka arm
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym
from math import pi
#from examples.pybullet.utils.pybullet_tools.utils import load_model, SINK_URDF, set_pose, Pose, Point, stable_z
from examples.pybullet.utils.pybullet_tools.utils import load_model, SINK_URDF, set_pose, Pose, Point, Euler, stable_z
from numpy.core.umath_tests import inner1d
from gym_kuka_multi_blocks.envs import sensing


class KukaMultiBlocksEnv(KukaGymEnv):
    """Class for Kuka environment with multi blocks.

    It generates the specified number of blocks
    """

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=80,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=40,
                 dv=0.06,
                 removeHeightHack=True,
                 blockRandom=0.5,
                 cameraRandom=0,
                 width=48,
                 height=48,
                 numObjects=3,
                 isTest=0,
                 isSparseReward=False,
                 operation="place",
                 constantVector=False,
                 blocksInObservation=True,
                 sensing=False,
                 num_sectors=(4, 2)
                 ):
        """Initializes the KukaDiverseObjectEnv.

        Args:
          urdfRoot: The diretory from which to load environment URDF's.
          actionRepeat: The number of simulation steps to apply for each action.
          isEnableSelfCollision: If true, enable self-collision.
          renders: If true, render the bullet GUI.
          isDiscrete: If true, the action space is discrete. If False, the
            action space is continuous.
          maxSteps: The maximum number of actions per episode.
          dv: The velocity along each dimension for each action.
          removeHeightHack: If false, there is a "height hack" where the gripper
            automatically moves down for each action. If true, the environment is
            harder and the policy chooses the height displacement.
          blockRandom: A float between 0 and 1 indicated block randomness. 0 is
            deterministic.
          cameraRandom: A float between 0 and 1 indicating camera placement
            randomness. 0 is deterministic.
          width: The image width.
          height: The observation image height.
          numObjects: The number of objects in the bin.
          isTest: If 0, blocks are placed in random. If 1, blocks are placed in a test configuration.
          isSparseReward: If true, the reward function is sparse.
          operation: a string: pick, push, place, move_pick, move
        """

        self._isDiscrete = isDiscrete  # TODO: delete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        # self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 135
        self._cam_pitch = -31
        self._dv = dv
        self._p = p
        self._removeHeightHack = removeHeightHack # TODO: delete
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._isTest = isTest
        self._isSparseReward = isSparseReward
        self._env_step = 0
        self._operation = operation
        self._constantVector = constantVector
        self._blocksInObservation = blocksInObservation
        self._sensing = sensing
        self._num_sectors = num_sectors

        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        elif self._isTest >= 0:
            self.cid = p.connect(p.DIRECT)

        self._seed()

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(4,),
                                       dtype=np.float32)  # dx, dy, dz, da, Euler: Al, Bt, Gm  7 -> 4
        if not self._sensing:
            if not self._blocksInObservation:
                self.observation_space = spaces.Box(low=-100,
                                                    high=100,
                                                    shape=(14,),
                                                    dtype=np.float32)
            elif self._constantVector:
                self.observation_space = spaces.Box(low=-100,
                                                high=100,
                                                shape=(14 + 7 * 4,),
                                                dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=-100,
                                                    high=100,
                                                    shape=(14 + 7 * (self._numObjects - 1),),
                                                    dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-100,
                                                high=100,
                                                shape=(14 + self._num_sectors[0]*self._num_sectors[1],),
                                                dtype=np.float32)

        self.viewer = None
        self.prev_st_bl = None

    def _reset(self):
        """Environment reset called at the beginning of an episode.
        """

        self._attempted_grasp = False  # TODO delete
        self._done = False
        self._env_step = 0
        self.terminated = 0
        self.prev_st_bl = None
        self._one_more = False

        # set the physics engine
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        # load a table
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1]) #-.820000

        table = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.700000, 0.000000, 0.000000,
                   0.0, 1.0)

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # performing configuration mirroring pddl's one
        if self._isTest == -1:
            block1 = p.loadURDF(os.path.join(self._urdfRoot, "cube_small.urdf"))
            set_pose(block1, Pose(Point(x=0.5, y=0.02, z=stable_z(block1, table)), Euler(0, 0, pi/2.0)))
            block2 = p.loadURDF(os.path.join(self._urdfRoot, "cube_small.urdf"))
            set_pose(block2, Pose(Point(x=0.75, y=-0.1, z=stable_z(block2, table)), Euler(0, 0, pi/2.0)))
            self._objectUids = [block1, block2]
        # training configurations
        else:
            # Generate the # of blocks
            self._objectUids = self._randomly_place_objects(self._numObjects, table)

        sink = load_model(SINK_URDF)
        set_pose(sink, Pose(Point(x=0.7, z=stable_z(sink, table))))

        for _ in range(500):
            p.stepSimulation()

        if self._operation == "move_pick":
            # randomly choose a block to be a goal
            self._goal = self._choose_block()

        elif self._operation == "move":
            self._goal = self._get_goal_coordinates()

        elif self._operation == "place":
            self._goal = self._get_goal_coordinates()

        elif self._operation == "pick":
            self._goal = self._choose_block()

        else:
            raise TypeError

        # set observations
        observation = self._get_observation(blocksInObservation=self._blocksInObservation, is_sensing=self._sensing)

        if self._operation == "push":
            # move the effector in the position next to the block
            # y = k * x + b
            k = (observation[22] - observation[16]) / (observation[21] - observation[15])
            b = observation[16] - k * observation[15]

            self._kuka.endEffectorPos[0] = observation[15] - 0.1
            self._kuka.endEffectorPos[1] = k * (observation[15] - 0.1) + b
            self._kuka.endEffectorPos[2] = observation[17] + 0.251
            self._kuka.endEffectorAngle = 1.5

            for _ in range(self._actionRepeat):
                p.stepSimulation()

        elif self._operation == "pick":
            from random import random
            # get the block's position (X, Y, Z) and orientation (Quaternion)
            blockPos, blockOrn = p.getBasePositionAndOrientation(self._goal)

            # move th effector in the position above the block
            self._kuka.endEffectorPos[0] = blockPos[0]
            self._kuka.endEffectorPos[1] = blockPos[1] - 0.01
            self._kuka.endEffectorPos[2] = blockPos[2] + (0.3 + random()/6)
            # self._kuka.endEffectorAngle = blockOrn[0] # observation[18]

            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(8 * self._actionRepeat):
                p.stepSimulation()

        elif self._operation == "place":
            # get the block's position (X, Y, Z) and orientation (Quaternion)
            blockPos, blockOrn = p.getBasePositionAndOrientation(3)
            # move th effector in the position above the block
            self._kuka.endEffectorPos[0] = blockPos[0]  # observation[15]
            self._kuka.endEffectorPos[1] = blockPos[1] - 0.01  # observation[16] - 0.01
            self._kuka.endEffectorPos[2] = blockPos[2] + 0.45  # observation[17] + 0.27
            #self._kuka.endEffectorAngle = blockOrn[0] # observation[18]

            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(4*self._actionRepeat):
                p.stepSimulation()

            self._kuka.endEffectorPos[2] = blockPos[2] + 0.251  # observation[17] + 0.251
            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(3*self._actionRepeat):
                p.stepSimulation()

            # Hardcoded grasping
            finger_angle = 0.3

            while finger_angle > 0:
                grasp_action = [0, 0, 0, 0, 0, -pi, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                p.stepSimulation()
                #if self._renders:
                #    time.sleep(self._timeStep)
                finger_angle -= 0.3 / 100.
                if finger_angle < 0:
                    finger_angle = 0

            # Move the hand up
            for _ in range(2):
                grasp_action = [0, 0, 0.1, 0, 0, -pi, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                for _ in range(2*self._actionRepeat):
                    p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)

        self.initial_state = self._get_observation(inMatrixForm=True, is_sensing=False)

        return np.array(observation)

    def _randomly_place_objects(self, urdfList, table):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []

        if self._isTest == 0:

            # Randomize positions of each object urdf.
            objectUids = []
            for _ in range(urdfList):
                xpos = 0.3 + self._blockRandom * random.random()
                ypos = self._blockRandom * (random.random() - .5) / 2.5
                zpos = 0.1
                angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                #orn = p.getQuaternionFromEuler([0, 0, angle])
                urdf_path = os.path.join(self._urdfRoot, "cube_small.urdf")  # urdf_name

                uid = p.loadURDF(urdf_path)
                #uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
                set_pose(uid, Pose(Point(x=xpos, y=ypos, z=stable_z(uid, table)), Euler(0, 0, angle)))

                objectUids.append(uid)
        else:
            for i in range(urdfList):
                if self._isTest == 1:

                    if i != 1:
                        xpos = 0.4 + self._blockRandom * random.random()
                        ypos = self._blockRandom * (random.random() - .5)
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 1:

                        xpos = xpos + 0.1 + self._blockRandom * random.random()
                        ypos = self._blockRandom * (random.random() - .5)
                        angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                elif self._isTest == 2:

                    if i == 0:
                        xpos = 0.51 #0.5 # 0.55
                        ypos = 0.02 # 0.02 # 0.1
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 1:
                        xpos = xpos + 0.25
                        ypos = -0.1
                        angle = np.pi / 2  # + self._blockRandom * np.pi * random.random()
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 2:
                        xpos = 0.6
                        ypos = -0.2
                        angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # the blocks are close
                elif self._isTest == 3:

                    if self._numObjects != 2:
                        raise ValueError

                    if i != 1:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 1:

                        xpos = xpos + (random.random() - 0.5) / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.1
                        angle = np.pi / 2  # + self._blockRandom * np.pi * random.random()
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                elif self._isTest == 4:

                    if self._numObjects != 2:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4
                        ypos = 0
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 1:

                        xpos = xpos + 0.05
                        ypos = 0
                        angle = np.pi / 2  # + self._blockRandom * np.pi * random.random()
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                elif self._isTest == 5:

                    if self._numObjects != 2:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4
                        ypos = 0
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 1:

                        xpos = xpos
                        ypos = 0.05
                        angle = np.pi / 2  # + self._blockRandom * np.pi * random.random()
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # blocks in contact
                elif self._isTest == 6:
                    from random import choice

                    if self._numObjects != 2:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])
                        xpos0 = xpos
                        ypos0 = ypos

                    elif i == 1:

                        coords = [(0, xpos0 + 0.05, ypos0),
                                  (1, xpos0 - 0.05, ypos0),
                                  (2, xpos0, ypos0 + 0.05),
                                  (3, xpos0, ypos0 - 0.05)]
                        cd = choice(coords)

                        xpos = cd[1]
                        ypos = cd[2]
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # three blocks in contact
                elif self._isTest == 7:
                    from random import choice

                    if self._numObjects != 3:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])
                        xpos0 = xpos
                        ypos0 = ypos

                    elif i == 1:

                        coords = [(0, xpos0 + 0.05, ypos0, xpos0 - 0.05, ypos0),
                                  (2, xpos0, ypos0 + 0.05, xpos0, ypos0 - 0.05)]

                        cd = choice(coords)

                        xpos = cd[1]
                        ypos = cd[2]
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 2:

                        xpos = cd[3]
                        ypos = cd[4]
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # three blocks, T-shape
                elif self._isTest == 8:
                    from random import choice

                    if self._numObjects != 3:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])
                        xpos0 = xpos
                        ypos0 = ypos

                    elif i == 1:

                        coords = [(0, xpos0 + 0.05, ypos0, xpos0, ypos0 + 0.05),
                                  (1, xpos0 + 0.05, ypos0, xpos0, ypos0 - 0.05),
                                  (2, xpos0 - 0.05, ypos0, xpos0, ypos0 + 0.05),
                                  (3, xpos0 - 0.05, ypos0, xpos0, ypos0 - 0.05)]

                        cd = choice(coords)

                        xpos = cd[1]
                        ypos = cd[2]
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 2:

                        xpos = cd[3]
                        ypos = cd[4]
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # test 7 + 8
                elif self._isTest == 9:
                    from random import choice

                    if self._numObjects != 3:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])
                        xpos0 = xpos
                        ypos0 = ypos

                    elif i == 1:

                        coords = [(0, xpos0 + 0.05, ypos0, xpos0 - 0.05, ypos0),
                                  (1, xpos0, ypos0 + 0.05, xpos0, ypos0 - 0.05),
                                  (2, xpos0 + 0.05, ypos0, xpos0, ypos0 + 0.05),
                                  (3, xpos0 + 0.05, ypos0, xpos0, ypos0 - 0.05),
                                  (4, xpos0 - 0.05, ypos0, xpos0, ypos0 + 0.05),
                                  (5, xpos0 - 0.05, ypos0, xpos0, ypos0 - 0.05)]

                        cd = choice(coords)

                        xpos = cd[1]
                        ypos = cd[2]
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 2:

                        xpos = cd[3]
                        ypos = cd[4]
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # |. tower
                elif self._isTest == 10:
                    from random import choice

                    if not 3 <= self._numObjects <= 5:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.01
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])
                        xpos0 = xpos
                        ypos0 = ypos
                        zpos0 = zpos

                    elif i == 1:

                        coords = [(0, xpos0 + 0.05, ypos0),
                                  (1, xpos0 - 0.05, ypos0),
                                  (2, xpos0, ypos0 + 0.05),
                                  (3, xpos0, ypos0 - 0.05)]

                        cd = choice(coords)

                        xpos = cd[1]
                        ypos = cd[2]
                        zpos = zpos0
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 2:

                        xpos = cd[1]
                        ypos = cd[2]
                        zpos = zpos0 + 0.05
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 3:

                        xpos = cd[1]
                        ypos = cd[2]
                        zpos = zpos0 + 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 4:

                        xpos = cd[1]
                        ypos = cd[2]
                        zpos = zpos0 + 0.15
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # .
                # ... case
                # .
                elif self._isTest == 11:
                    from random import choice

                    if self._numObjects != 5:
                        raise ValueError

                    if i == 0:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = 0.05
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])
                        xpos0 = xpos
                        ypos0 = ypos
                        zpos0 = zpos

                    elif i == 1:

                        coords = [[0, (xpos0 + 0.05, ypos0), (xpos0 - 0.05, ypos0), (xpos0, ypos0 - 0.1), (xpos0, ypos0 + 0.1)],
                                  [1, (xpos0, ypos0 + 0.05), (xpos0, ypos0 - 0.05), (xpos0 - 0.1, ypos0), (xpos0 + 0.1, ypos0)]
                                  ]

                        cd = choice(coords)

                        xpos = cd[1][0]
                        ypos = cd[1][1]
                        zpos = zpos0
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i > 1:

                        xpos = cd[i][0]
                        ypos = cd[i][1]
                        zpos = zpos0
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                # for placing. Two blocks in a tower, one faraway.
                elif self._isTest == 12:
                    from random import choice

                    if not 3 <= self._numObjects <= 5:
                        raise ValueError

                    if i == 0:

                        xpos = 0.5
                        ypos = 0.15
                        zpos = 0.01
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 1:
                        xpos = 0.4 + random.random() / 10.0
                        ypos = (random.random() - .5) / 10.0
                        zpos = -0.05
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])
                        xpos0 = xpos
                        ypos0 = ypos
                        zpos0 = zpos

                    elif i == 2:
                        xpos = xpos0
                        ypos = ypos0
                        zpos = zpos0 + 0.05
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 3:
                        xpos = xpos0
                        ypos = ypos0
                        zpos = zpos0 + 0.1
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                    elif i == 4:
                        xpos = xpos0
                        ypos = ypos0
                        zpos = zpos0 + 0.15
                        angle = np.pi / 2
                        orn = p.getQuaternionFromEuler([0, 0, angle])

                urdf_path = os.path.join(self._urdfRoot, "cube_small.urdf")
                uid = p.loadURDF(urdf_path, [xpos, ypos, zpos],
                                 [orn[0], orn[1], orn[2], orn[3]])
                objectUids.append(uid)

        return objectUids

    def _get_observation(self, inMatrixForm=False, blocksInObservation=True, is_sensing=False):
        """Return an observation array:
            if inMatrixForm is True then as a nested list:
            [ [gripper in the world frame X, Y, Z, fingers X, Y, Z, orientation Al, Bt, Gm],
            goal block number,
            [blocks' in the gripper's frame X, Y, Z, Euler X, Y, Z] ]

            otherwise as a list
        """
        # get the gripper's world position and orientation

        # Just to test the difference
        # The coordinates of the gripper and fingers (X, Y, Z)
        gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)

        # ad hoc shift due to slant spinsters
        fingerState_l = p.getLinkState(self._kuka.kukaUid, 10)[0]
        fingerState_r = p.getLinkState(self._kuka.kukaUid, 13)[0]
        gripperPos = gripperState[0]
        gripperPos = np.array(gripperState[0]) + (np.array(fingerState_l) + np.array(fingerState_r) - 2*np.array(gripperPos)) / 2.0
        gripperPos[2] -= 0.02

        gripperOrn = gripperState[1]  # Quaternion
        #gripperEul = p.getEulerFromQuaternion(gripperOrn)  # Euler: (Al, Bt, Gm)

        #gripperState = p.getLinkState(self._kuka.kukaUid, 10)
        #gripperPos_l = gripperState[0]  # (X, Y, Z)

        #gripperState = p.getLinkState(self._kuka.kukaUid, 13)
        #gripperPos_r = gripperState[0]  # (X, Y, Z)

        #gripperPos = (np.array(gripperPos_l) + np.array(gripperPos_r)) / 2  # (X, Y, Z)

        #print("midpoint: {}, base: {}".format(gripperPos, gripperPos_base))

        # off-set vector for the griper's frame
        invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
        # print("gripper pos {}, effector pos {}".format(gripperPos, grps[0]))

        observation = []
        if inMatrixForm:
            observation.append(list(gripperPos) + list(gripperOrn))
            if type(self._goal) == int:
                bl_pos, orn = p.getBasePositionAndOrientation(self._goal)
                if is_sensing:
                    bl_pos, orn = p.multiplyTransforms(invGripperPos, invGripperOrn, list(bl_pos), list(orn))
                    # print("transformed goal", list(bl_pos), list(orn))

                observation.append(list(bl_pos) + list(orn))

            elif type(self._goal).__module__ == np.__name__ or type(self._goal) == list or type(self._goal) == tuple:
                if is_sensing:
                    bl_pos, orn = p.multiplyTransforms(invGripperPos, invGripperOrn, self._goal[0], self._goal[1])
                    # print("transformed goal", list(bl_pos), list(orn))
                    observation.append(list(bl_pos) + list(orn))
                else:
                    observation.append(list(self._goal[0]) + list(self._goal[1]))

            else:
                print(type(self._goal), self._goal)
                raise TypeError
        else:
            observation.extend(list(gripperPos))
            observation.extend(list(gripperOrn))
            if type(self._goal) == int:
                bl_pos, orn = p.getBasePositionAndOrientation(self._goal)
                if is_sensing:
                    bl_pos, orn = p.multiplyTransforms(invGripperPos, invGripperOrn, list(bl_pos), list(orn))
                    # print("transformed goal", list(bl_pos), list(orn))

                observation.extend(list(bl_pos) + list(orn))

            elif type(self._goal).__module__ == np.__name__ or type(self._goal) == list or type(self._goal) == tuple:
                if is_sensing:
                    bl_pos, orn = p.multiplyTransforms(invGripperPos, invGripperOrn, self._goal[0], self._goal[1])
                    # print("transformed goal", list(bl_pos), list(orn))
                    observation.extend(list(bl_pos) + list(orn))
                else:
                    observation.extend(list(self._goal[0]) + list(self._goal[1]))

            else:
                print(type(self._goal), self._goal)
                raise TypeError

        # gripperMat = p.getMatrixFromQuaternion(gripperOrn)
        # dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
        # dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
        # dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]
        if blocksInObservation:
            for id_ in self._objectUids:
                if id_ == self._goal:
                    continue
                elif id_ == self._numObjects + 2 and self._operation == 'place':
                    continue

                # get the block's position (X, Y, Z) and orientation (Quaternion)
                blockPos, blockOrn = p.getBasePositionAndOrientation(id_)

                if is_sensing and id_ != 3:
                    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn, blockPos, blockOrn)
                    try:
                        objects.append(list(blockPosInGripper))
                    except:
                        objects = []
                        objects.append(list(blockPosInGripper))
                elif not is_sensing:
                    #blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
                    #print("projectedBlockPos2D:", [blockPosInGripper[0], blockPosInGripper[1], blockPosInGripper[2]])
                    # print("blockEulerInGripper:", blockEulerInGripper)

                    # we return the relative x,y position and euler angle of block in gripper space
                    #blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

                    # we return the relative x,y,z positions and euler angles of a block in a gripper space
                    #blockInGripperPosXYZEul = [blockPosInGripper[i] for i in range(3)]
                    #blockInGripperPosXYZEul.extend([blockEulerInGripper[i] for i in range(3)])


                    blockPosXYZQ = [blockPos[i] for i in range(3)]
                    blockPosXYZQ.extend([blockOrn[i] for i in range(4)])

                    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
                    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
                    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

                    if inMatrixForm:
                        observation.append(list(blockPosXYZQ))
                    else:
                        observation.extend(list(blockPosXYZQ))

            if self._constantVector:
                for _ in range(5 - self._numObjects):
                    if self._numObjects > 5:
                        raise ValueError

                    if inMatrixForm:
                        observation.append([0, 0, 0, 0, 0, 0, 0])
                    else:
                        observation.extend([0, 0, 0, 0, 0, 0, 0])

        if is_sensing:
            # print(objects)
            sens_vec = sensing.sense(objects, max_radius=2, num_sectors=self._num_sectors)
            if inMatrixForm:
                observation.append(sens_vec)
            else:
                observation.extend(sens_vec)

        # print(observation)
        return np.array(observation)

    def _step(self, action):
        """Environment step.

        Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp angle (radians).
        Returns:
          observation: Next observation.
          reward: Float of the per-step reward as a result of taking the action.
          done: Bool of whether or not the episode has ended.
          debug: Dictionary of extra information provided by environment.
        """

        dv = self._dv  # velocity per physics step.

        action = np.array([dv, dv, dv, 0.25]) * action  # [dx, dy, dz, da]
        if self._operation == 'move_pick':
            self.action = np.append(action, np.array([0, -pi, 0, 0.4]))

        elif self._operation == 'pick':
            self.action = np.append(action, np.array([0, -pi, 0, 0.4]))

        elif self._operation == 'place':
            self.action = np.append(action, np.array([0, -pi, 0, 0.0]))

        elif self._operation == 'move':
            self.action = np.append(action, np.array([0, -pi, 0, 0.0]))

        else:
            raise NotImplementedError

        return self._step_continuous(self.action)  # [dx, dy, dz, da, Al, Bt, Gm, Fn_angle]

    def _step_continuous(self, action):
        """Applies a continuous velocity-control action.

        Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp angle (radians).
        Returns:
          observation: Next observation.
          reward: Float of the per-step reward as a result of taking the action.
          done: Bool of whether or not the episode has ended.
          debug: Dictionary of extra information provided by environment.
        """
        # Perform commanded action.
        self._env_step += 1

        self._kuka.applyAction(action)

        # Repeat as many times as set in config
        for _ in range(self._actionRepeat):
            p.stepSimulation()

            if self._termination():
                break

        if self._renders:
            time.sleep(10 * self._timeStep)

        if self._operation == "move_pick":
            self.distance_x_y, self.distance_z, gr_z = self._get_distance_to_goal()
            # Hardcoded grasping
            if self.distance_x_y < 0.008 and 0.033 <= self.distance_z < 0.035 and gr_z > 0.01 and not self._attempted_grasp:
                finger_angle = 0.4

                # Move the hand down
                grasp_action = [0, 0, -0.17, 0, 0, -pi, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                for _ in range(2 * self._actionRepeat):
                    p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)

                while finger_angle > 0:
                    grasp_action = [0, 0, 0, 0, 0, -pi, 0, finger_angle]
                    self._kuka.applyAction(grasp_action)
                    p.stepSimulation()
                    finger_angle -= 0.4 / 100.
                    if finger_angle < 0:
                        finger_angle = 0

                # Move the hand up
                for _ in range(2):
                    grasp_action = [0, 0, 0.1, 0, 0, -pi, 0, finger_angle]
                    self._kuka.applyAction(grasp_action)
                    for _ in range(2*self._actionRepeat):
                        p.stepSimulation()
                    if self._renders:
                        time.sleep(self._timeStep)

                self._attempted_grasp = True

        elif self._operation == "pick":
            self.distance_x_y, self.distance_z, gr_z = self._get_distance_to_goal()

            if self.distance_x_y < 0.008 and self.distance_z < 0.0006 and not self._attempted_grasp:
                finger_angle = 0.4
                while finger_angle > 0:
                    grasp_action = [0, 0, 0, 0, 0, -pi, 0, finger_angle]
                    self._kuka.applyAction(grasp_action)
                    p.stepSimulation()
                    finger_angle -= 0.4 / 100.
                    if finger_angle < 0:
                        finger_angle = 0

                # Move the hand up
                for _ in range(2):
                    grasp_action = [0, 0, 0.1, 0, 0, -pi, 0, finger_angle]
                    self._kuka.applyAction(grasp_action)
                    for _ in range(2*self._actionRepeat):
                        p.stepSimulation()
                    if self._renders:
                        time.sleep(self._timeStep)

                self._attempted_grasp = True

        elif self._operation == "place":
            self.distance_x_y, self.distance_z = self._get_distance_to_goal()

        elif self._operation == "move":
            self.distance = self._get_distance_to_goal()

        observation = self._get_observation(blocksInObservation=self._blocksInObservation, is_sensing=self._sensing)
        reward = self._reward()
        done = self._termination()

        if self._operation == "move_pick":
            debug = {
                'goal_id': self._goal,
                'distance_x_y': self.distance_x_y,
                'distance_z': self.distance_z,
                'operation': self._operation,
                'disturbance': self.get_disturbance()
            }
        elif self._operation == "pick":
            debug = {
                'goal_id': self._goal,
                'distance_x_y': self.distance_x_y,
                'distance_z': self.distance_z,
                'operation': self._operation
            }
        elif self._operation == 'place':
            debug = {
                'goal_id': self._goal,
                'distance_x_y': self.distance_x_y,
                'distance_z': abs(self.distance_z - 0.0075),
                'operation': self._operation,
                'disturbance': self.get_disturbance()
            }
        elif self._operation == 'move':
            debug = {
                'goal_id': self._goal[0],
                'distance': self.distance,
                'operation': self._operation
            }
        else:
            print(self._operation)
            raise NotImplementedError

        return observation, reward, done, debug

    def _reward(self):
        """Calculates the reward for the episode.
        :return: float
        """
        if self._operation == "move_pick":
            return self._reward_move_pick()
        elif self._operation == "move":
            return self._reward_move()
        elif self._operation == "push":
            return self._reward_push()
        elif self._operation == "place":
            return self._reward_place()
        elif self._operation == "pick":
            return self._reward_pick()
        else:
            print('operation:', self._operation)
            raise NotImplementedError

    def _reward_move_pick(self):
        """Dense reward function for picking
        :return: float
        """

        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, is_sensing=False)

        # Get the goal block's coordinates
        x, y, z, *rest = block_pos[0]

        # to prevent the surrounding blocks from moving
        #if self.prev_st_bl is None:
        #    self.prev_st_bl = np.array(block_pos[1:][:])
        #    block_norm = 0.0
        #else:
        #    a = np.array(block_pos[1:][:]) - self.prev_st_bl
        #    self.prev_st_bl = np.array(block_pos[1:][:])
        #    block_norm = inner1d(a, a)[0]

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])

        #print("Z table:", z)
        # The distance to the goal block plus negative reward for an every step

        block_norm = self.get_disturbance()
        #print(100 * block_norm)

        if grip_pos[2] < 0.0:
            return -1

        if self._attempted_grasp:
            # If the block is above the ground, provide extra reward
            #print("Z tried:", z)
            if z > 0.1:
                #print("Z + 50:", z)
                return 50.0 - 100/36 * block_norm
            return -1.0
        else:
            return - 10*self.distance_x_y - 10*abs(self.distance_z - 0.0345) - action_norm - 100/36 * block_norm

    def _reward_pick(self):

        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True)

        # Get the goal block's coordinates
        x, y, z, *rest= block_pos[0]

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])

        #print("Z table:", z)
        # The distance to the goal block plus negative reward for an every step

        #if grip_pos[2] < 0.0:
        #    return -1

        if self._attempted_grasp:
            # If the block is above the ground, provide extra reward
            #print("Z tried:", z)
            if z > 0.1:
                #print("Z + 50:", z)
                return 50.0
            return -1.0
        else:
            return - 10*self.distance_x_y - 10*self.distance_z - action_norm

    def _reward_push(self):
        """
        Reward function for pushing
        :return: float
        """
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True)

        # Get the goal block's coordinates
        x, y, z, _, _, _ = block_pos[self._goal - 2]

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])
        # a hack to be fixed in future
        action_fingers = (0.0 - self.action[7]) ** 2 + (0.0 - self.action[4]) ** 2 +\
                         (-pi - self.action[5]) ** 2 + (0.0 - self.action[6]) ** 2

        #print("DISTANCE", self.distance, "BL BL DST", self.bl_bl_distance)
        '''
        if self.distance1 < 0.01:
            if not self._isInProximity:
                self._isInProximity = True
                return 50.0
            else:
                if self.bl_bl_distance < 0.01:
                    self._attempted_grasp = True
                    return 50
                else:
                    return 1 - self.bl_bl_distance / self._bl_bl_dist_origin - action_norm - action_fingers
        elif self.distance1 > 0.01 and self._isInProximity:
            return -5
        else:
            return - 10 * self.distance1 - action_norm - action_fingers
        '''
        if self.bl_bl_distance < 0.01:
            self._attempted_grasp = True
            return 50
        else:
            return - self.bl_bl_distance - action_norm - action_fingers

    def _reward_place(self):
        """
        Reward function for placing
        :return: float
        """
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, is_sensing=False)

        # Get the goal block's coordinates
        #x, y, z, _, _, _ = block_pos[1]

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])

        block_norm = self.get_disturbance()
        #print(100 * block_norm)

        if self._one_more:
            self._done = True
            if block_pos[1][2] - block_pos[0][2] > 0:
                return 50.0 - 100/16 * block_norm
            else:
                return -1.0

        if self.distance_x_y < 0.001 and 0.005 <= self.distance_z < 0.01:
            self._one_more = True
            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(self._actionRepeat):
                p.stepSimulation()
            self._kuka.applyAction([0, 0, 0.1, 0, 0, -pi, 0, 0.4])
            for _ in range(self._actionRepeat):
                p.stepSimulation()

        return -10*self.distance_x_y - 10*abs(self.distance_z - 0.0075) - action_norm - 100/16 * block_norm

    def _reward_move(self):
        """Dense reward function for picking
        :return: float
        """

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])

        if self.distance < 0.001:
            self._done = True
            return 50.0
        else:
            return -10*self.distance - action_norm

    def _choose_block(self):
        """
        Choose a random block ID
        :return: the block's ID (int)
        """
        #print(self._objectUids)
        id_ = 3 # random.choice(self._objectUids)

        if self._isTest >= 0:
            # change the colour of the goal block
            p.changeVisualShape(id_, -1, rgbaColor=[0, 0.1, 1, 1])
        #p.changeVisualShape(4, -1, rgbaColor=[1, 0.1, 0, 1])

        return id_

    def _get_goal_coordinates(self):
        from random import random
        id_ = self._numObjects + 2
        blockPos, blockOrn = p.getBasePositionAndOrientation(id_)

        if self._operation == 'move':
            blockPos_z = blockPos[2] + (0.3 + random()/6)
            return [[blockPos[0], blockPos[1], blockPos_z], list(blockOrn)]
        # elif self._operation == 'place':
            # blockPos_z = blockPos[2] +

        return [[blockPos[0], blockPos[1], blockPos[2]], list(blockOrn)]

    def _get_distance_to_goal(self):
        """
        To get the distance from the effector to the goal
        :return: list of floats
        """
        if self._operation == "move_pick":
            return self._get_distance_pick()

        elif self._operation == "pick":
            return self._get_distance_pick()

        elif self._operation == "move":
            return self._get_distance_move()

        elif self._operation == "push":
            raise NotImplementedError

        elif self._operation == "place":
            return self._get_distance_place()

        else:
            raise NotImplementedError

    def _get_distance_pick(self):
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True)

        # Get the goal block's coordinates
        x, y, z, *rest = block_pos[0]

        # Distance: gripper - block
        gr_bl_distance_x_y = (x - grip_pos[0]) ** 2 + (y - grip_pos[1]) ** 2
        gr_bl_distance_z = (z - grip_pos[2]) ** 2

        return gr_bl_distance_x_y, gr_bl_distance_z, grip_pos[2]

    def _get_distance_move(self):
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True)

        # Get the goal coordinates
        x, y, z, *rest = block_pos[0]

        # Distance: gripper - block
        gr_gl_distance = (x - grip_pos[0]) ** 2 + (y - grip_pos[1]) ** 2 + (z - grip_pos[2]) ** 2

        return gr_gl_distance

    def _get_distance_place(self):
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True)

        # Get the goal block's coordinates
        x, y, z, *rest = block_pos[0]

        # Distance: gripper - block
        gr_bl_distance_x_y = (x - grip_pos[0]) ** 2 + (y - grip_pos[1]) ** 2
        gr_bl_distance_z = (z - grip_pos[2]) ** 2

        return gr_bl_distance_x_y, gr_bl_distance_z

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        if self._operation == "move_pick" or self._operation == "pick":
            return self._attempted_grasp or self._env_step >= self._maxSteps
        else:
            return self._done or self._env_step >= self._maxSteps

    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
          num_objects:
            Number of graspable objects.

        Returns:
          A list of urdf filenames.
        """
        if test:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[^0]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects),
                                            num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        reset = _reset

        step = _step

    def set_goal(self, goal, operation):
        self._goal = goal
        self._operation = operation
        self._attempted_grasp = False
        self._done = False
        self._env_step = 0

    def get_disturbance(self):
        """
        Get the distance representing how far were the surrounding blocks moved
        :return:
        """
        import itertools
        last_step = self._get_observation(inMatrixForm=True, is_sensing=False)
        a = np.array(last_step[2:]) - np.array(self.initial_state[2:])
        b = list(itertools.chain(*a))

        return np.sqrt(inner1d(b, b))
