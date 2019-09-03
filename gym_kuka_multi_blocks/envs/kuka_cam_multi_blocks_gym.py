from gym_kuka_multi_blocks.envs.kukaGymEnv import KukaGymEnv
import random
import os
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


class KukaCamMultiBlocksEnv(KukaGymEnv):
    """Class for Kuka environment with multi blocks.

    It generates the specified number of blocks
    """

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=80,
                 isEnableSelfCollision=True,
                 renders=False,
                 maxSteps=40,
                 dv=0.06,
                 blockRandom=0.5,
                 width=84,
                 height=84,
                 numObjects=3,
                 isTest=0,
                 operation="place"
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
        """

        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 135
        self._cam_pitch = -31
        self._dv = dv
        self._p = p
        self._blockRandom = blockRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._isTest = isTest
        self._operation = operation

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

        # pixels
        self.observation_space = spaces.Box(0, 255, [height, width, 9], dtype=np.uint8)

        self.viewer = None

        # Pre-compute the camera settings #

        # a view from z
        look = [0.4, 0.0, 0.54]  # [0.23, 0.2, 0.54]
        distance = 1.3  # 1.
        yaw = 180  # 245
        pitch = -90  # -56
        roll = 0

        self._view_matrix = np.array(p.computeViewMatrixFromYawPitchRoll(
                                            look, distance, yaw, pitch, roll, 2))

        # from y
        look = [0.4, 0.0, 0.2]
        distance = 2.0
        yaw = 180
        pitch = 0
        roll = 0

        self._view_matrix = np.append(self._view_matrix,
                                      p.computeViewMatrixFromYawPitchRoll(
                                                look, distance, yaw, pitch, roll, 2))

        # from x
        look = [0.4, 0.0, 0.2]
        distance = 2.0
        yaw = 90
        pitch = 0
        roll = 0

        self._view_matrix = np.append(self._view_matrix,
                                      p.computeViewMatrixFromYawPitchRoll(
                                                look, distance, yaw, pitch, roll, 2))

        self._view_matrix = self._view_matrix.reshape([3, 16]).T

        fov = 20.
        aspect = self._width / self._height
        near = 0.01
        far = 10

        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def _reset(self):
        """Environment reset called at the beginning of an episode.
        """

        # Set all the parameters
        self._env_step = 0
        self._done = False

        # Set the physics engine
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        # load a table
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        table = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.700000, 0.000000,
                           0.000000,
                           0.0, 1.0)

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # Generate the blocks
        self._objectUids = self._randomly_place_objects(self._numObjects, table)

        if self._isTest != 13:
            for _ in range(100):
                p.stepSimulation()

        # FIXME: a more comprehensive goal state
        self._goal = 0  # self._get_goal()

        # set observations
        self._observation = self._get_observation()

        # return observations
        return self._observation

    def _randomly_place_objects(self, urdfList, table):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []

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

                if self._numObjects < 2:
                    raise ValueError

                if i == 0:
                    xpos = 0.4
                    ypos = -0.1
                    zpos = 0.0
                    angle = np.pi / 2
                    orn = p.getQuaternionFromEuler([0, 0, angle])

                elif i > 0:

                    xpos = 0.4  # xpos + 0.1
                    ypos = ypos + 0.1
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

            # three blocks, L-shape
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
                    zpos = -0.05
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

                if not 3 <= self._numObjects <= 6:
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

                else:
                    xpos = xpos
                    ypos = ypos
                    zpos = zpos + 0.05
                    angle = np.pi / 2
                    orn = p.getQuaternionFromEuler([0, 0, angle])

            # for placing. Two blocks in a tower, one on a sphere.
            elif self._isTest == 13:
                from random import choice

                if not 2 <= self._numObjects:
                    raise ValueError

                if i == 0:
                    xpos, ypos, zpos, xpos0, ypos0, zpos0 = self._place_on_sphere(radius=0.2)
                    angle = np.pi / 2
                    orn = p.getQuaternionFromEuler([0, 0, angle])

                elif i == 1:
                    xpos = xpos0
                    ypos = ypos0
                    zpos = zpos0
                    angle = np.pi / 2
                    orn = p.getQuaternionFromEuler([0, 0, angle])

                else:
                    xpos = xpos
                    ypos = ypos
                    zpos = zpos + 0.05
                    angle = np.pi / 2
                    orn = p.getQuaternionFromEuler([0, 0, angle])

            urdf_path = os.path.join(self._urdfRoot, "cube_small.urdf")
            uid = p.loadURDF(urdf_path, [xpos, ypos, zpos],
                             [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)

        return objectUids

    def _get_observation(self):
        """Return the observation as an image.
        """
        import matplotlib.pyplot as plt
        np_img_arr = np.zeros((self._height, self._width, 9), dtype=np.uint8)

        for i in range(3):
            img_arr = p.getCameraImage(width=self._width,
                                       height=self._height,
                                       viewMatrix=self._view_matrix[:, i],
                                       projectionMatrix=self._proj_matrix)
            rgb = img_arr[2]
            rgb = np.reshape(rgb, (self._height, self._width, 4))
            np_img_arr[:, :, 3*i:3+3*i] = rgb[:, :, :-1]

        # plt.imshow(np_img_arr[:, :, 0:3])
        # plt.show()
        # plt.imshow(np_img_arr[:, :, 3:6])
        # plt.show()
        # plt.imshow(np_img_arr[:, :, 6:9])
        # plt.show()

        assert np_img_arr.shape == (self._width, self._height, 9)
        assert np_img_arr.dtype.char in np.typecodes['AllInteger']

        return np_img_arr

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

        # TODO: add finger's angle into actions
        # TODO: rewrite all of this
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
            # FIXME: here
            self.distance_x_y, self.distance_z, gr_z = 0, 0, 0  # self._get_distance_to_goal()
            # Hardcoded grasping
            if self.distance_x_y < 0.008 and 0.033 <= self.distance_z < 0.035 and gr_z > 0.01:
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

        elif self._operation == "place":
            # FIXME: here
            self.distance_x_y, self.distance_z = 0, 0  # self._get_distance_to_goal()

        elif self._operation == "move":
            # FIXME: here
            self.distance = 0  # self._get_distance_to_goal()

        observation = self._get_observation()
        reward = self._reward()
        done = self._termination()

        if self._operation == "move_pick":
            debug = {
                'goal_id': self._goal,
                'distance_x_y': self.distance_x_y,
                'distance_z': self.distance_z,
                'operation': self._operation,
                # 'disturbance': self.get_disturbance()
            }
        elif self._operation == 'place':
            debug = {
                'goal_id': self._goal,
                'distance_x_y': self.distance_x_y,
                'distance_z': abs(self.distance_z - 0.0075),
                'operation': self._operation,
                # 'disturbance': self.get_disturbance(),
                'num_blocks': self._numObjects,
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

        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """

        return 0

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        if self._operation == "move_pick":
            return self._done or self._env_step >= self._maxSteps
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

    def _get_distance_to_goal(self):
        """
        To get the distance from the effector to the goal
        :return: list of floats
        """
        pass  # do nothing