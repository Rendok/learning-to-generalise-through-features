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
from examples.pybullet.utils.pybullet_tools.utils import load_model, SINK_URDF, set_pose, Pose, Point, stable_z
from numpy.core.umath_tests import inner1d


class KukaMultiBlocksEnv(KukaGymEnv):
    """Class for Kuka environment with multi blocks.

    It generates the specified number of blocks
    """

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=80,  # <---- was 80?
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=40,  # <---- was 20?
                 dv=0.06,
                 removeHeightHack=True,
                 blockRandom=0.3,
                 cameraRandom=0,
                 width=48,
                 height=48,
                 numObjects=3,
                 isTest=0,
                 isSparseReward=False,
                 operation="place",
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
          operation: a string: pick, push, place
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

        #if self._operation == "pick":
        self.observation_space = spaces.Box(low=-100,
                                            high=100,
                                            shape=(7 + 8 + 6 * self._numObjects,),
                                            dtype=np.float32)
        #elif self._operation == "place":
        #    self.observation_space = spaces.Box(low=-100,
        #                                        high=100,
        #                                        shape=(7 + 8 + 6 * self._numObjects,),
        #                                        dtype=np.float32)

        self.viewer = None

        #self._isInProximity = False
        #self._bl_bl_dist_origin = None

    def _reset(self):
        """Environment reset called at the beginning of an episode.
        """

        self._attempted_grasp = False  # TODO delete
        self._done = False
        self._env_step = 0
        self.terminated = 0

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

        if self._isTest == -1:
            block1 = p.loadURDF(os.path.join(self._urdfRoot, "cube_small.urdf"))
            block2 = p.loadURDF(os.path.join(self._urdfRoot, "cube_small.urdf"))
            set_pose(block1, Pose(Point(x=0.5, y=0.02, z=stable_z(block1, table))))
            set_pose(block2, Pose(Point(x=0.75, y=-0.1, z=stable_z(block1, table))))
            self._objectUids = [block1, block2]
        else:
            # Generate the # of blocks
            self._objectUids = self._randomly_place_objects(self._numObjects)

        sink = load_model(SINK_URDF)
        set_pose(sink, Pose(Point(x=1.0, z=stable_z(sink, table))))

        for _ in range(500):
            p.stepSimulation()

        if self._operation == "pick":
            # randomly choose a block to be a goal
            self._goal = self._choose_block()
        elif self._operation == "place":
            self._goal = self._get_goal_coordinates()
        else:
            raise TypeError

        # set observations
        observation = self._get_observation(isGripperIndex=True)  # FIXME: self._observation was changed to ...

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

        elif self._operation == "place":
            # move th effector in the position above the block
            self._kuka.endEffectorPos[0] = observation[15]
            self._kuka.endEffectorPos[1] = observation[16] - 0.01
            self._kuka.endEffectorPos[2] = observation[17] + 0.27
            self._kuka.endEffectorAngle = observation[18]

            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(6*self._actionRepeat):
                p.stepSimulation()

            self._kuka.endEffectorPos[2] = observation[17] + 0.251
            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(self._actionRepeat):
                p.stepSimulation()

        if self._operation == "place":
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
            for _ in range(1):
                grasp_action = [0, 0, 0.4, 0, 0, -pi, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                for _ in range(2*self._actionRepeat):
                    p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)

        return np.array(observation)

    def _randomly_place_objects(self, urdfList):
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
                xpos = 0.2 + self._blockRandom * random.random()
                ypos = self._blockRandom * (random.random() - .5) / 2.0
                angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                orn = p.getQuaternionFromEuler([0, 0, angle])
                urdf_path = os.path.join(self._urdfRoot, "cube_small.urdf")  # urdf_name
                uid = p.loadURDF(urdf_path, [xpos, ypos, .15],
                                 [orn[0], orn[1], orn[2], orn[3]])
                objectUids.append(uid)
                # Let each object fall to the tray individual, to prevent object
                # intersection.
                #for _ in range(500):
                #    p.stepSimulation()

        else:
            for i in range(urdfList):
                if self._isTest == 1:

                    if i != 1:
                        xpos = 0.4 + self._blockRandom * random.random()
                        ypos = self._blockRandom * (random.random() - .5)
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

                urdf_path = os.path.join(self._urdfRoot, "cube_small.urdf")  # urdf_name
                uid = p.loadURDF(urdf_path, [xpos, ypos, .15],
                                 [orn[0], orn[1], orn[2], orn[3]])
                objectUids.append(uid)
                # Let each object fall to the tray individual, to prevent object
                # intersection.
                #for _ in range(500):
                #    p.stepSimulation()

        return objectUids

    def _get_observation(self, inMatrixForm=False, isGripperIndex=True):
        """Return an observation array:
            if inMatrixForm is True then as a nested list:
            [ [gripper in the world frame X, Y, Z, fingers X, Y, Z, orientation Al, Bt, Gm],
            goal block number,
            [blocks' in the gripper's frame X, Y, Z, Euler X, Y, Z] ]

            otherwise as a list
        """
        # get the gripper's world position and orientation

        # Just to test the difference
        if isGripperIndex:
            # The coordinates of the gripper and fingers (X, Y, Z)
            gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
            fingerState_l = p.getLinkState(self._kuka.kukaUid, 10)[0]
            fingerState_r = p.getLinkState(self._kuka.kukaUid, 13)[0]

            #gripperPos = gripperState[0]

            #print((0.2385243302547786 - 0.4359219376500988 + 0.2369296963620491 - 0.4359219376500988)/2.0)

            gripperPos = np.array(gripperState[0] + np.array([0.0,  0.02399222398656322, -0.20819492434168495]))  # [0.00028128,  0.02405984, -0.19820549]
            gripperOrn = gripperState[1]  # Quaternion
            gripperEul = p.getEulerFromQuaternion(gripperOrn)  # Euler: (Al, Bt, Gm)

            #gripperState = p.getLinkState(self._kuka.kukaUid, 10)
            #gripperPos_l = gripperState[0]  # (X, Y, Z)

            #gripperState = p.getLinkState(self._kuka.kukaUid, 13)
            #gripperPos_r = gripperState[0]  # (X, Y, Z)

            #gripperPos = (np.array(gripperPos_l) + np.array(gripperPos_r)) / 2  # (X, Y, Z)

        else: # TODO: delete
            gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
            raise NotImplementedError

        #print("midpoint: {}, base: {}".format(gripperPos, gripperPos_base))

        observation = []
        if inMatrixForm:
            to_add = list(gripperPos)
            to_add.extend(list(fingerState_l + fingerState_r + gripperEul))
            observation.append(to_add)
            if type(self._goal) == int:
                #observation.append([self._goal - 3, 0, 0])
                bl_pos, _ = p.getBasePositionAndOrientation(self._goal)
                observation.append(list(bl_pos))

            elif type(self._goal).__module__ == np.__name__ or type(self._goal) == list:
                observation.append(list(self._goal))

            else:
                raise TypeError
        else:
            observation.extend(list(gripperPos))
            observation.extend(list(fingerState_l + fingerState_l + gripperEul))
            if type(self._goal) == int:
                #observation.extend([self._goal - 3, 0, 0])
                bl_pos, _ = p.getBasePositionAndOrientation(self._goal)
                observation.extend(list(bl_pos))

            elif type(self._goal).__module__ == np.__name__ or type(self._goal) == list:
                observation.extend(list(self._goal))

            else:
                print(type(self._goal), self._goal)
                raise TypeError
            #blockPos1, _ = p.getBasePositionAndOrientation(self._goal)
            #observation.extend(list(blockPos1))

        #invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
        #print("gripper pos {}, effector pos {}".format(gripperPos, grps[0]))

        # gripperMat = p.getMatrixFromQuaternion(gripperOrn)
        # dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
        # dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
        # dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

        for id_ in self._objectUids:
            # get the block's position (X, Y, Z) and orientation (Quaternion)
            blockPos, blockOrn = p.getBasePositionAndOrientation(id_)
            #print("blockPos: {}, gr - bl {}".format(blockPos, [gripperPos[i] - blockPos[i] for i in range(3)]))

            '''blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn, blockPos,
                                                                        blockOrn)
            blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
            #print("projectedBlockPos2D:", [blockPosInGripper[0], blockPosInGripper[1], blockPosInGripper[2]])
            # print("blockEulerInGripper:", blockEulerInGripper)

            # we return the relative x,y position and euler angle of block in gripper space
            #blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

            # we return the relative x,y,z positions and euler angles of a block in a gripper space
            blockInGripperPosXYZEul = [blockPosInGripper[i] for i in range(3)]
            blockInGripperPosXYZEul.extend([blockEulerInGripper[i] for i in range(3)])
            '''

            blockPosXYZEul = [blockPos[i] for i in range(3)]
            blockPosXYZEul.extend([blockOrn[i] for i in range(3)])

            # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
            # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
            # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

            if inMatrixForm:
                observation.append(list(blockPosXYZEul))
            else:
                observation.extend(list(blockPosXYZEul))
            #print('block', blockPosXYZEul[0:3])
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
        if self._operation == 'pick':
            self.action = np.append(action, np.array([0, -pi, 0, 0.4]))
        elif self._operation == 'place':
            self.action = np.append(action, np.array([0, -pi, 0, 0.0]))
        else:
            raise NotImplementedError
        #action = np.array([dv, dv, dv, 0.25, 2*pi, 2*pi, 2*pi]) * action  # [dx, dy, dz, da, Euler]
        #action = np.append(action, 0.3)  # [finger angle]

        return self._step_continuous(self.action) # [dx, dy, dz, da, Al, Bt, Gm, Fn_angle]

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

        if self._operation == "pick":
            # contains only one distance in that case
            self.distance_x_y, self.distance_z = self._get_distance_to_goal()
            # Hardcoded grasping
            if self.distance_x_y < 0.008 and 0.034 <= self.distance_z < 0.035 and not self._attempted_grasp:
                finger_angle = 0.3

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
                    #if self._renders:
                    #    time.sleep(self._timeStep)
                    finger_angle -= 0.3 / 100.
                    if finger_angle < 0:
                        finger_angle = 0

                # Move the hand up
                for _ in range(2):  # range(1) for place
                    grasp_action = [0, 0, 0.1, 0, 0, -pi, 0, finger_angle]
                    self._kuka.applyAction(grasp_action)
                    for _ in range(2*self._actionRepeat):
                        p.stepSimulation()
                    if self._renders:
                        time.sleep(self._timeStep)

                self._attempted_grasp = True  # TODO: delete attempted_grasp

        elif self._operation == "place":
            # contains only three distances so far
            #self.distance1, self.distance2, self.bl_bl_distance = self._get_distance_to_goal()
            self.distance1 = self._get_distance_to_goal()

        observation = self._get_observation(isGripperIndex=True)
        reward = self._reward()
        done = self._termination()
        #print("_________INTERNAL REWARD________", reward)

        if self._operation == "pick":
            debug = {
                'goal_id': self._goal,
                'distance_x_y': self.distance_x_y,
                'distance_z': self.distance_z,
                'operation': self._operation
            }
        elif self._operation =='place':
            debug = {
                'goal_id': self._goal,
                'distance1': self.distance1,
                'operation': self._operation
            }
        return observation, reward, done, debug

    def _reward(self):
        """Calculates the reward for the episode.
        :return: float
        """
        if self._operation == "pick":
            return self._reward_pick()
        elif self._operation == "push":
            return self._reward_push()
        elif self._operation == "place":
            return self._reward_place()

    def _reward_pick(self):
        """Dense reward function for picking
        :return: float
        """


        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

        #print("grip pos", grip_pos)

        # Get the goal block's coordinates
        x, y, z, _, _, _ = block_pos[self._goal - 2]

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])
        # a hack to be fixed in future
        #action_fingers = abs(0.4 - self.action[7])
        #print("DISTANCE", self.distance_x_y, abs(self.distance_z - 0.0345), "NORMS ACTION", action_norm)

        # One over distance reward
        #reward = max(reward, 0.01 / (0.25 + d))

        #print("Z table:", z)
        # The distance to the goal block plus negative reward for an every step
        if self._attempted_grasp:
            # If the block is above the ground, provide extra reward
            #print("Z tried:", z)
            if z > 0.1:
                #print("Z + 50:", z)
                return 50.0 + z * 10.0
            return -1.0
        else:
            return - 10*self.distance_x_y - 10*abs(self.distance_z - 0.0345) - action_norm
            #print("Delta d: {}, d: {}, ".format(self.pr_step_distance - d, d))


    def _reward_push(self):
        """
        Reward function for pushing
        :return: float
        """
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

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
        #grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

        # Get the goal block's coordinates
        #x, y, z, _, _, _ = block_pos[self._goal - 2]

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])
        # a hack to be fixed in future
        action_fingers = (0.0 - self.action[7]) ** 2 + (0.0 - self.action[4]) ** 2 +\
                         (-pi - self.action[5]) ** 2 + (0.0 - self.action[6]) ** 2

        #if self.bl_bl_distance[0] < 0.001 and self.bl_bl_distance[1] < 0.01:
        if self.distance1 < 0.01:
            self._done = True
            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(self._actionRepeat):
                p.stepSimulation()
            return 50
        else:
            return - self.distance1 - action_norm - action_fingers

    def _choose_block(self):
        """
        Choose a random block ID
        :return: the block's ID (int)
        """
        #print(self._objectUids)
        id_ = random.choice(self._objectUids)

        if self._isTest >= 0:
            # change the colour of the goal block
            p.changeVisualShape(id_, -1, rgbaColor=[0, 0.1, 1, 1])
        #p.changeVisualShape(4, -1, rgbaColor=[1, 0.1, 0, 1])

        return id_

    def _get_goal_coordinates(self):
        id_ = 4
        blockPos, blockOrn = p.getBasePositionAndOrientation(id_)
        #print(blockPos)

        return [blockPos[0], blockPos[1], blockPos[2] + 0.1]

    def _get_distance_to_goal(self):
        """
        To get the distance from the effector to the goal
        :return: float
        """
        if self._operation == "pick":
            return self._get_distance_pick()

        elif self._operation == "push":
            raise NotImplementedError

        elif self._operation == "place":
            return self._get_distance_place()

        else:
            raise NotImplementedError

    def _get_distance_to_goal_old(self):
        """
        To get the distance from the effector to the goal block

        Depreciated
        :return: float
        """

        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

        # Get the goal block's coordinates
        x, y, z, _, _, _ = block_pos[self._goal - 2]

        # Get the second block's coordinates
        x1, y1, z1, _, _, _ = block_pos[self._goal - 1]

        # Distance
        gr_bl1_distance = (x - grip_pos[0]) ** 2 + (y - grip_pos[1]) ** 2 + (z - grip_pos[2]) ** 2

        # Distance
        gr_bl2_distance = (x1 - grip_pos[0]) ** 2 + (y1 - grip_pos[1]) ** 2 + (z1 - grip_pos[2]) ** 2
        #gr_bl2_distance_x_y = (x1 - grip_pos[0]) ** 2 + (y1 - grip_pos[1]) ** 2
        #gr_bl2_distance_z = (z1 - grip_pos[2]) ** 2

        # Distance
        #bl_bl_distance = (x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2
        bl_bl_distance_x_y = (x1 - x) ** 2 + (y1 - y) ** 2
        bl_bl_distance_z = (z1 - z) ** 2

        return gr_bl1_distance, gr_bl2_distance, [bl_bl_distance_x_y, bl_bl_distance_z]

    def _get_distance_pick(self):
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

        # Get the goal block's coordinates
        x, y, z, _, _, _ = block_pos[self._goal - 2]

        # Distance: gripper - block
        gr_bl_distance_x_y = (x - grip_pos[0]) ** 2 + (y - grip_pos[1]) ** 2
        gr_bl_distance_z = (z - grip_pos[2]) ** 2

        return gr_bl_distance_x_y, gr_bl_distance_z

    def _get_distance_place(self):
        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

        # Get the goal coordinates
        x, y, z = block_pos[0]
        #x1, y1, z1, *rest = block_pos[1]

        # Distance
        gr_bl1_distance = (x - grip_pos[0]) ** 2 + (y - grip_pos[1]) ** 2 + (z - grip_pos[2]) ** 2
        #gr_bl2_distance = (x1 - grip_pos[0]) ** 2 + (y1 - grip_pos[1]) ** 2 + (z1 - grip_pos[2]) ** 2

        return gr_bl1_distance

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        if self._operation == "pick":
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
