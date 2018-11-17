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
                 maxSteps=20,  # <---- was 20?
                 dv=0.06,
                 removeHeightHack=True,
                 blockRandom=0.3,
                 cameraRandom=0,
                 width=48,
                 height=48,
                 numObjects=3,
                 isTest=0,
                 isSparseReward=False,
                 operation="put"):
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
          operation: a string: pick, push, put
        """

        self._isDiscrete = isDiscrete
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
        self._removeHeightHack = removeHeightHack
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._isTest = isTest
        self._isSparseReward = isSparseReward

        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        else:
            self.cid = p.connect(p.DIRECT)

        self._seed()

        if self._isDiscrete:
            if self._removeHeightHack:
                self.action_space = spaces.Discrete(9)
            else:
                self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # dx, dy, da
            if self._removeHeightHack:
                self.action_space = spaces.Box(low=-1,
                                               high=1,
                                               shape=(4,),
                                               dtype=np.float32)  # dx, dy, dz, da, Euler: Al, Bt, Gm  7 -> 4

        self.observation_space = spaces.Box(low=-100,
                                            high=100,
                                            shape=(7 + 6 + 6 * self._numObjects,),
                                            dtype=np.float32)

        self.viewer = None

        # how many times the environment is repeated
        #self._num_env_rep = 0 # TODO: delete

        #self._isInProximity = False
        #self._bl_bl_dist_origin = None
        self._operation = operation

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

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.700000, 0.000000, 0.000000,
                   0.0, 1.0)

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # Choose the objects in the bin.
        # urdfList = self._get_random_object(
        #    self._numObjects, self._isTest)
        # self._objectUids = self._randomly_place_objects(urdfList)

        # Generate the # of blocks
        self._objectUids = self._randomly_place_objects(self._numObjects)

        # randomly choose a block to be a goal
        self._goal = self._choose_block()

        # set observations
        observation = self._get_observation(isGripperIndex=True)  # FIXME: self._observation was changed to ...

        from math import pi

        if self._operation == "push":
            # move the effector in the position next to the block
            # y = k * x + b
            k = (observation[20] - observation[14]) / (observation[19] - observation[13])
            b = observation[14] - k * observation[13]

            self._kuka.endEffectorPos[0] = observation[13] - 0.1
            self._kuka.endEffectorPos[1] = k * (observation[13] - 0.1) + b
            self._kuka.endEffectorPos[2] = observation[15] + 0.251
            self._kuka.endEffectorAngle = 1.5

            for _ in range(self._actionRepeat):
                p.stepSimulation()

        elif self._operation == "put":
            # move th effector in the position above the block
            self._kuka.endEffectorPos[0] = observation[13]
            self._kuka.endEffectorPos[1] = observation[14] - 0.01
            self._kuka.endEffectorPos[2] = observation[15] + 0.27
            self._kuka.endEffectorAngle = observation[16]

            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(5*self._actionRepeat):
                p.stepSimulation()

            self._kuka.endEffectorPos[2] = observation[15] + 0.251
            self._kuka.applyAction([0, 0, 0, 0, 0, -pi, 0, 0.4])
            for _ in range(self._actionRepeat):
                p.stepSimulation()

        if self._operation == "put":
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
                grasp_action = [0, 0, 0.1, 0, 0, -pi, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)


        #self._num_env_rep += 1 TODO: delete

        return np.array(observation)  # FIXME: ditto

    def _randomly_place_objects(self, urdfList):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """
        '''
        # Randomize positions of each object urdf.
        objectUids = []
        for _ in range(urdfList):
            xpos = 0.5 + self._blockRandom * random.random()
            ypos = self._blockRandom * (random.random() - .5)
            angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, "cube_small.urdf")  # urdf_name
            uid = p.loadURDF(urdf_path, [xpos, ypos, .15],
                             [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(500):
                p.stepSimulation()
        '''
        # Randomize positions of each object urdf.
        objectUids = []
        i = 0
        for _ in range(urdfList):

            if self._isTest == 0:

                if i != 1:
                    xpos = 0.4 + self._blockRandom * random.random()
                    ypos = self._blockRandom * (random.random() - .5)
                    angle = np.pi / 2
                    orn = p.getQuaternionFromEuler([0, 0, angle])

                elif i == 1:

                    xpos = xpos + 0.22 + self._blockRandom * random.random()
                    ypos = self._blockRandom * (random.random() - .5)
                    angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                    orn = p.getQuaternionFromEuler([0, 0, angle])

            elif self._isTest == 1:

                if i == 0:
                    xpos = 0.5
                    ypos = 0.1
                    angle = np.pi / 2
                    orn = p.getQuaternionFromEuler([0, 0, angle])

                elif i == 1:
                    xpos = xpos + 0.3
                    ypos = -0.1
                    angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                    orn = p.getQuaternionFromEuler([0, 0, angle])

                elif i == 2:
                    xpos = 0.6
                    ypos = -0.2
                    angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                    orn = p.getQuaternionFromEuler([0, 0, angle])

            #angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
            #orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, "cube_small.urdf")  # urdf_name
            uid = p.loadURDF(urdf_path, [xpos, ypos, .15],
                             [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(500):
                p.stepSimulation()

            i += 1
        return objectUids

    def _get_observation(self, inMatrixForm=False, isGripperIndex=True):
        """Return an observation array:
            [ gripper's in the world frame X, Y, Z, Al, Bt, Gm, Blocks' in the gripper's frame X, Y, Z, Euler X, Y, Z]

            if inMatrixForm is True then
            [ [gripper's in the world frame X, Y, Z, Al, Bt, Gm], [Blocks' in the gripper's frame X, Y, Z, Euler X, Y, Z]]
        """
        # get the gripper's world position and orientation

        # Just to test the difference
        if isGripperIndex:
            # The coorditates of the gripper and fingers (X, Y, Z)
            gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
            fingerState_l = p.getLinkState(self._kuka.kukaUid, 10)[0]
            fingerState_r = p.getLinkState(self._kuka.kukaUid, 13)[0]

            #gripperPos = gripperState[0]

            gripperPos = np.array(gripperState[0] + np.array([0.00028128,  0.02405984, -0.19820549]))
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
            to_add.extend(list(fingerState_l + fingerState_l + gripperEul))
            observation.append(to_add)
            observation.append(self._goal - 3)
            #blockPos1, _ = p.getBasePositionAndOrientation(self._goal)
            #observation.append(list(blockPos1))
        else:
            observation.extend(list(gripperPos))
            observation.extend(list(fingerState_l + fingerState_l + gripperEul))
            observation.append(self._goal - 3)
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
            #print(list(blockInGripperPosXYZEul))
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

        from math import pi

        dv = self._dv  # velocity per physics step.
        if self._isDiscrete:
            # Static type assertion for integers.
            assert isinstance(action, int)
            if self._removeHeightHack:
                raise NotImplementedError
                dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
                da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
                action = np.array([dx, dy, dz, da])
                action = np.append(action, action[4:])
                action = np.append(action, 0.3)
            else:
                raise NotImplementedError
                dx = [0, -dv, dv, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0][action]
                dz = -dv
                da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
                action = np.array([dx, dy, dz, da])
                action = np.append(action, action[4:])
                action = np.append(action, 0.3)
        else:
            if self._removeHeightHack:
                action = np.array([dv, dv, dv, 0.25]) * action  # [dx, dy, dz, da]
                self.action = np.append(action, np.array([0, -pi, 0, 0.0]))
                #action = np.array([dv, dv, dv, 0.25, 2*pi, 2*pi, 2*pi]) * action  # [dx, dy, dz, da, Euler]
                #action = np.append(action, 0.3)  # [finger angle]
            else:
                raise NotImplementedError
                dx = dv * action[0]
                dy = dv * action[1]
                dz = -dv
                da = 0.25 * action[2]
                action = np.array([dx, dy, dz, da])
                action = np.append(action, action[4:])
                self.action = np.append(action, 0.3)

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


        self.distance1, self.distance2, self.bl_bl_distance = self._get_distance_to_goal()


        if self._operation == "pick":
            # Hardcoded grasping
            if self.distance1 < 0.005 and not self._attempted_grasp:
                from math import pi
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
                    grasp_action = [0, 0, 0.1, 0, 0, -pi, 0, finger_angle]
                    self._kuka.applyAction(grasp_action)
                    p.stepSimulation()
                    if self._renders:
                        time.sleep(self._timeStep)

                self._attempted_grasp = True  # TODO: delete attempted_grasp

        observation = self._get_observation(isGripperIndex=True)
        reward = self._reward()
        done = self._termination()
        #print("_________INTERNAL REWARD________", reward)

        debug = {
            'goal_id': self._goal,
            'distance1': self.distance1,
            'distance2': self.distance2
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
        elif self._operation == "put":
            return self._reward_put()

    def _reward_pick(self):
        """Dense reward function for picking
        :return: float
        """
        from numpy.core.umath_tests import inner1d

        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

        # Get the goal block's coordinates
        x, y, z, _, _, _ = block_pos[self._goal - 2]

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])
        # a hack to be fixed in future
        action_fingers = abs(0.3 - self.action[7])
        #print("DISTANCE", self.distance, "NORMS ACTION", action_norm, "FINGERS NORM", action_fingers, "FINGERS", self.action[7])

        # One over distance reward
        #reward = max(reward, 0.01 / (0.25 + d))

        #print("Z table:", z)
        # The distance to the goal block plus negative reward for an every step
        if self._attempted_grasp:
            # If the block is above the ground, provide extra reward
            #print("Z tried:", z)
            if z > 0.05:
                #print("Z + 50:", z)
                return 50.0 + z * 10.0
            return -1.0
        else:
            return - self.distance - action_norm - action_fingers
            #print("Delta d: {}, d: {}, ".format(self.pr_step_distance - d, d))


    def _reward_push(self):
        """
        Reward function for pushing
        :return: float
        """
        from numpy.core.umath_tests import inner1d
        from math import pi

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

    def _reward_put(self):
        """
        Reward function for putting
        :return: float
        """
        from numpy.core.umath_tests import inner1d
        from math import pi

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
            self._done = True
            return 50
        else:
            return - self.bl_bl_distance - self.distance1 - action_norm - action_fingers


    def _choose_block(self):
        """
        Choose a random block ID
        :return: the block's ID (int)
        """
        '''
        if not self._isTest:
            # to train to pick one block at a time
            if self._num_env_rep < 18000:
                id_ = 3
            elif self._num_env_rep < 36000:
                id_ = 4
            elif self._num_env_rep < 54000:
                id_ = 5
            else:
                # choose randomly a goal block
                id_ = random.choice(self._objectUids)
        else:
            id_ = 3
        '''
        id_ = 3 #random.choice(self._objectUids)
        # change the colour of the goal block
        p.changeVisualShape(id_, -1, rgbaColor=[0, 0.1, 1, 1])
        p.changeVisualShape(4, -1, rgbaColor=[1, 0.1, 0, 1])

        return id_

    def _get_distance_to_goal(self):
        """
        To get the distance from the effector to the goal block
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

        # Distance
        bl_bl_distance = (x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2

        return gr_bl1_distance, gr_bl2_distance, bl_bl_distance

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
