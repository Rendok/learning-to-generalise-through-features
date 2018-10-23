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
                 isTest=False,
                 isSparseReward=False):
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
          isTest: If true, use the test set of objects. If false, use the train
            set of objects.
          isSparseReward: If true, the reward function is sparse.
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
                                            shape=(7 + 6 * self._numObjects,),
                                            dtype=np.float32)

        self.viewer = None

        # the distance to the nearest block recoded in a previous step
        self.pr_step_distance = None

    def _reset(self):
        """Environment reset called at the beginning of an episode.
        """
        self._attempted_grasp = False  # TODO delete
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

        return np.array(observation)  # FIXME: ditto

    def _randomly_place_objects(self, urdfList):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """

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
            gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
            gripperPos = np.array(gripperState[0] + np.array([0.00028128,  0.02405984, -0.19820549]))
            gripperOrn = gripperState[1]  # Quaternion
            gripperEul = p.getEulerFromQuaternion(gripperOrn)  # Euler: (Al, Bt, Gm)

            #gripperState = p.getLinkState(self._kuka.kukaUid, 10)
            #gripperPos_l = gripperState[0]  # (X, Y, Z)

            #gripperState = p.getLinkState(self._kuka.kukaUid, 13)
            #gripperPos_r = gripperState[0]  # (X, Y, Z)

            #gripperPos = (np.array(gripperPos_l) + np.array(gripperPos_r)) / 2  # (X, Y, Z)

        else:
            gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
            raise ImportError
        #gripperPos = gripperState[0]  # (X, Y, Z)
        #gripperOrn = gripperState[1]  # Quaternion
        #gripperEul = p.getEulerFromQuaternion(gripperOrn)  # Euler: (Al, Bt, Gm)
        # print("gripperEul:", gripperEul)

        #print("midpoint: {}, base: {}".format(gripperPos, gripperPos_base))

        observation = []
        if inMatrixForm:
            to_add = list(gripperPos)
            to_add.extend(list(gripperEul))
            observation.append(to_add)
            observation.append(self._goal)
        else:
            observation.extend(list(gripperPos))
            observation.extend(list(gripperEul))
            observation.append(self._goal)

        invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
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
                dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
                da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
                action = np.array([dx, dy, dz, da])
                action = np.append(action, action[4:])
                action = np.append(action, 0.3)
            else:
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
                self.action = np.append(action, np.array([0, -pi, 0, 0.3]))
                #action = np.array([dv, dv, dv, 0.25, 2*pi, 2*pi, 2*pi]) * action  # [dx, dy, dz, da, Euler]
                #action = np.append(action, 0.3)  # [finger angle]
            else:
                dx = dv * action[0]
                dy = dv * action[1]
                dz = -dv
                da = 0.25 * action[2]
                action = np.array([dx, dy, dz, da])
                action = np.append(action, action[4:])
                self.action = np.append(action, 0.3)

        return self._step_continuous(self.action)

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

        '''# If we are close to the bin, attempt grasp.
        state = p.getLinkState(self._kuka.kukaUid,
                               self._kuka.kukaEndEffectorIndex)
        end_effector_pos = state[0]
        '''

        self.distance = self._get_distance_to_goal()

        # Hardcoded grasping
        #if end_effector_pos[2] <= 0.23:  # Z coordinate
        if self.distance < 0.005: #0.045:  # Z coordinate
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

            # Move the hand up TODO: delete
            for _ in range(250):
                grasp_action = [0, 0, 0.001, 0, 0, -pi, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                #finger_angle -= 0.3 / 100.
                #if finger_angle < 0:
                #    finger_angle = 0
        
            self._attempted_grasp = True  # TODO: delete attempted_grasp

        observation = self._get_observation(isGripperIndex=True)
        done = self._termination()
        reward = self._reward()
        #print("_________INTERNAL REWARD________", reward)

        debug = {
            'grasp_success': self._graspSuccess,
            'goal_id': self._goal,
            'distance': self.distance
        }
        return observation, reward, done, debug

    def _reward(self):
        """Calculates the reward for the episode.

        """
        if self._isSparseReward:
            return self._sparse_reward()
        else:
            return self._dense_reward()

    def _dense_reward(self):
        """Dense reward function
        :return: float
        """
        #from math import sqrt
        from numpy.core.umath_tests import inner1d

        self._graspSuccess = 0

        # Unpack the block's coordinate
        grip_pos, *block_pos = self._get_observation(inMatrixForm=True, isGripperIndex=True)

        # Get the goal block's coordinates
        x, y, z, _, _, _ = block_pos[self._goal - 2]

        #max_distance = 1.0

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])
        # a hack to be fixed in future
        action_fingers = abs(0.3 - self.action[7])
        #print("DISTANCE", self.distance, "NORMS ACTION", action_norm, "FINGERS NORM", action_fingers, "FINGERS", self.action[7])
        #print("DISTANCE", self.distance, "REWARD", 1 - self.distance / max_distance, "NORMS ACTION", action_norm, "ACTION", self.action[0:4])

        # One over distance reward
        #reward = max(reward, 0.01 / (0.25 + d))

        #print("Z table:", z)
        # The distance to the goal block plus negative reward for an every step
        if self._attempted_grasp:
            # If the block is above the ground, provide extra reward
            #print("Z tried:", z)
            if z > 0.05:
                self._graspSuccess += 1
                #print("Z + 50:", z)
                return 50.0 + z * 10.0
            return -1.0
        else:
            return - self.distance - action_norm - action_fingers
            #print("Delta d: {}, d: {}, ".format(self.pr_step_distance - d, d))


    def _choose_block(self):
        """
        Choose a random block ID
        :return: the block's ID (int)
        """
        #import random

        #random.seed()
        id_ = 3 #random.choice(self._objectUids)

        # change the colour of the goal block
        p.changeVisualShape(id_, -1, rgbaColor=[0, 0.1, 1, 1])

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

        # Distance
        distance = (x - grip_pos[0]) ** 2 + (y - grip_pos[1]) ** 2 + (z - grip_pos[2]) ** 2
        return distance

    def _sparse_reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """
        reward = 0
        self._graspSuccess = 0
        for uid in self._objectUids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            # If any block is above height, provide reward.
            if pos[2] > 0.2:
                self._graspSuccess += 1
                reward = 1
                break
        return reward

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        # return self._env_step >= self._maxSteps  # TODO: add new termination requirements
        return self._attempted_grasp or self._env_step >= self._maxSteps  # TODO: delete attempted_grasp

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
