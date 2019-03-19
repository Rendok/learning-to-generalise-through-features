from gym_kuka_multi_blocks.envs.kukaGymEnv import KukaGymEnv
import pybullet_data
import pybullet as p
from gym import spaces
import numpy as np
from . import kuka  # implementation of a kuka arm

import sys
sys.path.append("/Users/dgrebenyuk/Research/pddlstream")
#sys.path.insert(0, '/Users/dgrebenyuk/Research/pddlstream')
from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect, enable_gravity, simulate_for_duration,\
    step_simulation, reset_simulation

from pddlstream.utils import print_solution, read, INF, get_file_path, find_unique

from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, \
            Pose, \
            Point, set_default_camera, stable_z, \
            BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
            disconnect, DRAKE_IIWA_URDF, get_bodies, user_input, HideOutput, KUKA_IIWA_URDF, get_model_path

from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, get_movable_collision_test

from pddlstream.language.generator import from_gen_fn, from_fn, empty_gen

USE_SYNTHESIZERS = False

PLAN = None
COST = None
EVALUATIONS = None

class KukaHRLEnv(KukaGymEnv):
    """Class for Kuka environment with multi blocks.

    It generates the specified number of blocks
    """

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=80,
                 is_enable_self_collision=True,
                 renders=False,
                 max_steps=20,
                 dv=0.06,
                 block_random=0.3,
                 camera_random=0,
                 width=48,
                 height=48,
                 num_objects=3):
        """Initializes the KukaDiverseObjectEnv.

        Args:
          urdf_root: The diretory from which to load environment URDF's.
          action_repeat: The number of simulation steps to apply for each action.
          is_enable_self_collision: If true, enable self-collision.
          renders: If true, render the bullet GUI.
          max_steps: The maximum number of actions per episode.
          dv: The velocity along each dimension for each action.
          block_random: A float between 0 and 1 indicated block randomness. 0 is
            deterministic.
          camera_random: A float between 0 and 1 indicating camera placement
            randomness. 0 is deterministic.
          width: The image width.
          height: The observation image height.
          num_objects: The number of objects in the bin.
        """

        self._timeStep = 1. / 240.
        self._urdfRoot = urdf_root
        self._actionRepeat = action_repeat
        self._is_enable_self_collision = is_enable_self_collision
        # self._observation = []
        self._renders = renders
        self._maxSteps = max_steps
        self._cam_dist = 3.3
        self._cam_yaw = 150
        self._cam_pitch = -31
        self._dv = dv
        self._p = p
        self._block_random = block_random
        self._camera_random = camera_random
        self._width = width
        self._height = height
        self._num_objects = num_objects
        self.plan = None

        #if self._renders:
        #    self.cid = p.connect(p.SHARED_MEMORY)
        #    if (self.cid < 0):
        #        self.cid = p.connect(p.GUI)
        #    p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        #else:
        #    self.cid = p.connect(p.DIRECT)

        self.cid = connect(use_gui=self._renders)
        p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])

        self._seed()

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(4,),
                                       dtype=np.float32)  # dx, dy, dz, da, Euler: Al, Bt, Gm  7 -> 4

        self.observation_space = spaces.Box(low=-100,
                                            high=100,
                                            shape=(36,), #TODO: fix the size
                                            dtype=np.float32)

        self.viewer = None
        self._done = False
        self._rl_goal = None
        self._rl_action = None
        self._env_step = 0
        self.distance = 10.

        enable_gravity()

        # set the physics engine
        # p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)

        # We want to find a plan only once
        global PLAN
        global COST
        global EVALUATIONS

        if PLAN is None:
            PLAN, COST, EVALUATIONS = self.solve(load_world=self.load_world,
                                                 pddlstream_from_problem=self.pddlstream_from_problem,
                                                 teleport=True)
            self.plan = PLAN

            self.reset()

            #self.cid = connect(use_gui=False)
            #enable_gravity()
            #p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        else:
            self.plan = PLAN

            # to train an RL
            self.execute_rl_action(self.plan[0])
            #print("NOT SOLVED", self.plan[:1], ",...")

    def reset(self, render=False):
        """
        Environment reset called at the beginning of an episode.

        :return observation (np.array) (12 joint configurations)
        """
        self._done = False
        self._env_step = 0
        #self.cid = connect(use_gui=render)
        reset_simulation()


        # to train an RL
        self.execute_rl_action(self.plan[0])

        self.robot, self.body_names, self.movable_bodies = self.load_world()

        # simulate_for_duration(5)
        #for _ in range(500):
        #    step_simulation()

        observation = self.get_observation()

        return np.array(observation)

    def load_world(self):
        """
        Loads the environment initial state
        :return: robot, body_names, movable_bodies
        """
        import os

        from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
            get_stable_gen, get_ik_fn, get_free_motion_gen, \
            get_holding_motion_gen, get_movable_collision_test


        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        # set robot ID
        self.robot = 0

        #for i in range(13):
        #    print(p.getJointInfo(robot, i))

        #robot = load_model(os.path.join(self._urdfRoot,"kuka_iiwa/kuka_with_gripper2.sdf"))


        table = load_model(os.path.join(self._urdfRoot, "table/table.urdf"))
        sink = load_model(SINK_URDF)
        #stove = load_model(STOVE_URDF, pose=Pose(Point(x=+1.5)))
        block1 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block2 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)

        set_pose(block1, Pose(Point(x = 0.1, z=stable_z(block1, table))))
        set_pose(block2, Pose(Point(y = 0.1, z=stable_z(block1, table))))
        set_pose(self.robot, Pose(Point(x=-0.5, z=stable_z(self.robot, table))))
        set_pose(sink, Pose(Point(x=0.5, z=stable_z(sink, table))))

        # the pose of the last loaded object cannot be set, hence floor
        floor = load_model(os.path.join(self._urdfRoot, "plane.urdf"))


        #print(stable_z(cup, table))

        body_names = {
            sink: 'sink',
            #stove: 'stove',
            block1: 'block1',
            block2: 'block2',
            table: 'table',
            #cup: 'cup',
            floor: 'floor',
        }
        movable_bodies = [block1, block2]


        #set_default_camera()

        return self.robot, body_names, movable_bodies

    def pddlstream_from_problem(self, robot, movable=[], teleport=False, movable_collisions=False, grasp_name='top'):
        # assert (not are_colliding(tree, kin_cache))
        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}

        # initial state
        print('Robot:', robot)
        conf = BodyConf(robot, get_configuration(robot))
        # add a robot configuration
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        fixed = self.get_fixed(robot, movable)
        print('Movable:', movable)
        print('Fixed:', fixed)
        for body in movable:
            pose = BodyPose(body, get_pose(body))
            # add the poses of all movable objects
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]
            # add all placement surfaces with a sampled pose
            for surface in fixed:
                init += [('Stackable', body, surface)]
                if is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]

        for body in fixed:
            name = get_body_name(body)
            if 'sink' in name:
                init += [('Sink', body)]
            if 'stove' in name:
                init += [('Stove', body)]

        # goal
        body = movable[0]
        goal = ('and',
                ('AtConf', conf),  # move the arm to the initial configuration
                # ('Holding', body),
                ('On', movable[0], fixed[1]),
                ('On', movable[1], fixed[1]),
                ('Cleaned', movable[0]),
                ('Cleaned', movable[1]),
                #('Cooked', body),
                )

        stream_map = {
            'sample-pose': from_gen_fn(get_stable_gen(fixed)),
            'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
            'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
            'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
            'TrajCollision': get_movable_collision_test(),
        }

        if USE_SYNTHESIZERS:
            stream_map.update({
                'plan-free-motion': empty_gen(),
                'plan-holding-motion': empty_gen(),
            })

        return domain_pddl, constant_map, stream_pddl, stream_map, init, goal

    def solve(self, load_world, pddlstream_from_problem, viewer=False, display=True, simulate=False, teleport=False):

        import cProfile
        import pstats
        from pddlstream.language.synthesizer import StreamSynthesizer
        from pddlstream.algorithms.focused import solve_focused

        #robot, names, movable = self.robot, self.body_names, self.movable_bodies  # load_world()
        robot, names, movable = load_world()

        #saved_world = WorldSaver()
        # dump_world()

        pddlstream_problem = pddlstream_from_problem(robot, movable=movable, movable_collisions=True, teleport=teleport)
        _, _, _, stream_map, init, goal = pddlstream_problem
        synthesizers = [
            StreamSynthesizer('safe-free-motion', {'plan-free-motion': 1, 'trajcollision': 0},
                              from_fn(self.get_free_motion_synth(robot, movable, teleport))),
            StreamSynthesizer('safe-holding-motion', {'plan-holding-motion': 1, 'trajcollision': 0},
                              from_fn(self.get_holding_motion_synth(robot, movable, teleport))),
        ] if USE_SYNTHESIZERS else []

        print('Names:', names)
        print('Init:', init)
        print('Goal:', goal)
        print('Streams:', stream_map.keys())
        print('Synthesizers:', stream_map.keys())

        # initialize and measure performance
        pr = cProfile.Profile()
        pr.enable()
        solution = solve_focused(pddlstream_problem, synthesizers=synthesizers, max_cost=INF, visualize=True)
        print_solution(solution)
        # plan, cost, evaluations = solution
        pr.disable()
        # print stats
        pstats.Stats(pr).sort_stats('tottime').print_stats(10)

        #disconnect()

        return solution

    def execute(self, plan, viewer=False, display=True, simulate=False, teleport=False):

        if plan is None:
            return

        #if (not display) or (plan is None):
        #    disconnect()
        #    return

        #if viewer:
        #    # disconnect()
        #    connect(use_gui=True)
        #    load_world()
        #else:
        #    pass  # saved_world.restore()

        connect(use_gui=True)
        self.reset()
        p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])

        a = plan[0]
        b = a[1]
        print(b[0].configuration, b[1], b[2].body_paths[0].path)
        #print(b[1].pose, b[2].grasp_pose)
        print(type(b[0]), type(b[1]), type(b[2]))

        command = self.postprocess_plan(plan)

        # user_input('Execute?')
        if simulate:
            command.control()
        else:
            #command.step()
            command.execute(time_step=0.3)
            #command.refine(num_steps=10).execute(time_step=0.05)

    def step(self, action):
        """
        Environment step.

        :arg
            5-vector parameterizing XYZ offset, vertical angle offset
            (radians), and grasp angle (radians).
        :return
            observation: Next observation.
            reward: Float of the per-step reward as a result of taking the action.
            done: Bool of whether or not the episode has ended.
            debug: Dictionary of extra information provided by environment.
        """

        from math import pi
        import time

        dv = self._dv  # velocity per physics step.

        action = np.array([dv, dv, dv, 0.25]) * action  # [dx, dy, dz, da]
        self.action = np.append(action, np.array([0, -pi, 0, 0.0])) # [dx, dy, dz, da, Al, Bt, Gm, Fn_angle]
        #action = np.array([dv, dv, dv, 0.25, 2*pi, 2*pi, 2*pi]) * action  # [dx, dy, dz, da, Euler]
        #action = np.append(action, 0.3)  # [finger angle]


        self._kuka.applyAction(self.action)

        # Repeat as many times as set in config
        for _ in range(self._actionRepeat):
            p.stepSimulation()

            if self._termination():
                break

        if self._renders:
            time.sleep(10 * self._timeStep)

        #self.distance1, self.distance2, self.bl_bl_distance = self._get_distance_to_goal()

        # Perform commanded action.
        self._env_step += 1

        observation = self.get_observation()
        reward = self._reward()
        done = self._termination()

        debug = {
            'action_name': self._rl_action,
            'goal': self._rl_goal,
            #'distance2': self.distance2
        }
        return observation, reward, done, debug

    def _reward(self):
        """Calculates the reward for the episode.
        :return: float
        """
        if self._rl_action == 'move_free' or self._rl_action == 'move_holding':
            return self._reward_move()
        else:
            raise NotImplementedError

    def _reward_move(self):
        """

        :return: float
        """
        from numpy.core.umath_tests import inner1d
        from math import sqrt

        obs = self.get_observation(inMatrixForm=True)

        # abs(final - initial configuration)
        delta = abs(obs[1] - obs[2])
        self.distance = sqrt(inner1d(delta, delta))

        # Negative reward for every extra action
        action_norm = inner1d(self.action[0:4], self.action[0:4])
        # a hack to be fixed in future
        action_fingers = abs(0.3 - self.action[7])

        return - self.distance - action_norm - action_fingers

    def get_observation(self, inMatrixForm=False):

        return self._get_observation_move(inMatrixForm=inMatrixForm)

    def _get_observation_move(self, inMatrixForm=False):
        """Return an observation array:
            [ gripper's in the world frame X, Y, Z, Al, Bt, Gm, Blocks' in the gripper's frame X, Y, Z, Euler X, Y, Z]

            if inMatrixForm is True then
            [ [gripper's in the world frame X, Y, Z, Al, Bt, Gm], [Blocks' in the gripper's frame X, Y, Z, Euler X, Y, Z]]
        """
        # get the gripper's world position and orientation

        # The coorditates of the gripper and fingers (X, Y, Z)
        gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
        fingerState_l = p.getLinkState(self._kuka.kukaUid, 10)[0]
        fingerState_r = p.getLinkState(self._kuka.kukaUid, 13)[0]

        #gripperPos = gripperState[0]

        gripperPos = np.array(gripperState[0] + np.array([0.00028128,  0.02405984, -0.19820549]))
        gripperOrn = gripperState[1]  # Quaternion
        gripperEul = p.getEulerFromQuaternion(gripperOrn)  # Euler: (Al, Bt, Gm)

        observation = []
        if inMatrixForm:
            temp = list(gripperPos)
            temp.extend(list(fingerState_l + fingerState_r + gripperEul))
            observation.append(temp)
            observation.append(list(get_configuration(self.robot)))
            observation.append(self._get_rl_goal())
        else:
            observation.extend(list(gripperPos))
            observation.extend(list(fingerState_l + fingerState_r + gripperEul))
            observation.extend(list(get_configuration(self.robot)))
            observation.extend(self._get_rl_goal())

        return np.array(observation)

    def execute_rl_action(self, plan_action):
        """
        Used to switch to a appropriate RL for the current action
        :param plan_action:
        :return:
        """

        name, params = plan_action

        self._rl_action = name

        if name == 'move_free' or name == 'move_holding':
            self._set_rl_goal(params[1].configuration)  # arm's joint configuration
        else:
            raise ImportError


        # TODO: save an initial world state for training

    def _termination(self):
        """
        Terminates the episode if either the goal or maxSteps steps is achieved.
        """
        return self.distance <= 0.1 or self._env_step >= self._maxSteps


    def _set_rl_goal(self, goal):
        self._rl_goal = goal

    def _get_rl_goal(self):
        return self._rl_goal


    # Utility methods --------------------------------------

    def get_fixed(self, robot, movable):
        rigid = [body for body in get_bodies() if body != robot]
        fixed = [body for body in rigid if body not in movable]

        return fixed

    def get_free_motion_synth(self, robot, movable=[], teleport=False):
        fixed = self.get_fixed(robot, movable)

        def fn(outputs, certified):
            assert (len(outputs) == 1)
            q0, _, q1 = find_unique(lambda f: f[0] == 'freemotion', certified)[1:]
            obstacles = fixed + self.place_movable(certified)
            free_motion_fn = get_free_motion_gen(robot, obstacles, teleport)
            return free_motion_fn(q0, q1)

        return fn

    def get_holding_motion_synth(self, robot, movable=[], teleport=False):
        fixed = self.get_fixed(robot, movable)

        def fn(outputs, certified):
            assert (len(outputs) == 1)
            q0, _, q1, o, g = find_unique(lambda f: f[0] == 'holdingmotion', certified)[1:]
            obstacles = fixed + self.place_movable(certified)
            holding_motion_fn = get_holding_motion_gen(robot, obstacles, teleport)
            return holding_motion_fn(q0, q1, o, g)

        return fn

    def place_movable(self, certified):
        placed = []
        for literal in certified:
            if literal[0] == 'not':
                fact = literal[1]
                if fact[0] == 'trajcollision':
                    _, b, p = fact[1:]
                    set_pose(b, p.pose)
                    placed.append(b)
        return placed

    def postprocess_plan(self, plan):
        paths = []
        for name, args in plan:
            if name == 'place':
                paths += args[-1].reverse().body_paths
            elif name in ['move', 'move_free', 'move_holding', 'pick']:
                paths += args[-1].body_paths


        return Command(paths)

#========================================

    def one_block(self):
        """
        Loads the environment initial state
        :return: robot, body_names, movable_bodies
        """
        import os

        from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
            get_stable_gen, get_ik_fn, get_free_motion_gen, \
            get_holding_motion_gen, get_movable_collision_test

        from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, \
            Pose, \
            Point, set_default_camera, stable_z, \
            BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
            disconnect, DRAKE_IIWA_URDF, get_bodies, user_input, HideOutput, KUKA_IIWA_URDF, get_model_path

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        # set robot ID
        robot = 0

        table = load_model(os.path.join(self._urdfRoot, "table/table.urdf"))
        sink = load_model(SINK_URDF)

        block1 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)

        set_pose(block1, Pose(Point(x=0.1, z=stable_z(block1, table))))
        set_pose(robot, Pose(Point(x=-0.5, z=stable_z(robot, table))))
        set_pose(sink, Pose(Point(x=0.5, z=stable_z(sink, table))))

        # the pose of the last loaded object cannot be set, hence floor
        floor = load_model(os.path.join(self._urdfRoot, "plane.urdf"))

        # print(stable_z(cup, table))

        body_names = {
            sink: 'sink',
            # stove: 'stove',
            block1: 'block1',
            table: 'table',
            # cup: 'cup',
            floor: 'floor',
        }
        movable_bodies = [block1]

        return robot, body_names, movable_bodies

    def two_blocks(self):
        """
        Loads the environment initial state
        :return: robot, body_names, movable_bodies
        """
        import os

        from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
            get_stable_gen, get_ik_fn, get_free_motion_gen, \
            get_holding_motion_gen, get_movable_collision_test

        from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, \
            Pose, \
            Point, set_default_camera, stable_z, \
            BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
            disconnect, DRAKE_IIWA_URDF, get_bodies, user_input, HideOutput, KUKA_IIWA_URDF, get_model_path

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        # set robot ID
        robot = 0

        table = load_model(os.path.join(self._urdfRoot, "table/table.urdf"))
        sink = load_model(SINK_URDF)

        block1 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block2 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)

        set_pose(block1, Pose(Point(x=0.1, z=stable_z(block1, table))))
        set_pose(block2, Pose(Point(y=0.1, z=stable_z(block1, table))))
        set_pose(robot, Pose(Point(x=-0.5, z=stable_z(robot, table))))
        set_pose(sink, Pose(Point(x=0.5, z=stable_z(sink, table))))

        # the pose of the last loaded object cannot be set, hence floor
        floor = load_model(os.path.join(self._urdfRoot, "plane.urdf"))

        # print(stable_z(cup, table))

        body_names = {
            sink: 'sink',
            # stove: 'stove',
            block1: 'block1',
            block2: 'block2',
            table: 'table',
            # cup: 'cup',
            floor: 'floor',
        }
        movable_bodies = [block1, block2]

        return robot, body_names, movable_bodies

    def five_blocks(self):
        """
        Loads the environment initial state
        :return: robot, body_names, movable_bodies
        """
        import os

        from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
            get_stable_gen, get_ik_fn, get_free_motion_gen, \
            get_holding_motion_gen, get_movable_collision_test

        from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, \
            Pose, \
            Point, set_default_camera, stable_z, \
            BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
            disconnect, DRAKE_IIWA_URDF, get_bodies, user_input, HideOutput, KUKA_IIWA_URDF, get_model_path

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        # set robot ID
        robot = 0

        table = load_model(os.path.join(self._urdfRoot, "table/table.urdf"))
        sink = load_model(SINK_URDF)

        block1 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block2 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block3 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block4 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block5 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)

        set_pose(block1, Pose(Point(x=0.1, z=stable_z(block1, table))))
        set_pose(block2, Pose(Point(y=0.1, z=stable_z(block1, table))))
        set_pose(block3, Pose(Point(x=0.12, z=stable_z(block1, table))))
        set_pose(block4, Pose(Point(y=0.12, z=stable_z(block1, table))))
        set_pose(block5, Pose(Point(x=0.15, z=stable_z(block1, table))))
        set_pose(robot, Pose(Point(x=-0.5, z=stable_z(robot, table))))
        set_pose(sink, Pose(Point(x=0.5, z=stable_z(sink, table))))

        # the pose of the last loaded object cannot be set, hence floor
        floor = load_model(os.path.join(self._urdfRoot, "plane.urdf"))

        # print(stable_z(cup, table))

        body_names = {
            sink: 'sink',
            # stove: 'stove',
            block1: 'block1',
            block2: 'block2',
            block3: 'block3',
            block4: 'block4',
            block5: 'block5',
            table: 'table',
            # cup: 'cup',
            floor: 'floor',
        }
        movable_bodies = [block1, block2, block3, block4, block5]
        return robot, body_names, movable_bodies

    def ten_blocks(self):
        """
        Loads the environment initial state
        :return: robot, body_names, movable_bodies
        """
        import os

        from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
            get_stable_gen, get_ik_fn, get_free_motion_gen, \
            get_holding_motion_gen, get_movable_collision_test

        from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, \
            Pose, \
            Point, set_default_camera, stable_z, \
            BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
            disconnect, DRAKE_IIWA_URDF, get_bodies, user_input, HideOutput, KUKA_IIWA_URDF, get_model_path

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        # set robot ID
        robot = 0

        table = load_model(os.path.join(self._urdfRoot, "table/table.urdf"))
        sink = load_model(SINK_URDF)

        block1 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block2 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block3 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block4 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block5 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block6 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block7 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block8 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block9 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block10 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)

        set_pose(block1, Pose(Point(x=0.1, z=stable_z(block1, table))))
        set_pose(block2, Pose(Point(y=0.1, z=stable_z(block1, table))))
        set_pose(block3, Pose(Point(x=0.12, z=stable_z(block1, table))))
        set_pose(block4, Pose(Point(y=0.12, z=stable_z(block1, table))))
        set_pose(block5, Pose(Point(x=0.15, z=stable_z(block1, table))))
        set_pose(block6, Pose(Point(y=0.15, z=stable_z(block1, table))))
        set_pose(block7, Pose(Point(x=0.18, z=stable_z(block1, table))))
        set_pose(block8, Pose(Point(y=0.18, z=stable_z(block1, table))))
        set_pose(block9, Pose(Point(x=0.21, z=stable_z(block1, table))))
        set_pose(block10, Pose(Point(y=0.21, z=stable_z(block1, table))))
        set_pose(robot, Pose(Point(x=-0.5, z=stable_z(robot, table))))
        set_pose(sink, Pose(Point(x=0.5, z=stable_z(sink, table))))

        # the pose of the last loaded object cannot be set, hence floor
        floor = load_model(os.path.join(self._urdfRoot, "plane.urdf"))

        # print(stable_z(cup, table))

        body_names = {
            sink: 'sink',
            # stove: 'stove',
            block1: 'block1',
            block2: 'block2',
            block3: 'block3',
            block4: 'block4',
            block5: 'block5',
            block6: 'block6',
            block7: 'block7',
            block8: 'block8',
            block9: 'block9',
            block10: 'block10',
            table: 'table',
            # cup: 'cup',
            floor: 'floor',
        }
        movable_bodies = [block1, block2, block3, block4, block5, block6, block7, block8, block9, block10]
        return robot, body_names, movable_bodies

    def pddl_one_block(self, robot, movable=[], teleport=False, movable_collisions=False, grasp_name='top'):
        # assert (not are_colliding(tree, kin_cache))
        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}

        # initial state
        print('Robot:', robot)
        conf = BodyConf(robot, get_configuration(robot))
        # add a robot configuration
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        fixed = self.get_fixed(robot, movable)
        print('Movable:', movable)
        print('Fixed:', fixed)
        for body in movable:
            pose = BodyPose(body, get_pose(body))
            # add the poses of all movable objects
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]
            # add all placement surfaces with a sampled pose
            for surface in fixed:
                init += [('Stackable', body, surface)]
                if is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]

        for body in fixed:
            name = get_body_name(body)
            if 'sink' in name:
                init += [('Sink', body)]
            if 'stove' in name:
                init += [('Stove', body)]

        # goal
        body = movable[0]
        goal = ('and',
                ('AtConf', conf),  # move the arm to the initial configuration
                ('On', movable[0], fixed[1]),
                ('Cleaned', movable[0]),
                )

        stream_map = {
            'sample-pose': from_gen_fn(get_stable_gen(fixed)),
            'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
            'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
            'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
            'TrajCollision': get_movable_collision_test(),
        }

        if USE_SYNTHESIZERS:
            stream_map.update({
                'plan-free-motion': empty_gen(),
                'plan-holding-motion': empty_gen(),
            })

        return domain_pddl, constant_map, stream_pddl, stream_map, init, goal

    def pddl_two_blocks(self, robot, movable=[], teleport=False, movable_collisions=False, grasp_name='top'):
        # assert (not are_colliding(tree, kin_cache))
        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}

        # initial state
        print('Robot:', robot)
        conf = BodyConf(robot, get_configuration(robot))
        # add a robot configuration
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        fixed = self.get_fixed(robot, movable)
        print('Movable:', movable)
        print('Fixed:', fixed)
        for body in movable:
            pose = BodyPose(body, get_pose(body))
            # add the poses of all movable objects
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]
            # add all placement surfaces with a sampled pose
            for surface in fixed:
                init += [('Stackable', body, surface)]
                if is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]

        for body in fixed:
            name = get_body_name(body)
            if 'sink' in name:
                init += [('Sink', body)]
            if 'stove' in name:
                init += [('Stove', body)]

        # goal
        body = movable[0]
        goal = ('and',
                ('AtConf', conf),  # move the arm to the initial configuration
                ('On', movable[0], fixed[1]),
                ('On', movable[1], fixed[1]),
                ('Cleaned', movable[0]),
                ('Cleaned', movable[1]),
                )

        stream_map = {
            'sample-pose': from_gen_fn(get_stable_gen(fixed)),
            'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
            'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
            'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
            'TrajCollision': get_movable_collision_test(),
        }

        if USE_SYNTHESIZERS:
            stream_map.update({
                'plan-free-motion': empty_gen(),
                'plan-holding-motion': empty_gen(),
            })

        return domain_pddl, constant_map, stream_pddl, stream_map, init, goal

    def pddl_five_blocks(self, robot, movable=[], teleport=False, movable_collisions=False, grasp_name='top'):
        # assert (not are_colliding(tree, kin_cache))
        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}

        # initial state
        print('Robot:', robot)
        conf = BodyConf(robot, get_configuration(robot))
        # add a robot configuration
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        fixed = self.get_fixed(robot, movable)
        print('Movable:', movable)
        print('Fixed:', fixed)
        for body in movable:
            pose = BodyPose(body, get_pose(body))
            # add the poses of all movable objects
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]
            # add all placement surfaces with a sampled pose
            for surface in fixed:
                init += [('Stackable', body, surface)]
                if is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]

        for body in fixed:
            name = get_body_name(body)
            if 'sink' in name:
                init += [('Sink', body)]
            if 'stove' in name:
                init += [('Stove', body)]

        # goal
        body = movable[0]
        goal = ('and',
                ('AtConf', conf),  # move the arm to the initial configuration
                ('On', movable[0], fixed[1]),
                ('On', movable[1], fixed[1]),
                ('On', movable[2], fixed[1]),
                ('On', movable[3], fixed[1]),
                ('On', movable[4], fixed[1]),
                ('Cleaned', movable[0]),
                ('Cleaned', movable[1]),
                ('Cleaned', movable[2]),
                ('Cleaned', movable[3]),
                ('Cleaned', movable[4]),
                )

        stream_map = {
            'sample-pose': from_gen_fn(get_stable_gen(fixed)),
            'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
            'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
            'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
            'TrajCollision': get_movable_collision_test(),
        }

        if USE_SYNTHESIZERS:
            stream_map.update({
                'plan-free-motion': empty_gen(),
                'plan-holding-motion': empty_gen(),
            })

        return domain_pddl, constant_map, stream_pddl, stream_map, init, goal

    def pddl_ten_blocks(self, robot, movable=[], teleport=False, movable_collisions=False, grasp_name='top'):
        # assert (not are_colliding(tree, kin_cache))
        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}

        # initial state
        print('Robot:', robot)
        conf = BodyConf(robot, get_configuration(robot))
        # add a robot configuration
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        fixed = self.get_fixed(robot, movable)
        print('Movable:', movable)
        print('Fixed:', fixed)
        for body in movable:
            pose = BodyPose(body, get_pose(body))
            # add the poses of all movable objects
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]
            # add all placement surfaces with a sampled pose
            for surface in fixed:
                init += [('Stackable', body, surface)]
                if is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]

        for body in fixed:
            name = get_body_name(body)
            if 'sink' in name:
                init += [('Sink', body)]
            if 'stove' in name:
                init += [('Stove', body)]

        # goal
        body = movable[0]
        goal = ('and',
                ('AtConf', conf),  # move the arm to the initial configuration
                ('On', movable[0], fixed[1]),
                ('On', movable[1], fixed[1]),
                ('On', movable[2], fixed[1]),
                ('On', movable[3], fixed[1]),
                ('On', movable[4], fixed[1]),
                ('On', movable[5], fixed[1]),
                ('On', movable[6], fixed[1]),
                ('On', movable[7], fixed[1]),
                ('On', movable[8], fixed[1]),
                ('On', movable[9], fixed[1]),
                ('Cleaned', movable[0]),
                ('Cleaned', movable[1]),
                ('Cleaned', movable[2]),
                ('Cleaned', movable[3]),
                ('Cleaned', movable[4]),
                ('Cleaned', movable[5]),
                ('Cleaned', movable[6]),
                ('Cleaned', movable[7]),
                ('Cleaned', movable[8]),
                ('Cleaned', movable[9]),
                )

        stream_map = {
            'sample-pose': from_gen_fn(get_stable_gen(fixed)),
            'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
            'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
            'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
            'TrajCollision': get_movable_collision_test(),
        }

        if USE_SYNTHESIZERS:
            stream_map.update({
                'plan-free-motion': empty_gen(),
                'plan-holding-motion': empty_gen(),
            })

        return domain_pddl, constant_map, stream_pddl, stream_map, init, goal

    def four_blocks_row(self):
        """
        Loads the environment initial state
        :return: robot, body_names, movable_bodies
        """
        import os

        from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
            get_stable_gen, get_ik_fn, get_free_motion_gen, \
            get_holding_motion_gen, get_movable_collision_test

        from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, \
            Pose, \
            Point, set_default_camera, stable_z, \
            BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
            disconnect, DRAKE_IIWA_URDF, get_bodies, user_input, HideOutput, KUKA_IIWA_URDF, get_model_path

        # load a kuka arm
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        # set robot ID
        robot = 0

        table = load_model(os.path.join(self._urdfRoot, "table/table.urdf"))
        sink = load_model(SINK_URDF)

        block1 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block2 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block3 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)
        block4 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=False)


        base1 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                           fixed_base=True)
        base2 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=True)
        base3 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=True)
        base4 = load_model(os.path.join(self._urdfRoot, "cube_small.urdf"),
                            fixed_base=True)

        set_pose(block1, Pose(Point(x=-0.23, y=0.25, z=stable_z(block1, table))))
        set_pose(block2, Pose(Point(x=-0.26, y=0.25, z=stable_z(block2, table))))
        set_pose(block3, Pose(Point(x=-0.29, y=0.25, z=stable_z(block3, table))))
        set_pose(block4, Pose(Point(x=-0.31, y=0.25, z=stable_z(block4, table))))

        set_pose(base1, Pose(Point(y=-0.1, x=0, z=stable_z(base1, table))))
        set_pose(base2, Pose(Point(y=-0.05, x=0, z=stable_z(base2, table))))
        set_pose(base3, Pose(Point(y=-0., x=0, z=stable_z(base3, table))))
        set_pose(base4, Pose(Point(y=0.05, x=0, z=stable_z(base4, table))))

        set_pose(robot, Pose(Point(x=-0.5, z=stable_z(robot, table))))
        set_pose(sink, Pose(Point(x=0.5, z=stable_z(sink, table))))

        # the pose of the last loaded object cannot be set, hence floor
        floor = load_model(os.path.join(self._urdfRoot, "plane.urdf"))

        # print(stable_z(cup, table))

        body_names = {
            table: 'table',
            sink: 'sink',
            # stove: 'stove',
            block1: 'block1',
            block2: 'block2',
            block3: 'block3',
            block4: 'block4',
            base1: 'base1',
            base2: 'base2',
            base3: 'base3',
            base4: 'base4',
            floor: 'floor',
        }
        movable_bodies = [block1, block2, block3, block4]
        return robot, body_names, movable_bodies

    def pddl_four_blocks_row(self, robot, movable=[], teleport=False, movable_collisions=False, grasp_name='top'):
        # assert (not are_colliding(tree, kin_cache))
        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}

        # initial state
        print('Robot:', robot)
        conf = BodyConf(robot, get_configuration(robot))
        # add a robot configuration
        init = [('CanMove',),
                ('Conf', conf),
                ('AtConf', conf),
                ('HandEmpty',)]

        fixed = self.get_fixed(robot, movable)
        print('Movable:', movable)
        print('Fixed:', fixed)
        for body in movable:
            pose = BodyPose(body, get_pose(body))
            # add the poses of all movable objects
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]
            # add all placement surfaces with a sampled pose
            for surface in fixed:
                init += [('Stackable', body, surface)]
                if is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]

        for body in fixed:
            name = get_body_name(body)
            if 'sink' in name:
                init += [('Sink', body)]
            if 'stove' in name:
                init += [('Stove', body)]

        # goal
        goal = ('and',
                ('AtConf', conf),  # move the arm to the initial configuration
                ('On', movable[0], fixed[3]),
                )

        stream_map = {
            'sample-pose': from_gen_fn(get_stable_gen(fixed)),
            'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
            'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
            'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
            'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
            'TrajCollision': get_movable_collision_test(),
        }

        if USE_SYNTHESIZERS:
            stream_map.update({
                'plan-free-motion': empty_gen(),
                'plan-holding-motion': empty_gen(),
            })

        return domain_pddl, constant_map, stream_pddl, stream_map, init, goal


