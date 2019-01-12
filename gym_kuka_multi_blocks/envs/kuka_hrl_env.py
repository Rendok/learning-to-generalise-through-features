from gym_kuka_multi_blocks.envs.kukaGymEnv import KukaGymEnv
import pybullet_data
import pybullet as p
from gym import spaces
import numpy as np
from . import kuka  # implementation of a kuka arm

import sys
sys.path.append("/Users/dgrebenyuk/Research/pddlstream")
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
        self._action_repeat = action_repeat
        self._is_enable_self_collision = is_enable_self_collision
        # self._observation = []
        self._renders = renders
        self._max_steps = max_steps
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

        #if self._renders:
        #    self.cid = p.connect(p.SHARED_MEMORY)
        #    if (self.cid < 0):
        #        self.cid = p.connect(p.GUI)
        #    p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        #else:
        #    self.cid = p.connect(p.DIRECT)

        self.cid = connect(use_gui=renders)
        p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])

        self._seed()

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(4,),
                                       dtype=np.float32)  # dx, dy, dz, da, Euler: Al, Bt, Gm  7 -> 4

        self.observation_space = spaces.Box(low=-100,
                                            high=100,
                                            shape=(7 + 6 + 6 * self._num_objects,),
                                            dtype=np.float32)

        self.viewer = None

    def reset(self):
        """Environment reset called at the beginning of an episode.
        """
        self._done = False
        reset_simulation()

        enable_gravity()

        # set the physics engine
        #p.resetSimulation()
        #p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)

        # simulate_for_duration(5)
        for _ in range(500):
            step_simulation()

        #robot, body_names, movable_bodies = self.load_world()

        return 1

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
        robot = 0

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
        set_pose(robot, Pose(Point(x=-0.5, z=stable_z(robot, table))))
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

        return robot, body_names, movable_bodies

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

    def solve(self, viewer=False, display=True, simulate=False, teleport=False):

        import cProfile
        import pstats
        from pddlstream.language.synthesizer import StreamSynthesizer
        from pddlstream.algorithms.focused import solve_focused


        robot, names, movable = self.load_world()

        #saved_world = WorldSaver()
        # dump_world()

        pddlstream_problem = self.pddlstream_from_problem(robot, movable=movable, movable_collisions=True, teleport=teleport)
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

        disconnect()

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
        self.load_world()

        command = self.postprocess_plan(plan)

        # user_input('Execute?')
        if simulate:
            command.control()
        else:
            #command.step()
            command.execute(time_step=0.1)
            #command.refine(num_steps=10).execute(time_step=0.05)


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

        print(paths)

        return Command(paths)

