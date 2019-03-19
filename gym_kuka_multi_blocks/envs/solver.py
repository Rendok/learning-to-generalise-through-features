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

def load_world():
    """
    Loads the environment initial state
    :return: robot, body_names, movable_bodies
    """

    import os

    from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
        get_stable_gen, get_ik_fn, get_free_motion_gen, \
        get_holding_motion_gen, get_movable_collision_test

    _urdfRoot = pybullet_data.getDataPath()
    _timeStep = 1. / 240.
    _cam_dist = 3.3
    _cam_yaw = 150
    _cam_pitch = -31

    connect(use_gui=False)
    p.resetDebugVisualizerCamera(_cam_dist, _cam_yaw, _cam_pitch, [0.52, -0.2, -0.33])

    # load a kuka arm
    _kuka = kuka.Kuka(urdfRootPath=_urdfRoot, timeStep=_timeStep)
    # set robot ID
    robot = 0


    table = load_model(os.path.join(_urdfRoot, "table/table.urdf"))
    # stove = load_model(STOVE_URDF, pose=Pose(Point(x=+1.5)))
    sink = load_model(SINK_URDF)
    block1 = load_model(os.path.join(_urdfRoot, "cube_small.urdf"),
                        fixed_base=False)
    block2 = load_model(os.path.join(_urdfRoot, "cube_small.urdf"),
                        fixed_base=False)



    set_pose(block1, Pose(Point(x=0.1, z=stable_z(block1, table))))
    set_pose(block2, Pose(Point(y=0.1, z=stable_z(block1, table))))
    set_pose(robot, Pose(Point(x=-0.5, z=stable_z(robot, table))))
    set_pose(sink, Pose(Point(x=0.5, z=stable_z(sink, table))))

    # the pose of the last loaded object cannot be set, hence floor
    floor = load_model(os.path.join(_urdfRoot, "plane.urdf"))

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

    # set_default_camera()

    return robot, body_names, movable_bodies


def pddlstream_from_problem(robot, movable=[], teleport=False, movable_collisions=False, grasp_name='top'):
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

    fixed = get_fixed(robot, movable)
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


def solve(load_world, pddlstream_from_problem, viewer=False, display=True, simulate=False, teleport=False):

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
                          from_fn(get_free_motion_synth(robot, movable, teleport))),
        StreamSynthesizer('safe-holding-motion', {'plan-holding-motion': 1, 'trajcollision': 0},
                          from_fn(get_holding_motion_synth(robot, movable, teleport))),
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


# Utility methods --------------------------------------

def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]

    return fixed

def get_free_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)

    def fn(outputs, certified):
        assert (len(outputs) == 1)
        q0, _, q1 = find_unique(lambda f: f[0] == 'freemotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        free_motion_fn = get_free_motion_gen(robot, obstacles, teleport)
        return free_motion_fn(q0, q1)

    return fn

def get_holding_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)

    def fn(outputs, certified):
        assert (len(outputs) == 1)
        q0, _, q1, o, g = find_unique(lambda f: f[0] == 'holdingmotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        holding_motion_fn = get_holding_motion_gen(robot, obstacles, teleport)
        return holding_motion_fn(q0, q1, o, g)

    return fn

def place_movable(certified):
    placed = []
    for literal in certified:
        if literal[0] == 'not':
            fact = literal[1]
            if fact[0] == 'trajcollision':
                _, b, p = fact[1:]
                set_pose(b, p.pose)
                placed.append(b)
    return placed
