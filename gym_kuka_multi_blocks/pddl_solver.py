from __future__ import print_function

import sys

sys.path.append("/Users/dgrebenyuk/Research/pddlstream")

import cProfile
import pstats
import argparse

from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, get_movable_collision_test
from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, \
    BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, get_bodies, user_input, HideOutput
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.generator import from_gen_fn, from_fn, empty_gen
from pddlstream.language.synthesizer import StreamSynthesizer
from pddlstream.utils import print_solution, read, INF, get_file_path, find_unique

import pybullet as p
import pybullet_data
import os

USE_SYNTHESIZERS = False


def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed


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


#######################################################

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
    body = movable[1]
    goal = ('and',
            ('AtConf', conf),  # move the arm to the initial configuration
            # ('Holding', body),
            # ('On', body, fixed[1]),
            # ('On', body, fixed[2]),
            # ('Cleaned', movable[1]),
            ('Cooked', body),
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


#######################################################

def load_world():
    # TODO: store internal world info here to be reloaded
    with HideOutput():
        robot = load_model(DRAKE_IIWA_URDF)
        # robot = load_model(KUKA_IIWA_URDF)
        floor = load_model('models/short_floor.urdf')
        sink = load_model(SINK_URDF, pose=Pose(Point(x=-0.5)))
        stove = load_model(STOVE_URDF, pose=Pose(Point(x=+0.5)))
        block = load_model(BLOCK_URDF, fixed_base=False)

        block1 = load_model(BLOCK_URDF, fixed_base=False)
        block2 = load_model(BLOCK_URDF, fixed_base=False)
        block3 = load_model(BLOCK_URDF, fixed_base=False)
        block4 = load_model(BLOCK_URDF, fixed_base=False)
        block5 = load_model(BLOCK_URDF, fixed_base=False)
        block6 = load_model(BLOCK_URDF, fixed_base=False)
        block7 = load_model(BLOCK_URDF, fixed_base=False)

        _urdfRoot = pybullet_data.getDataPath()
        #urdf_path = os.path.join(_urdfRoot, "cube_small.urdf")  # urdf_name

        #angle = 3.1481 / 2
        #orn = p.getQuaternionFromEuler([0, 0, angle])
        #uid = p.loadURDF(urdf_path, [0.55, 0.1, 1.15], orn)

        cup = load_model('models/cup.urdf',  #'models/dinnerware/cup/cup_small.urdf'
                         fixed_base=False)

    body_names = {
        sink: 'sink',
        stove: 'stove',
        block: 'celery',
        cup: 'cup',
    }
    movable_bodies = [block, cup, block1, block2, block3, block4, block5, block6, block7]

    set_pose(block, Pose(Point(x=0.1, y=0.5, z=stable_z(block, floor))))

    set_pose(block1, Pose(Point(x=-0.1, y=0.5, z=stable_z(block1, floor))))
    set_pose(block2, Pose(Point(y=0.35, z=stable_z(block2, floor))))
    set_pose(block3, Pose(Point(y=0.65, z=stable_z(block3, floor))))
    set_pose(block4, Pose(Point(x=0.1, y=0.65, z=stable_z(block4, floor))))
    set_pose(block5, Pose(Point(x=-0.1, y=0.65, z=stable_z(block5, floor))))
    set_pose(block6, Pose(Point(x=0.1, y=0.35, z=stable_z(block6, floor))))
    set_pose(block7, Pose(Point(x=-0.1, y=0.35, z=stable_z(block7, floor))))

    set_pose(cup, Pose(Point(y=0.5, z=stable_z(cup, floor))))
    set_default_camera()

    return robot, body_names, movable_bodies


def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            paths += args[-1].body_paths

    return Command(paths)


#######################################################

def main(viewer=False, display=True, simulate=False, teleport=False):
    # TODO: fix argparse & FastDownward
    # parser = argparse.ArgumentParser()  # Automatically includes help
    # parser.add_argument('-viewer', action='store_true', help='enable viewer.')
    # parser.add_argument('-display', action='store_true', help='enable viewer.')
    # args = parser.parse_args()
    # TODO: getopt

    connect(use_gui=viewer)
    #connect(p.DIRECT)

    p.setPhysicsEngineParameter(numSolverIterations=150)
    #_timeStep = 1. / 240.
    #p.setTimeStep(_timeStep)
    p.setGravity(0, 0, -10)

    robot, names, movable = load_world()

    #for _ in range(500):
    #    p.stepSimulation()

    saved_world = WorldSaver()
    # dump_world()

    #print(p.getBasePositionAndOrientation(6))

    pddlstream_problem = pddlstream_from_problem(robot, movable=movable,
                                                 teleport=teleport, movable_collisions=True)
    _, _, _, stream_map, init, goal = pddlstream_problem
    synthesizers = [
        StreamSynthesizer('safe-free-motion', {'plan-free-motion': 1, 'trajcollision': 0},
                          from_fn(get_free_motion_synth(robot, movable, teleport))),
        StreamSynthesizer('safe-holding-motion', {'plan-holding-motion': 1, 'trajcollision': 0},
                          from_fn(get_holding_motion_synth(robot, movable, teleport))),
    ] if USE_SYNTHESIZERS else []
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', stream_map.keys())
    print('Synthesizers:', stream_map.keys())
    print('Names:', names)

    # initialize and measure performance
    pr = cProfile.Profile()
    pr.enable()
    solution = solve_focused(pddlstream_problem, synthesizers=synthesizers, max_cost=INF)
    print_solution(solution)
    plan, cost, evaluations = solution
    pr.disable()
    # print stats
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)

    if plan is None:
        return

    if (not display) or (plan is None):
        disconnect()
        return

    if not viewer:  # TODO: how to reenable the viewer
        disconnect()
        connect(use_gui=True)
        load_world()
    else:
        saved_world.restore()

    command = postprocess_plan(plan)

    #user_input('Execute?')
    if simulate:
        command.control()
    else:
        # command.step()
        command.refine(num_steps=10).execute(time_step=0.001)

    # wait_for_interrupt()
    #user_input('Finish?')
    disconnect()


if __name__ == '__main__':
    main()
