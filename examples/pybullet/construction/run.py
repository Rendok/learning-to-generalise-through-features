from __future__ import print_function

import cProfile
import pstats
import numpy as np
from itertools import product

from examples.pybullet.construction.debug import get_test_cfree, test_grasps, test_print
from examples.pybullet.construction.utils import parse_elements, parse_node_points, parse_ground_nodes, create_elements, \
    load_extrusion, TOOL_NAME, DISABLED_COLLISIONS, check_trajectory_collision, get_grasp_pose, load_world, \
    prune_dominated, get_element_neighbors, get_node_neighbors, sample_direction, draw_element
from examples.pybullet.utils.pybullet_tools.utils import connect, dump_body, disconnect, wait_for_interrupt, \
    get_movable_joints, get_sample_fn, set_joint_positions, link_from_name, add_line, inverse_kinematics, \
    get_link_pose, multiply, wait_for_duration, set_color, has_gui, add_text, angle_between, plan_joint_motion, \
    get_pose, invert, point_from_pose, get_distance, get_joint_positions, wrap_angle, \
    get_collision_fn, compute_joint_weights
from pddlstream.algorithms.incremental import solve_exhaustive, solve_incremental
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import PDDLProblem, And
from pddlstream.language.generator import from_test, from_gen_fn
from pddlstream.utils import read, get_file_path, print_solution, user_input, irange, topological_sort, neighbors_from_orders

SUPPORT_THETA = np.math.radians(10)  # Support polygon

JOINT_WEIGHTS = [0.3078557810844393, 0.443600199302506, 0.23544367607317915,
                 0.03637161028426032, 0.04644626184081511, 0.015054267683041092]

class MotionTrajectory(object):
    def __init__(self, robot, joints, path):
        self.robot = robot
        self.joints = joints
        self.path = path
    def __repr__(self):
        return 'm({},{})'.format(len(self.joints), len(self.path))

class PrintTrajectory(object):
    def __init__(self, robot, joints, path, element, reverse, colliding=set()):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.n1, self.n2 = reversed(element) if reverse else element
        self.element = element
        self.colliding = colliding
    def __repr__(self):
        return '{}->{}'.format(self.n1, self.n2)

def get_other_node(node1, element):
    assert node1 in element
    return element[node1 == element[0]]

def is_ground(element, ground_nodes):
    return any(n in ground_nodes for n in element)

def draw_model(elements, node_points, ground_nodes):
    handles = []
    for element in elements:
        color = (0, 0, 1) if is_ground(element, ground_nodes) else (1, 0, 0)
        handles.append(draw_element(node_points, element, color=color))
    return handles

def get_supported_orders(elements, node_points):
    node_neighbors = get_node_neighbors(elements)
    orders = set()
    for node in node_neighbors:
        supporters = {e for e in node_neighbors[node] if element_supports(e, node, node_points)}
        printers = {e for e in node_neighbors[node] if is_start_node(node, e, node_points) and not doubly_printable(e, node_points)}
        orders.update((e1, e2) for e1 in supporters for e2 in printers)
    return orders

def element_supports(e, n1, node_points): # A property of nodes
    # TODO: support polygon (ZMP heuristic)
    # TODO: recursively apply as well
    # TODO: end-effector force
    # TODO: allow just a subset to support
    # TODO: construct using only upwards
    n2 = get_other_node(n1, e)
    delta = node_points[n2] - node_points[n1]
    theta = angle_between(delta, [0, 0, -1])
    return theta < (np.pi / 2 - SUPPORT_THETA)

def is_start_node(n1, e, node_points):
    return not element_supports(e, n1, node_points)

def doubly_printable(e, node_points):
    return all(is_start_node(n, e, node_points) for n in e)

def retrace_supporters(element, incoming_edges, supporters):
    for element2 in incoming_edges[element]:
        if element2 not in supporters:
            retrace_supporters(element2, incoming_edges, supporters=supporters)
            supporters.append(element2)

##################################################

def get_pddlstream(trajectories, element_bodies, ground_nodes):
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    constant_map = {}

    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    stream_map = {
        'test-cfree': from_test(get_test_cfree(element_bodies)),
    }

    init = []
    for n in ground_nodes:
        init.append(('Connected', n))
    for t in trajectories:
        init.extend([
            ('Node', t.n1),
            ('Node', t.n2),
            ('Element', t.element),
            ('Traj', t),
            ('Connection', t.n1, t.element, t, t.n2),
        ])

    goal_literals = [('Printed', e) for e in element_bodies]
    goal = And(*goal_literals)
    # TODO: weight or order these in some way
    # TODO: instantiation slowness is due to condition effects. Change!

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)



def get_pddlstream2(robot, obstacles, node_points, element_bodies, ground_nodes, trajectories=[]):
    domain_pddl = read(get_file_path(__file__, 'regression.pddl')) # progression | regression
    constant_map = {}

    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    stream_map = {
        'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-print': get_wild_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes),
    }

    # TODO: assert that all elements have some support
    init = []
    for n in ground_nodes:
        init.append(('Grounded', n))
    for e in element_bodies:
        for n in e:
            if element_supports(e, n, node_points):
                init.append(('Supports', e, n))
            if is_start_node(n, e, node_points):
                init.append(('StartNode', n, e))
    for e in element_bodies:
        n1, n2 = e
        init.extend([
            ('Node', n1),
            ('Node', n2),
            ('Element', e),
            ('Printed', e),
            ('Edge', n1, e, n2),
            ('Edge', n2, e, n1),
            #('StartNode', n1, e),
            #('StartNode', n2, e),
        ])
        #if is_ground(e, ground_nodes):
        #    init.append(('Grounded', e))
    #for e1, neighbors in get_element_neighbors(element_bodies).items():
    #    for e2 in neighbors:
    #        init.append(('Supports', e1, e2))
    for t in trajectories:
        init.extend([
            ('Traj', t),
            ('PrintAction', t.n1, t.element, t),
        ])

    goal = And(*[('Removed', e) for e in element_bodies])

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes, trajectories=[]):
    if trajectories is None:
        return None
    pddlstream_fn = get_pddlstream2 # get_pddlstream | get_pddlstream2

    # TODO: Iterated search
    # http://www.fast-downward.org/Doc/Heuristic#h.5Em_heuristic
    # randomize_successors=True
    pr = cProfile.Profile()
    pr.enable()
    pddlstream_problem = pddlstream_fn(robot, obstacles, node_points, element_bodies,
                                       ground_nodes, trajectories=trajectories)
    #solution = solve_exhaustive(pddlstream_problem, planner='goal-lazy', max_time=300, debug=True)
    #solution = solve_incremental(pddlstream_problem, planner='add-random-lazy', max_time=600,
    #                             max_planner_time=300, debug=True)
    stream_info = {
        #'sample-print': StreamInfo(PartialInputs(unique=True)),
    }
    planner = 'ff-ehc'
    #planner = 'ff-wastar1000' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    #planner = 'ff-eager-wastar1000'
    #planner = 'ff-wastar5'
    solution = solve_focused(pddlstream_problem, stream_info=stream_info, max_time=600,
                             effort_weight=1, unit_efforts=True, use_skeleton=False, #unit_costs=True,
                             planner=planner, max_planner_time=15, debug=True)
    # solve_exhaustive | solve_incremental
    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    # Random restart based strategy here
    print_solution(solution)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)
    plan, _, _ = solution

    if plan is None:
        return None
    if pddlstream_fn == get_pddlstream:
        return [t for _, (n1, e, t, n2) in plan]
    elif pddlstream_fn == get_pddlstream2:
        return [t for _, (n1, e, t) in reversed(plan)]
    else:
        raise ValueError(pddlstream_fn)

##################################################

def optimize_angle(robot, link, element_pose, translation, direction, reverse, initial_angles,
                   collision_fn, max_error=1e-2):
    movable_joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, movable_joints)
    best_error, best_angle, best_conf = max_error, None, None
    for i, angle in enumerate(initial_angles):
        grasp_pose = get_grasp_pose(translation, direction, angle, reverse)
        target_pose = multiply(element_pose, invert(grasp_pose))
        conf = inverse_kinematics(robot, link, target_pose)
        # if conf is None:
        #    continue
        #if pairwise_collision(robot, robot):
        conf = get_joint_positions(robot, movable_joints)
        if not collision_fn(conf):
            link_pose = get_link_pose(robot, link)
            error = get_distance(point_from_pose(target_pose), point_from_pose(link_pose))
            if error < best_error:  # TODO: error a function of direction as well
                best_error, best_angle, best_conf = error, angle, conf
            # wait_for_interrupt()
        if i != len(initial_angles)-1:
            set_joint_positions(robot, movable_joints, initial_conf)
    #print(best_error, translation, direction, best_angle)
    if best_conf is not None:
        set_joint_positions(robot, movable_joints, best_conf)
        #wait_for_interrupt()
    return best_angle, best_conf


def compute_direction_path(robot, length, reverse, element_body, direction, collision_fn):
    step_size = 0.0025 # 0.005
    #angle_step_size = np.pi / 128
    angle_step_size = np.math.radians(0.25)
    angle_deltas = [-angle_step_size, 0, angle_step_size]
    #num_initial = 12
    num_initial = 1

    steps = np.append(np.arange(-length / 2, length / 2, step_size), [length / 2])
    #print('Length: {} | Steps: {}'.format(length, len(steps)))

    #initial_angles = [wrap_angle(angle) for angle in np.linspace(0, 2*np.pi, num_initial, endpoint=False)]
    initial_angles = [wrap_angle(angle) for angle in np.random.uniform(0, 2*np.pi, num_initial)]
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)
    set_joint_positions(robot, movable_joints, sample_fn())
    link = link_from_name(robot, TOOL_NAME)
    element_pose = get_pose(element_body)
    current_angle, current_conf = optimize_angle(robot, link, element_pose,
                                                 steps[0], direction, reverse, initial_angles, collision_fn)
    if current_conf is None:
        return None
    # TODO: constrain maximum conf displacement
    # TODO: alternating minimization for just position and also orientation
    trajectory = [current_conf]
    for translation in steps[1:]:
        #set_joint_positions(robot, movable_joints, current_conf)
        initial_angles = [wrap_angle(current_angle + delta) for delta in angle_deltas]
        current_angle, current_conf = optimize_angle(robot, link, element_pose,
                                                     translation, direction, reverse, initial_angles, collision_fn)
        if current_conf is None:
            return None
        trajectory.append(current_conf)
    return trajectory


def sample_print_path(robot, length, reverse, element_body, collision_fn):
    #max_directions = 10
    #max_directions = 16
    max_directions = 1
    #for direction in np.linspace(0, 2*np.pi, 10, endpoint=False):
    directions = [sample_direction() for _ in range(max_directions)]
    #directions = np.linspace(0, 2*np.pi, 10, endpoint=False)
    #directions = np.random.uniform(0, 2*np.pi, max_directions)
    for direction in directions:
        trajectory = compute_direction_path(robot, length, reverse, element_body, direction, collision_fn)
        if trajectory is not None:
            return trajectory
    return None

##################################################

def get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes):
    max_attempts = 150
    max_trajectories = 10
    check_collisions = True
    # 50 doesn't seem to be enough

    movable_joints = get_movable_joints(robot)
    disabled_collisions = {tuple(link_from_name(robot, link) for link in pair) for pair in DISABLED_COLLISIONS}
    #element_neighbors = get_element_neighbors(element_bodies)
    node_neighbors = get_node_neighbors(element_bodies)
    incoming_supporters, _ = neighbors_from_orders(get_supported_orders(element_bodies, node_points))
    # TODO: print on full sphere and just check for collisions with the printed element
    # TODO: can slide a component of the element down

    def gen_fn(node1, element, fluents=[]):
        # TODO: sample collisions while making the trajectory
        reverse = (node1 != element[0])
        element_body = element_bodies[element]
        n1, n2 = reversed(element) if reverse else element
        delta = node_points[n2] - node_points[n1]
        # if delta[2] < 0:
        #    continue
        length = np.linalg.norm(delta)  # 5cm

        #supporters = {e for e in node_neighbors[n1] if element_supports(e, n1, node_points)}
        supporters = []
        retrace_supporters(element, incoming_supporters, supporters)
        elements_order = [e for e in element_bodies if (e != element) and (e not in supporters)]
        bodies_order = [element_bodies[e] for e in elements_order]
        collision_fn = get_collision_fn(robot, movable_joints, obstacles + [element_bodies[e] for e in supporters],
                                        attachments=[], self_collisions=True,
                                        disabled_collisions=disabled_collisions, use_limits=True)
        trajectories = []
        for num in irange(max_trajectories):
            for attempt in range(max_attempts):
                path = sample_print_path(robot, length, reverse, element_body, collision_fn)
                if path is None:
                    continue
                if check_collisions:
                    collisions = check_trajectory_collision(robot, path, bodies_order)
                    colliding = {e for k, e in enumerate(elements_order) if (element != e) and collisions[k]}
                else:
                    colliding = set()
                if (node_neighbors[n1] <= colliding) and not any(n in ground_nodes for n in element):
                    continue
                print_traj = PrintTrajectory(robot, movable_joints, path, element, reverse, colliding)
                trajectories.append(print_traj)
                if print_traj not in trajectories:
                    continue
                print('{}) {}->{} ({}) | {} | {} | {}'.format(num, n1, n2, len(supporters), attempt, len(trajectories),
                                                              sorted(len(t.colliding) for t in trajectories)))
                yield print_traj,
                if not colliding:
                    return
            else:
                print('{}) {}->{} ({}) | {} | Failure!'.format(num, len(supporters), n1, n2, max_attempts))
                break
    return gen_fn

def get_wild_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes):
    gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)
    def wild_gen_fn(node1, element, fluents=[]):
        for t, in gen_fn(node1, element, fluents=fluents):
            outputs = [(t,)]
            facts = [('Collision', t, e2) for e2 in t.colliding]
            #outputs = []
            #facts.append(('PrintAction', node1, element, t))
            yield outputs, facts
    return wild_gen_fn

##################################################

def sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes):
    gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)
    all_trajectories = []
    for index, (element, element_body) in enumerate(element_bodies.items()):
        add_text(element[0], position=(0, 0, -0.05), parent=element_body)
        add_text(element[1], position=(0, 0, +0.05), parent=element_body)
        trajectories = []
        for node1 in element:
            for traj, in gen_fn(node1, element):
                trajectories.append(traj)
        all_trajectories.extend(trajectories)
        if not trajectories:
            #if has_gui():
            #    for e, body in element_bodies.items():
            #        if e == element:
            #            set_color(body, (0, 1, 0, 1))
            #        elif e in element_neighbors[e]:
            #            set_color(body, (1, 0, 0, 1))
            #        else:
            #            set_color(body, (1, 0, 0, 0))
            #    wait_for_interrupt()
            return None
    return all_trajectories

def compute_motions(robot, fixed_obstacles, element_bodies, initial_conf, trajectories):
    # TODO: can just plan to initial and then shortcut
    # TODO: backoff motion
    # TODO: reoptimize for the sequence that have the smallest movements given this
    # TODO: sample trajectories
    # TODO: more appropriate distance based on displacement/volume
    if trajectories is None:
        return None
    weights = np.array(JOINT_WEIGHTS)
    resolutions = np.divide(0.005*np.ones(weights.shape), weights)
    movable_joints = get_movable_joints(robot)
    disabled_collisions = {tuple(link_from_name(robot, link) for link in pair) for pair in DISABLED_COLLISIONS}
    printed_elements = []
    current_conf = initial_conf
    all_trajectories = []
    for i, print_traj in enumerate(trajectories):
        set_joint_positions(robot, movable_joints, current_conf)
        goal_conf = print_traj.path[0]
        obstacles = fixed_obstacles + [element_bodies[e] for e in printed_elements]
        path = plan_joint_motion(robot, movable_joints, goal_conf, obstacles=obstacles,
                                 self_collisions=True, disabled_collisions=disabled_collisions,
                                 weights=weights, resolutions=resolutions,
                                 restarts=5, iterations=50, smooth=100)
        if path is None:
            print('Failed to find a path!')
            return None
        motion_traj = MotionTrajectory(robot, movable_joints, path)
        print('{}) {}'.format(i, motion_traj))
        all_trajectories.append(motion_traj)
        current_conf = print_traj.path[-1]
        printed_elements.append(print_traj.element)
        all_trajectories.append(print_traj)
    # TODO: return to initial?
    return all_trajectories

def display_trajectories(ground_nodes, trajectories, time_step=0.05):
    if trajectories is None:
        return
    connect(use_gui=True)
    floor, robot = load_world()
    wait_for_interrupt()
    movable_joints = get_movable_joints(robot)
    #element_bodies = dict(zip(elements, create_elements(node_points, elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    connected = set(ground_nodes)
    for trajectory in trajectories:
        if isinstance(trajectory, PrintTrajectory):
            print(trajectory, trajectory.n1 in connected, trajectory.n2 in connected,
                  is_ground(trajectory.element, ground_nodes), len(trajectory.path))
            connected.add(trajectory.n2)
        #wait_for_interrupt()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []
        for conf in trajectory.path:
            set_joint_positions(robot, movable_joints, conf)
            if isinstance(trajectory, PrintTrajectory):
                current_point = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_NAME)))
                if last_point is not None:
                    color = (0, 0, 1) if is_ground(trajectory.element, ground_nodes) else (1, 0, 0)
                    handles.append(add_line(last_point, current_point, color=color))
                last_point = current_point
            wait_for_duration(time_step)
        #wait_for_interrupt()
    #user_input('Finish?')
    wait_for_interrupt()
    disconnect()

##################################################

def main(viewer=False):
    # TODO: setCollisionFilterGroupMask
    # TODO: only produce collisions rather than collision-free
    # TODO: return collisions using wild-stream functionality
    # TODO: handle movements between selected edges
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)

    # https://github.mit.edu/yijiangh/assembly-instances
    #read_minizinc_data(os.path.join(root_directory, 'print_data.txt'))
    #return

    # djmm_test_block | mars_bubble | sig_artopt-bunny | topopt-100 | topopt-205 | topopt-310 | voronoi
    elements, node_points, ground_nodes = load_extrusion('voronoi')

    node_order = list(range(len(node_points)))
    #np.random.shuffle(node_order)
    node_order = sorted(node_order, key=lambda n: node_points[n][2])
    elements = sorted(elements, key=lambda e: min(node_points[n][2] for n in e))

    #node_order = node_order[:100]
    ground_nodes = [n for n in ground_nodes if n in node_order]
    elements = [element for element in elements if all(n in node_order for n in element)]

    print('Nodes: {} | Ground: {} | Elements: {}'.format(
        len(node_points), len(ground_nodes), len(elements)))
    #plan = plan_sequence_test(node_points, elements, ground_nodes)

    connect(use_gui=viewer)
    floor, robot = load_world()
    obstacles = [floor]
    initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    #dump_body(robot)
    #if has_gui():
    #    draw_model(elements, node_points, ground_nodes)
    #    wait_for_interrupt('Continue?')

    #joint_weights = compute_joint_weights(robot, num=1000)

    #elements = elements[:5]
    #elements = elements[:10]
    #elements = elements[:25]
    #elements = elements[:35]
    #elements = elements[:50]
    #elements = elements[:75]
    #elements = elements[:100]
    #elements = elements[:150]
    #elements = elements[150:]

    # TODO: prune if it collides with any of its supports
    # TODO: prioritize choices that don't collide with too many edges
    # TODO: accumulate edges along trajectory

    #test_grasps(robot, node_points, elements)
    #test_print(robot, node_points, elements)
    #return

    #for element in elements:
    #    color = (0, 0, 1) if doubly_printable(element, node_points) else (1, 0, 0)
    #    draw_element(node_points, element, color=color)
    #wait_for_interrupt('Continue?')

    # TODO: topological sort
    #node = node_order[40]
    #node_neighbors = get_node_neighbors(elements)
    #for element in node_neighbors[node]:
    #    color = (0, 1, 0) if element_supports(element, node, node_points) else (1, 0, 0)
    #    draw_element(node_points, element, color)

    #element = elements[-1]
    #draw_element(node_points, element, (0, 1, 0))
    #incoming_edges, _ = neighbors_from_orders(get_supported_orders(elements, node_points))
    #supporters = []
    #retrace_supporters(element, incoming_edges, supporters)
    #for e in supporters:
    #    draw_element(node_points, e, (1, 0, 0))
    #wait_for_interrupt('Continue?')

    #for name, args in plan:
    #    n1, e, n2 = args
    #    draw_element(node_points, e)
    #    user_input('Continue?')
    #test_ik(robot, node_order, node_points)

    element_bodies = dict(zip(elements, create_elements(node_points, elements)))

    precompute = False
    if precompute:
        pr = cProfile.Profile()
        pr.enable()
        trajectories = sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes)
        pr.disable()
        pstats.Stats(pr).sort_stats('tottime').print_stats(10)
        user_input('Continue?')
    else:
        trajectories = []

    motions = False
    plan = plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes, trajectories=trajectories)
    if motions:
        plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan)
    disconnect()
    display_trajectories(ground_nodes, plan)

    # Collisions at the end points?


if __name__ == '__main__':
    main()