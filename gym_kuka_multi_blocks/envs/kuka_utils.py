"""
Utilities for Kuka environment
"""
import pybullet as p
import math



def inverse_kinematics(robot, link, pose, max_iterations=200, tolerance=1e-3):
    """
    Calculate inverse kinematics

    :param robot:
    :param link:
    :param pose:
    :param max_iterations:
    :param tolerance:
    :return:
    """
    # lower limits for null space
    ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    # upper limits for null space
    ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    # joint ranges for null space
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    # restposes for null space
    rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # joint damping coefficents
    jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
          0.00001, 0.00001, 0.00001]


    (target_pos, target_orn) = pose
    #movable_joints = get_movable_joints(robot)
    kinematic_conf = p.calculateInverseKinematics(robot, link, target_pos, target_orn,
                                                  ll, ul, jr, rp)

    if (kinematic_conf is None) or any(map(math.isnan, kinematic_conf)):
        return None
    #set_joint_positions(robot, movable_joints, kinematic_conf)
    return kinematic_conf
    #return (target_pos, target_orn)
