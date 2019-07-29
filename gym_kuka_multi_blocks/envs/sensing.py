import numpy as np
from math import pi
import itertools


def append_spherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[0]**2 + xyz[1]**2
    ptsnew[3] = np.sqrt(xy + xyz[2]**2)
    ptsnew[4] = np.arctan2(np.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    # ptsnew[4] = np.arctan2(xyz[2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[5] = np.arctan2(xyz[1], xyz[0])
    return ptsnew


def is_in_sector(cd, angles, radius, num_sectors):
    """
    :param array cd: An object's coordinates, [x, y, z, r, phi, theta]
    :param array angles: The array containing the left bounds of a sectors, [theta, phi]
    :param float radius: The radius of a sensing sphere, r > 0
    :param tuple num_sectors: The number of sectors, (theta, phi)
    :return bool: Confirms if the object is in the sector
    """

    r = cd[3]
    phi = cd[4]
    theta = cd[5]

    if angles[0] < theta <= angles[0] + 2*pi / num_sectors[0] and \
       angles[1] < phi <= angles[1] + pi / num_sectors[1] and \
       r <= radius:
        # print("Point (", cd, ") in the sector (", angles[0], angles[0] + 2*pi / num_sectors[0],"), (", angles[1], angles[1] + pi / num_sectors[1], ")")
        return True
    else:
        # print("Point (", cd[0:3], ") not in the sector", angles)
        return False


def sense(objects, max_radius, num_sectors=(4, 2)):

    sectors = list(itertools.product(np.linspace(-np.pi, np.pi - 2*np.pi / num_sectors[0], num_sectors[0]),
                                     np.linspace(0, np.pi - np.pi / num_sectors[1], num_sectors[1])))

    distance = [max_radius] * num_sectors[0] * num_sectors[1]

    for o in objects:
        o = append_spherical_np(np.array(o))
        for i, s in enumerate(sectors):
            if is_in_sector(o, angles=s, radius=max_radius, num_sectors=num_sectors):
                distance[i] = min(o[3], distance[i])

    return distance


# a = [[0, 1, 1], [0, 1, 0], [1, 0, 0]]
# print(sense(a, 2, (8, 4)))
