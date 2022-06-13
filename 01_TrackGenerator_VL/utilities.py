import numpy as np
from random import uniform


def get_normal_vector(vec):

    return (-vec[1], vec[0])


def get_tangent_angle(xys):
    sx = xys[-2][0]
    sy = xys[-2][1]
    ex = xys[-1][0]
    ey = xys[-1][1]

    return np.arctan2((ey - sy), (ex - sx))


def magnitude(vec):
    """Calculates magnitude of (x,y) tuple as if it were a vector"""
    (a, b) = vec
    return np.sqrt(a * a + b * b)


def subtract_vectors(a, b):
    """Given two tuples a and b, find a-b"""
    return (a[0] - b[0], a[1] - b[1])


def get_distance(a, b):
    """Returns the distance between two points"""
    return magnitude(subtract_vectors(a, b))


def get_tangent_vector(xys):
    angle = get_tangent_angle(xys)
    return (np.cos(angle), np.sin(angle))


def add_vectors(a, b):
    """Given two tuples a and b, add them together"""
    return (a[0] + b[0], a[1] + b[1])


def scale_vector(vec, scale):
    """Multiplies vec by scale"""
    return (vec[0] * scale, vec[1] * scale)


def normalize_vec(vec):
    """
    Calculates unit vector pointing in the direction
    that the input tuple would be if it were a vector
    """
    (a, b) = vec
    mag = np.sqrt(a * a + b * b)
    return (a / mag, b / mag)


def rad_to_deg(rad):
    """
    Calculates degree to radiant angle
    """
    deg = rad*180/np.pi
    return deg


def deg_to_rad(deg):
    """
    Calculates radiant to degree angle
    """
    rad = np.pi*deg/180
    return rad

def cap_angle(ang):
        """Returns angle between 0 and 2*math.pi"""
        if ang < 0:
                return cap_angle(ang + 2 * np.pi)
        elif ang >= 2*np.pi:
                return cap_angle(ang - 2 * np.pi)
        return ang

def calculate_tangent_angle(xys):
    """
    Calculate direction of outgoing tangent of a set of points
    Is an angle!
    """
    sx = xys[-2][0]
    sy = xys[-2][1]
    ex = xys[-1][0]
    ey = xys[-1][1]
    return np.arctan2((ey - sy), (ex - sx))


def calculate_tangent_vector(xys):
    angle = calculate_tangent_angle(xys)
    return (np.cos(angle), np.sin(angle))


def orientation(p, q, r):
    """
    to find the orientation of an ordered triplet (p,q,r)
    function returns the following values:
    0 : Collinear points
    1 : Clockwise points
    2 : Counterclockwise
    """
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - \
        (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0


def onSegment(p, q, r):
    """
    Given three collinear points p, q, r, the function checks if point q lies on line segment 'pr'
    """
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False
