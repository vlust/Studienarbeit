"""
Created on Thu Oct 15 20:20:11 2020

@author: Karl Grossmann
"""

#import rospy
import numpy as np
from scipy.spatial import Delaunay  # used for Delauney Triangulation
import matplotlib.pyplot as plt
import math

# Parameters
step = 0.001  # configures the detail of the trajectory
origin = [0, 0]  # configures the origin point


class Node:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def __str__(self):
        return 'x: ' + str(self.x) + ' - y: ' + str(self.y) + ' - parent: ' + self.parent


#   Generate Filtered Delauney Grid
def get_waypoint_coordinate(x, y, p1, p2, draw = False):
    """
    Connect points
    """
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
    if draw is True:
        plt.plot([x1, x2], [y1, y2], 'k-.')
    return point


def generate_waypoints(left_cones, right_cones, draw):
    coordinates = np.append(left_cones, right_cones, axis=0)
    #print(coordinates)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    # print(x)
    # print(y)
    # Delauney
    tri = Delaunay(coordinates)
    triangles = tri.simplices
    waypoints = np.zeros((triangles.shape[0] * 2, 2))  # assumption: every triangle generates 2 waypoints

    # print(waypoints)
    no_new_waypoints = []
    viz_triangles = []


    # left_cones = [x for xs in left_cones for x in xs]
    # right_cones = [x for xs in right_cones for x in xs]
    for i in range(triangles.shape[0]):
        line = triangles[i]
        element0 = list(coordinates[line[0]])
        element1 = list(coordinates[line[1]])
        element2 = list(coordinates[line[2]])

        triangle_waypoints = []
        # if (element0 in left_cones):
        #     print('smth')

        if element0 in left_cones and element1 in left_cones and element2 in  left_cones or element0 in right_cones \
                and element1 in right_cones and element2 in right_cones:
            no_new_waypoints.append(i)
            no_new_waypoints.append(i + 1)
            continue

        if element0 in right_cones and element1 in right_cones or element0 in left_cones and element1 in left_cones:
            pass
        else:
            triangle_waypoints.append(get_waypoint_coordinate(x, y, line[0], line[1], draw))
            viz_triangles.append([element0, element1])

        if element1 in right_cones and element2 in right_cones or element1 in left_cones and element2 in left_cones:
            pass
        else:
            triangle_waypoints.append(get_waypoint_coordinate(x, y, line[1], line[2], draw))
            viz_triangles.append([element1, element2])

        if element2 in right_cones and element0 in right_cones or element2 in left_cones and element0 in left_cones:
            pass
        else:
            triangle_waypoints.append(get_waypoint_coordinate(x, y, line[2], line[0], draw))
            viz_triangles.append([element2, element0])

        waypoints[i] = triangle_waypoints[0]
        waypoints[i + 1] = triangle_waypoints[1]

    waypoints = np.delete(waypoints, no_new_waypoints, axis=0)  # delete triangles where no new waypoints were added
    waypoints = np.unique(waypoints, axis=0)  # Delete duplicated values

    return waypoints, viz_triangles


#   Rapid Tree Generator
def generate_tree(waypoints):
    start_node = Node(0, 0)
    node_list = [start_node]

    for i in range(waypoints.shape[0]):
        next_node = find_nearest_node(node_list[i], waypoints)
        if i > 0 and next_node[0] == node_list[i - 1].x and next_node[1] == node_list[i - 1].y:
            next_node = find_sec_nearest_node(node_list[i], next_node, waypoints)
            if i > 1 and next_node[0] == node_list[i - 2].x and next_node[1] == node_list[i - 2].y:
                break
        new_node = Node(next_node[0], next_node[1])
        new_node.parent = node_list[i]
        node_list.append(new_node)

    return node_list


def find_nearest_node(node, waypoints):
    index = 0
    dif = 20000
    for idx, val in enumerate(waypoints):
        if val[0] == node.x and val[1] == val[1]:
            continue
        dist = calculate_distance(node.x, node.y, val[0], val[1])
        if dist < dif:
            dif = dist
            index = idx
        else:
            continue
    return waypoints[index]


def find_sec_nearest_node(node, parent_node, waypoints):
    index = 0
    dif = 20000
    for idx, val in enumerate(waypoints):
        if val[0] == node.x and val[1] == val[1]:
            continue
        if val[0] == parent_node[0] and val[1] == parent_node[1]:
            continue
        dist = calculate_distance(node.x, node.y, val[0], val[1])
        if dist < dif:
            dif = dist
            index = idx
        else:
            continue
    return waypoints[index]


def calculate_distance(x1, y1, x2, y2):
    dx = (x1 - x2) ** 2
    dy = (y1 - y2) ** 2
    distance = math.sqrt((dx + dy))
    return distance


def plot_tree(node_list):
    for i in range(len(node_list)):
        if i == len(node_list) - 1:
            continue
        x_values = [node_list[i].x, node_list[i + 1].x]
        y_values = [node_list[i].y, node_list[i + 1].y]
        plt.plot(x_values, y_values)


#   Trajectory Generation
def linear_curve(p1, p2):
    """
    Linear Curve connecting 2 points
    :param p1: first point
    :param p2: second point
    :return: array of point cloud defining trajectory
    """
    spline = []
    for i in np.arange(0, 1, step):
        quotation_x = p1[0] + i * p2[0]
        quotation_y = p1[1] + i * p2[1]
        spline.append([quotation_x, quotation_y])

    return spline


def quadratic_bezier(p1, p2, p3):
    """
    Quadratic Bezier Curve connecting 3 points
    :param p1: first point
    :param p2: second point
    :param p3: third point
    :return: array of point cloud defining trajectory
    """
    spline = []
    for i in np.arange(0, 1, step):
        quotation_x = pow(1 - i, 2) * p1[0] + (1 - i) * 2 * i * p2[0] + i * i * p3[0]
        quotation_y = pow(1 - i, 2) * p1[1] + (1 - i) * 2 * i * p2[1] + i * i * p3[1]
        spline.append([quotation_x, quotation_y])

    return spline


def cubic_bezier(p1, p2, p3, p4):
    """
    Cubic Bezier curve connecting 4 points
    :param p1: first point
    :param p2: second point
    :param p3: third point
    :param p4: fourth point
    :return: array of point cloud defining trajectory
    """
    spline = []
    for i in np.arange(0, 1, step):
        quotation_x = pow(1 - i, 3) * p1[0] + pow(1 - i, 2) * 3 * i * p2[0] + (1 - i) * 3 * i * i * p3[0] + i * i * i * \
                      p4[0]
        quotation_y = pow(1 - i, 3) * p1[1] + pow(1 - i, 2) * 3 * i * p2[1] + (1 - i) * 3 * i * i * p3[1] + i * i * i * \
                      p4[1]
        spline.append([quotation_x, quotation_y])

    return spline


def catmull_spline(waypoints):
    """
    Catmull-Rom curve connecting more than 4 points
    :param waypoints: array of points
    :return: array of point cloud defining trajectory
    """
    spline = []
    for i in range(len(waypoints) - 2):
        if i == 0:
            spline.append(catmull_section(np.array([0, 0]), np.array(waypoints[i]), np.array(waypoints[i + 1]),
                                          np.array(waypoints[i + 2])))
        else:
            spline.append(
                catmull_section(np.array(waypoints[i - 1]), np.array(waypoints[i]), np.array(waypoints[i + 1]),
                                np.array(waypoints[i + 2])))

    spline = np.reshape(spline, (-1, 2))
    return spline


def catmull_section(p0, p1, p2, p3):
    """
    :param p0: first point
    :param p1: second point
    :param p2: third point
    :param p3: fourth point
    :return: array of point cloud defining the catmull spline section
    """
    arr = []
    for t in np.arange(0.0, 1.0, 0.01):
        # Formula for Point on Catmull-Rom Spline
        # dynamic parameter: t (0.0 =< t <= 1.0)
        s1 = (2 * p1)
        s2 = (-p0 + p2) * t
        s3 = (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
        s4 = (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
        new_point = 0.5 * (s1 + s2 + s3 + s4)
        arr.append(new_point)

    return arr


def generate_trajectory(waypoints):
    """
    :param waypoints: array of points
    :return: trajectory out of those points
    """
    num = len(waypoints)
    if num == 0 or num == 1:
        #rospy.logwarn('[Planning] There are no waypoints for calculation.')
        return 0
    elif num == 2:
        return linear_curve(waypoints[0], waypoints[1])
    elif num == 3:
        return quadratic_bezier(waypoints[0], waypoints[1], waypoints[2])
    elif num == 4:
        return cubic_bezier(waypoints[0], waypoints[1], waypoints[2], waypoints[3])
    elif num >= 5:
        return catmull_spline(waypoints)


#  Search for absolute distance to point
def find_point_in_distance(array, distance, deviation):
    """
    :param array: array of points
    :param distance: distance where the point should be located
    :param deviation: deviation
    :return: point in given distance
    """
    for idx in range(len(array)):
        dist = calculate_distance(0, 0, array[idx][0], array[idx][1])
        if (distance - deviation) < dist < (distance + deviation):
            return array[idx]


#   Trajectory Generation for situations where only one color of cones is available
def sort_by_proximity(array, origin):
    """
    :param array: array of points
    :param origin: origin point
    :return: array of points sorted by proximity to (0, 0)
    """
    cones_proximity_array = []
    for i in range(array.shape[0]):
        cones_proximity_array.append((i, calculate_distance(origin[0], origin[1], array[i, 0], array[i, 1])))

    # Sort created array by proximity
    cones_proximity_array = sorted(cones_proximity_array, key=lambda proximity: proximity[1])
    sorted_array = []
    for i in range(0, len(cones_proximity_array)):
        index = cones_proximity_array[i][0]
        sorted_array.append(array[index])

    return sorted_array


def shift_points_by_vector(array, vector):
    """
    :param array: array of points
    :param vector: vector by which the points should be shifted
    :return: array of point cloud defining trajectory
    """
    new_array = []
    for i in range(0, len(array)):
        x_coordinate = array[i][0] - vector[0]
        y_coordinate = array[i][1] - vector[1]
        new_array.append([x_coordinate, y_coordinate])
    return new_array


def generate_trajectory_without_cones(array, origin):
    """
    :param array: array of points
    :param origin: (0, 0)
    :return: array of point cloud defining trajectory
    """
    sorted_points = sort_by_proximity(array, origin)
    shifted_points = shift_points_by_vector(sorted_points, sorted_points[0])
    trajectory = generate_trajectory(shifted_points)
    return trajectory


#   Start of Application
def planning_main(left_cones, right_cones, radius, plot):
    if len(left_cones) == 0 and len(right_cones) != 0:
        spline = generate_trajectory_without_cones(right_cones, origin)
        viz_triangles = []
    elif len(right_cones) == 0 and len(left_cones) != 0:
        spline = generate_trajectory_without_cones(left_cones, origin)
        viz_triangles = []
    else:
        # Search all Waypoints
        waypoints, viz_triangles = generate_waypoints(left_cones, right_cones, plot)
        # Build Tree
        tree_list = generate_tree(waypoints)
        print(tree_list)
        # Build Catmull-Rom Spline
        points_for_catmull = []
        for i in range(len(tree_list)):
            points_for_catmull.append([tree_list[i].x, tree_list[i].y])
        # Generate Spline
        spline = generate_trajectory(np.array(points_for_catmull))

    if plot:
        for point in left_cones:
            plt.plot(point[0], point[1], 'yo', label='Left Cones')  # plot left cones using yellow circle markers
        for point in right_cones:
            plt.plot(point[0], point[1], 'bo', label='Right Cones')  # plot right cones using blue circle markers
        for point in waypoints:
            plt.plot(point[0], point[1], 'ro', label='Waypoints')
        plot_tree(tree_list)  # Plot Tree of Rapid Tree Generator
        spline_np = np.array(spline)
        spline_np = spline_np.reshape(-1, 2)
        plt.plot(spline_np[:, 0], spline_np[:, 1], color='b')  # Plot Catmull-Rom Spline

        # Plot Circle
        theta = np.linspace(0, 2 * np.pi, 100)
        a = radius * np.cos(theta)
        b = radius * np.sin(theta)
        #plt.plot(a, b)
        plt.gca().set_aspect('equal', adjustable='box')
        # Draw Lookahead Point
        # plt.plot(lookahead[0], lookahead[1], 'go', label='Lookahead Point')
        # Draw Line to Lookahead
        # plt.plot([0, lookahead[0]], [0, lookahead[1]], 'g')
        plt.show()

    return viz_triangles, spline