from math import dist
from re import I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
from scipy.spatial import distance as spatialdistance
# from velocity import Velocity
import time
import csv
from datetime import datetime
from filtering import Track_Filtering
import pandas as pd
# import helper_funcs_glob
# import trajectory_planning_helpers as tph

class optimizing_Handler():

    def __init__(self,
                vehicle_width=1.15,
                cone_width=0.285,
                track_width_min=3,
                max_curvature=1/3,
                max_track_length=500):
        self.vehicle_width = vehicle_width
        self.cone_width = cone_width
        self.track_width_min = track_width_min
        self.max_curvature = max_curvature
        self.max_track_length = max_track_length

    def filter(raw, filter_index=5):
        raw = np.append(raw[-filter_index+1:], raw).reshape((-1,2))
        raw = raw.reshape((-1,2))
        result = np.array([])
        for i in np.arange(filter_index-1, len(raw), 1):
            result = np.append(result, np.mean(raw[i-filter_index+1:i+1,1]))
        return result

    def filter_log(raw, filter_index=5):
        raw = np.append(raw[-filter_index+1:], raw).reshape((-1,2))
        #print(raw[-filter_index+1:])
        #raw = np.append(raw, raw[:filter_index-1])
        raw = raw.reshape((-1,2))
        result = np.array([])
        for i in np.arange(filter_index-1, len(raw), 1):
            result = np.append(result, np.mean(raw[i-filter_index+1:i+1,1]))
        return result

    def generateTree(self,
                    points,
                    start_point=None):
        """
            .. description::
                Methode to bring points into right order. Starts with vehicle position and tries to find nearest point for each upcoming point.
            .. inputs::
                :np.ndarray points:             Array containing position of points user wants to order. Shape: (-1,2)
                :np.ndarray midpoint_start:     Array containing 
    def run(self, left_cones, right_cones):
        self.left_cones = left_cones
        self.right_cones = right_cones

        midpoint_start = np.mean(self.orange_cones, axis=0)

        self.centerline = self.generateTree(self.centerline, midpoint_start)
        self.left_cones = self.generateTree(self.left_cones, midpoint_start)
        self.right_cones = self.generateTree(self.right_cones, midpoint_start)

        self.center_line= self.Delauney(self.left_cones, self.right_cones)position of the first point. Shape: [x, y]
            .. outputs::
                :np.ndarray points_ordered:     Array containing positions of ordered points. Shape: (-1,2)
            .. STATUS::
                IN PROGRESS
        """
        # create one source and one result array for filtering of points. Origin will be deleted later!
        points_organized = np.copy(start_point)
        self.points_unorganized = np.copy(points)
        # itterate trough all points
        while(len(self.points_unorganized) != 0):
            # call Nearest Neigbor Methode with last organized point
            nearest_neigbor = self.findNearestPoint([points_organized[-2], points_organized[-1]])
            points_organized = np.append(points_organized, nearest_neigbor)
        # reshape result array to [[x,y],[x,y], ... ]
        points_organized = points_organized.reshape((-1,2))
        # Delete duplicated values
        #points_organized = np.unique(points_organized, axis=0)
        # delet vehicle position
        points_organized = np.delete(points_organized, 0, 0)

        return points_organized

    def Distance(self,
                points):
        """
            .. description::
                Methode to calculate the Euclidean distance between an input array.

            .. inputs::
                :np.ndarray points:     Points to calculate the distance

            .. outputs::
                :float Distance:        Cummulated distance between all points. Starting wit zero for first point 0+d_1 for econd point and 0+d_1+d_2 for third point and so on.

            .. STATUS::
                DONE
        """
        distance = (points[1:]-points[:-1])
        distance = np.append(0, np.cumsum(np.sqrt(distance[:,0]**2 + distance[:,1]**2)))

        return distance

    def CubicSpline(self,
                points,
                factor=1,
                endpoint=True):
        """
            .. description::
                Methode to calculate a cubic spline of the input points. Also calculates the curvature of the spline

            .. inputs::
                :np.array points:           Array of points as input for cubicspline
                :int factor:                Factor for interpolation between points.
                :bool endpoint:             Wether to use the last point or not

            .. outputs::
                :np.ndarray curvature:      Array with curvature values of the spline interpolation. Shape: ((-1)) -> [[c_1, c_2, ...]]
                :np.ndarray normalvectors:  Array with normalvectors of the spline interpolation. Shape: ((-1,c)) -> [[x, y]]
                :np.ndarray new_points:     Array of new points from cubicspline. Shape: ((-1,2)) -> [[x, y]]

            .. STATUS::
                Done.
        """
        # create step-itterator array
        i = np.arange(0, len(points), 1)
        i_new = np.linspace(0, len(points)-1, (len(points)-1)*factor+1, endpoint=endpoint)

        # generate cubic Spline
        self.spline_x = CubicSpline(i, points[:,0], bc_type='natural')
        self.spline_y = CubicSpline(i, points[:,1], bc_type='natural')

        # get spline arrays
        x_new = self.spline_x(i_new)
        y_new = self.spline_y(i_new)
        x_1_new = self.spline_x(i_new,1)
        y_1_new = self.spline_y(i_new,1)
        x_2_new = self.spline_x(i_new,2)
        y_2_new = self.spline_y(i_new,2)

        # create resulting points
        new_points = np.c_[x_new, y_new]

        # calulate curvature: c = (x'*y'' - x''*y') / (x'**2 + y'**2)**3
        curvature = (x_1_new*y_2_new - x_2_new*y_1_new) / np.sqrt(x_1_new**2+y_1_new**2)**3

        # calculate normalvector: n= [-y', x']/np.sqrt(x'**2 + y'**2)
        normalvectors = [-y_1_new, x_1_new]/np.sqrt(x_1_new**2 + y_1_new**2)
        normalvectors = np.c_[normalvectors[0], normalvectors[1]]

        return curvature, normalvectors, new_points

    def run(self,
                left_cones=None,
                right_cones=None,
                orange_cones=None,
                hires_centerline=False):
            """
                .. description::
                    Main function of the gloabal track algorithm. Methode got following steps:
                        1. find midpoints with delauney traingulation
                        2. sort midpoints with generate tree algorithem (first point is the one nearest to start and finish line, econd one have to be calulated with the direction)
                        3. calculate trackwidth to left and right for each midpoint
                        4. call minicurve optimizer

                .. inputs::
                    :np.ndarray left_cones:     Positions of left (blue) cones with id=0. Shape:((-1,2))
                    :np.ndarray right_cones:    Positions of right (blue) cones with id=1. Shape:((-1,2))
                    :np.ndarray orange_cones:   Positions of start and end (orange) cones with id=2. Shape:((-1,2))

                .. outputs::
                    :np.ndarray centerline:     Positions of centerline points. Shape:((-1,2))
                    :np.ndarray raceline:       Positions of raceline points. Shape:((-1,2))
            """

            self.left_cones = left_cones
            self.right_cones = right_cones
            self.orange_cones = orange_cones

            # call delauny traingulation
            self.centerline = self.Delauney(self.left_cones, self.right_cones)

            # find the midpoint between the start/finish cones
            midpoint_start = np.mean(self.orange_cones, axis=0)

            # sort centerline, left cones and right cones
            self.centerline = self.generateTree(self.centerline, midpoint_start)
            self.left_cones = self.generateTree(self.left_cones, midpoint_start)
            self.right_cones = self.generateTree(self.right_cones, midpoint_start)

            # check if right direction is choosen
            normalvector_start = self.left_cones[0] - self.centerline[0]
            direction = np.array([normalvector_start[1], -normalvector_start[0]])
            direction_start = self.centerline[1] - self.centerline[0]

            unit_vector_1 = direction / np.linalg.norm(direction)
            unit_vector_2 = direction_start / np.linalg.norm(direction_start)
            dot_product = np.dot(unit_vector_2, unit_vector_1)
            angle = np.arccos(dot_product)
            if (angle > np.pi*1.5/4):
                self.centerline = np.append(self.centerline[0], np.flip(self.centerline, 0)[:-1]).reshape((-1,2))

            # create high resolution splines of left and right cones
            left_cones_spline = self.CubicSpline(np.append(self.left_cones, self.left_cones[0]).reshape((-1,2)), factor=20, endpoint=False)[2]
            right_cones_spline = self.CubicSpline(np.append(self.right_cones, self.right_cones[0]).reshape((-1,2)), factor=20, endpoint=False)[2]

            # find nearest left and right border point for trackwidth calculation
            self.trackwidths = np.array([])
            for point in self.centerline:

                # calculate distance from midpoint to ervery border point
                distance_left = np.max([0, np.min(np.sqrt((left_cones_spline[:,0] - point[0])**2 + (left_cones_spline[:,1] - point[1])**2) - self.vehicle_width/2 - self.cone_width/2)])
                distance_right = np.max([0, np.min(np.sqrt((right_cones_spline[:,0] - point[0])**2 + (right_cones_spline[:,1] - point[1])**2) - self.vehicle_width/2 - self.cone_width/2)])

                # append to trackwidth array
                self.trackwidths = np.append(self.trackwidths, [distance_right, distance_left])

            # resahpe trackwidth
            self.trackwidths = self.trackwidths.reshape((-1,2))

            # # calculate Raceline
            # self.raceline = self.getRaceline(self.centerline, self.trackwidths)

            # # get spline approximation of centerline
            # reftrack_interp, self.normalvectors = self.getRaceline(self.centerline, self.trackwidths, optimize=False, regression=2)
            # self.centerline = reftrack_interp[:,:2]
            # self.trackwidths = reftrack_interp[:,2:]

            # # get curvature of centerlinde and raceline
            # self.centerline_curvature, x, centerline_for_velocity = self.CubicSpline(self.centerline, factor=10)
            # self.raceline_curvature, x, raceline_for_velocity = self.CubicSpline(self.raceline, factor=10)
            # #self.centerline_curvature, x, self.centerline = self.CubicSpline(self.centerline, factor=1)
            # #self.raceline_curvature, x, self.raceline = self.CubicSpline(self.raceline, factor=1)

            # # get distance for centerline and raceline
            # self.centerline_distance = self.Distance(centerline_for_velocity)
            # self.raceline_distance = self.Distance(raceline_for_velocity)
            # #self.centerline_distance = self.Distance(self.centerline)
            # #self.raceline_distance = self.Distance(self.raceline)

            # return values
            return self.centerline,self.trackwidths  #self.raceline, self.raceline_distance, self.raceline_curvature, self.trackwidths


    def Delauney(self,
                    left_cones,
                    right_cones,
                    hairpin=False):
            """
                .. description::
                    Using delauney triangulation from scipy to filter the cones. Delauney-algorithem takes in a finite amount of cones and generates triangles under the term of maximizing the inner angle of all triangles. For further information see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
                    Methode returns midpoints and corresponding left and right cones as class attributes. Have to be filtered by Tree Generator!

                .. inputs::
                    :np.ndarray left_cones:     Positions of left (blue) cones with id=0. Shape:((-1,2))
                    :np.ndarray right_cones:    Positions of right (blue) cones with id=1. Shape:((-1,2))

                .. outputs::
                    :np.ndarray centerline:     Positions of centerpoints. Shape:((-1,2))

                .. STATUS::
                    DONE -> Maybe one can finde a solution where the order of the cones are corresponding to those of the midpoints!
            """
            # Variable for needed points
            self.centerline = np.array([])
            self.right_cones_filtered = np.array([])
            self.left_cones_filtered = np.array([])

            # Create array for x and y values of cones
            c = np.append(left_cones, right_cones).reshape((-1, 2))
            coordinates = c.tolist()
            x = c[:, 0]
            y = c[:, 1]

            # Delauney
            tri = Delaunay(coordinates)
            triangles = tri.simplices

            # Filter Points from Delauney Triangles
            self.viz_triangles = []

            # itterate through triangles
            for line in triangles:
                # point-coordinate of ervery cone
                element0 = coordinates[line[0]]
                element1 = coordinates[line[1]]
                element2 = coordinates[line[2]]

                # check if first and second point are both right or both left
                if (element0 in right_cones) and (element1 in right_cones):
                    pass
                elif (element0 in left_cones) and (element1 in left_cones):
                    pass
                else:
                    self.centerline = np.append(self.centerline, self.getWaypointCoordinate(element0, element1))
                    self.viz_triangles.append([element0, element1])

                    # check which point is left and right
                    if (element0 in right_cones):
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element0))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element1))
                    else:
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element1))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element0))


                # check if second and third point are both right or both left
                if (element1 in right_cones) and (element2 in right_cones):
                    pass
                elif (element1 in left_cones) and (element2 in left_cones):
                    pass
                else:
                    self.centerline = np.append(self.centerline, self.getWaypointCoordinate(element1, element2))
                    self.viz_triangles.append([element1, element2])

                    # check which point is left and right
                    if (element1 in right_cones):
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element1))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element2))
                    else:
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element2))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element1))


                # check if first and third point are both right or both left
                if (element2 in right_cones) and (element0 in right_cones):
                    pass
                elif (element2 in left_cones) and (element0 in left_cones):
                    pass
                else:
                    self.centerline = np.append(self.centerline, self.getWaypointCoordinate(element2, element0))
                    self.viz_triangles.append([element2, element0])

                    # check which point is left and right
                    if (element0 in right_cones):
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element0))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element2))
                    else:
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element2))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element0))

            # Rebuilding arrays
            self.centerline = np.reshape(self.centerline, (-1, 2))  # Reshape to [[x,y], [x,y]] Form
            self.right_cones = np.reshape(self.right_cones_filtered, (-1, 2))  # Reshape to [[x,y], [x,y]] Form
            self.left_cones = np.reshape(self.left_cones_filtered, (-1, 2))  # Reshape to [[x,y], [x,y]] Form
            self.centerline = np.unique(self.centerline, axis=0)  # Delete duplicated values

            #self.right_cones_filtered = np.delete(self.right_cones_filtered, double_indize,axis=0)  # Delete duplicated values
            self.right_cones = np.unique(self.right_cones,axis=0)  # Delete duplicated values

            #self.left_cones_filtered = np.delete(self.left_cones_filtered, double_indize, axis=0)  # Delete duplicated values
            self.left_cones = np.unique(self.left_cones, axis=0)  # Delete duplicated values

            # return values
            return self.centerline

    def read_Data(self, filepath):

        with open(filepath, 'r') as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)
                fileinput = (np.array(list(reader)).astype(float))

        self.distance = np.array([])

        # save Track for velocity-profile with Dimension: [[x_1,y_1], [x_2,y_2], ... , [x_n,y_n]]
        self.track = np.c_[fileinput[:,0], fileinput[:,1]]
        self.trackwidths = np.c_[fileinput[:,2], fileinput[:,3]]

        # scale to FSD Track
        self.distance = self.Distance(self.track)
        factor = self.distance[-1]/self.max_track_length
        self.track /= factor
        self.distance /= factor

        # scale trackwidth to FSD size
        min_width = np.min(self.trackwidths[:,0]+self.trackwidths[:,1])
        width_factor = min_width/self.track_width_min
        self.trackwidths /= width_factor
        print('max. Trackwidth: ', np.max(self.trackwidths[:,0]+self.trackwidths[:,1]))
        print('min. Trackwidth: ', np.min(self.trackwidths[:,0]+self.trackwidths[:,1]))

        # remove waypoint, so that distance between waypoints is around 3 meters
        last = 0
        track_FSD = np.array([self.track[0]])
        distance_FSD = np.array([self.distance[0]])
        trackwidths_FSD = np.array([self.trackwidths[0]])
        for i in range(1, len(self.distance)):
            if(self.distance[i]-self.distance[last] >= 3):
                track_FSD = np.append(track_FSD, self.track[i])
                distance_FSD = np.append(distance_FSD, self.distance[i])
                trackwidths_FSD = np.append(trackwidths_FSD, self.trackwidths[i])
                last = i

        self.track = np.reshape(track_FSD, (-1,2))
        self.distance = np.reshape(distance_FSD, (-1))
        self.trackwidths = np.reshape(trackwidths_FSD, (-1,2))

        self.left_cones, self.right_cones, self.orange_cones = self.Boundaries_Formula_E(self.track, self.trackwidths)

        # return values
        return self.left_cones, self.right_cones, self.orange_cones

    # def getRaceline(self,
    #                 centerline,
    #                 trackwidths,
    #                 optimize=True,
    #                 regression=120):
    #     """
    #         .. description::

    #         .. inputs::

    #         .. outputs::
    #     """

    #     # put all arrays together. Shape: [x_m, y_m, w_tr_right_m, w_tr_left_m]
    #     reftrack = np.c_[centerline, trackwidths]

    #     ### spline regression smooth options
    #     # k_reg:                        [-] order of B-Splines -> standard: 3
    #     # s_reg:                        [-] smoothing factor, range [1.0, 100.0]

    #     reg_smooth_opts={"k_reg": 3,
    #                     "s_reg": regression}

    #     ### stepsize options
    #     # stepsize_prep:               [m] used for linear interpolation before spline approximation
    #     # stepsize_reg:                [m] used for spline interpolation after spline approximation (stepsize during opt.)
    #     # stepsize_interp_after_opt:   [m] used for spline interpolation after optimization

    #     stepsize_opts={"stepsize_prep": 0.5,
    #                 "stepsize_reg": 1.5,
    #                 "stepsize_interp_after_opt": 1.0}

    #     # create dictonary
    #     pars={}
    #     pars["reg_smooth_opts"] = reg_smooth_opts
    #     pars["stepsize_opts"] = stepsize_opts

    #     #####################################
    #     #                                   #
    #     #       RACELINE OPTIMIZATION       #
    #     #                                   #
    #     #####################################

    #     # use TRACK PREPERATION from TUM
    #     reftrack_interp, normvec_normalized_interp, a_interp = \
    #     helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack,
    #                                                 reg_smooth_opts=pars["reg_smooth_opts"],
    #                                                 stepsize_opts=pars["stepsize_opts"],
    #                                                 debug=False,
    #                                                 min_width=None)[0:3]

    #     if (optimize != False):
    #         # call optimization from TUM
    #         alpha_opt = tph.opt_min_curv.opt_min_curv(reftrack=reftrack_interp,
    #                                             normvectors=normvec_normalized_interp,
    #                                             A=a_interp,
    #                                             kappa_bound=self.max_curvature,
    #                                             w_veh=0,
    #                                             print_debug=False,
    #                                             plot_debug=False)[0]

    #         self.raceline = reftrack_interp[:,0:2] + normvec_normalized_interp * alpha_opt.reshape((-1,1))

    #         # return values
    #         return self.raceline

    #     else:
    #         return reftrack_interp, normvec_normalized_interp


    def findNearestPoint(self,
                        origin):
        """
            .. description::
                Methode to find nearest point to last origin point. Deletes used points from points_unorgaized class attribute

            .. inputs::
                :np.ndarray origin:         Position of last known point. [x, y]

            .. outputs::
                :np.ndarray nearest_point:  Position of next ordered point.

            .. STATUS::
                IN PROGRESS
        """
        result = origin
        # check if it's the last point
        if(len(self.points_unorganized) == 1):
            result = self.points_unorganized[0]
            self.points_unorganized = np.delete(self.points_unorganized, 0, 0)
            return result

        else:
            # calculate distance to new origin point
            new_x = self.points_unorganized[:,0] - origin[0]
            new_y = self.points_unorganized[:,1] - origin[1]
            new_distance = np.sqrt(new_x**2 + new_y**2)

            # get index of points with minimum distance to new origin
            i_min = np.argmin(new_distance)

            # ToDo: maybe check if there are more than one nearest point. If true delete both points
            if(i_min.size == 1):
                # append nearest point to result array
                result = self.points_unorganized[i_min]

            # delete all used points out of source array
            self.points_unorganized = np.delete(self.points_unorganized, i_min, 0)

            return result


    def getWaypointCoordinate(self, p1, p2):
        """
            .. description::
                Calculates the midpoint of two incomming points.

            .. inputs::
                :list p1:               coordinates of first point [x, y].
                :list p2:               coordinates of second point [x, y].

            .. outputs::
                :np.ndarray midpoint    coordinates of midpoint [x, y]
        """

        midpoint = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
        return midpoint

    def Boundaries_Formula_E(self,
                            track,
                            trackwidths):
        """
            .. description::
                Calcultes the position of left, right and start- and finish-line cones.

            .. inputs::
                :np.ndarray track:          Array with coordinates of centerpoints. Shape: ((-1,2)) -> [[x, y]]
                :np.ndarray trackwidths:    Array with trackwidths to the left and right side from centerpoint. Shape: ((-1,2)) -> [[w_r, w_l]]

            .. outputs::
                :np.ndarray left_cones:     Positions of left (blue) cones with id=0. Shape:((-1,2))
                :np.ndarray right_cones:    Positions of right (blue) cones with id=1. Shape:((-1,2))
                :np.ndarray orange_cones:   Positions of start and end (orange) cones with id=2. Shape:((-1,2))
        """
        # call cubic spline to get the normalvectors of the track
        curvature, normalvectors, new_points = self.CubicSpline(track)

        # calculate the left and right cone positions
        left_cones = track + normalvectors * trackwidths[:,1].reshape((-1,1))
        right_cones = track - normalvectors * trackwidths[:,0].reshape((-1,1))

        # calculate 4 orange cones at start- and finish line
        start_direction = np.array([normalvectors[0,1], -normalvectors[0,0]])
        orange_cones = np.array([left_cones[0] + start_direction * 0.5,
                                right_cones[0] + start_direction * 0.5,
                                left_cones[0] - start_direction * 0.5,
                                right_cones[0] - start_direction * 0.5]).reshape((4,2))

        # return values
        return left_cones, right_cones, orange_cones

def read_input(file):
    df = pd.read_csv(file)
    right=[]
    left=[]
    orange=[]
    for enum, row in df.iterrows():
        if row['color']=='1':
            right.append((row['x'],row['y']))
        if row['color']=='2':
            left.append((row['x'],row['y']))
        if row['color']=='3':
            orange.append((row['x'],row['y']))
    return right, left, orange

if __name__ == '__main__':

    handler = optimizing_Handler()
    filepath=r"C:\Users\lustv\Downloads\global_tracks_batch_1\global_tracks_batch_1\track#8.csv"
    #left, right, orange = handler.read_Data(filepath='/workspace/as_ros/src/global_motion_planning/scripts/track#8.csv')
    right, left, orange=read_input(filepath)

    #raceline, raceline_distance, raceline_curvature, track_width = handler.run(left_cones=left, right_cones=right, orange_cones=orange)
    midpoints, track_width = handler.run(left_cones=left, right_cones=right, orange_cones=orange)

    # Raceline_Velocity = Velocity(g_x_max=1.5, g_y_max=1.5)

    # Raceline_Velocity.run(raceline, raceline_curvature, raceline_distance, v_initial=10.5, plot=False)

    #print('Centerline-Time:', Centerline_Velocity.Time,'sec')
    # print('Raceline-Time:', Raceline_Velocity.Time,'sec')

    #print('Centerline-Distance:', Global_Track.centerline_distance[-1],'m')
    # print('Raceline-Distance:', handler.raceline_distance[-1],'m')
    print((track_width))
    print((midpoints))
    plt.plot(midpoints[:,0], midpoints[:,1], '*')
    plt.show()
    print('smth')
