from operator import length_hint
from re import I
import numpy as np
from numpy.core.defchararray import center
from numpy.core.function_base import linspace
from numpy.core.shape_base import vstack
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
from scipy.spatial import distance as spatialdistance
from scipy.interpolate import interp1d
import math
# import quadprog
# import trajectory_planning_helpers as tph
# import helper_funcs_glob
# from pyclothoids import Clothoid
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.special import expit
import random
import time
from scipy.signal import find_peaks
# from mpl_toolkits import mplot3d

class Track_Filtering():
    """
        .. author::
            Birk Blumhoff (CURE Mannheim e.V.)

        .. description::
            Finding the local LAP out of a local Map. Optional one can use a local Path-Algorithem if it's not done by the underlaying MPC.

        .. features::
            :np.ndarray left_cones:     array of cones positions for left cones no matter whih color. Shape: (-1,2) -> [x, y]
            :np.ndarray left_right:     array of cones positions for right cones no matter whih color. Shape: (-1,2) -> [x, y]

        .. methodes::
            :__init__:
            :np.ndarray run:        executes whole local Planning Algorithem. Returns coordinates of LAP.
            :np.ndarray run:        executes whole local Planning Algorithem. Returns array of path.
            :bool preprocess:
            :bool delauney:         methode to calculate the 
            :bool fallback:         method to gernerate a array of midpoints if there are only cones with one color
            :bool cubicspline:
            :bool curvature:
            :bool normalvector:
            :bool lookahead:        

            :bool local_path:       Optimization methode for generating the raceline. Writes the result in class feature. Returns 'True' or 'False' if optimization fails.
            :bool plot:             Plot all available and produced data in several plots.
            :bool save:             Save the generated path in .csv-file.

        .. STATUS:: in Progress
    """

    def __init__(self,
                    vehicle_width=1.15,
                    track_width_min=3,
                    max_curvature=2/3):
        """
            .. description::
                methode for initialization local Planning Object.

            .. inputs::
                :float vehicle_width:       width of the vehicle, also called trackwidth. Default: 1.15 meter
                :float track_width_min:     minimum width of the racetrack, defines ba the rules. Default: 3 meter
                :string planner:            Argument wether to calculate the "LAP", the local "PATH" or th optimal path "OPTIMAL".

            .. outputs::
                :None:
        """
        # add position of car in local map to attributes
        self.origin = np.array([0.0, 0.0])
        self.vehicle_width = vehicle_width
        self.track_width_min = track_width_min
        self.max_curvature = max_curvature

        """
        self.map_kind = None
        self.left_cones = np.array([])
        self.right_cones = np.array([])
        self.right_cones_filtered = np.array([])
        self.left_cones_filtered = np.array([])
        self.midpoints = np.array([])
        self.midpoints_organized = np.array([])
        self.midpoints_unorganized = np.array([])
        """

    def run(self,
            left_cones: np.ndarray,
            right_cones: np.ndarray,
            print_debug: bool = False):
        """
            .. description::
                Methode to run local track filter. Inputs are left and right cones. Outputs are centerline of filtered track an legal area of the track. Algorithem follow 3 steps:
                    I. Find the midpoints of the track. Algo seperates between the seven cases listed below.
                    II. Interpolate and approximate centerline from found midpoints. Calculate normalvectors.
                    III. Defines the legal track width for each point and calcultes borderline on the right and left.
                For first step the Algo have to decide which kind of Midpoint generator should be used:
                    1. Only one cones:
                        1.1 left:                   DynamicWindow
                        1.2 right:                  DynamicWindow
                    2. Only one color:
                        2.1 left:                   BorderShift
                        2.2 right:                  BorderShift
                    3. Two colors:
                        3.1 One cone each:          GateFallback
                        3.2 One cone on one side:   BorderShift
                        3.2 All other:              Delauney

            .. inputs::
                :np.ndarray left_cones:     array of cones positions for left cones no matter whih color. Shape (-1,2): [x, y]
                :np.ndarray right_cones:    array of cones positions for right cones no matter whih color. Shape (-1,2): [x, y]
                :bool print_debug:          Indicator if debug messages should be printed

            .. outputs::
                :np.ndarray midpoints:      array containing the x an y position of each midpoint. Shape (-1,2): [x, y]
                :np.ndarray border_left     array containing x and y position of left border. Shape (-1,2): [x, y]
                :np.ndarray border_right    array containing x and y position of right border. Shape (-1,2): [x, y]
                :np.ndarray trackwidths     array containing left and right legal trackwidth for each midpoint. Shape (-1,2): [w_left, w_right]
                :np.ndarray normalvectors   array containing x and y coordination of normalvectors on centerline. Shape (-1,2); [x_n, y_n]

            .. STATUS::
                IN PROGRESS

            .. Notes::
                For case 3.2 the threshold should be modified. Also a other stepsize for interpolating and approximation could serve better results.
        """

        # check for unusable shape
        if (len(left_cones) != 0 and len(left_cones[0]) == 0):
            left_cones = np.array([])
        if (len(right_cones) != 0 and len(right_cones[0]) == 0):
            right_cones = np.array([])

        # add cones to class attributes
        self.left_cones = left_cones
        self.right_cones = right_cones
        self.print_debug = print_debug

        #####################################
        #                                   #
        #          Case Filtering           #
        #                                   #
        #####################################

        #####################################
        #       1.1 ONE LEFT CONE           #
        #####################################
        if ((len(self.right_cones) == 0) and (len(self.left_cones) == 1)):

            # run Dynamic Window
            if(self.DynamicWindow(side="ONE_LEFT") != True):

                #print error message
                if (self.print_debug != True):
                    print('1.1 ERROR: Dynamic Window failed.')
                return False

        #####################################
        #       1.2 ONE RIGHT CONE          #
        #####################################
        elif ((len(self.right_cones) == 1) and (len(self.left_cones) == 0)):

            # run Dynamic Window
            if(self.DynamicWindow(side="ONE_RIGHT") != True):

                #print error message
                if (self.print_debug != False):
                    print('1.2 ERROR: Dynamic Window failed.')
                return False

        #####################################
        #     2.1 MULTIPLE LEFT CONES       #
        #####################################
        elif ((len(self.right_cones) == 0) and (len(self.left_cones) != 0)):

            # run Bordershift
            if(self.BorderShift(side="LEFT") != True):

                #print error message
                if (self.print_debug != False):
                    print('2.1 ERROR: Bordershift failed.')
                return False

        #####################################
        #     2.2 MULTIPLE RIGHT CONES      #
        #####################################
        elif ((len(self.left_cones) == 0) and (len(self.right_cones) != 0)):

            # run Bordershift
            if(self.BorderShift(side="RIGHT") != True):

                #print error message
                if (self.print_debug != False):
                    print('2.2 ERROR: Bordershift failed.')
                return False

        #####################################
        #        3.1 ONE CONE EACH          #
        #####################################
        elif ((len(self.left_cones) == 1) and (len(self.right_cones) == 1)):

            # run Bordershift
            if(self.GateFallback() != True):

                #print error message
                if (self.print_debug != False):
                    print('2.2 ERROR: Gate Fallback failed.')
                return False

        #####################################
        #     3.2 ONE CONE ON ONE SIDE      #
        #####################################
        elif (len(self.right_cones) == 1):

            # run Bordershift
            if(self.BorderShift(side="LEFT") != True):

                #print error message
                if (self.print_debug != False):
                    print('3.2 ERROR: Bordershift failed.')
                return False

        elif (len(self.left_cones) == 1):

            # run Bordershift
            if(self.BorderShift(side="RIGHT") != True):

                #print error message
                if (self.print_debug != False):
                    print('3.2 ERROR: Bordershift failed.')
                return False

        #####################################
        #        3.3 MULTIPLE CONES         #
        #####################################
        elif ((len(self.left_cones) > 1) and (len(self.right_cones) > 1)):

            #run full pathplanner
            if (self.FullPathfilter() != True):
                # logging message
                if (self.print_debug != False):
                    print('3.3 ERROR: Full Pathfilter failed.')
                return False

        #####################################
        #              0 ERROR              #
        #####################################
        else:
            # print error message
            if (self.print_debug != False):
                print("0 ERROR: Youre car is blind. Stop the car!")

            #return empty arrays
            self.midpoints = np.array([])
            self.track_widths = np.array([])
            self.border_left = np.array([])
            self.border_right = np.array([])
            self.normalvectors = np.array([])

            return False

        return True



        """
        #####################################
        #                                   #
        #         TRACK PREPARATION         #
        #                                   #
        #####################################
        #self.drawEllipse()
        #self.drawClothoid([0,0], first=True)
        result = self.bestPath()
        self.drawClothoid(result)#, first=True)
        return True
        """

    def DynamicWindow(self,
                    side: str = "ORANGE"):
        """
            .. description::
                Method for midpoint generating if there is only one single cone detected. Calculates the normalvector on the support-vector to the cone. With the the minimum half track width one can estimate the midpoint. Returns coorsinates of midpoint.

            .. inputs::
                :string side:       Srtring indicating if cone is "ONE_RIGHT" or "ONE_LEFT".

            .. outputs::
                :bool status:       True if Dynamic Window works fine!

            .. STATUS::
                DONE!
        """
        try:
            # Variable for needed points
            self.midpoints = np.array([0, 0])
            self.normalvectors = np.array([0, 1])
            width = 0
            self.track_widths = np.array([width, width])

            # check if it's a left cone
            if(side == "ONE_LEFT"):
                direction = self.left_cones[0] - self.origin

                # normalvector on line to cone
                normalvector = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)

                # append to midpoints
                self.midpoints = np.append(self.midpoints, (self.left_cones[0] - self.track_width_min/2 * normalvector))

            # it's a right cone:
            elif (side == "ONE_RIGHT"):
                direction = self.right_cones[0] - self.origin

                # normalvector on line to cone
                normalvector = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)

                # append to midpoints
                self.midpoints = np.append(self.midpoints, (self.right_cones[0] + normalvector * self.track_width_min/2))

            # it's a orange cone
            else:
                # print error message
                if (self.print_debug != False):
                    print('ERROR: You see only one cone and its a god damn orange one. BRAKE!')

                #return empty arrays
                self.midpoints = np.array([])
                self.track_widths = np.array([])
                self.border_left = np.array([])
                self.border_right = np.array([])
                self.normalvectors = np.array([])

                return False

            # reshape midpoints array to useable shape
            self.midpoints = self.midpoints.reshape((-1,2))

            # calculate Normalvector on new centerline
            self.normalvectors = np.append(self.normalvectors, np.array([-self.midpoints[-1][1], self.midpoints[-1][0]]) / np.linalg.norm(self.midpoints[-1]))
            self.normalvectors = self.normalvectors.reshape((-1,2))

            # calculate trackwidth (left and right both the same) -> see notebook for documentation
            width = np.max([0, np.abs((np.linalg.norm(direction) * self.track_width_min/2) / np.linalg.norm(self.midpoints[-1]) - self.vehicle_width/2)])
            self.track_widths = np.append(self.track_widths, np.array([width, width]))
            self.track_widths = self.track_widths.reshape((-1,2))

            # calculate borderpoints
            self.border_left = self.midpoints + self.normalvectors * self.track_widths[:,0].reshape((-1,1))
            self.border_right = self.midpoints - self.normalvectors * self.track_widths[:,1].reshape((-1,1))

            return True
        except:
            #return empty arrays
            self.midpoints = np.array([])
            self.track_widths = np.array([])
            self.border_left = np.array([])
            self.border_right = np.array([])
            self.normalvectors = np.array([])

            return False

    def BorderShift(self,
                side: str,
                track_width: float = None):
        """
            .. description::
                Methode shifting trackborder points to the estimatet middle of the track. Resulting in new centerline and a legal trackwidth for each centerpoint

            .. inputs::
                :string side:               String indicating if cone is "ONE_RIGHT" or "ONE_LEFT".
                :float track_width:         describes the real track width. Distance from borderpoint to estimated midpoint. Default is 1.5 m (half min. track width).

                :np.ndarray border:         Array containing the position of every borderpoint. In the right order. Shape:(-1,2)
                :np.ndarray:                Array containing normalvectors of every borderpoint. Shape:(-1,2)
                :float track_width:         

            .. outputs::
                :bool status:       True if Dynamic Window works fine!

                :np.ndarray midpoints:      Array containing estimated midpoints. Shape:(-1,2)
                :np.ndarray track_widths:   Array containing legal track width to left and right side of centerpoint. Shape: (-1,2)
                :np.ndarray border_left:    Array containing legal track borders on the left side. Shape: (-1,2)
                :np.ndarray border_right:   Array containing legal track borders on the right side. Shape: (-1,2)

            .. STATUS::
                DONE
        """
        try:
            
            # check for Input
            if (track_width == None):
                track_width = self.track_width_min/2

            # check if left or right border given
            if (side == "LEFT"):

                # bring border points in true order
                points = self.generateTree(self.left_cones)
            elif (side == "RIGHT"):

                # bring border points in true order
                points = self.generateTree(self.right_cones)
            else:

                # print error message
                if (self.print_debug != False):
                    print('ERROR: You got a bunch of orange cones. Better stop the car!')
                return False

            # Get normalvectors of borderpoints (if Polyfit is choosen, points are smoothed)
            try:
                points, self.normalvectors = self.CubicSpline(points, factor=2)

            except:
                order = np.min([5, len(points) // 2 * 2 - 1])
                points, self.normalvectors = self.Polyfit(points, order=order)

            if (side == "LEFT"):

                # calculate new midpoints for left border cones
                self.midpoints = points - self.normalvectors * track_width
            else:

                # calculate new midpoints for right border cones
                self.midpoints = points + self.normalvectors * track_width

            # calculate track widths array
            legal_width = track_width - self.vehicle_width/2
            self.track_widths = np.full_like(self.midpoints, [legal_width, legal_width]).reshape((-1,2))

            # calculate borderpoints
            self.border_left = self.midpoints + self.normalvectors * self.track_widths[:,0].reshape((-1,1))
            self.border_right = self.midpoints - self.normalvectors * self.track_widths[:,1].reshape((-1,1))

            # append origin coordinates and normalvector
            self.midpoints = np.append(self.origin, self.midpoints).reshape((-1,2))
            self.normalvectors = np.append(np.array([0, 1]), self.normalvectors).reshape([-1,2])

            # append borders 
            self.border_left = np.append([0,0], self.border_left).reshape((-1,2))
            self.border_right = np.append([0,0], self.border_right).reshape((-1,2))
            width = np.array([0,0])
            self.track_widths = np.append(width, self.track_widths).reshape((-1,2))

            return True
        except:
            #return empty arrays
            self.midpoints = np.array([])
            self.track_widths = np.array([])
            self.border_left = np.array([])
            self.border_right = np.array([])
            self.normalvectors = np.array([])

            return False

    def GateFallback(self):
        """
            .. description::
                Methode shifting trackborder points to the estimatet middle of the track. Resulting in new centerline and a legal trackwidth for each centerpoint

            .. inputs::
                :None:

            .. outputs::
                :bool status:               Returns true if GateFallback finished succesfully!

            .. STATUS::
                DONE
        """

        try:
            # Variable for needed points
            self.midpoints = np.array([0, 0])
            self.normalvectors = np.array([0, 1])
            width = 0
            self.track_widths = np.array([width, width])

            # get centerpoint between cones
            self.midpoints = np.append(self.midpoints, self.getWaypointCoordinate(self.left_cones[0], self.right_cones[0])).reshape((-1,2))
            #length_of_shift = np.linalg.norm(self.left_cones[0] - self.midpoints[-1])

            # calculate Normalvector on new centerline
            self.normalvectors = np.append(self.normalvectors, np.array([-self.midpoints[-1][1], self.midpoints[-1][0]]) / np.linalg.norm(self.midpoints[-1]))
            self.normalvectors = self.normalvectors.reshape((-1,2))

            # calculate left and right trackwidth (both the same) -> see notebook for documentation
            width = np.max([0, (np.linalg.norm(self.left_cones[0] - self.midpoints[-1])-self.vehicle_width/2)])
            self.track_widths = np.append(self.track_widths, np.array([width, width])).reshape((-1,2))

            # calculate borderpoints
            self.border_left = self.midpoints + self.normalvectors * self.track_widths[:,0].reshape((-1,1))
            self.border_right = self.midpoints - self.normalvectors * self.track_widths[:,1].reshape((-1,1))

            return True
        except:

            #return empty arrays
            self.midpoints = np.array([])
            self.track_widths = np.array([])
            self.border_left = np.array([])
            self.border_right = np.array([])
            self.normalvectors = np.array([])

            return False

    def FullPathfilter(self):
        """
            .. description::
                Methode for filtering a whole local track.

            .. inputs::
                :None:

            .. outputs::
                :bool status:               Returns true if FullPathplanner finished succesfully!

            .. STATUS::
                IN PROGRESS
        """

        try:
            # call Delauney to get useable cones and midpoints od the track
            if (self.Delauney() != True):

                # print error message
                if (self.print_debug != False):
                    print('ERROR: Delauney failed.')

                #return empty arrays
                self.midpoints = np.array([])
                self.track_widths = np.array([])
                self.border_left = np.array([])
                self.border_right = np.array([])
                self.normalvectors = np.array([])

                return False

            # generate Tree of midpoints and cones
            self.left_cones = self.generateTree(self.left_cones)
            self.right_cones = self.generateTree(self.right_cones)
            self.midpoints = self.generateTree(self.midpoints)

            # get high resolution spline of left cones -> Doesn't work if array got less than 3 points
            if (len(self.left_cones) > 2):
                left_cones, left_normalvectors = self.CubicSpline(self.left_cones, factor=10)
            else:
                left_cones, left_normalvectors = self.Polyfit(self.left_cones, order=1, factor=10)

            # get high resolution spline of right cones -> Doesn't work if array got less than 3 points
            if (len(self.right_cones) > 2):
                right_cones, right_normalvectors = self.CubicSpline(self.right_cones, factor=10)
            else:
                right_cones, right_normalvectors = self.Polyfit(self.right_cones, order=1, factor=10)

            # Interpolation of centerline
            #try:
            #self.midpoints, self.normalvectors = self.CubicSpline(self.midpoints, factor=2)

            #except:
            order = np.min([5, len(self.midpoints) // 2 * 2 - 1])

            # Get normalvectors of borderpoints (and smoothe centerline)
            self.midpoints, self.normalvectors = self.Polyfit(self.midpoints, order=order, factor=1)

            # check curvature of centerline. If curvature is above threshold (1/3m) points will be deleted.
            i_kappa = np.argwhere(np.absolute(self.curvature_centerline) > self.max_curvature)
            #if (len(i_kappa > 0)):
            i_kappa = i_kappa[i_kappa != 0]

            # check if there are some to delete
            if (len(i_kappa) > 0):
                i_kappa = np.min(i_kappa)
                self.midpoints = self.midpoints[:i_kappa]
                self.normalvectors = self.normalvectors[:i_kappa]

            # find nearest left and right border point for trackwidth calculation
            self.track_widths = np.array([])
            for point in self.midpoints:

                # calculate distance from midpoint to ervery border point
                #distance_left = np.linalg.norm(left_cones - point) - self.vehicle_width/2
                #distance_right = np.linalg.norm(right_cones - point) - self.vehicle_width/2
                distance_left = np.max([0, np.min(np.sqrt((left_cones[:,0] - point[0])**2 + (left_cones[:,1] - point[1])**2) - self.vehicle_width/2)])
                distance_right = np.max([0, np.min(np.sqrt((right_cones[:,0] - point[0])**2 + (right_cones[:,1] - point[1])**2) - self.vehicle_width/2)])

                # append to trackwidth array
                self.track_widths = np.append(self.track_widths, [distance_left, distance_right])

            # resahpe trackwidth
            self.track_widths = self.track_widths.reshape((-1,2))

            # reshape array and append origin
            self.midpoints = np.append(self.origin, self.midpoints).reshape((-1,2))
            self.normalvectors = np.append(np.array([0,1]), self.normalvectors).reshape((-1,2))
            self.track_widths = np.append(np.array([0,0]), self.track_widths).reshape((-1,2))

            # calculate borderpoints
            self.border_left = self.midpoints + self.normalvectors * self.track_widths[:,0].reshape((-1,1))
            self.border_right = self.midpoints - self.normalvectors * self.track_widths[:,1].reshape((-1,1))

            return True
        except:

            #return empty arrays
            self.midpoints = np.array([])
            self.track_widths = np.array([])
            self.border_left = np.array([])
            self.border_right = np.array([])
            self.normalvectors = np.array([])

            return False

    def Delauney(self,
                hairpin=False):
        """
            .. description::
                Using delauney triangulation from scipy to filter the cones. Delauney-algorithem takes in a finite amount of cones and generates triangles under the term of maximizing the inner angle of all triangles. For further information see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
                Methode returns midpoints and corresponding left and right cones as class attributes. Have to be filtered by Tree Generator!

            .. inputs::
                :None:

            .. outputs::
                :bool status:       True if delauney works fine!

            .. STATUS::
                DONE -> Maybe one can finde a solution where the order of the cones are corresponding to those of the midpoints!
        """
        try:
            # Variable for needed points
            self.midpoints = np.array([])
            self.right_cones_filtered = np.array([])
            self.left_cones_filtered = np.array([])

            # Create array for x and y values of cones
            c = np.append(self.left_cones, self.right_cones)
            c = c.reshape(-1, 2)
            coordinates = c.tolist()
            x = c[:, 0]
            y = c[:, 1]

            # Delauney
            tri = Delaunay(coordinates)
            triangles = tri.simplices

            # Filter Points from Delauney Triangles
            self.viz_triangles = np.array([])
            self.viz_triangle = np.array([])

            # itterate through triangles
            for line in triangles:
                # point-coordinate of ervery cone
                element0 = coordinates[line[0]]
                element1 = coordinates[line[1]]
                element2 = coordinates[line[2]]
                self.viz_triangle = np.append(self.viz_triangle, np.array([element1, element0]))
                self.viz_triangle = np.append(self.viz_triangle, np.array([element2, element0]))
                self.viz_triangle = np.append(self.viz_triangle, np.array([element2, element1]))

                # check if first and second point are both right or both left
                if (element0 in self.right_cones.tolist()) and (element1 in self.right_cones.tolist()):
                    pass
                elif (element0 in self.left_cones.tolist()) and (element1 in self.left_cones.tolist()):
                    pass
                else:
                    self.midpoints = np.append(self.midpoints, self.getWaypointCoordinate(element0, element1))
                    self.viz_triangles = np.append(self.viz_triangles, np.array([element0, element1]))

                    # check which point is left and right
                    if (element0 in self.right_cones.tolist()):
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element0))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element1))
                    else:
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element1))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element0))


                # check if second and third point are both right or both left
                if (element1 in self.right_cones.tolist()) and (element2 in self.right_cones.tolist()):
                    pass
                elif (element1 in self.left_cones.tolist()) and (element2 in self.left_cones.tolist()):
                    pass
                else:
                    self.midpoints = np.append(self.midpoints, self.getWaypointCoordinate(element1, element2))
                    self.viz_triangles = np.append(self.viz_triangles, np.array([element1, element2]))

                    # check which point is left and right
                    if (element1 in self.right_cones.tolist()):
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element1))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element2))
                    else:
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element2))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element1))


                # check if first and third point are both right or both left
                if (element2 in self.right_cones.tolist()) and (element0 in self.right_cones.tolist()):
                    pass
                elif (element2 in self.left_cones.tolist()) and (element0 in self.left_cones.tolist()):
                    pass
                else:
                    self.midpoints = np.append(self.midpoints, self.getWaypointCoordinate(element2, element0))
                    self.viz_triangles = np.append(self.viz_triangles, np.array([element2, element0]))

                    # check which point is left and right
                    if (element0 in self.right_cones.tolist()):
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element0))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element2))
                    else:
                        self.right_cones_filtered = np.append(self.right_cones_filtered, np.array(element2))
                        self.left_cones_filtered = np.append(self.left_cones_filtered, np.array(element0))

            # Rebuilding arrays
            self.viz_triangles = self.viz_triangles.reshape((-1,2,2))
            self.viz_triangles = self.viz_triangle.reshape((-1,2,2))
            self.midpoints = np.reshape(self.midpoints, (-1, 2))  # Reshape to [[x,y], [x,y]] Form
            self.right_cones_filtered = np.reshape(self.right_cones_filtered, (-1, 2))  # Reshape to [[x,y], [x,y]] Form
            self.left_cones_filtered = np.reshape(self.left_cones_filtered, (-1, 2))  # Reshape to [[x,y], [x,y]] Form
            self.midpoints, double_indize = np.unique(self.midpoints, return_index=True, axis=0)  # Delete duplicated values

            #self.right_cones_filtered = np.delete(self.right_cones_filtered, double_indize,axis=0)  # Delete duplicated values
            self.right_cones_filtered = np.unique(self.right_cones_filtered,axis=0)  # Delete duplicated values

            #self.left_cones_filtered = np.delete(self.left_cones_filtered, double_indize, axis=0)  # Delete duplicated values
            self.left_cones_filtered = np.unique(self.left_cones_filtered, axis=0)  # Delete duplicated values

            plt.subplots(figsize=(8, 8))

            print(self.viz_triangle)
            for e in self.viz_triangle:
                plt.plot([e[0,0], e[1,0]], [e[0,1], e[1,1]], '-', color='gray')
            
            # plot vehicle
            plt.plot(self.origin[0], self.origin[1], 'o', color='black', label='car')
            # plot left cones
            if (len(self.left_cones) != 0):
                plt.plot(self.left_cones[:,0], self.left_cones[:,1], 'o', color='blue', label='left')

            # plot right cones
            if (len(self.right_cones) != 0):
                plt.plot(self.right_cones[:,0], self.right_cones[:,1], 'o', color='gold', label='right')

            #plt.plot(self.midpoints[:,0], self.midpoints[:,1], '.', color='black')

            plt.xlim(-1, 19)
            plt.ylim(-15, 5)
            plt.show()

            plt.subplots(figsize=(8, 8))
            for e in self.viz_triangles:
                plt.plot([e[0,0], e[1,0]], [e[0,1], e[1,1]], '-', color='gray')
            
            # plot vehicle
            plt.plot(self.origin[0], self.origin[1], 'o', color='black', label='car')
            # plot left cones
            if (len(self.left_cones) != 0):
                plt.plot(self.left_cones[:,0], self.left_cones[:,1], 'o', color='blue', label='left')

            # plot right cones
            if (len(self.right_cones) != 0):
                plt.plot(self.right_cones[:,0], self.right_cones[:,1], 'o', color='gold', label='right')

            plt.plot(self.midpoints[:,0], self.midpoints[:,1], '.', color='black')

            plt.xlim(-1, 19)
            plt.ylim(-15, 5)
            plt.show()

            return True
        except:
            return False

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

    def generateTree(self,
                    points):
        """
            .. description::
                Methode to bring points into right order. Starts with vehicle position and tries to find nearest point for each upcoming point.

            .. inputs::
                :np.ndarray points:             Array containing position of points user wants to order. Shape: (-1,2)

            .. outputs::
                :np.ndarray points_ordered:     Array containing positions of ordered points. Shape: (-1,2)

            .. STATUS::
                IN PROGRESS
        """

        # create one source and one result array for filtering of points. Origin will be deleted later!
        points_organized = np.copy(self.origin)
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

    def Polyfit(self,
                points,
                order=5,
                factor=1):
        """
            .. description::
                Methode calculating a Polyfit from numpy for a given set of points. Also calculates normalcevtors for each point

            .. inputs::
                :np.ndarray points:         Array containing the position of every cone. In the right order. Shape:(-1,2)
                :int order:                 Highest order of Polynomial. Needs to be an odd number.
                :int factor:                Scaling factor for approximation. 10 means 10 times more points in approximation.

            .. outputs::
                :np.ndarray points_new:     Array containing new 'better' fitted points. Shape:(-1,2)
                :np.ndarray normalvectors:  Array containing normalvector for each new point. Shape:(-1,2)

            .. STATUS::
                DONE
        """

        # create itterator and array for x and y
        i = np.arange(0, len(points), 1)
        i_new = np.linspace(0, len(points)-1, (len(points)-1)*factor+1, endpoint=True)
        x = points[:,0]
        y = points[:,1]

        # create Polynomial fit for x and y
        poly_x = np.poly1d(np.polyfit(i,x,deg=order))
        poly_y = np.poly1d(np.polyfit(i,y,deg=order))

        # calculate new points
        x_new = poly_x(i_new)
        y_new = poly_y(i_new)
        points_new = np.c_[x_new, y_new]

        # calculate normalvector: n= [-y', x']/np.sqrt(x'**2 + y'**2)
        x_1_new = poly_x.deriv()(i_new)
        y_1_new = poly_y.deriv()(i_new)
        normalvectors = [-y_1_new, x_1_new]/np.sqrt(x_1_new**2 + y_1_new**2)
        normalvectors = np.c_[normalvectors[0], normalvectors[1]]

        # calculate curvature
        x_2_new = poly_x.deriv().deriv()(i_new)
        y_2_new = poly_y.deriv().deriv()(i_new)
        self.curvature_centerline = (x_1_new*y_2_new - x_2_new*y_1_new) / np.sqrt(x_1_new**2+y_1_new**2)**3

        return points_new, normalvectors

    def CubicSpline(self,
                    points,
                    factor=1):
        """
            .. description::
                Methode for calculating a stepwise defined cubic spline. Second derivitive in start and end point are zero. Returns array of new points with the wished scaling factor. And one for the normalvectors.

            .. inputs::
                :np.ndarray points:         Array containing the position of every cone. In the right order. Shape:(-1,2)
                :int factor:                Scaling factor for approximation. 10 means 10 times more points in approximation.

            .. outputs::
                :np.ndarray points_new:     Array containing new 'better' fitted points. Shape:(-1,2)
                :np.ndarray normalvectors:  Array containing normalvector for each new point. Shape:(-1,2)

            .. STATUS::
                ToDo
        """

        # create step-itterator array
        i = np.arange(0, len(points), 1)
        i_new = np.linspace(0, len(points)-1, (len(points)-1)*factor+1, endpoint=True)

        # generate cubic Spline
        spline_x = CubicSpline(i, points[:,0], bc_type='natural')
        spline_y = CubicSpline(i, points[:,1], bc_type='natural')

        # calculate spline points
        x_new = spline_x(i_new)
        y_new = spline_y(i_new)
        points_new = np.c_[x_new, y_new]

        # calculate normalvector: n= [-y', x']/np.sqrt(x'**2 + y'**2)
        x_1_new = spline_x(i_new,1)
        y_1_new = spline_y(i_new,1)
        normalvectors = [-y_1_new, x_1_new]/np.sqrt(x_1_new**2 + y_1_new**2)
        normalvectors = np.c_[normalvectors[0], normalvectors[1]]

        return points_new, normalvectors

    def plot(self):
        plt.subplots(figsize=(8, 8))

        # plot vehicle
        plt.plot(self.origin[0], self.origin[1], 'o', color='black', label='car')
        # plot left cones
        if (len(self.left_cones) != 0):
            plt.plot(self.left_cones[:,0], self.left_cones[:,1], 'o', color='blue', label='left')

        # plot right cones
        if (len(self.right_cones) != 0):
            plt.plot(self.right_cones[:,0], self.right_cones[:,1], 'o', color='gold', label='right')

        # plot midpoints_organized
        if (len(self.midpoints) != 0):
            plt.plot(self.midpoints[:,0], self.midpoints[:,1], '.', color='gray', label='mid')

        # plot border points
        if (len(self.border_right) != 0):
            plt.plot(self.border_right[:,0], self.border_right[:,1], '-', color='red', label='right')
        if (len(self.border_left) != 0):
            plt.plot(self.border_left[:,0], self.border_left[:,1], '-', color='red', label='left')
        
        #if (len(self.raceline) != 0):
        #    plt.plot(self.raceline[:,0], self.raceline[:,1], '+-', color='green', label='race')
        
        #plt.plot(self.Ellipse[:,0], self.Ellipse[:,1], '-+')

        #n = len(self.midpoints)
        #plt.plot(self.clothoid.SampleXY(n)[0], self.clothoid.SampleXY(n)[1], '-+')
        

        # plot centerline??

        plt.xlim(-1, 19)
        plt.ylim(-15, 5)
        #plt.legend()
        plt.show()

        """
        plt.plot(self.i, self.rho_midpoints, color='red', label='midpoints')
        plt.grid()
        plt.legend()
        plt.show()
        """
