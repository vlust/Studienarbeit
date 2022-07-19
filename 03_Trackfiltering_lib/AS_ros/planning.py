import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
from scipy.spatial import distance as spatialdistance
from scipy.interpolate import interp1d
import math
from pyclothoids import Clothoid
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.special import expit
import random
import time
from scipy.signal import find_peaks
from mpl_toolkits import mplot3d
from planning.velocity import Velocity

class Planning():
    """
        .. author::
            Birk Blumhoff (CURE Mannheim e.V.)

        .. description::
            Finding the time or curvature optimal path in a local filtered map. Until now only the 'ClothoidTime' optimizer runs stable.

        .. methodes::                                   Only used methodes are described
            :__init__:
            :np.ndarray run:                            executes whole local Planning Algorithem. Returns array of path, velocity and accelerations.
            :np.ndarray bestClothoid_by_heuristic:
            :float Clothoid_by_heuristic:
            :float cost_by_distance:
            :float cost_by_distance_2:
            :np.ndarray bestClothoid_by_time:           Starts optimization of clothoid by racetime
            :float Clothoid_by_time:                    Calculates the costfunction of one specific clothoid
            :float collision_by_time:                   Calculates the penaltys of the current clothoid because of collisions with the track boundarie
            :np.ndarray minitime:
            :float minimizer:
            :np.ndarray CubicSpline:                    Calculates a piecewise CubicSpline of a given set of points. Returns also the normalvectors and the curvature
            :np.ndarray Distance:                       Calculates the piecewise accumulated path distance.
            :np.ndarray Curvature:
            :bool MidpointsByCurvature:
            :None plot:                                 Plots the result of the local path planning

        .. STATUS:: in Progress
    """

    def __init__(self,
                vehicle_width=1.15,
                track_width_min=3,
                max_curvature=1/3,
                farest_point=10):
        """
            .. description::
                methode for initialization local Planning Object.

            .. inputs::
                :float vehicle_width:       width of the vehicle, also called trackwidth. Default: 1.15 meter
                :float track_width_min:     minimum width of the racetrack, defines ba the rules. Default: 3 meter
                :float max_curvature:       Max value of curvature, defined by rules, pathe-points with that value will not be taken into account.
                :int farest_point:          Indize of farest midpoint used for optimization. Keeps the optiization time low.

            .. outputs::
                :None:
        """

        # safe class attributes
        self.origin = np.array([0.0, 0.0])
        self.vehicle_width = vehicle_width
        self.track_width_min = track_width_min
        self.max_curvature = max_curvature
        self.farest_point = farest_point

        # initialize velocity_profile class
        self.My_Velocity = Velocity()
        self.path = np.array([])

    def run(self,
            midpoints: np.ndarray,
            border_left: np.ndarray,
            border_right: np.ndarray,
            track_widths: np.ndarray,
            normalvectors: np.ndarray,
            v_initial: float = None,
            v_end: float = None,
            optimizer: str = 'ClothoidTime',
            print_debug: bool = False):
        """
            .. description::
                Methode to run the local path planner. User can choose between 3 optimizers:
                    + 'ClothoidHeuristic'       Optimizes the local oath of the clothoid by a heuristic costfunction
                    + 'ClothoidTime'            Minimizes racetime along the clothoid and takes track boundaries into account. Best methode yet.
                    + 'MinTime'                 Minimizes the racetime by shifting each midpoint within the boundaries. High computational time.

            .. inputs::
                :np.ndarray midpoints:          Local coordinates of midpoints. Shape: ((-1,2)) -> [[x, y]]
                :np.ndarray border_left:        Local coordinates of left border point. Bound the legal track. Shape: ((-1,2)) -> [[x, y]]
                :np.ndarray border_right:       Local coordinates of right border point. Bound the legal track. Shape: ((-1,2)) -> [[x, y]]
                :np.ndarray track_widths:       Float values of trackwidth to left and right. Real (not legal) trackwidth. Shape: ((-1,2)) -> [[w_left, w_right]]
                :np.ndarray normalvectors:      Normalvectors on the centerline. Shape: ((-1,2)) -> [[x, y]]
                :float v_initaial:              Initial velocity of vehicle. Used in 2nd and 3rd optimizer. Have to be calculated from wheelspeedsensors.
                :float v_end:                   Velocity at the end of the local path. Used in 2nd and 3rd optimizer. Should be very low for safety reasons.
                :str optimizer:                 Indicator to choose the path optimization methode: 'ClothoidHeuristic', 'ClothoidTime', 'MiniTime'
                :bool print_debug:              Attribute wether to print debug messages or not.

            .. outputs::
                :np.ndarray Path:               Array of the gernerated local Path. Shape: ((-1,2))

            .. STATUS::
                IN PROGRESS
        """

        # safe class attributes
        self.midpoints = midpoints
        self.border_left = border_left
        self.border_right = border_right
        self.track_widths = track_widths
        self.normalvectors = normalvectors
        self.v_initial = v_initial
        self.v_end = v_end

        # start timer: if time of 1/30Hz is elapsed the centerline should be taken into account for the local path
        time_1 = time.time()

        # try optimization
        """
                Clothoide Heuristic optimization: Used a heuristic costfunction to optimize clothoid.
        """
        if (optimizer == 'ClothoidHeuristic'):

            # optimize clothoid by heuristic
            result = self.bestClothoid_by_heuristic()
            self.Clothoid_by_heuristic(result)

            """
                    Clothoide Time optimization: Uses Velocity-Profile to minimize racetime along the clothoide. Cares abou Trackboundaries by additional time.
            """
        elif (optimizer == 'ClothoidTime'):

            # optimize clothoid by time
            optimization_result = self.bestClothoid_by_time('SLSQP')

            """
                Time optimization: Uses Velocity-Profile to minimize racetime along the track. Midpoints can be shifted within left and right track boundaries. High computing time.
            """
        elif (optimizer == 'MiniTime'):
            result = self.minitime()
            self.path = self.midpoints_for_optimization + self.normalvectors_for_optimization*result.reshape((-1,1))

        else:
            # time.time() - time_1 > 1/30

            # set midpoints to new raceline
            curvature, normalvectors, self.path = self.CubicSpline(self.midpoints)

            # get distance of path
            self.distance = self.Distance(self.path)

            # call velocity profile to get velocity and accelerations
            self.velocity, self.acceleration_longitudinal, self.acceleration_lateral = self.My_Velocity.run(Path=self.path,
                                                                                                            Curvature=curvature,
                                                                                                            Distance=self.distance,
                                                                                                            v_initial=self.v_initial,
                                                                                                            v_end= self.v_end)

            if (print_debug != False):
                print('ERROR: Optimization Timeout.')

        # print debug_messges
        fps = 1/round(time.time() - time_1, 5)
        if (fps < 30 and print_debug != False):
            print('TIME:',str(round(time.time() - time_1,5)), 'FPS:',1/round(time.time() - time_1,5) )

        return self.path, self.velocity, self.acceleration_longitudinal, self.acceleration_lateral

    def bestClothoid_by_heuristic(self):
        """
            .. description::
                Methode to calculate the best clothoide. Uses scipy minimize to get the th BEST path, using:
                    - Squared sum of curvature          Reason: a minmimum curve path will be driven with the maximumm velocity
                    - Length                            Reason: a short track is faster than a long one
                    - distance                          Reason: a path outside of the legal track should be vorbidden

            .. inputs::
                :None:

            .. outputs::
                :np.ndarray result:     kappa and beta with lowest cost
        """
        # set boundaries for arguments
        i = np.min([self.farest_point, len(self.midpoints)-1])
        #bounds = ((-self.track_widths[i,1], self.track_widths[i,0]),(-np.pi/2,np.pi/2),)
        #bounds = ((0,0),(-np.pi/2,np.pi/2),)
        bounds = ((-self.track_widths[i,1], self.track_widths[i,0]),(0,0),)

        # create minimizer arguments
        minimizer_kwargs = {"method":"SLSQP","bounds":bounds}
        
        # minimize costfunction
        #result = basinhopping(self.drawClothoid, x0=[0,0], minimizer_kwargs=minimizer_kwargs, niter=10)
        #result = minimize(self.drawClothoid, x0=[random.uniform(-self.track_widths[-1,1], self.track_widths[-1,0]), random.uniform(-np.pi/4,np.pi/4)], method='SLSQP', bounds=bounds)
        result = minimize(self.Clothoid_by_heuristic, x0=[0,0], method='SLSQP', bounds=bounds)

        return result.x

    def Clothoid_by_heuristic(self, x0):
        """
            .. description::
                Methode to draw a clothoid with given start-, end-point and start-, end-tangente. Calculates the cost of the created clothoid:
                    - Squared sum of curvature          Reason: a minmimum curve path will be driven with the maximumm velocity
                    - Length                            Reason: a short track is faster than a long one
                    - hit                               Reason: a path outside of the legal track should be vorbidden
                Saves the clothoid as class attribute.

            .. inputs::
                :np.ndarray x0:     Deviation along the normalvector of the last midpoint (beta), defines the new endposition. Deviation from endpoint-tangente, difing the new end-tangente.

            .. outputs::
                : float:            cost of that specific clothoid (will be minimized by the optimizer)
        """
        # get deviation for endpoint and endtangente from input
        beta = x0[0]
        kappa = x0[1]

        # limit max number of points
        i = np.min([self.farest_point, len(self.midpoints)-1])

        # get endpoint and endtangente
        point = self.midpoints[i] + self.normalvectors[i]*beta
        normal = self.normalvectors[i]
        tangent = np.array([normal[1], -normal[0]])
        psi = np.arctan2(tangent[1], tangent[0])+kappa

        # create Clothoid
        self.clothoid_by_heuristic = Clothoid.G1Hermite(0,0,0, point[0], point[1], psi)

        # Calculate squared curvature: (x_1*y_2 - x_2*y_1) / np.sqrt(x_1**2+y_1**2)**3
        curv_squared = 0
        for l in np.linspace(0,self.clothoid_by_heuristic.length, len(self.midpoints)):
            curv_squared += (self.clothoid_by_heuristic.XD(l)*self.clothoid_by_heuristic.YDD(l) - self.clothoid_by_heuristic.XDD(l)*self.clothoid_by_heuristic.YD(l))**2/(self.clothoid_by_heuristic.XD(l)**2 + self.clothoid_by_heuristic.YD(l)**2)

        # calculate length
        length = self.clothoid_by_heuristic.length

        # calculate hits
        distance = self.cost_by_distance(self.clothoid_by_heuristic)

        # return Costfunction
        return curv_squared * distance * length

    def cost_by_distance(self, clothoid):
        """
            Maybe add some more points to centerline and clothoid.
        """
        # create itterator
        j = np.min([self.farest_point, len(self.midpoints)-1])

        # get x and y position of clothoid
        Clothoid = np.c_[clothoid.SampleXY(j)[0], clothoid.SampleXY(j)[1]]

        cost = 0
        # find nearest midpoint for each clothoid point
        for point in Clothoid:
            # calculate distance from clothoide point to every midpoint
            distance_mid = np.sqrt((self.midpoints[:j,0] - point[0])**2 + (self.midpoints[:j,1] - point[1])**2)
            cost += np.amin(distance_mid)**2
        return cost

    def cost_by_distance_2(self, clothoid):
        """
            Maybe add some more points to centerline and clothoid.
        """
        # create itterator
        j = np.min([self.farest_point, len(self.midpoints)-1])

        # get x and y position of clothoid
        Clothoid = np.c_[clothoid.SampleXY(j)[0], clothoid.SampleXY(j)[1]]

        cost = 0
        # find nearest midpoint for each clothoid point
        for point in Clothoid:
            # calculate distance from clothoide point to every midpoint
            distance_mid = np.sqrt((self.midpoints[:j,0] - point[0])**2 + (self.midpoints[:j,1] - point[1])**2)
            cost += np.argmin(distance_mid)**2
        return cost

    def bestClothoid_by_time(self, method):
        """
            .. description::
                Methode to calculate the best clothoide. Uses scipy minimize to get the the BEST path, using:
                    - racetime          Reason: we're building a racecar?!?
                    - penalty           Reason: hittinh the track boundaries (cones) will be punished with time penaltys.

            .. inputs::
                :None:

            .. outputs::
                :np.ndarray result:     kappa and beta with lowest cost
        """
        # set boundaries for arguments
        i = np.min([self.farest_point, len(self.midpoints)])
        bounds = ((-self.track_widths[i-1,1], self.track_widths[i-1,0]),)

        # create minimizer arguments
        minimizer_kwargs = {"method":"SLSQP","bounds":bounds}

        # create options for optimizer
        options={'ftol': 1e-02,
                'eps': 1.4901161193847656e-08}
        
        # minimize costfunction
        result = minimize(self.Clothoid_by_time, x0=[0], method=method, bounds=bounds, options=options)

        return result.x

    def Clothoid_by_time(self, beta, print_debug=False):
        """
            .. description::
                Methode to draw a clothoid with given start-, end-point and start-, end-tangente. Calculates the racetime of the created clothoid.
                Saves the clothoid as class attribute.

            .. inputs::
                :np.ndarray beta:     Deviation along the normalvector of the last midpoint (beta), defines the new endposition. Deviation from endpoint-tangente, difing the new end-tangente.

            .. outputs::
                : float:            cost of that specific clothoid (will be minimized by the optimizer)
        """
        # limit max number of points
        i = np.min([self.farest_point, len(self.midpoints)])

        # get endpoint and endtangente
        point = self.midpoints[i-1] + self.normalvectors[i-1]*beta
        normal = self.normalvectors[i-1]
        tangent = np.array([normal[1], -normal[0]])
        psi = np.arctan2(tangent[1], tangent[0])

        # create Clothoid
        self.clothoid_by_time = Clothoid.G1Hermite(0,0,0, point[0], point[1], psi)

        # get curvature profile:
        self.curvature = np.array([])
        self.distance = np.array([])
        self.path = np.array([])

        # Calculate curvature: (x_1*y_2 - x_2*y_1) / np.sqrt(x_1**2+y_1**2)**3
        curv = 0
        for l in np.linspace(0, self.clothoid_by_time.length, 10):
            curv += (self.clothoid_by_time.XD(l)*self.clothoid_by_time.YDD(l) - self.clothoid_by_time.XDD(l)*self.clothoid_by_time.YD(l))/(self.clothoid_by_time.XD(l)**2 + self.clothoid_by_time.YD(l)**2)**3
            self.curvature = np.append(self.curvature, curv)
            self.distance = np.append(self.distance, l)
            self.path = np.append(self.path, np.array([self.clothoid_by_time.X(l), self.clothoid_by_time.Y(l)]))

        self.path = self.path.reshape((-1,2))

        # call distance to centerline
        penalty = self.collision_by_time(self.clothoid_by_time)

        # call velocity
        self.velocity, self.acceleration_longitudinal, self.acceleration_lateral = self.My_Velocity.run(self.path, self.curvature, self.distance, v_initial=self.v_initial, v_end=self.v_end, plot=False)

        # print debug message
        if (print_debug != False):
            print('CLOTHOID_TIME:',self.velocity[0], self.acceleration_longitudinal[0], self.acceleration_lateral[0])

        # retrun costfunction
        cost = (self.My_Velocity.Time + penalty)
        return cost

    def collision_by_time(self,
                    clothoid):
        """
            .. description::
                Methode to calculate the cost of a collision with the trackboundaries by time. In the moment a linear costfuction is used. Later this can be expanded by taking the penalty from FSD Rules into account:
                    Costs for knockt over cones (DOO) or vehicle outside of the track (OC):
                        DOO:
                            Accel: 2s
                            Skidpad: 0.2s
                            Autocross: 2s
                            Trackdrive: 2s
                        OC:
                            Accel: DNF
                            Skidpad: DNF
                            Autocross: 10s
                            Trackdrive: 10s

            .. inputs::
                :obj clothoid:      Pyclothoid object containing a clothoid.

            .. outputs::
                :float time:        penalty time of the clothoid path, by hitting track boundaries

            .. STATUS::
                IN PROGRESS
        """
        # get x and y position of clothoid
        j = np.min([self.farest_point, len(self.midpoints)])
        Clothoid = np.c_[clothoid.SampleXY(j)[0], clothoid.SampleXY(j)[1]]
        #Clothoid = np.c_[clothoid.SampleXY(12)[0], clothoid.SampleXY(12)[1]]

        time = 0
        # find nearest midpoint for each clothoid point
        for point in Clothoid:
            # calculate distance from clothoide point to every midpoint
            
            distance_mid = np.sqrt((self.midpoints[:j,0] - point[0])**2 + (self.midpoints[:j,1] - point[1])**2)
            i = np.argmin(distance_mid)
        
            # translate and rotate path position
            transform_point = point - self.midpoints[i]
            psi = np.arctan2(self.normalvectors[i][1], self.normalvectors[i][0]) - np.pi/2
            rot = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            result = np.dot(transform_point, rot)
            distance = result[1]

            if (distance > self.track_widths[i,0]):
                time += np.absolute(distance-self.track_widths[i,0])
            elif (distance < -self.track_widths[i,1]):
                time += np.absolute(distance+self.track_widths[i,1])
            #if ((distance >= self.border_left[i]) or (distance <= self.border_right[i])):
            #    time += 10
        return time

    def minitime(self):
        """
            .. description::

            .. inputs::

            .. outputs::

            .. STATUS::
                ToDo
        """
        i = len(self.midpoints)
        if (i > 2):
            self.midpoints_for_optimization = np.array([self.midpoints[0],
                                                        self.midpoints[int(i/2)],
                                                        self.midpoints[i-1]])
            self.track_widths_for_optimization = np.array([self.track_widths[0],
                                                        self.track_widths[int(i/2)],
                                                        self.track_widths[i-1]])
            self.normalvectors_for_optimization = np.array([self.normalvectors[0],
                                                        self.normalvectors[int(i/2)],
                                                        self.normalvectors[i-1]])
            print('MIDPOINTS:',self.midpoints_for_optimization)
        else:
            self.midpoints_for_optimization = self.midpoints
            self.track_widths_for_optimization = self.track_widths
            self.normalvectors_for_optimization = self.normalvectors

        

        bounds = tuple(map(tuple, np.c_[-self.track_widths_for_optimization[:,1], self.track_widths_for_optimization[:,0]]))
        minimizer_kwargs = {"method":"SLSQP","bounds":bounds}
        #midpoints_for optimization
        x0 = np.zeros(len(self.track_widths_for_optimization[:])).tolist()
        #result = basinhopping(self.minimizer, x0=x0, minimizer_kwargs=minimizer_kwargs, niter=10)
        result = minimize(self.minimizer, x0=x0, method='SLSQP', bounds=bounds)
        #print(self.minimizer(x0))

        return result.x

    def minimizer(self, x0):
        """
            .. description::

            .. inputs::

            .. outputs::

            .. STATUS::
                ToDo
        """
        width = np.asarray(x0)
        Path = self.midpoints_for_optimization + self.normalvectors_for_optimization*width.reshape((-1,1))

        # get curvature
        curvature = np.absolute(self.Polyfit(Path))

        # get distance
        distance = self.distance(Path**2)

        # call velocity profile
        self.My_Velocity.run(Path, curvature, distance)
        cost = self.My_Velocity.Time
        print(cost)
        #cost = np.sum(curvature)

        return cost

    def CubicSpline(self,
                points,
                factor = 1):
        """
            .. description::
                Methode to calculate a cubic spline of the input points. Also calculates the curvature of the spline

            .. inputs::
                :np.array points:           Array of points as input for cubicspline
                :int factor:                Factor for interpolation between points.

            .. outputs::
                :np.ndarray curvature:      Array with curvature values of the spline interpolation. Shape: ((-1)) -> [[c_1, c_2, ...]]
                :np.ndarray normalvectors:  Array with normalvectors of the spline interpolation. Shape: ((-1,c)) -> [[x, y]]
                :np.ndarray new_points:     Array of new points from cubicspline. Shape: ((-1,2)) -> [[x, y]]

            .. STATUS::
                Done.
        """
        # create step-itterator array
        i = np.arange(0, len(points), 1)
        i_new = np.linspace(0, len(points)-1, (len(points)-1)*factor+1, endpoint=True)

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
        Distance = (points[1:]-points[:-1])
        Distance = np.append(0, np.cumsum(np.sqrt(Distance[:,0]**2 + Distance[:,1]**2)))

        return Distance

    def Curvature(self,
                    factor=1):
        """
            .. description::
                Calculates curvature for given array of midpoints.

            .. inputs::
                :int factor:        Scaling factor for approximation. 10 means 10 times more points in interpolation.

            .. outputs::
                :bool status:       True if curvature calculation finished

            .. STATUS::
                Not used until now.
        """
        # create step-itterator array
        i = np.arange(0, len(self.midpoints), 1)
        i_new = np.arange(0, int(len(self.midpoints)-1/factor), 1/factor)

        # generate cubic Spline
        spline_x = CubicSpline(i, self.midpoints[:,0], bc_type='natural')
        spline_y = CubicSpline(i, self.midpoints[:,1], bc_type='natural')

        # calculate first and secon derivation of the spline
        x_1 = spline_x(i_new,1)
        y_1 = spline_y(i_new,1)
        x_2 = spline_x(i_new,2)
        y_2 = spline_y(i_new,2)

        #self.normalvector = [-y_1, x_1]/np.sqrt(x_1**2 + y_1**2)

        # calculate curvature
        curvature = (x_1*y_2 - x_2*y_1) / np.sqrt(x_1**2+y_1**2)**3

        return curvature

    def MidpointsByCurvature(self,
                            max_curve=1/3,
                            min_curve=0.4/32**2):
        """
            .. description::
                Shifts the Midpoints by a curvature dependig factor to the outter site of the track

            .. inputs::
                :float max_curve:   max curvature of corner with min raidus um 3 m.
                :float min_curve:   min curvature where vehicle can drive with fullspeed

            .. outputs::
                :bool status:       True if Shift calculation finished

            .. STATUS::
                Not used until now.
        """
        try:
            # replace litte curvatures by zero
            curvature = np.place(self.curvature, np.absolute(self.curvature)<=min_curve, 0)

            # calculate shift factor: a = v² * rho ->
            alpha = (-curvature)*1/max_curve
            alpha = np.clip(alpha, -1, 1)

            self.raceline = self.midpoints + self.normalvectors * alpha

            return True
        except:
            return False

    def plot(self,
            left_cones=None,
            right_cones=None):
        """
            .. description::
                Plot of the local track with track boundaries and generated path

            .. inputs::
                :None:

            .. outputs::
                :None:

            .. STATUS::
                DONE
        """
        plt.subplots(figsize=(8, 8))

        # plot vehicle
        plt.plot(self.origin[0], self.origin[1], 'o', color='black', label='car')

        # plot left cones
        if (len(left_cones) != 0):
            plt.plot(left_cones[:,0], left_cones[:,1], 'o', color='blue', label='left')

        # plot right cones
        if (len(right_cones) != 0):
            plt.plot(right_cones[:,0], right_cones[:,1], 'o', color='gold', label='right')

        # plot midpoints
        if (len(self.midpoints) != 0):
            plt.plot(self.midpoints[:,0], self.midpoints[:,1], '.', color='gray', label='centerline')

        # plot border points
        if (len(self.border_right) != 0):
            plt.plot(self.border_right[:,0], self.border_right[:,1], '-', color='red', label='borders')
        if (len(self.border_left) != 0):
            plt.plot(self.border_left[:,0], self.border_left[:,1], '-', color='red')

        # plot generated path
        if (len(self.path) != 0):
            plt.plot(self.path[:,0], self.path[:,1], '-', label='raceline')

        plt.xlim(-1, 20)
        plt.ylim(-10, 10)
        #plt.grid()
        plt.legend()
        plt.show()
