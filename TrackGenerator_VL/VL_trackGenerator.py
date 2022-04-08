import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from scipy.special import binom
from random import uniform, choice

class TrackGenerator:
        FIDELITY = 50
        TRACK_WIDTH = 3.5
        MIN_STRAIGHT = 5
        MAX_STRAIGHT = 80
        MIN_CONSTANT_TURN = 10
        MAX_CONSTANT_TURN = 45
        MAX_TRACK_LENGTH = 105
        MAX_ELEMENTS = 3

        ################################################################
        #MAKRO GENERATOR FUNCTIONS
        ################################################################
        
        def generate_randomTrack():
                """
                generates random track with max amount of TrackGenerator.MAX_ELEMENTS track elements
                """
                #Starting position
                point_in = [0,0]
                tangent_in = [1,0]
                normal_in = [0,1]

                #Trackdata
                track_data = []
                cur_track_data= []

                #track elements
                elementList=[999]
                elementCounter = 0

                # exit conditions
                #error=False
                failedCounter = 0
                failedElement = False
                finished = False

                #loop for generating track elemnts
                while finished is False:
                        if failedCounter == 10:
                                
                                return None, None, True #Generation failed due to to many tries
                        if elementCounter == TrackGenerator.MAX_ELEMENTS:
                                break #generation finished due to max number of elements

                        cur_track_data = []
                        cur_track_data = track_data.copy()

                        if failedElement:
                                data_out, tangent_out, normal_out, finished, elementType= TrackGenerator.randomElement(point_in, tangent_in, normal_in)
                        else:
                                data_out, tangent_out, normal_out, finished, elementType= TrackGenerator.randomElement(point_in, tangent_in, normal_in)
                        if finished:
                                break
                        cur_track_data.extend(data_out[1:])
                        #print(f"TD_len {len(track_data)}")
                        if TrackGenerator.check_if_viable(cur_track_data, elementType, elementList[-1]):
                                failedElement=False
                                track_data=cur_track_data
                                elementList.append(elementType)

                                #prep for new data
                                point_in = data_out[-1]
                                tangent_in = tangent_out
                                normal_in = normal_out
                                elementCounter += 1
                                continue
                        else:
                                failedCounter += 1
                                failedElement=True
                                continue
                for point in track_data:
                        point = (round(point[0], 2), round(point[1], 2))
                track_data=[(round(point[0], 2), round(point[1], 2)) for point in track_data]
                        
                conedata=TrackGenerator.get_cones(track_data)
                return track_data, conedata, elementList, False

        def randomElement(point_in, tangent_in, normal_in, newElement=None):
                """
                Adds new random Track element (if newElement is TRUE then empty track element is not an option)
                """
                if newElement is None:
                        newElement=True

                track_element = 0
                

                finished=False #last track element?
                if newElement:
                        functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
                        #functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn]
                        i = choice(range(len(functions)))
                        data_out, tangent_out, normal_out=(functions)[i](point_in, tangent_in, normal_in)
                        track_element = i
                else:
                        functions = [TrackGenerator.random_Bezier, TrackGenerator.add_straight, TrackGenerator.add_constant_turn, TrackGenerator.emptyElement]
                        i = choice(range(len(functions)))
                        data_out, tangent_out, normal_out=(functions)[i](point_in, tangent_in, normal_in)
                        track_element = i
                if data_out is None:
                        finished=True
                        track_element = i
                #print(f"element{track_element}")

                return data_out, tangent_out, normal_out, finished, track_element

                
        def check_if_viable(toCheck_track_data, newElement, lastElement):
                """
                checks if added track element is viable
                """
                doubleStraight = newElement == lastElement == 1
                return not TrackGenerator.intersectsWithSelf(toCheck_track_data) and not doubleStraight

        ################################################################
        #TRACKELEMENT FUNCTIONS
        ################################################################
        def emptyElement(point_in,
                                tangent_in,
                                normal_in):
                return None, tangent_in, normal_in

        def add_straight(point_in,
                        tangent_in,
                        normal_in,
                        params={}):
                """
                Creates a straight line.

                params["length"]: specifies lenth of straight
                """

                # Load in params        
                if "length" in params:
                        length = params["length"]
                else:
                        length = uniform(
                                TrackGenerator.MIN_STRAIGHT,
                                TrackGenerator.MAX_STRAIGHT
                        )

                # Prepare outputs
                straight_func = TrackGenerator.parametric_straight(tangent_in, point_in, length)
                tangent_out = tangent_in
                normal_out = normal_in
                #added_length = length

                return (
                        TrackGenerator.de_parameterize(straight_func),
                        normalize_vec(tangent_out),
                        normalize_vec(normal_out)
                        
                )

        def add_bezier(point_in,
                     point_out,
                     tangent_in,
                     tangent_out,
                     params={}
                    ):
                """
                Creates a cubic bezier.
                params["scale_in"]: specifies how straight the input line is
                params["scale_out"]: specifies how straight the output line is
                """

                # Load in params
                if "scale_in" in params:
                        scale_in = params["scale_in"]
                else:
                        scale_in = uniform(10, 25)
                        #scale_in = 25
                if "scale_out" in params:
                        scale_out = params["scale_out"]
                else:
                        scale_out = uniform(10, 25)
                        #scale_out = 25

                p0_to_p1 = scale_vector(tangent_in, scale_in)
                p0 = point_in
                p1 = (p0[0] + p0_to_p1[0], p0[1] + p0_to_p1[1])
                
                pn_1_to_pn = scale_vector(tangent_out, scale_out)
                pn = point_out
                pn_1 = (pn[0] - pn_1_to_pn[0], pn[1] - pn_1_to_pn[1])

                p0=point_in

                control_points = [p0, p1, pn_1, pn]


                # Prepare outputs
                bez_out = TrackGenerator.parametric_bezier(control_points)
                normal_out = get_normal_vector(tangent_out)
                

                return (
                        TrackGenerator.de_parameterize(bez_out),
                        normalize_vec(tangent_out),
                        normalize_vec(normal_out)
                        
                )
        def random_Bezier(point_in, tangent_in, normal_in):
                """
                creates random beziew element with point on arc with random degree alpha and random distance r and random output tangent beta.
                """
                MAX_ALPHA=deg_to_rad(75)
                MAX_BETA=-deg_to_rad(60)
                r=uniform(30, 50)
                #POINT_UT FROM ALPHA
                alpha=uniform(-MAX_ALPHA,MAX_ALPHA)
                newTan=(tangent_in[0]* np.cos(alpha) + tangent_in[1] *np.sin(alpha), -tangent_in[0]*np.sin(alpha) + tangent_in[1]* np.cos(alpha))   #direction towards point out
                point_out= (point_in[0]+newTan[0]*r,point_in[1]+newTan[1]*r)    #move by r

                #TANGENT_OUT FROM BETA
                beta=uniform(-MAX_BETA,MAX_BETA)
                #beta=MAX_BETA
                tangent_out=(newTan[0]* np.cos(beta) + newTan[1] *np.sin(beta), -newTan[0]*np.sin(beta) + newTan[1]* np.cos(beta))
                      

                return TrackGenerator.add_bezier(point_in, point_out,tangent_in,tangent_out)



        def add_constant_turn(point_in,
                                tangent_in,
                                normal_in,
                                params={}):
                """
                Creates a circle segment.

                params["turn_against_normal"]: boolean, when true then the
                                        circle will have the opposite normal.
                params["radius"]: radius of the circle
                params["circle_percent"]: percent of a full turn the circle should undergo
                """

                # Load in params
                if "turn_against_normal" in params:
                        turn_against_normal = params["turn_against_normal"]
                else:
                        turn_against_normal = uniform(0, 1) < 0.5

                if turn_against_normal:
                        #flip reference frame.
                        normal_in = scale_vector(normal_in, -1)

                if "radius" in params:
                        radius = params["radius"]
                else:
                        radius = uniform(
                                TrackGenerator.MIN_CONSTANT_TURN,
                                TrackGenerator.MAX_CONSTANT_TURN
                        )

                if "circle_percent" in params:
                        circle_percent = params["circle_percent"]
                else:
                        
                        circle_percent = uniform(0.1, 0.5)

                center = add_vectors(point_in, scale_vector(normal_in, radius))

                #je nach Orientierung des Tracks muss die Richtung der gezeichneten Kreise geändert werden
                def sgn(x):
                        """-1 if x is negative, +1 if positive, 0 if 0."""
                        return -1 if x < 0 else 1 if x > 0 else 0
                handedness = sgn(tangent_in[0] * normal_in[1] - tangent_in[1] * normal_in[0])
                turn_angle = handedness * circle_percent * np.pi * 2

                # And now grab output points
                circle_function = TrackGenerator.parametric_circle(point_in, center, turn_angle)
                points_out = TrackGenerator.de_parameterize(circle_function)

                # Calculate total length
                #added_length = turn_angle * radius

                # Now we want to find the new normal vector,
                normal_out = normalize_vec((
                        points_out[-1][0] - center[0],
                        points_out[-1][1] - center[1]
                ))

                # And finally recalculate the tangent:
                tangent_out = calculate_tangent_vector(points_out)

                # Returns a list of points and the new edge of the racetrack and the change in length
                return (points_out, normalize_vec(tangent_out), normalize_vec(normal_out))

        #####################################
        #FUNCTIONS IN PARAMETER FORM
        #####################################
        def de_parameterize(func):
                """Given a parametric function, turn it into a list of points"""
                return [func(1.0 * t / (TrackGenerator.FIDELITY - 1))
                        for t in range(0, TrackGenerator.FIDELITY)]

        def parametric_circle(start_point, center_point, delta_angle):
                """Returns parametric function of circle => returns point for given t \elem[0,1]"""
                def output(s, c, a):
                        (sx, sy) = s
                        (cx, cy) = c
                        cos_a = np.cos(a)
                        sin_a = np.sin(a)
                        del_x = sx - cx
                        del_y = sy - cy
                        result_x = cos_a * del_x - sin_a * del_y + cx
                        result_y = sin_a * del_x + cos_a * del_y + cy
                        
                        return (result_x, result_y)
                return  Parametrization(lambda t: output(start_point, center_point, t * delta_angle))
        
        def parametric_straight(slope_vec, start_point, line_length):
                """Returns the parametric function of a line given a slope, start point and length"""
                def to_return(slope, start, length, t):
                        return add_vectors(start, scale_vector(slope, 1.0 * t * line_length))
                return Parametrization(lambda t: to_return(slope_vec, start_point, line_length, t))

        def parametric_bezier(control_points):
                """
                This function will itself return a function of a parameterized bezier
                
                """
                def to_return(cp, t):
                        the_sum_x = 0
                        the_sum_y = 0
                        n = len(cp)
                        for i in range(n):
                                coefficient = binom(n-1, i) * (1 - t)**(n - i - 1) * t**i
                                the_sum_x += coefficient * cp[i][0]
                                the_sum_y += coefficient * cp[i][1]
                        return (the_sum_x, the_sum_y)
                return Parametrization(lambda t: to_return(control_points, t))

                
        def get_cones(xys, track_width=None):
                
        
                """ 
                Takes in a list of points, returns a list of triples dictating x, y position
                of cone and color  [(x, y, color)].
                """

                if track_width is None:
                        cone_normal_distance = TrackGenerator.TRACK_WIDTH
                else:
                        cone_normal_distance = track_width
                
                #print(cone_normal_distance)

                # How close can cones be from those on the same side.
                min_cone_distance_sameSide = 4

                cone_cross_closeness_parameter = cone_normal_distance * 3 / 4 - 1

                all_points_aSide = [(0, 0)]
                all_points_bSide = [(0, 0)]
                to_return = []

                # This is used to check if the yellow and blue cones suddenly swapped.
                last_tangent_normal = (0, 0)
                


                for i in range(len(xys)):
                        # Skip first 
                        if i == 0:
                                continue

                        # calculate first tangent then normals
                        cur_point = xys[i]
                        cur_tangent_angle = get_tangent_angle(xys[:(i+1)])
                        
                        
                        cur_tangent_normal = (
                        np.ceil(
                                cone_normal_distance * np.sin(cur_tangent_angle)
                        ),
                        np.ceil(
                                -cone_normal_distance * np.cos(cur_tangent_angle)
                        )
                        )

                        # Check if normal vector direction flipped
                        scal_product = (
                                last_tangent_normal[0] * cur_tangent_normal[0] +
                                last_tangent_normal[1] * cur_tangent_normal[1]
                        )
                        

                        
                        # skalarprodukt negativ => stumpfer Winkel ->Normalvektor ist vertauscht
                        if scal_product < 0:
                                cur_tangent_normal = scale_vector(cur_tangent_normal, -1)

                        last_tangent_normal = cur_tangent_normal

                        #store cone points

                        a_side_point = ((cur_point[0] + cur_tangent_normal[0]),
                                (cur_point[1] + cur_tangent_normal[1]))
                        b_side_point = ((cur_point[0] - cur_tangent_normal[0]),
                                (cur_point[1] - cur_tangent_normal[1]))

                        #get distance to cones on same side
                        distance_to_last_aSide = min([
                                                        (a_side_point[0] - prev_point_aSide[0])**2 +
                                                        (a_side_point[1] - prev_point_aSide[1])**2
                                                        for prev_point_aSide in all_points_aSide
                        ])
                        distance_to_last_bSide = min([
                                                        (b_side_point[0] - prev_point_bSide[0])**2 +
                                                        (b_side_point[1] - prev_point_bSide[1])**2
                                                        for prev_point_bSide in all_points_bSide
                        ])

                        cross_distance_aSide = min([
                                         (a_side_point[0] - prev_point_bSide[0])**2 +
                                         (a_side_point[1] - prev_point_bSide[1])**2
                                         for prev_point_bSide in all_points_bSide
                        ])
                        cross_distance_bSide = min([
                                                (b_side_point[0] - prev_point_aSide[0])**2 +
                                                (b_side_point[1] - prev_point_aSide[1])**2
                                                for prev_point_aSide in all_points_aSide
                        ])
                        rel_xys = xys[
                        max([0, i - 100]):
                        min([len(xys), i + 100])
                        ]
                        distance_pA = min([
                                        (a_side_point[0] - xy[0])**2 +
                                        (a_side_point[1] - xy[1])**2
                                        for xy in rel_xys
                        ])
                        distance_pB = min([
                                        (b_side_point[0] - xy[0])**2 +
                                        (b_side_point[1] - xy[1])**2
                                        for xy in rel_xys
                        ])

                        aSide_OK = (
                        distance_to_last_aSide > min_cone_distance_sameSide**2 and
                        cross_distance_aSide > cone_cross_closeness_parameter**2 and
                        distance_pA > cone_cross_closeness_parameter**2
                        )

                        bSide_OK = (
                        distance_to_last_bSide > min_cone_distance_sameSide**2  and
                        cross_distance_bSide > cone_cross_closeness_parameter**2 and
                        distance_pB > cone_cross_closeness_parameter**2
                                 
                        )

                        # And when they are OK store points
                        if (aSide_OK):
                                x = round((a_side_point[0]), 2)
                                y = round((a_side_point[1]), 2)
                                to_return.append((x, y, "Y"))
                                #.append((cur_point[0], cur_point[1],"YM"))
                                all_points_aSide.append(a_side_point)
                        if (bSide_OK):
                                x = round((b_side_point[0]), 2)
                                y = round((b_side_point[1]), 2)
                                to_return.append((x, y, "B"))
                                #to_return.append((cur_point[0], cur_point[1],"BM"))
                                all_points_bSide.append(b_side_point)     
                return to_return

        def __doIntersect(p1,q1,p2,q2):
                """
                The main function that returns true if
                the line segment 'p1q1' and 'p2q2' intersect.
                """
                # Find the 4 orientations required for
                # the general and special cases
                o1 = orientation(p1, q1, p2)
                o2 = orientation(p1, q1, q2)
                o3 = orientation(p2, q2, p1)
                o4 = orientation(p2, q2, q1)
                
                # General case
                if ((o1 != o2) and (o3 != o4)):
                        return True
                
                # Special Cases
                
                # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
                if ((o1 == 0) and onSegment(p1, p2, q1)):
                        return True
                
                # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
                if ((o2 == 0) and onSegment(p1, q2, q1)):
                        return True
                
                # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
                if ((o3 == 0) and onSegment(p2, p1, q2)):
                        return True
                
                # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
                if ((o4 == 0) and onSegment(p2, q1, q2)):
                        return True
                
                # If none of the cases
                return False

        def intersectsWithSelf(curvePoints):
                points = curvePoints
                x=False
                #print(len(points))
                for i in range(len(points) - 1):

                        #Point pair 1
                        p1 = points[i]
                        p2 = points[i+1]

                        for j in range(len(points) - 1):
                                #Point pair 2
                                p3 = points[j]
                                p4 = points[j+1]

                                if i != j-1 and i != j and i != j+1:
                                        if TrackGenerator.__doIntersect(p1,p2, p3,p4): 
                                                print(str(i), " ", str(j))
                                                return True 
                                        else:
                                                continue
                
                return False
        
        #######################################################
        #VISUALIZE Tracks
        #######################################################
        def visualize_all(trackdata, conedata):
                #   sort track data             
                x, y = TrackGenerator.visualize_track(trackdata)
                yellow_x, yellow_y, blue_x, blue_y=TrackGenerator.visualize_cones(conedata)

                plt.plot(x, y)
                plt.plot(yellow_x,yellow_y,'*',color='orange')
                plt.plot(blue_x,blue_y,'*',color='blue')

                plt.axis('scaled')
                plt.show()

        def show_track(trackdata):
                x, y = TrackGenerator.visualize_track(trackdata)
                plt.plot(x, y)
                plt.axis('scaled')
                plt.show()

        def visualize_track(track_data):
                return map(list, zip(*track_data))
                
        def visualize_cones(conedata):
                yellow_cones=[x for x in conedata if x[2]=='Y']
                blue_cones=[x for x in conedata if x[2]=='B']
                yellow_x, yellow_y, _=map(list, zip(*yellow_cones))
                blue_x, blue_y, _=map(list, zip(*blue_cones))

                return yellow_x, yellow_y, blue_x, blue_y
                
        def show_cones(conedata):
                yellow_x, yellow_y, blue_x, blue_y=TrackGenerator.visualize_cones(conedata)
                plt.plot(yellow_x,yellow_y,'*',color='orange')
                plt.plot(blue_x,blue_y,'*',color='blue')
                plt.axis('scaled')
                plt.show()



                        

                        
     
class Parametrization:
        """
        Parametrization functions are defined as functions which take a number \elem [0,1]
        and return a point.  It is used to define geometric curves in terms of a percentage.
        So if you had a circle parametrization function, and gave it an input 0.25, it would
        return the point 25% of the way around the circle segment.
        """
        def __init__(self, func):
                self._func = func

        def __call__(self, num):
                return self._func(num)

        @staticmethod
        def compose(others):
                """
                Combines two parametrizations in a way such that the composite
                still only takes a number from 0 to 1.
                """
                amount = len(others)
                threshold = 1.0/amount

                def composite(num):
                        # Check out of bounds
                        if num <= 0:
                                return others[0](0)
                        elif num >= 1:
                                return others[-1](1)

                        # Acts as composition of components
                        i = 0
                        while True:
                                if num < threshold:
                                        return others[i](amount * num)
                                else:
                                        num -= threshold
                                        i += 1
                return Parametrization(composite)



# p0=[0,0]
# p1=[0,30]
# pn_1=[30,0]
# pn=[30,30]
# control_points = [p0, p1, pn_1, pn]

# p2=[0,0]
# p3=[0,1]
# pn_3=[1,0]
# pn=[30,30]

# d=1

# ######################## DATA ############################
# #data=list(TrackGenerator.de_parameterize(TrackGenerator.parametric_bezier(control_points)))
# data, _, _, _=TrackGenerator.add_constant_turn(p0, p3, pn_3)
# #data=list(TrackGenerator.de_parameterize(TrackGenerator.parametric_circle([0,0],[0,d*20],d*np.pi/2)))
# #data=list(TrackGenerator.de_parameterize(TrackGenerator.parametric_straight([1,0],[0,0],50)))

# TrackGenerator.visualize(data)


        # def orient_constant_turn(point_in,
        #                     point_out,
        #                     tangent_in,
        #                     tangent_out,
        #                     normal_in):
        #         xys = []
        #         """
        #         Complicated connector function that attempts to abide by contest regulations.

        #         Warning: Sometimes the circles are not of allowed radius!
        #         """

        #         total_length = 0
        #         final_tangent_out = tangent_out
        #         tangent_out = scale_vector(tangent_out, -1)

        #         # We need to calculate the turn angle (angle between 2 vectors):
        #         outer_turn_angle = np.arccos(
        #                 - tangent_in[0] * tangent_out[0] - tangent_in[1] * tangent_out[1]
        #         )
        #         circle_turn_angle = np.pi - outer_turn_angle
        #         circle_turn_percent = circle_turn_angle / (2 * np.pi)
        #         circle_radius = uniform(
        #                 TrackGenerator.MIN_CONSTANT_TURN,
        #                 TrackGenerator.MAX_CONSTANT_TURN
        #         )

        #         # Now we draw this circle:
        #         (points_out, tangent_out, normal_out, added_length) = TrackGenerator.add_constant_turn(
        #                 point_in,
        #                 tangent_in,
        #                 normal_in,
        #                 params={
        #                         "turn_against_normal": False,
        #                         "circle_percent":      circle_turn_percent,
        #                         "radius":              circle_radius
        #                 }
        #         )
        #         total_length += added_length
        #         xys.extend(points_out)

        #         # And now we add a half-turn to point us directly back to the start.
        #         # Radius is calculated by finding distance when projected along the normal
        #         tangent_in = tangent_out
        #         normal_in = normal_out
        #         point_in = points_out[-1]
        #         tangent_out = final_tangent_out
        #         diff = subtract_vectors(point_in, point_out)
        #         circle_radius = (diff[0] * normal_in[0] + diff[1] * normal_in[1])/2

        #         # Now we draw the circle:
        #         (points_out, tangent_out, normal_out, added_length) = TrackGenerator.add_constant_turn(
        #                 point_in,
        #                 tangent_in,
        #                 normal_in,
        #                 params={
        #                         "turn_against_normal": False,
        #                         "circle_percent":      0.5,
        #                         "radius":              abs(circle_radius)
        #                 }
        #         )
        #         total_length += added_length
        #         xys.extend(points_out)

        #         # And then add a straight to connect back to the start
        #         tangent_in = tangent_out
        #         normal_in = normal_out
        #         point_in = points_out[-1]
        #         straight_length = get_distance(point_in, point_out) * 1.1
        #         (points_out, tangent_out, normal_out, added_length) = TrackGenerator.add_straight(
        #                 point_in,
        #                 tangent_in,
        #                 normal_in,
        #                 params={
        #                         "length": straight_length
        #                 }
        #         )
        #         total_length += added_length
        #         xys.extend(points_out)

        #         return (xys, tangent_out, normal_out, total_length)
