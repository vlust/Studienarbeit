import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from scipy.special import binom
from random import uniform

class TrackGenerator:
        FIDELITY = 350
        TRACK_WIDTH = 3.5
        MIN_STRAIGHT = 20
        MAX_STRAIGHT = 80
        MIN_CONSTANT_TURN = 10
        MAX_CONSTANT_TURN = 25
        MAX_TRACK_LENGTH = 105
 
        ################################################################
        #FUNKTIONEN UM TRACKKOMPONENTE HINZUFÜGEN
        ################################################################

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
                normal_out = tangent_out
                

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
                #r=30
                #POINT_UT FROM ALPHA
                alpha=uniform(-MAX_ALPHA,MAX_ALPHA)
                #alpha=MAX_ALPHA
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
        #FUNKTIONEN IN PARAMETER FORM 
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
                                x = (a_side_point[0])
                                y = (a_side_point[1])
                                to_return.append((x, y, "A"))
                                to_return.append((cur_point[0], cur_point[1],"AM"))
                                all_points_aSide.append(a_side_point)
                        if (bSide_OK):
                                x = (b_side_point[0])
                                y = (b_side_point[1])
                                to_return.append((x, y, "B"))
                                to_return.append((cur_point[0], cur_point[1],"BM"))
                                all_points_bSide.append(b_side_point)     
                return to_return
        def check_if_overlap(points):
                """
                Naive check to see if track overlaps itself

                (Won't catch overlaps due to track width, only if track center overlaps)
                """

                # Remove end points as in theory that should also be the start point
                # (I remove extra to be a little generous to it as a courtesy)
                #points = points[:-10]
                # We want to add in the diagonally-connected points, otherwise you can imagine
                # that two tracks moving diagonally opposite could cross eachother inbetween the pixels,
                # fooling our test.

                for index in range(1, len(points)):
                        (sx, sy) = points[index - 1]
                        (ex, ey) = points[index]
                        manhattan_distance = abs(ex - sx) + abs(ey - sy)
                        if (manhattan_distance > 1):
                                # moved diagonally, insert an extra point for it at the end!
                                points.append((sx + 1, sy) if ex > sx else (sx - 1, sy))
                return len(set(points)) != len(points)

        def onSegment(p, q, r):
                """
                Given three collinear points p, q, r, the function checks if point q lies on line segment 'pr'
                """
                if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
                        (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
                        return True
                return False
                
        def orientation(p, q, r):
                """
                to find the orientation of an ordered triplet (p,q,r)
                function returns the following values:
                0 : Collinear points
                1 : Clockwise points
                2 : Counterclockwise
                """
                val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
                if (val > 0):
                        # Clockwise orientation
                        return 1
                elif (val < 0):
                        # Counterclockwise orientation
                        return 2
                else:  
                        # Collinear orientation
                        return 0


        def doIntersect(p1,q1,p2,q2):
                """
                The main function that returns true if
                the line segment 'p1q1' and 'p2q2' intersect.
                """
                # Find the 4 orientations required for
                # the general and special cases
                o1 = TrackGenerator.orientation(p1, q1, p2)
                o2 = TrackGenerator.orientation(p1, q1, q2)
                o3 = TrackGenerator.orientation(p2, q2, p1)
                o4 = TrackGenerator.orientation(p2, q2, q1)
                
                # General case
                if ((o1 != o2) and (o3 != o4)):
                        return True
                
                # Special Cases
                
                # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
                if ((o1 == 0) and TrackGenerator.onSegment(p1, p2, q1)):
                        return True
                
                # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
                if ((o2 == 0) and TrackGenerator.onSegment(p1, q2, q1)):
                        return True
                
                # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
                if ((o3 == 0) and TrackGenerator.onSegment(p2, p1, q2)):
                        return True
                
                # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
                if ((o4 == 0) and TrackGenerator.onSegment(p2, q1, q2)):
                        return True
                
                # If none of the cases
                return False
        # def ccw(A,B,C):
        #         return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        # # Return true if line segments AB and CD intersect
        # def intersect(A,B,C,D):
        #         return TrackGenerator.ccw(A,C,D) != TrackGenerator.ccw(B,C,D) and TrackGenerator.ccw(A,B,C) != TrackGenerator.ccw(A,B,D)

        def intersectsWithSelf(curvePoints):
                points = curvePoints
                for i in range(len(points) - 1):

                        #Point pair 1
                        p1 = points[i]
                        p2 = points[i+1]

                        for j in range(len(points) - 1):
                                #Point pair 2
                                p3 = points[j]
                                p4 = points[j+1]

                                if j is not i-1 and j is not i and j is not i+1:
                                        #print(str(i), " ", str(j))
                                        if TrackGenerator.doIntersect(p1,p2, p3,p4): return True 
                return False
                
        def visualize(trackdata):
                #   sort track data             
                x = np.empty(len(trackdata))
                y = np.empty(len(trackdata))
                for i in range(len(trackdata)):
                        x[i] = trackdata[i][0]
                        y[i] = trackdata[i][1] 
                conedata=TrackGenerator.get_cones(trackdata)
                ax=[]
                ay=[]
                bx=[]
                by=[]

                amx=[]
                amy=[]
                bmx=[]
                bmy=[]

                #sort cone data
                for i in range(len(conedata)):
                        if(conedata[i][2]=='A'):
                                ax.append(conedata[i][0])
                                ay.append(conedata[i][1])
                        if(conedata[i][2]=='AM'):
                                amx.append(conedata[i][0])
                                amy.append(conedata[i][1])
                                
                        if(conedata[i][2]=='B'):
                                bx.append(conedata[i][0])
                                by.append(conedata[i][1])
                        if(conedata[i][2]=='BM'):
                                bmx.append(conedata[i][0])
                                bmy.append(conedata[i][1])
                #plots
                plt.plot(x,y)
                plt.plot(ax,ay,'*',color='orange')
                plt.plot(bx,by,'*',color='blue')
                # plt.scatter(bmx,bmy,color='red')   
                # plt.scatter(amx,amy,color='green')           
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
