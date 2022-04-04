import numpy as np
import matplotlib.pyplot as plt

from random import uniform

FIDELITY = 350


def add_vectors(a, b):
        """Given two tuples a and b, add them together"""
        return (a[0] + b[0], a[1] + b[1])


def scale_vector(vec, scale):
        """Multiplies vec by scale"""
        return (vec[0] * scale, vec[1] * scale)


def de_parameterize(func):
        """Given a parametric function, turn it into a list of points"""
        return [func(1.0 * t / (FIDELITY - 1))
                for t in range(0, FIDELITY)]


def parametric_straight(slope_vec, start_point, line_length):
        """Returns the parametric function of a line given a slope and start point"""
        def to_return(slope, start, length, t):
                return add_vectors(start, scale_vector(slope, 1.0 * t * line_length))
        return Parametrization(lambda t: to_return(slope_vec, start_point, line_length, t))


def parametric_circle(start_point, center_point, delta_angle):
        """
        Returns a function in terms of t \elem [0,1] to parameterize a circle

        We can calculate points on a circle using a rotation matrix
        R(a)*[S-C]+C gives us any point on a circle starting at S with center C
        with counterclockwise angular dstance 'a'
        """
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
        return Parametrization(lambda t: output(start_point, center_point, t * delta_angle))


class Parametrization:
        """
        Class that is used to seemlessly store and combine parametrization functions
        for the generator.

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
# for i in range(100):
#     #print(parametric_circle([0,0],[1,1],np.pi)(i/100))
#     print(
#         de_parameterize(parametric_circle([0,0],[1,1],np.pi))
#     )
#     #list.append(parametric_circle([0,0],[1,1],np.pi)(i/100))
data=list(de_parameterize(parametric_circle([0,0],[0,-1],1)))

# x=(x[0] for x in data)
# y=(x[1] for x in data)

# print(data.count)

# #print(data[1][0])
# print(zip(*data))

a = np.empty(FIDELITY)
b = np.empty(FIDELITY)
for i in range(FIDELITY):
    a[i] = data[i][0]
    b[i] = data[i][1]


plt.plot(a,b)
plt.show()