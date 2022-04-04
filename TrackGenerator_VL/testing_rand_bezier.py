import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from random import uniform
from utilities import *
from VL_trackGenerator import *

def randomBezier(point_in, tangent_in):
    MAX_ALPHA=np.pi/2
    MAX_BETA=np.pi/4
    r=50
    #POINT_UT FROM ALPHA
    alpha=uniform(-MAX_ALPHA,MAX_ALPHA)
    newTan=(tangent_in[0]* np.cos(alpha) + tangent_in[1] *np.sin(alpha), -tangent_in[0]*np.sin(alpha) + tangent_in[1]* np.cos(alpha))   #direction towards point out
    point_out= (point_in[0]+newTan[0]*r,point_in[1]+newTan[1]*r)    #move by r

    #TANGENT_OUT FROM BETA
    beta=uniform(-MAX_BETA,MAX_BETA)
    tangent_out=(newTan[0]* np.cos(beta) + newTan[1] *np.sin(beta), -newTan[0]*np.sin(beta) + newTan[1]* np.cos(beta))
    tangent_out=scale_vector(tangent_out, -1)       #flip vektor (point in beta towards old point)
    #add_bezier()
    return point_out, tangent_out

for i in range(50):
    a, b= randomBezier((0,0), (1,0))
    plt.arrow(a[0],a[1],b[0], b[1])
plt.ylim(-10,10)
plt.xlim(-10,10)
plt.show()