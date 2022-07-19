#from VL_trackGenerator import *
import pandas as pd
import json
import os
import time
import functools
from multiprocessing import Pool
from random import uniform, choice, choices, randrange
from sklearn.utils import shuffle
from tqdm import tqdm
from planning_lib import *

path = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(path+'\Tracks_Batch2\\track#0.csv')

plt.plot(df['x'], df['y'],'*')
#plt.show()

left_cones = []
right_cones = []
for i, row in df.iterrows():
    #print(row['color'])
    #print(row.keys())
    if  row['color'] == '1':
        x=float(row['x'])
        y=float(row['y'])
        left_cones.append([x,y])
        
        #print('HERE')
    if  row['color'] == '2':
        x=float(row['x'])
        y=float(row['y'])
        right_cones.append([x,y])


# print(right_cones)
#waypoints, viz_triangles = generate_waypoints(left_cones, right_cones, True)
import timeit


startTime = time.time()
viz_triangles, spline = planning_main(left_cones, right_cones,60,True)
endTime = time.time()
howMuchTime = endTime - startTime
print(str(howMuchTime) + " sec")
# x=[point[0] for point in viz_triangles]
# y=[point[1] for point in viz_triangles]
# print(waypoints)

