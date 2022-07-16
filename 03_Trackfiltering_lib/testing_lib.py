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

df = pd.read_csv(path+'/Tracks_Batch2/track#0.csv')

plt.plot(df['x'], df['y'],'*')
plt.show()
print(df)
left_cones = []
right_cones = []
for i, row in df.iterrows():
    #print(row['color'])
    #print(row.keys())
    if  row['color'] == '1':
        left_cones.append((row['x'],row['y']))
        #print('HERE')
    if  row['color'] == '2':
        right_cones.append((row['x'],row['y']))

print(left_cones)

print(right_cones)
waypoints, viz_triangles = generate_waypoints(left_cones, right_cones, True)

