from VL_trackGenerator import *
import random
# import matplotlib.pyplot as plt
import os
import json
import pandas as pd

params={"radius":4}
point_out=(-100,-20)
#data,_,_=TrackGenerator.add_refocus((0,0),point_out,(1,0),params)
#data,_,_=TrackGenerator.generate_to_next_checkpoint([(0,0)],point_out,(1,0), 5)
i=0
notviable=True
while notviable:
    data,check_points,success =TrackGenerator.generate_random_track((0,0))
    notviable= not success
    print(check_points)
    print(f'TRY NR {i}')
    i+=1
    out=list(zip(*data))
    plt.plot(out[0], out[1])
    for point in check_points:
        plt.scatter(point[0],point[1])
    plt.axis('scaled')
    plt.show()
    

