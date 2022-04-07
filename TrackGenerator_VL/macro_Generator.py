"""
Script to generate multiple tracks with TrackGenerator
"""

from VL_trackGenerator import *
import pandas as pd
import json
import os

NUMBER_OF_TRACKS = 10
path=os.path.dirname(os.path.abspath(__file__))

for i in range (NUMBER_OF_TRACKS):
    track, cones, elements, error=TrackGenerator.generate_randomTrack()
    x, y = TrackGenerator.visualize_track(track)
    yellow_x, yellow_y, blue_x, blue_y=TrackGenerator.visualize_cones(cones)
    plt.plot(x, y)
    plt.plot(yellow_x,yellow_y,'*',color='orange')
    plt.plot(blue_x,blue_y,'*',color='blue')
    filename=f"/tracks/track#{i}.png"
    plt.savefig(path+filename)
    plt.clf()
    print(f"Fig_{i} done")


