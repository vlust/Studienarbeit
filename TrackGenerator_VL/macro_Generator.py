"""
Script to generate multiple tracks with TrackGenerator
"""

from VL_trackGenerator import *
import pandas as pd
import json
import os

NUMBER_OF_TRACKS = 10
path=os.path.dirname(os.path.abspath(__file__))

def save_csv_allpoints(track, cones, elemets):
    df = pd.DataFrame(cones, columns =['x', 'y', 'color'])
    track_l=[]
    for point in track:
        point_l=list(point)
        point_l.extend("M")
        track_l.append(point_l)
    df_track =pd.DataFrame(track_l[0::5], columns =['x', 'y', 'color'])
    df=df.append(df_track)
    str_elements=""
    for element in elements:
        str_elements+="_"+str(element)

    filename=f"/tracks/track#{i}_elements"+str_elements+".csv"
    df.to_csv(path+filename, encoding='utf-8', index=False)

def savefig(track, cones):
    x, y = TrackGenerator.visualize_track(track)
    yellow_x, yellow_y, blue_x, blue_y=TrackGenerator.visualize_cones(cones)
    plt.plot(x, y)
    plt.plot(yellow_x,yellow_y,'*',color='orange')
    plt.plot(blue_x,blue_y,'*',color='blue')
    plt.axis('scaled')
    filename=f"/tracks/track#{i}.png"
    plt.savefig(path+filename)
    plt.clf()
    print(f"Fig_{i} done")

for i in range (NUMBER_OF_TRACKS):
    track, cones, elements, error=TrackGenerator.generate_randomTrack()
    if not error:
        save_csv_allpoints(track, cones, elements)
        savefig(track, cones)
    else:
        print("track failed")

