from VL_trackGenerator import *
import random
# import matplotlib.pyplot as plt
import os
import json
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))


def save_csv_allpoints(track, cones, i):
    df = pd.DataFrame(cones, columns=['x', 'y', 'color', 'target'])
    track_l = []
    for point in track:
        point_l = list(point)
        point_l.extend("M")
        track_l.append(point_l)
    df_track = pd.DataFrame(track_l[0::5], columns=['x', 'y', 'color'])
    df = pd.concat([df, df_track])

    TrackGenerator.show_cones(df.values.tolist())
    filename = f"/tracks/track#{i}.csv"
    df.to_csv(path+filename, encoding='utf-8', index=False)


def savefig(track, cones, i):
    x, y = TrackGenerator.visualize_track(track)
    yellow_x, yellow_y, blue_x, blue_y, orange_x, orange_y = TrackGenerator.visualize_cones(
        cones)
    plt.plot(x, y)
    plt.plot(yellow_x, yellow_y, '*', color='orange')
    plt.plot(blue_x, blue_y, '*', color='blue')
    plt.plot(orange_x, orange_y, '*', color='red')
    plt.axis('scaled')
    filename = f"/tracks/track#{i}.png"
    plt.savefig(path+filename)
    plt.clf()


params = {"radius": 4}
point_out = (-100, -20)
# data,_,_=TrackGenerator.add_refocus((0,0),point_out,(1,0),params)
#data,_,_=TrackGenerator.generate_to_next_checkpoint([(0,0)],point_out,(1,0), 5)
i = 0

num_tracks=1
while num_tracks<10:
    notviable = True
    while notviable:
        data, check_points, success, dir_point = TrackGenerator.generate_random_track(
            (0, 0))
        notviable = not success
        if not notviable:
            cones = TrackGenerator.get_cones(data)
            savefig(data, cones, i)
            save_csv_allpoints(data, cones, i)
            # print(check_points)
            TrackGenerator.visualize_all(data, cones)

            print(f'TRY NR {i}')
            i += 1
            num_tracks+=1
    # out=list(zip(*data))
    # plt.plot(out[0], out[1])
    # for point in check_points:
    #     plt.scatter(point[0],point[1])
    # plt.scatter(dir_point[0],dir_point[1])
    # plt.axis('scaled')
    # plt.show()
