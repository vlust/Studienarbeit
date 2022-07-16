"""
Script to generate multiple tracks with TrackGenerator
"""

from VL_trackGenerator import *
import pandas as pd
import json
import os
import time
import functools
from multiprocessing import Pool
from random import uniform, choice, choices, randrange
from sklearn.utils import shuffle
from tqdm import tqdm


NUMBER_OF_TRACKS = 20
NUMBER_OF_BATCHES = 1
path = os.path.dirname(os.path.abspath(__file__))

def get_length_track(track):
    length=0
    lastpoint=None
    first=True
    for point in track:
        if first:
            first=False
        else:
            length += np.hypot(point[0] - lastpoint[0], point[1] - lastpoint[1])
        lastpoint=point
        
    #print(f'length{length}')
    return length

# track=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10)]
# length = get_length_track(track)
# perc = uniform(15,17)/length
# num_cones = 10
# num_cones = num_cones / 2
# num_points_mid = num_cones*perc

# points_len_new = int(len(track)*uniform(4,6)/length)
# track = track[:points_len_new]
# num = int(num_points_mid/len(track))
# track = track[0::num]

# print(track)
# length=get_length_track(track)
# points_len_new=int(len(track)*uniform(17,18)/length)
# print(points_len_new)
def save_csv_planning_track(track, cones, file):

    norms=[]

    for i in range(len(track)):
            # Skip first
            if i == 0:
                continue

            # calculate first tangent then normals
            cur_point = track[i]
            cur_tangent_angle = get_tangent_angle(track[:(i+1)])
            #print(cur_tangent_angle)

            # cur_tangent_normal = (
            #     np.ceil(
            #         np.sin(cur_tangent_angle)
            #     ),
            #     np.ceil(
            #         np.cos(cur_tangent_angle)
            #     )
            # )
            cur_tangent_normal = (
            
                round(np.sin(cur_tangent_angle),2)
            ,
          
                round(np.cos(cur_tangent_angle),2)
            )
            
            norms.append(cur_tangent_normal)
    print(norms)

    length = get_length_track(track)
    perc = uniform(15,17)/length
    #print(f'perc{perc}')
    num_cones = len(cones)
    #print(f'num_cones{num_cones}')
    num_cones = num_cones / 2
    num_points_mid = num_cones*perc
    #print(f'num_points_mid{num_points_mid}')

    points_len_new = int(len(track)*uniform(15,17)/length)
    track = track[:points_len_new]
    num = int(len(track)/uniform(5,7))
    #print(f'num{num}')
    track = track[0::num]
    norms = norms[0::num]

    
    track_l = []
    for enum, point in enumerate(track):
        point_l = list(point)
        nurms_l = list(norms[enum])
        point_l.extend(nurms_l)
        point_l.extend([1.5])
        track_l.append(point_l)
    df = pd.DataFrame(track_l, columns=['x', 'y', 'nx', 'ny', 'dist'])
    # df = pd.concat([df, df_track])
    #TrackGenerator.show_cones(df_track.values.tolist())
    plt.plot(df['x'],df['y'])
    
    #plt.show()
    filename = f"/tracks/track#{file}.csv"
    df.to_csv(path+filename, encoding='utf-8', index=False)

def save_csv_allpoints(track, cones, elements, i):
    df = pd.DataFrame(cones, columns=['x', 'y', 'color', 'target'])
    track_l = []
    for point in track:
        point_l = list(point)
        point_l.extend("M")
        track_l.append(point_l)
    df_track = pd.DataFrame(track_l[0::5], columns=['x', 'y', 'color'])
    df = pd.concat([df, df_track])
    str_elements = ""
    for element in elements:
        str_elements += "_"+str(element)
    TrackGenerator.show_cones(df.values.tolist())
    filename = f"/tracks/track#{i}.csv"
    df.to_csv(path+filename, encoding='utf-8', index=False)


def save_csv(cones, filenumber, i):
    df = pd.DataFrame(cones, columns=['x', 'y', 'color', 'target'])
    df = df[:50]
    df_shuffled = df.iloc[np.random.permutation(len(df))]
    df_shuffled = df_shuffled.reset_index(drop=True)

    #df['no'] = i

    filled_df = fill_df_zero(df_shuffled)
    # TrackGenerator.show_cones(filled_df.values.tolist())
    filename = f"/tracks/temp/tracks_batch#{filenumber}.csv"
    filled_df.to_csv(path+filename, mode='a', index=False, header=False)


def fill_df_rand(df):
    for i in range(50-len(df.index)):
        df = pd.concat([df, df.iloc[[randrange(0, len(df.index))]]])
    df = df[:50]
    return df


def fill_df_zero(df):
    df_padded = df.reindex(range(50), fill_value=0)

def savefig(track, cones, i):
    x, y = TrackGenerator.visualize_track(track)
    yellow_x, yellow_y, blue_x, blue_y,orange_x,orange_y = TrackGenerator.visualize_cones(cones)
    plt.plot(x, y)
    plt.plot(yellow_x, yellow_y, '*', color='orange')
    plt.plot(blue_x, blue_y, '*', color='blue')
    plt.plot(orange_x, orange_y, '*', color='red')
    plt.axis('scaled')
    filename = f"/tracks/track#{i}.png"
    plt.savefig(path+filename)
    plt.clf()


def macro_track_to_csv(filenumber):
    header = pd.DataFrame(columns=['x', 'y', 'color', 'target'])
    filename = f"/tracks/temp/tracks_batch#{filenumber}.csv"
    header.to_csv(path+filename, index=False)
    counter = 0
    for i in tqdm(range(NUMBER_OF_TRACKS)):
        #t0 = time.time()
        track, cones, elements, error = TrackGenerator.generate_random_local_track()
        counter += 1
        if counter == 100:
            # print(f"*********************** track for file #{filenumber+1} added {i+1} *******************************")
            counter = 0
        if not error:
            save_csv_planning_track(track, cones, i)
            #save_csv_allpoints(track, cones, elements, filenumber)
        #t1 = time.time()
        #print(f"time: {t1-t0}\n")


# if __name__ == '__main__':
#     list_ranges = list(range(NUMBER_OF_BATCHES))
#     #t0 = time.time()
#     pool = Pool(processes=len(list_ranges))
#     pool.map(macro_track_to_csv, list_ranges)
#     # t1 = time.time()
#     # print(f"time: {t1-t0}\n")

macro_track_to_csv(1)
