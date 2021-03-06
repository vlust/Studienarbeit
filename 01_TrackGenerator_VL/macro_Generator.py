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


NUMBER_OF_TRACKS = 4000
NUMBER_OF_BATCHES = 10
path = os.path.dirname(os.path.abspath(__file__))


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
    dfcut = df_padded[:50]
    return dfcut


def savefig(track, cones, i):
    x, y = TrackGenerator.visualize_track(track)
    yellow_x, yellow_y, blue_x, blue_y = TrackGenerator.visualize_cones(cones)
    plt.plot(x, y)
    plt.plot(yellow_x, yellow_y, '*', color='orange')
    plt.plot(blue_x, blue_y, '*', color='blue')
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
            save_csv(cones, filenumber, i)
            #save_csv_allpoints(track, cones, elements, filenumber)
        #t1 = time.time()
        #print(f"time: {t1-t0}\n")


if __name__ == '__main__':
    list_ranges = list(range(NUMBER_OF_BATCHES))
    #t0 = time.time()
    pool = Pool(processes=len(list_ranges))
    pool.map(macro_track_to_csv, list_ranges)
    # t1 = time.time()
    # print(f"time: {t1-t0}\n")

# macro_track_to_csv(1)
