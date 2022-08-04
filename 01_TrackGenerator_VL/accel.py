import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from scipy.special import binom
from random import uniform, choice, choices
from math import dist
import time
from VL_trackGenerator import *
import pandas as pd
import os

TRACK_WIDTH = 2
path = os.path.dirname(os.path.abspath(__file__))

def start_line(point_in):
    x = point_in[0]
    y = point_in[1]
    dist = 0.15
    cones = [[x+dist, y+TRACK_WIDTH, 3, 1], [x-dist, y+TRACK_WIDTH, 3, 1],
             [x+dist, y-TRACK_WIDTH, 3, 1], [x-dist, y-TRACK_WIDTH, 3, 1]]
    return cones


def return_straight(point_in):
    cones = []
    x = point_in[0]
    y = point_in[1]
    for i in range(14):
        cones.append([x+5*(i+1), y+TRACK_WIDTH, 1, 1])
        cones.append([x+5*(i+1), y-TRACK_WIDTH, 2, 1])
    return cones


def return_straight_or(point_in):
    cones = []
    x = point_in[0]
    y = point_in[1]
    for i in range(15):
        cones.append([x+5*(i+1), y+TRACK_WIDTH, 3, 1])
        cones.append([x+5*(i+1), y-TRACK_WIDTH, 3, 1])
    return cones


def return_endline(point_in):
    x = point_in[0]
    y = point_in[1]
    abs = 0.7
    cones = [[x, y, 3, 1], [x, y, 3, 1],
             [x, y+abs, 3, 1], [x, y+abs, 3, 1],
             [x, y-abs, 3, 1], [x, y-abs, 3, 1],
             [x, y+2*abs, 3, 1], [x, y+2*abs, 3, 1],
             [x, y-2*abs, 3, 1], [x, y-2*abs, 3, 1]]
    return cones

def save_csv_allpoints(cones, i):
    df = pd.DataFrame(cones, columns=['x', 'y', 'color', 'target'])
    TrackGenerator.show_cones(df.values.tolist())
    filename = f"/tracks/track#{i}.csv"
    df.to_csv(path+filename, encoding='utf-8', index=False)


cones = []
print(len(return_straight((0, 0))))
print(len(return_straight_or((75, 0))))
cones.extend(start_line((0, 0)))
cones.extend(return_straight((0, 0)))
cones.extend(start_line((75, 0)))
cones.extend(return_straight_or((75, 0)))
cones.extend(return_endline((150, 0)))


TrackGenerator.show_cones(cones)
save_csv_allpoints(cones, 1)
print('smth')
