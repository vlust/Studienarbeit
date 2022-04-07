"""
Script to generate multiple tracks with TrackGenerator
"""

from VL_trackGenerator import *
import csv

NUMBER_OF_TRACKS = 100

for i in range (NUMBER_OF_TRACKS):
    TrackGenerator.generate_randomTrack()
    with open('filename.txt') as fp:
        #nothing
        x=0


