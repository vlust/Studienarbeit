import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from random import uniform, choice, choices
from math import dist
import os
import glob
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))
path_all = path+'/tracks'
os.chdir(path_all)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]


tracks = [pd.read_csv(f) for f in all_filenames]

print(tracks[0])
# np.sqrt((x2 - x1)^2 + (y2 - y1)^2)


