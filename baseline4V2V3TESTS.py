from __future__ import print_function, division
from glob import glob
from astropy.io import fits
import numpy as np
import os
import time
import random

# other code
import TA_functions as TAf
import coords_transform as ct


# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# May 2016 - Version 1.0: initial version completed


stars_sample = [101, 103, 105, 108, 109, 111, 113, 114, 133, 136, 147, 150, 157, 158, 161, 181, 184, 185, 186, 194, 199]
detector = 491
scene = 1
arcsecs = True
bkgd_method = 'frac'
shutters = 'rapid'
noise = 'real'
verbose = True
debug = False


scene2study = "Scene"+str(scene)+"_"

# get benckmark values
benchmark_data, magnitudes = TAf.read_star_param_files(scene2study)
bench_P1, bench_P2 = benchmark_data
allbench_starP1, allbench_xP1, allbench_yP1, allbench_V2P1, allbench_V3P1, allbench_xLP1, allbench_yLP1 = bench_P1
allbench_starP2, allbench_xP2, allbench_yP2, allbench_V2P2, allbench_V3P2, allbench_xLP2, allbench_yLP2 = bench_P2
allbench_stars = allbench_starP1.tolist()

# get the index for the sample stars
star_idx_list = []
for st in stars_sample:
    st_idx = allbench_stars.index(st)
    star_idx_list.append(st_idx)

# get the benchmark for star sample
bench_starP1, bench_xP1, bench_yP1, bench_V2P1, bench_V3P1, bench_xLP1, bench_yLP1 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
bench_starP2, bench_xP2, bench_yP2, bench_V2P2, bench_V3P2, bench_xLP2, bench_yLP2 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
for i in star_idx_list:
    bench_starP1 = np.append(bench_starP1, allbench_starP1[i])
    bench_xP1 = np.append(bench_xP1, allbench_xP1[i])
    bench_yP1 = np.append(bench_yP1, allbench_yP1[i])
    bench_V2P1 = np.append(bench_V2P1, allbench_V2P1[i])
    bench_V3P1 = np.append(bench_V3P1, allbench_V3P1[i])
    bench_xLP1 = np.append(bench_xLP1, allbench_xLP1[i])
    bench_yLP1 = np.append(bench_yLP1, allbench_yLP1[i])
    bench_starP2 = np.append(bench_starP2, allbench_starP2[i])
    bench_xP2 = np.append(bench_xP2, allbench_xP2[i])
    bench_yP2 = np.append(bench_yP2, allbench_yP2[i])
    bench_V2P2 = np.append(bench_V2P2, allbench_V2P2[i])
    bench_V3P2 = np.append(bench_V3P2, allbench_V3P2[i])
    bench_xLP2 = np.append(bench_xLP2, allbench_xLP2[i])
    bench_yLP2 = np.append(bench_yLP2, allbench_yLP2[i])
if arcsecs:
    bench_V2P1 = bench_V2P1 * 3600.0
    bench_V3P1 = bench_V3P1 * 3600.0
    bench_V2P2 = bench_V2P2 * 3600.0
    bench_V3P2 = bench_V3P2 * 3600.0

trueVsP1 = [bench_V2P1, bench_V3P1]
trueVsP2 = [bench_V2P2, bench_V3P2]
LoLeftCornersP1 = [bench_xLP1, bench_yLP1]
LoLeftCornersP2 = [bench_xLP2, bench_yLP2]


# get pixel space data
if bkgd_method == 'frac':
    bkgd_method = 'Frac'
input_files_list = glob('../resultsXrandomstars/centroid_txt_files/*'+bkgd_method+"*.txt")
print (input_files_list)
data_P1 = np.loadtxt(input_files_list[0], skiprows=5, usecols=(0,1,2,3,4,5,6,7), unpack=True)
stars, bg_value, x13, y13, x15, y15, x17, y17 = data_P1
data_P2 = np.loadtxt(input_files_list[1], skiprows=5, usecols=(0,1,2,3,4,5,6,7), unpack=True)
_, _, x23, y23, x25, y25, x27, y27 = data_P2


# perform tests
case = "Scene"+str(scene)+"_"+shutters+"_"+noise+'_'+bkgd_method
transf_direction = "forward"
filter_input = 'F140X'
tilt = False

# TEST 1

# TEST 2

# TEST 3:
# (a) Transform P1 and P2 individually to V2-V3
T3V2_P1, T3V3_P1 = ct.coords_transf(transf_direction, detector, filter_input, x13, y13, tilt, arcsecs, debug)
T3V2_P2, T3V3_P2 = ct.coords_transf(transf_direction, detector, filter_input, x23, y23, tilt, arcsecs, debug)
# (b) compare star by star and position by position
T3diffV2_P1, T3diffV3_P1, T3bench_V2_list_P1, T3bench_V3_list_P1 = TAf.compare2ref(case, bench_starP1, bench_V2P1,
                                                                                   bench_V3P1, stars_sample, T3V2_P1, T3V3_P1)
T3diffV2_P2, T3diffV3_P2, T3bench_V2_list_P2, T3bench_V3_list_P2 = TAf.compare2ref(case, bench_starP2, bench_V2P2,
                                                                                   bench_V3P2, stars_sample, T3V2_P2, T3V3_P2)

print ('star', 'BG', 'V2_Position1', 'V3_Position1', 'TrueV2_P1', 'TrueV3_P1', 'DiffV2_P1', 'DiffV3_P1')
for i, st in enumerate(stars_sample):
    print(st, bg_value[i], T3V2_P1[i], T3V3_P1[i], bench_V2P1[i], bench_V3P1[i], T3diffV2_P1[i], T3diffV3_P1[i])

print ('star', 'BG', 'V2_Position2', 'V3_Position2', 'TrueV2_P2', 'TrueV3_P2', 'DiffV2_P2', 'DiffV3_P2')
for i, st in enumerate(stars_sample):
    print(st, bg_value[i], T3V2_P2[i], T3V3_P2[i], bench_V2P2[i], bench_V3P2[i], T3diffV2_P1[i], T3diffV3_P1[i])
