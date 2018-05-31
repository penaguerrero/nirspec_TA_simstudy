from __future__ import print_function
import numpy as np
import os




# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Sept 2016 - Version 1.0: initial version completed


"""
This script calculates the differences of true - measured pixel and sky positions, and determines the mean of the
differences once these have been converted to absolute.
"""

# File name to be analyzed
file_name = "gooduglies_stars_centroids_Scene1_bgFrac_Position2.txt"
#file_name = "gooduglies_T3_results_2DetsScene1_rapid_real_bgFrac0.3_Nsigma2.5.txt"

# Columns of interest
cols = (0, 2, 3, 8, 9)
#cols = (0, 2, 3, 6, 7, 10, 11)

# Load the file
path4file2analyze = "../resultsXrandomstars/good_and_uglies/"
file2analyze = os.path.join(path4file2analyze, file_name)
star, xpos, ypos, truex, truey = np.loadtxt(file2analyze, skiprows=5, usecols=cols, unpack=True)
#star, xpos, ypos, truex, truey, diffv2, diffv3 = np.loadtxt(file2analyze, skiprows=5, usecols=cols, unpack=True)
diffx = np.abs(truex - xpos)
diffy = np.abs(truey - ypos)
meanx = np.mean(diffx)
meany = np.mean(diffy)
#meanv2 = np.mean(np.abs(diffv2))
#meanv3 = np.mean(np.abs(diffv3))

print (' Mean of absolute differences (true - measured) in X = ', meanx)
print (' Mean of absolute differences (true - measured) in Y = ', meany)
#print (' Mean of absolute differences (true - measured) in V2 = ', meanv2)
#print (' Mean of absolute differences (true - measured) in V3 = ', meanv3)

print (' \n  * Script position_files_stats.py finished.')
