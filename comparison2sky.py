from __future__ import print_function, division
from glob import glob
import numpy as np
import os
import collections
# other code
import coords_transform as ct
import testing_functions as tf 
import least_squares_iterate as lsi
print("Modules correctly imported! \n")



# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# May 2016 - Version 1.0: initial version completed


"""
This script tests which of the following options obtains the smaller difference to the true sky positions (V2-V3 space):
1. Average positions P1 and P2, transform to V2-V3 space, and compare to average reference positions (V2-V3 space)
2. Transform individual positions P1 and P2 to V2-V3 space, average V2-V3 space positions, and compare to 
   average reference positions.
3. Transform P1 and P2 individually to V2-V3 space and compare star by star and position by position.

It outputs 3 text file with results of test per case into directory TargetAcquisition/Coords_transforms/results/
* case depends on scene, noise, background value, and shutter velocity; results in 36 files per scene.
"""

#######################################################################################################################

# general settings
detector = 491           # detector, integer: 491 or 492
Nsigma = 3               # N-sigma rejection of bad stars, integer or float
max_iterations = 10      # Max number of iterations for N-sigma function, integer
bkgd_method = "None"     # background to test, string: all, None, fixed, frac  
filter_input = "F140X"   # Filter, string: for now only test case is F140X
full_detector = False    # Give resulting coordinates in terms of full detector: True or False
Pier_corr = True         # Include Pier's corrections to measured positions
show_positions = False   # Print positions on file and screen: True or False
tilt = False             # tilt angle: True or False
debug = True            # See screen print statements for intermediate answers: True or False 
save_txt_file = False    # Save text file with resulting transformations: True or False
arcsecs = True          # Print the differences in arcsecs? True or False (=degrees)
single_case = 103       # test only a particular case: integer number of star, else set to None
# Known bad stars in X and Y: 103, 105, 106, 112, 134, 152, 156, 170, 188

#######################################################################################################################

#  --> FUNCTIONS        
    
def get_mindiff(d1, d2, d3):
    """ This function determines the minimum difference from checkboxes 3, 5, and 7,
    and counts the number of repetitions. """
    min_diff = []
    for i, _ in enumerate(d1):
        diffs_list = [d1[i], d2[i], d3[i]]
        md = min(diffs_list)
        if md == d1[i]:
            m_diff = 3
        elif md == d2[i]:
            m_diff = 5
        elif md == d3[i]:
            m_diff = 7
        min_diff.append(m_diff)
    counter=collections.Counter(min_diff)
    return min_diff, counter


def find_best_fracbgvalue(diffV2, diffV3):
    """ This function uses the next 2 functions to determine the fractional background value that has
    the smallest difference with respect to true values for all 100 stars.
    Input:      - diffV2, diffV3  = numpy arrays of the fractional background True-Measured V2 
                                    and V3 positions for the 100 stars
    Output:     - list_best_frbg_value_V2, list_best_frbg_value_V3 = list of best fractional backgrounds 
                  (the best value is repeated 11 times for each star, so that the length is the same to
                  that of the differences arrays)
    """
    list_best_frbg_value_V2, list_best_frbg_value_V3 = [], []
    # Divide the whole array into a list of arrays according to star number
    arrsV2_list, arrsV3_list = divide_array_by_starNo(diffV2, diffV3)
    # Obtain the list of best fractional background for each star in the list
    for slice_diffV2, slice_diffV3 in zip(arrsV2_list, arrsV3_list):
        best_frbg_value_V2, best_frbg_value_V3 = get_best_fracvalue(slice_diffV2, slice_diffV3)
        list_best_frbg_value_V2.append(best_frbg_value_V2)
        list_best_frbg_value_V3.append(best_frbg_value_V3)
    # Flatten the list of lists into a single list 
    list_best_frbg_value_V2 = [item for sublist in list_best_frbg_value_V2 for item in sublist]
    list_best_frbg_value_V3 = [item for sublist in list_best_frbg_value_V3 for item in sublist]
    counterV2 = collections.Counter(list_best_frbg_value_V2)
    counterV3 = collections.Counter(list_best_frbg_value_V3)
    return list_best_frbg_value_V2, list_best_frbg_value_V3, counterV2, counterV3

def divide_array_by_starNo(arrX, arrY):
    """ This function divides the array of fractional background into slices corresponding to star number. 
    Input:     - arr        =  numpy array to be divided
    Output:    - arrsX_list  =  list of numpy X arrays corresponding to star number
               - arrsY_list  =  list of numpy Y arrays corresponding to star number
    """
    arrsX_list, arrsY_list = [], []
    frac_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counter = 0
    star_arrayX, star_arrayY = np.array([]), np.array([])
    for itemX, itemY in zip(arrX, arrY):
        counter += 1
        if counter <= len(frac_values):
            star_arrayX = np.append(star_arrayX, itemX)
            star_arrayY = np.append(star_arrayY, itemY)
        if counter == len(frac_values):
            arrsX_list.append(star_arrayX)
            arrsY_list.append(star_arrayY)
            counter = 0
            star_arrayX, star_arrayY = np.array([]), np.array([])
    return arrsX_list, arrsY_list
    
def get_best_fracvalue(slice_diffV2, slice_diffV3):
    """ This function determines which background fractional value has the smallest difference 
    with respect true sky positions. 
    Input:     diffV2, diffV3 = numpy arrays of the fractional background True-Measured V2 
                                and V3 positions for the same star (each array has 11 elements)
    Output:    best_frbg_value_V2 = list of same minimum difference of True-Measured V2
               best_frbg_value_V3 = list of same minimum difference of True-Measured V3
    """
    frac_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_diff_V2 = min(slice_diffV2)
    min_diff_V3 = min(slice_diffV3)
    #print (slice_diffV2)
    #print (slice_diffV3)
    #print ("mins for fractional backgrounds in V2 and V3: ", min_diff_V2, min_diff_V3)
    #raw_input()
    best_frbg_value_V2, best_frbg_value_V3 = [], []
    for dv2, dv3 in zip(slice_diffV2, slice_diffV3):
        if dv2 == min_diff_V2:
            idx2 = slice_diffV2.tolist().index(min_diff_V2)
            # make sure the smallest value is not 0.0
            if idx2 == 0:
                min_diff_V2 = second_smallest(slice_diffV2)
                idx2 = slice_diffV2.tolist().index(min_diff_V2)
            #print ("best fractional background value for V2 is: ", frac_values[idx])
        if dv3 == min_diff_V3:
            idx3 = slice_diffV3.tolist().index(min_diff_V3)
            # make sure the smallest value is not 0.0
            if idx3 == 0:
                min_diff_V3 = second_smallest(slice_diffV3)
                idx3 = slice_diffV3.tolist().index(min_diff_V3)
            #print ("best fractional background value for V3 is: ", frac_values[idx])
    # repeat each value 11 times
    for _ in frac_values:
        best_frbg_value_V2.append(frac_values[idx2])
        best_frbg_value_V3.append(frac_values[idx3])    
    return best_frbg_value_V2, best_frbg_value_V3

def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2
    

#######################################################################################################################

#  --> CODE

if single_case is not None:
    save_txt_file = False   # just in case, do not overwrite text file with only one case.

# Define the paths for results and inputs
path4results = "../Coords_transforms/detector_"+str(detector)+"/results/"
path4inputP1P2 = "../PFforMaria/detector_"+str(detector)+"_comparison_txt_positions/"

# Get true positions from Pierre's position files from Tony's star parameters file 
# Paths to Scenes 1 and 2 local directories: /Users/pena/Documents/AptanaStudio3/NIRSpec/TargetAcquisition/
path4starfiles = "../PFforMaria/"
path_scene1_slow = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRS/postage_redo"
path_scene1_slow_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRS no_noise/postage_redo"
path_scene1_rapid = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRSRAPID/postage_redo"
path_scene1_rapid_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRS no_noise/postage_redo"
path_scene1_slow_shifted = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRS/postage_redo"
path_scene1_slow_shifted_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRS no_noise/postage_redo"
path_scene1_rapid_shifted = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRSRAPID/postage_redo"
path_scene1_rapid_shifted_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRS no_noise/postage_redo"
path_scene2_slow = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRS/postage_redo"
path_scene2_slow_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRS no_noise/postage_redo"
path_scene2_rapid = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRSRAPID/postage_redo"
path_scene2_rapid_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRSRAPID no_noise/postage_redo"
path_scene2_slow_shifted = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRS/postage_redo"
path_scene2_slow_shifted_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRS no_noise/postage_redo"
path_scene2_rapid_shifted = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRSRAPID/postage_redo"
path_scene2_rapid_shifted_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRSRAPID no_noise/postage_redo"

paths_list = [path_scene1_slow, path_scene1_slow_nonoise, 
              path_scene1_rapid, path_scene1_rapid_nonoise,
              path_scene1_slow_shifted, path_scene1_slow_shifted_nonoise, 
              path_scene1_rapid_shifted, path_scene1_rapid_shifted_nonoise, 
              path_scene2_slow, path_scene2_slow_nonoise, 
              path_scene2_rapid, path_scene2_rapid_nonoise,
              path_scene2_slow_shifted, path_scene2_slow_shifted_nonoise, 
              path_scene2_rapid_shifted, path_scene2_rapid_shifted_nonoise]

# Get list of all input files of the background method given to test
if bkgd_method == "all":
    input_files_list = glob(path4inputP1P2+"*.txt")
else:
    input_files_list = glob(path4inputP1P2+"*"+bkgd_method+"*.txt")

#input_files_list = glob('../resultsXrandomstars/centroid_txt_files/*'+bkgd_method+"*.txt")

# Start TESTS the loop
for infile in input_files_list:
    # define the case to work with 
    case = os.path.basename(infile).replace(".txt", "")
    case = case.replace("_centroids", "")
    print ("* studying case: ", case)
    
    # get the benchmark data
    benchmark_data, magnitudes = tf.read_star_param_files(case, detector, path4starfiles, paths_list)
    bench_P1, bench_P2 = benchmark_data
    bench_starP1, bench_xP1, bench_yP1, bench_V2P1, bench_V3P1, bench_xLP1, bench_yLP1 = bench_P1
    bench_starP2, bench_xP2, bench_yP2, bench_V2P2, bench_V3P2, bench_xLP2, bench_yLP2 = bench_P2
    if single_case:
        bench_stars = bench_starP1.tolist()
        star_idx = bench_stars.index(single_case)
        bench_starP1 = np.array([bench_starP1[star_idx]])
        bench_xP1 = np.array([bench_xP1[star_idx]])
        bench_yP1 = np.array([bench_yP1[star_idx]])
        bench_V2P1 = np.array([bench_V2P1[star_idx]])
        bench_V3P1 = np.array([bench_V3P1[star_idx]])
        bench_xLP1 = np.array([bench_xLP1[star_idx]])
        bench_yLP1 = np.array([bench_yLP1[star_idx]])
        bench_starP2 = np.array([bench_starP2[star_idx]])
        bench_xP2 = np.array([bench_xP2[star_idx]])
        bench_yP2 = np.array([bench_yP2[star_idx]])
        bench_V2P2 = np.array([bench_V2P2[star_idx]])
        bench_V3P2 = np.array([bench_V3P2[star_idx]])
        bench_xLP2 = np.array([bench_xLP2[star_idx]])
        bench_yLP2 = np.array([bench_yLP2[star_idx]])
    #avg_benchX = (bench_xP1 + bench_xP2)/2.0
    #avg_benchY = (bench_yP1 + bench_yP2)/2.0
    if arcsecs:
        bench_V2P1 = bench_V2P1 * 3600.
        bench_V2P2 = bench_V2P2 * 3600.
        bench_V3P1 = bench_V3P1 * 3600.
        bench_V3P2 = bench_V3P2 * 3600.
    avg_benchV2 = (bench_V2P1 + bench_V2P2)/2.0
    avg_benchV3 = (bench_V3P1 + bench_V3P2)/2.0

        
    # read the measured detector centroids
    data = np.loadtxt(infile, skiprows=2, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack=True)
    stars, bg_value, x13, y13, x15, y15, x17, y17, x23, y23, x25, y25, x27, y27 = data
    #data_P1 = np.loadtxt(input_files_list[0], skiprows=5, usecols=(0,1,2,3,4,5,6,7), unpack=True)
    #stars, bg_value, x13, y13, x15, y15, x17, y17 = data_P1
    #data_P2 = np.loadtxt(input_files_list[1], skiprows=5, usecols=(0,1,2,3,4,5,6,7), unpack=True)
    #_, _, x23, y23, x25, y25, x27, y27 = data_P2
    if single_case:
        star_idx = stars.tolist().index(single_case)
        if "frac" in case:
            bg_value = np.array(bg_value[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            x13 = np.array(x13[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            y13 = np.array(y13[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            x15 = np.array(x15[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            y15 = np.array(y15[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            x17 = np.array(x17[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            y17 = np.array(y17[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            x23 = np.array(x23[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            y23 = np.array(y23[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            x25 = np.array(x25[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            y25 = np.array(y25[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            x27 = np.array(x27[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            y27 = np.array(y27[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
            stars = np.array(stars[ (stars==stars[star_idx]) & (stars <= stars[star_idx+11]) ])
        else:
            stars = np.array([stars[star_idx]])
            bg_value = np.array([bg_value[star_idx]])
            x13 = np.array([x13[star_idx]])
            y13 = np.array([y13[star_idx]])
            x15 = np.array([x15[star_idx]])
            y15 = np.array([y15[star_idx]])
            x17 = np.array([x17[star_idx]])
            y17 = np.array([y17[star_idx]])
            x23 = np.array([x23[star_idx]])
            y23 = np.array([y23[star_idx]])
            x25 = np.array([x25[star_idx]])
            y25 = np.array([y25[star_idx]])
            x27 = np.array([x27[star_idx]])
            y27 = np.array([y27[star_idx]])
            
    
    if debug or single_case:
        print ("Check that read BENCHMARK values correspond to expected for case: ", case)
        print ("Star, xP1, yP1, V2P1, V3P1, xLP1, yLP1")
        print (bench_starP1[0], bench_xP1[0], bench_yP1[0], bench_V2P1[0], bench_V3P1[0], bench_xLP1[0], bench_yLP1[0])
        print ("Star, xP2, yP2, V2P2, V3P2, xLP2, yLP2")
        print (bench_starP2[0], bench_xP2[0], bench_yP2[0], bench_V2P2[0], bench_V3P2[0], bench_xLP2[0], bench_yLP2[0])
        print ("Check that read MEASURED values correspond to expected for the same case: ", case)
        print ("   -> reading measured infro from: ", infile)
        print ("Star, BG, x13, y13, x15, y15, x17, y17, LoLeftP1 (x, y), TrueP1 (x, y)")
        print (stars[0], bg_value[0], x13[0], y13[0], x15[0], y15[0], x17[0], y17[0], bench_xLP1[0], bench_yLP1[0], bench_xP1[0], bench_yP1[0])
        print ("Star, BG, x23, y23, x25, y25, x27, y27, LoLeftP2 (x, y), TrueP2 (x, y)")
        print (stars[0], x23[0], y23[0], x25[0], y25[0], x27[0], y27[0], bench_xLP2[0], bench_yLP2[0], bench_xP2[0], bench_yP2[0])
        raw_input(" * press enter to continue... \n")

    # convert from 32x32 pixel to full detector coordinates
    P1P2data = [x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27]
    benchmark_xLyL_P1 = [bench_xLP1, bench_yLP1]
    benchmark_xLyL_P2 = [bench_xLP2, bench_yLP2]
    x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27 = tf.convert2fulldetector(detector, stars, P1P2data, bench_starP1, benchmark_xLyL_P1, benchmark_xLyL_P2, Pier_corr=Pier_corr)
    
    # TEST 1: (a) Avg P1 and P2, (b) transform to V2-V3, (c) compare to avg reference positions (V2-V3 space)
    transf_direction = "forward"
    # Step (a) - averages
    avgx3 = (x13+x23)/2.0
    avgy3 = (y13+y23)/2.0
    avgx5 = (x15+x25)/2.0
    avgy5 = (y15+y25)/2.0
    avgx7 = (x17+x27)/2.0
    avgy7 = (y17+y27)/2.0
    # Step (b) - transformations to degrees
    T1_V2_3, T1_V3_3 = ct.coords_transf(transf_direction, detector, filter_input, avgx3, avgy3, tilt, arcsecs, debug)
    T1_V2_5, T1_V3_5 = ct.coords_transf(transf_direction, detector, filter_input, avgx5, avgy5, tilt, arcsecs, debug)
    T1_V2_7, T1_V3_7 = ct.coords_transf(transf_direction, detector, filter_input, avgx7, avgy7, tilt, arcsecs, debug)
    # Step (c) - comparison
    T1_diffV2_3, T1_diffV3_3, T1bench_V2_list, T1bench_V3_list = tf.compare2ref(case, bench_starP1, avg_benchV2, avg_benchV3, stars, T1_V2_3, T1_V3_3)
    T1_diffV2_5, T1_diffV3_5, _, _ = tf.compare2ref(case, bench_starP1, avg_benchV2, avg_benchV3, stars, T1_V2_5, T1_V3_5)
    T1_diffV2_7, T1_diffV3_7, _, _ = tf.compare2ref(case, bench_starP1, avg_benchV2, avg_benchV3, stars, T1_V2_7, T1_V3_7)
    # get the minimum of the differences
    T1_min_diff, T1_counter = get_mindiff(T1_diffV2_3, T1_diffV2_5, T1_diffV2_7)
    # get the fractional value that has the smaller difference
    if "frac" in case:
        T1list_best_frbg_value_V2_3, T1list_best_frbg_value_V3_3, T1counterV2_3, T1counterV3_3 = find_best_fracbgvalue(T1_diffV2_3, T1_diffV3_3)
        T1list_best_frbg_value_V2_5, T1list_best_frbg_value_V3_5, T1counterV2_5, T1counterV3_5 = find_best_fracbgvalue(T1_diffV2_5, T1_diffV3_5)
        T1list_best_frbg_value_V2_7, T1list_best_frbg_value_V3_7, T1counterV2_7, T1counterV3_7 = find_best_fracbgvalue(T1_diffV2_7, T1_diffV3_7)
    # calculate standard deviations and means
    if not single_case:    
        T1stdev_V2_3, T1mean_V2_3 = tf.find_std(T1_diffV2_3)
        T1stdev_V2_5, T1mean_V2_5 = tf.find_std(T1_diffV2_5)
        T1stdev_V2_7, T1mean_V2_7 = tf.find_std(T1_diffV2_7)
        T1stdev_V3_3, T1mean_V3_3 = tf.find_std(T1_diffV3_3)
        T1stdev_V3_5, T1mean_V3_5 = tf.find_std(T1_diffV3_5)
        T1stdev_V3_7, T1mean_V3_7 = tf.find_std(T1_diffV3_7)
        T1bench_V2, T1bench_V3 = np.array(T1bench_V2_list), np.array(T1bench_V3_list)
        print ("For TEST 1: ")
        # to express in arcsecs multiply by 3600.0
        T1LSdeltas_3, T1LSsigmas_3, T1LSlines2print_3 = lsi.ls_fit_iter(max_iterations, T1_V2_3, T1_V3_3, T1bench_V2, T1bench_V3, Nsigma=Nsigma)
        T1LSdeltas_5, T1LSsigmas_5, T1LSlines2print_5 = lsi.ls_fit_iter(max_iterations, T1_V2_5, T1_V3_5, T1bench_V2, T1bench_V3, Nsigma=Nsigma)
        T1LSdeltas_7, T1LSsigmas_7, T1LSlines2print_7 = lsi.ls_fit_iter(max_iterations, T1_V2_7, T1_V3_7, T1bench_V2, T1bench_V3, Nsigma=Nsigma)
        # Do N-sigma rejection
        T1sigmaV2_3, T1meanV2_3, T1sigmaV3_3, T1meanV3_3, T1newV2_3, T1newV3_3, T1niter_3, T1lines2print_3 = tf.Nsigma_rejection(Nsigma, T1_diffV2_3, T1_diffV3_3, max_iterations)
        T1sigmaV2_5, T1meanV2_5, T1sigmaV3_5, T1meanV3_5, T1newV2_5, T1newV3_5, T1niter_5, T1lines2print_5 = tf.Nsigma_rejection(Nsigma, T1_diffV2_5, T1_diffV3_5, max_iterations)
        T1sigmaV2_7, T1meanV2_7, T1sigmaV3_7, T1meanV3_7, T1newV2_7, T1newV3_7, T1niter_7, T1lines2print_7 = tf.Nsigma_rejection(Nsigma, T1_diffV2_7, T1_diffV3_7, max_iterations)
    
    # TEST 2: (a) Transform individual P1 and P2 to V2-V3, (b) avg V2-V3 space positions, (c) compare to avg reference positions
    # Step (a) - transformations
    T2_V2_13, T2_V3_13 = ct.coords_transf(transf_direction, detector, filter_input, x13, y13, tilt, arcsecs, debug)
    T2_V2_15, T2_V3_15 = ct.coords_transf(transf_direction, detector, filter_input, x15, y15, tilt, arcsecs, debug)
    T2_V2_17, T2_V3_17 = ct.coords_transf(transf_direction, detector, filter_input, x17, y17, tilt, arcsecs, debug)
    T2_V2_23, T2_V3_23 = ct.coords_transf(transf_direction, detector, filter_input, x23, y23, tilt, arcsecs, debug)
    T2_V2_25, T2_V3_25 = ct.coords_transf(transf_direction, detector, filter_input, x25, y25, tilt, arcsecs, debug)
    T2_V2_27, T2_V3_27 = ct.coords_transf(transf_direction, detector, filter_input, x27, y27, tilt, arcsecs, debug)
    # Step (b) - averages
    T2_V2_3 = (T2_V2_13 + T2_V2_23)/2.0
    T2_V3_3 = (T2_V3_13 + T2_V3_23)/2.0
    T2_V2_5 = (T2_V2_15 + T2_V2_25)/2.0
    T2_V3_5 = (T2_V3_15 + T2_V3_25)/2.0
    T2_V2_7 = (T2_V2_17 + T2_V2_27)/2.0
    T2_V3_7 = (T2_V3_17 + T2_V3_27)/2.0
    # Step (c) - comparison
    T2_diffV2_3, T2_diffV3_3, T2bench_V2_list, T2bench_V3_list = tf.compare2ref(case, bench_starP1, avg_benchV2, avg_benchV3, stars, T2_V2_3, T2_V3_3)
    T2_diffV2_5, T2_diffV3_5, _, _ = tf.compare2ref(case, bench_starP1, avg_benchV2, avg_benchV3, stars, T2_V2_5, T2_V3_5)
    T2_diffV2_7, T2_diffV3_7, _, _ = tf.compare2ref(case, bench_starP1, avg_benchV2, avg_benchV3, stars, T2_V2_7, T2_V3_7)
    # get the minimum of the differences
    T2_min_diff, T2_counter = get_mindiff(T2_diffV2_3, T2_diffV2_5, T2_diffV2_7)
    # get the fractional value that has the smaller difference
    if "frac" in case:
        T2list_best_frbg_value_V2_3, T2list_best_frbg_value_V3_3, T2counterV2_3, T2counterV3_3 = find_best_fracbgvalue(T2_diffV2_3, T2_diffV3_3)
        T2list_best_frbg_value_V2_5, T2list_best_frbg_value_V3_5, T2counterV2_5, T2counterV3_5 = find_best_fracbgvalue(T2_diffV2_5, T2_diffV3_5)
        T2list_best_frbg_value_V2_7, T2list_best_frbg_value_V3_7, T2counterV2_7, T2counterV3_7 = find_best_fracbgvalue(T2_diffV2_7, T2_diffV3_7)
    # calculate standard deviations and means
    if not single_case:    
        T2stdev_V2_3, T2mean_V2_3 = tf.find_std(T2_diffV2_3)
        T2stdev_V2_5, T2mean_V2_5 = tf.find_std(T2_diffV2_5)
        T2stdev_V2_7, T2mean_V2_7 = tf.find_std(T2_diffV2_7)
        T2stdev_V3_3, T2mean_V3_3 = tf.find_std(T2_diffV3_3)
        T2stdev_V3_5, T2mean_V3_5 = tf.find_std(T2_diffV3_5)
        T2stdev_V3_7, T2mean_V3_7 = tf.find_std(T2_diffV3_7)
        T2bench_V2, T2bench_V3 = np.array(T2bench_V2_list), np.array(T2bench_V3_list)
        print ("For TEST 2: ")
        T2LSdeltas_3, T2LSsigmas_3, T2LSlines2print_3 = lsi.ls_fit_iter(max_iterations, T2_V2_3, T2_V3_3, T2bench_V2, T2bench_V3, Nsigma=Nsigma)
        T2LSdeltas_5, T2LSsigmas_5, T2LSlines2print_5 = lsi.ls_fit_iter(max_iterations, T2_V2_5, T2_V3_5, T2bench_V2, T2bench_V3, Nsigma=Nsigma)
        T2LSdeltas_7, T2LSsigmas_7, T2LSlines2print_7 = lsi.ls_fit_iter(max_iterations, T2_V2_7, T2_V3_7, T2bench_V2, T2bench_V3, Nsigma=Nsigma)
        # Do N-sigma rejection
        T2sigmaV2_3, T2meanV2_3, T2sigmaV3_3, T2meanV3_3, T2newV2_3, T2newV3_3, T2niter_3, T2lines2print_3 = tf.Nsigma_rejection(Nsigma, T2_diffV2_3, T2_diffV3_3, max_iterations)
        T2sigmaV2_5, T2meanV2_5, T2sigmaV3_5, T2meanV3_5, T2newV2_5, T2newV3_5, T2niter_5, T2lines2print_5 = tf.Nsigma_rejection(Nsigma, T2_diffV2_5, T2_diffV3_5, max_iterations)
        T2sigmaV2_7, T2meanV2_7, T2sigmaV3_7, T2meanV3_7, T2newV2_7, T2newV3_7, T2niter_7, T2lines2print_7 = tf.Nsigma_rejection(Nsigma, T2_diffV2_7, T2_diffV3_7, max_iterations)
    
    # TEST 3: (a) Transform P1 and P2 individually to V2-V3 (b) compare star by star and position by position
    # Step (a) - transformations
    T3_V2_13, T3_V3_13 = ct.coords_transf(transf_direction, detector, filter_input, x13, y13, tilt, arcsecs, debug)
    T3_V2_15, T3_V3_15 = ct.coords_transf(transf_direction, detector, filter_input, x15, y15, tilt, arcsecs, debug)
    T3_V2_17, T3_V3_17 = ct.coords_transf(transf_direction, detector, filter_input, x17, y17, tilt, arcsecs, debug)
    T3_V2_23, T3_V3_23 = ct.coords_transf(transf_direction, detector, filter_input, x23, y23, tilt, arcsecs, debug)
    T3_V2_25, T3_V3_25 = ct.coords_transf(transf_direction, detector, filter_input, x25, y25, tilt, arcsecs, debug)
    T3_V2_27, T3_V3_27 = ct.coords_transf(transf_direction, detector, filter_input, x27, y27, tilt, arcsecs, debug)
    # Step (b) - comparison
    T3_diffV2_13, T3_diffV3_13, T3bench_V2_listP1, T3bench_V3_listP1 = tf.compare2ref(case, bench_starP1, bench_V2P1, bench_V3P1, stars, T3_V2_13, T3_V3_13)
    T3_diffV2_23, T3_diffV3_23, T3bench_V2_listP2, T3bench_V3_listP2 = tf.compare2ref(case, bench_starP1, bench_V2P2, bench_V3P2, stars, T3_V2_23, T3_V3_23)
    T3_diffV2_15, T3_diffV3_15, _, _ = tf.compare2ref(case, bench_starP1, bench_V2P1, bench_V3P1, stars, T3_V2_15, T3_V3_15)
    T3_diffV2_25, T3_diffV3_25, _, _ = tf.compare2ref(case, bench_starP1, bench_V2P2, bench_V3P2, stars, T3_V2_25, T3_V3_25)
    T3_diffV2_17, T3_diffV3_17, _, _ = tf.compare2ref(case, bench_starP1, bench_V2P1, bench_V3P1, stars, T3_V2_17, T3_V3_17)
    T3_diffV2_27, T3_diffV3_27, _, _ = tf.compare2ref(case, bench_starP1, bench_V2P2, bench_V3P2, stars, T3_V2_27, T3_V3_27)
    # get the minimum of the differences
    T3_min_diff1, T3_counter1 = get_mindiff(T3_diffV2_13, T3_diffV2_15, T3_diffV2_17)
    T3_min_diff2, T3_counter2 = get_mindiff(T3_diffV2_23, T3_diffV2_25, T3_diffV2_27)
    # get the fractional value that has the smaller difference
    if "frac" in case:
        T3list_best_frbg_value_V2_13, T3list_best_frbg_value_V3_13, T3counterV2_13, T3counterV3_13 = find_best_fracbgvalue(T3_diffV2_13, T3_diffV3_13)
        T3list_best_frbg_value_V2_15, T3list_best_frbg_value_V3_15, T3counterV2_15, T3counterV3_15 = find_best_fracbgvalue(T3_diffV2_15, T3_diffV3_15)
        T3list_best_frbg_value_V2_17, T3list_best_frbg_value_V3_17, T3counterV2_17, T3counterV3_17 = find_best_fracbgvalue(T3_diffV2_17, T3_diffV3_17)
        T3list_best_frbg_value_V2_23, T3list_best_frbg_value_V3_23, T3counterV2_23, T3counterV3_23 = find_best_fracbgvalue(T3_diffV2_23, T3_diffV3_23)
        T3list_best_frbg_value_V2_25, T3list_best_frbg_value_V3_25, T3counterV2_25, T3counterV3_25 = find_best_fracbgvalue(T3_diffV2_25, T3_diffV3_25)
        T3list_best_frbg_value_V2_27, T3list_best_frbg_value_V3_27, T3counterV2_27, T3counterV3_27 = find_best_fracbgvalue(T3_diffV2_27, T3_diffV3_27)
    # calculate standard deviations and means
    if not single_case:    
        T3stdev_V2_13, T3mean_V2_13 = tf.find_std(T3_diffV2_13)
        T3stdev_V2_15, T3mean_V2_15 = tf.find_std(T3_diffV2_15)
        T3stdev_V2_17, T3mean_V2_17 = tf.find_std(T3_diffV2_17)
        T3stdev_V3_13, T3mean_V3_13 = tf.find_std(T3_diffV3_13)
        T3stdev_V3_15, T3mean_V3_15 = tf.find_std(T3_diffV3_15)
        T3stdev_V3_17, T3mean_V3_17 = tf.find_std(T3_diffV3_17)
        T3stdev_V2_23, T3mean_V2_23 = tf.find_std(T3_diffV2_23)
        T3stdev_V2_25, T3mean_V2_25 = tf.find_std(T3_diffV2_25)
        T3stdev_V2_27, T3mean_V2_27 = tf.find_std(T3_diffV2_27)
        T3stdev_V3_23, T3mean_V3_23 = tf.find_std(T3_diffV3_23)
        T3stdev_V3_25, T3mean_V3_25 = tf.find_std(T3_diffV3_25)
        T3stdev_V3_27, T3mean_V3_27 = tf.find_std(T3_diffV3_27)
        T3bench_V2P1, T3bench_V3P1 = np.array(T3bench_V2_listP1), np.array(T3bench_V3_listP1)
        T3bench_V2P2, T3bench_V3P2 = np.array(T3bench_V2_listP2), np.array(T3bench_V3_listP2)
        print ("For TEST 3: ")
        T3LSdeltas_13, T3LSsigmas_13, T3LSlines2print_13 = lsi.ls_fit_iter(max_iterations, T3_V2_13, T3_V3_13, T3bench_V2P1, T3bench_V3P1, Nsigma=Nsigma)
        T3LSdeltas_15, T3LSsigmas_15, T3LSlines2print_15 = lsi.ls_fit_iter(max_iterations, T3_V2_15, T3_V3_15, T3bench_V2P1, T3bench_V3P1, Nsigma=Nsigma)
        T3LSdeltas_17, T3LSsigmas_17, T3LSlines2print_17 = lsi.ls_fit_iter(max_iterations, T3_V2_17, T3_V3_17, T3bench_V2P1, T3bench_V3P1, Nsigma=Nsigma)
        T3LSdeltas_23, T3LSsigmas_23, T3LSlines2print_23 = lsi.ls_fit_iter(max_iterations, T3_V2_23, T3_V3_23, T3bench_V2P2, T3bench_V3P2, Nsigma=Nsigma)
        T3LSdeltas_25, T3LSsigmas_25, T3LSlines2print_25 = lsi.ls_fit_iter(max_iterations, T3_V2_25, T3_V3_25, T3bench_V2P2, T3bench_V3P2, Nsigma=Nsigma)
        T3LSdeltas_27, T3LSsigmas_27, T3LSlines2print_27 = lsi.ls_fit_iter(max_iterations, T3_V2_27, T3_V3_27, T3bench_V2P2, T3bench_V3P2, Nsigma=Nsigma)
        # Do N-sigma rejection
        T3sigmaV2_13, T3meanV2_13, T3sigmaV3_13, T3meanV3_13, T3newV2_13, T3newV3_13, T3niter_13, T3lines2print_13 = tf.Nsigma_rejection(Nsigma, T3_diffV2_13, T3_diffV3_13, max_iterations)
        T3sigmaV2_15, T3meanV2_15, T3sigmaV3_15, T3meanV3_15, T3newV2_15, T3newV3_15, T3niter_15, T3lines2print_15 = tf.Nsigma_rejection(Nsigma, T3_diffV2_15, T3_diffV3_15, max_iterations)
        T3sigmaV2_17, T3meanV2_17, T3sigmaV3_17, T3meanV3_17, T3newV2_17, T3newV3_17, T3niter_17, T3lines2print_17 = tf.Nsigma_rejection(Nsigma, T3_diffV2_17, T3_diffV3_17, max_iterations)
        T3sigmaV2_23, T3meanV2_23, T3sigmaV3_23, T3meanV3_23, T3newV2_23, T3newV3_23, T3niter_23, T3lines2print_23 = tf.Nsigma_rejection(Nsigma, T3_diffV2_23, T3_diffV3_23, max_iterations)
        T3sigmaV2_25, T3meanV2_25, T3sigmaV3_25, T3meanV3_25, T3newV2_25, T3newV3_25, T3niter_25, T3lines2print_25 = tf.Nsigma_rejection(Nsigma, T3_diffV2_25, T3_diffV3_25, max_iterations)
        T3sigmaV2_27, T3meanV2_27, T3sigmaV3_27, T3meanV3_27, T3newV2_27, T3newV3_27, T3niter_27, T3lines2print_27 = tf.Nsigma_rejection(Nsigma, T3_diffV2_27, T3_diffV3_27, max_iterations)
    
    if debug or single_case:
        print ("TEST 1: ")
        print ("transformations: detector (avgx, avgy),  sky (V2, V3),  true (avgV2, avgV3)")
        print ("            ChBx3: ", avgx3[0], avgy3[0], T1_V2_3[0], T1_V3_3[0], avg_benchV2[0], avg_benchV3[0])
        print ("            ChBx5: ", avgx5[0], avgy5[0], T1_V2_5[0], T1_V3_5[0], avg_benchV2[0], avg_benchV3[0])
        print ("            ChBx7: ", avgx7[0], avgy7[0], T1_V2_7[0], T1_V3_7[0], avg_benchV2[0], avg_benchV3[0])
        print ("TEST 2: ")
        print ("transformations: detector P1 and P2 (x, y),  sky (avgV2, avgV3),  true (avgV2, avgV3)")
        print ("            ChBx3: ", x13[0], y13[0], x23[0], y23[0], T2_V2_3[0], T2_V3_3[0], avg_benchV2[0], avg_benchV3[0])
        print ("            ChBx5: ", x15[0], y15[0], x25[0], y25[0], T2_V2_5[0], T2_V3_5[0], avg_benchV2[0], avg_benchV3[0])
        print ("            ChBx7: ", x17[0], y17[0], x27[0], y27[0], T2_V2_7[0], T2_V3_7[0], avg_benchV2[0], avg_benchV3[0])
        print ("TEST 3: ")
        print ("transformations: detector P1 and P2 (x, y),  sky P1 and P2 (V2, V3),  true P1 and P2 (V2, V3)")
        print ("            ChBx3: ", x13[0], y13[0], x23[0], y23[0], T3_V2_13[0], T3_V3_13[0], T3_V2_23[0], T3_V3_23[0], bench_V2P1[0], bench_V3P1[0], bench_V2P2[0], bench_V3P2[0])
        print ("            ChBx5: ", x15[0], y15[0], x25[0], y25[0], T3_V2_13[0], T3_V3_13[0], T3_V2_23[0], T3_V3_23[0], bench_V2P1[0], bench_V3P1[0], bench_V2P2[0], bench_V3P2[0])
        print ("            ChBx7: ", x17[0], y17[0], x27[0], y27[0], T3_V2_13[0], T3_V3_13[0], T3_V2_23[0], T3_V3_23[0], bench_V2P1[0], bench_V3P1[0], bench_V2P2[0], bench_V3P2[0])
        raw_input(" * press enter to continue... \n")

    # Print results to screen and save into a text file if told so
    # Text file 1
    line0 = "{}".format("Differences = diffs = True_Positions - Measured_Positions")
    if arcsecs:
        line0bis = "{}".format("***  units are arcsecs")
    else:
        line0bis = "{}".format("***  units are degrees")
    line1 = "{}\n {}".format("Test1: average P1 and P2, transform to V2-V3, calculate differences",
                             "  * Standard deviations and means ")
    if not single_case:    
        # print regular standard deviations and means
        line2a = "std_dev_V2_3 = {:<20}    std_dev_V3_3 = {:<20}".format(T1stdev_V2_3, T1stdev_V3_3)
        line2b = "std_dev_V2_5 = {:<20}    std_dev_V3_5 = {:<20}".format(T1stdev_V2_5, T1stdev_V3_5)
        line2c = "std_dev_V2_7 = {:<20}    std_dev_V3_7 = {:<20}".format(T1stdev_V2_7, T1stdev_V3_7)
        line3a = "   mean_V2_3 = {:<22}     mean_V3_3 = {:<22}".format(T1mean_V2_3, T1mean_V3_3)
        line3b = "   mean_V2_5 = {:<22}     mean_V3_5 = {:<22}".format(T1mean_V2_5, T1mean_V3_5)
        line3c = "   mean_V2_7 = {:<22}     mean_V3_7 = {:<22}".format(T1mean_V2_7, T1mean_V3_7)
        # Print number of repetitions to find best checkbox
        line3bisA = "Repetitions Diffs: {}".format(T1_counter)
    if "frac" in case:
        line3bisB = "Repetitions Fractional BG values ChBx3: V2-{}".format(T1counterV2_3)
        line3bisC = "                                        V3-{}".format(T1counterV3_3)
        line3bisD = "                                 ChBx5: V2-{}".format(T1counterV2_5)
        line3bisE = "                                        V3-{}".format(T1counterV3_5)
        line3bisF = "                                 ChBx7: V2-{}".format(T1counterV2_7)
        line3bisG = "                                        V3-{}".format(T1counterV3_7)
    if show_positions:
        line4 = "{:<5} {:<20} {:<40} {:<40} {:<38} {:<35} {:<40} {:<40} {:<24} {:<7}".format(
                        "Star", "BG_value", "Avg_Pos_Checkbox_3", "Avg_Pos_Checkbox_5", "Avg_PosCheckbox_7",
                        "AvgTrue_Pos", "Diff_Chbx_3", "Diff_Chbx_5", "Diff_Chbx_7", "MinDiff")
        line5 = "{:>25} {:>17} {:>22} {:>17} {:>22} {:>22} {:>17} {:>11} {:>18} {:>17} {:>22} {:>17} {:>22} {:>17}".format(
                        "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y")
    else:
        line4 = "{:<5} {:<20} {:<40} {:<40} {:<23} {:<9}".format("Star", "BG_value", "Checkbox=3", "Checkbox=5", "Checkbox=7", "MinDiff")
        line5 = "{:>25} {:>17} {:>22} {:>17} {:>22} {:>17}".format("x", "y", "x", "y", "x", "y")        
    if "frac" in case:
        line4 += " {:>23}".format("BestFracVal")
        line5 += " {:>25} {:>10} {:>10}".format("Cbx3", "Cbx5", "Cbx7")
    print (line0)
    print (line0bis)
    print (line1)
    if not single_case:    
        print (line2a)
        print (line2b)
        print (line2c)
        print (line3a)
        print (line3b)
        print (line3c)
        print (line3bisA)
    if "frac" in case:
        print (line3bisB)
        print (line3bisC)
        print (line3bisD)
        print (line3bisE)
        print (line3bisF)
        print (line3bisG)
    print (line4)
    print (line5)
    if save_txt_file:
        txt_out = path4results+"Test1_results_"+case+".txt"
        to = open(txt_out, "w+")
        to.write(line0+"\n")
        to.write(line0bis+"\n")
        to.write(line1+"\n")
        to.write(line2a+"\n")
        to.write(line2b+"\n")
        to.write(line2c+"\n")
        to.write(line3a+"\n")
        to.write(line3b+"\n")
        to.write(line3c+"\n")
        # print standard deviations from least squares routine
        to.write("  * From least squares routine:  \n")
        to.write("       Checkbox 3:  \n")
        for line2print in T1LSlines2print_3:
            to.write(line2print+"\n")
        to.write("       Checkbox 5:  \n")
        for line2print in T1LSlines2print_5:
            to.write(line2print+"\n")
        to.write("       Checkbox 7:  \n")
        for line2print in T1LSlines2print_7:
            to.write(line2print+"\n")
        # print standard deviations and means after n-sigma rejection
        to.write(" Checkbox 3:  \n")
        for line2print in T1lines2print_3:
            to.write(line2print+"\n")
        to.write(" Checkbox 5:  \n")
        for line2print in T1lines2print_5:
            to.write(line2print+"\n")
        to.write(" Checkbox 7:  \n")
        for line2print in T1lines2print_7:
            to.write(line2print+"\n")
        to.write(line3bisA+"\n")
        if "frac" in case:
            to.write(line3bisB+"\n")
            to.write(line3bisC+"\n")
            to.write(line3bisD+"\n")
            to.write(line3bisE+"\n")
            to.write(line3bisF+"\n")
            to.write(line3bisG+"\n")
        to.write(line4+"\n")
        to.write(line5+"\n")
    for i, st in enumerate(stars):
        st = int(st)
        if show_positions:
            line6 = "{:<5} {:<5} {:>20}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:>10}  {:<14} {:>18}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:<7}".format(
                        st, bg_value[i], 
                        T1_V2_3[i], T1_V3_3[i], T1_V2_5[i], T1_V3_5[i], T1_V2_7[i], T1_V3_7[i], 
                        T1bench_V2_list[i], T1bench_V3_list[i],
                        T1_diffV2_3[i], T1_diffV3_3[i], T1_diffV2_5[i], T1_diffV3_5[i], T1_diffV2_7[i], T1_diffV3_7[i], T1_min_diff[i])
        else:
            line6 = "{:<5} {:<5} {:>20}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:<7}".format(st, bg_value[i], 
                    T1_diffV2_3[i], T1_diffV3_3[i], T1_diffV2_5[i], T1_diffV3_5[i], T1_diffV2_7[i], T1_diffV3_7[i], T1_min_diff[i])
        if "frac" in case:
            line6 += " {:<4} {:<5} {:<4} {:<5} {:<4} {:<5}".format(
                T1list_best_frbg_value_V2_3[i], T1list_best_frbg_value_V3_3[i],
                T1list_best_frbg_value_V2_5[i], T1list_best_frbg_value_V3_5[i],
                T1list_best_frbg_value_V2_7[i], T1list_best_frbg_value_V3_7[i])
        print (line6)
        if save_txt_file:
            to.write(line6+"\n")
    if save_txt_file:
        to.close()
        print (" * Results saved in file: ", txt_out)
    #raw_input(" * Press enter to continue... \n")

    # Text file 2
    line0 = "{}".format("Differences = True_Positions - Measured_Positions")
    if arcsecs:
        line0bis = "{}".format("***  units are arcsecs")
    else:
        line0bis = "{}".format("***  units are degrees")
    line1 = "{}".format("Test2: P1 P2, average positions in V2-V3, calculate differences")
    if not single_case:    
        line2a = "std_dev_V2_3 = {:<20}   std_dev_V3_3 = {:<20}".format(T2stdev_V2_3, T2stdev_V3_3)
        line2b = "std_dev_V2_5 = {:<20}   std_dev_V3_5 = {:<20}".format(T2stdev_V2_5, T2stdev_V3_5)
        line2c = "std_dev_V2_7 = {:<20}   std_dev_V3_7 = {:<20}".format(T2stdev_V2_7, T2stdev_V3_7)
        line3a = "   mean_V2_3 = {:<22}    mean_V3_3 = {:<22}".format(T2mean_V2_3, T2mean_V3_3)
        line3b = "   mean_V2_5 = {:<22}    mean_V3_5 = {:<22}".format(T2mean_V2_5, T2mean_V3_5)
        line3c = "   mean_V2_7 = {:<22}    mean_V3_7 = {:<22}".format(T2mean_V2_7, T2mean_V3_7)
        line3bisA = "Repetitions Diffs1: {}".format(T2_counter)
    if "frac" in case:
        line3bisB = "Repetitions Fractional BG values ChBx3: V2-{}".format(T2counterV2_3)
        line3bisC = "                                        V3-{}".format(T2counterV3_3)
        line3bisD = "                                 ChBx5: V2-{}".format(T2counterV2_5)
        line3bisE = "                                        V3-{}".format(T2counterV3_5)
        line3bisF = "                                 ChBx7: V2-{}".format(T2counterV2_7)
        line3bisG = "                                        V3-{}".format(T2counterV3_7)
    if show_positions:
        line4 = "{:<5} {:<20} {:<40} {:<40} {:<35} {:<30} {:<40} {:<40} {:<23} {:<7}".format(
                        "Star", "BG_value", "Avg_Pos_Checkbox_3", "Avg_Pos_Checkbox_5", "Avg_PosCheckbox_7",
                        "AvgTrue_Pos", "Diff_Chbx_3", "Diff_Chbx_5", "Diff_Chbx_7", "MinDiff")
        line5 = "{:>25} {:>17} {:>22} {:>17} {:>22} {:>17} {:>17} {:>11} {:>18} {:>17} {:>22} {:>17} {:>22} {:>17}".format(
                        "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y")
    else:
        line4 = "{:<5} {:<20} {:<40} {:<40} {:<23} {:<9}".format("Star", "BG_value", "Checkbox=3", "Checkbox=5", "Checkbox=7", "MinDiff")
        line5 = "{:>25} {:>17} {:>22} {:>17} {:>22} {:>17}".format("x", "y", "x", "y", "x", "y")        
    if "frac" in case:
        line4 += " {:>23}".format("BestFracVal")
        line5 += " {:>25} {:>10} {:>10}".format("Cbx3", "Cbx5", "Cbx7")
    print (line0)
    print (line0bis)
    print (line1)
    if not single_case:    
        print (line2a)
        print (line2b)
        print (line2c)
        print (line3a)
        print (line3b)
        print (line3c)
        print (line3bisA)
    if "frac" in case:
        print (line3bisB)
        print (line3bisC)
        print (line3bisD)
        print (line3bisE)
        print (line3bisF)
        print (line3bisG)
    print (line4)
    print (line5)
    if save_txt_file:
        txt_out = path4results+"Test2_results_"+case+".txt"
        to = open(txt_out, "w+")
        to.write(line0+"\n")
        to.write(line0bis+"\n")
        to.write(line1+"\n")
        to.write(line2a+"\n")
        to.write(line2b+"\n")
        to.write(line2c+"\n")
        to.write(line3a+"\n")
        to.write(line3b+"\n")
        to.write(line3c+"\n")
        # print standard deviations from least squares routine
        to.write("  * From least squares routine:  \n")
        to.write("       Checkbox 3:  \n")
        for line2print in T2LSlines2print_3:
            to.write(line2print+"\n")
        to.write("       Checkbox 5:  \n")
        for line2print in T2LSlines2print_5:
            to.write(line2print+"\n")
        to.write("       Checkbox 7:  \n")
        for line2print in T2LSlines2print_7:
            to.write(line2print+"\n")
        # print standard deviations and means after n-sigma rejection
        to.write(" Checkbox 3:  \n")
        for line2print in T2lines2print_3:
            to.write(line2print+"\n")
        to.write(" Checkbox 5:  \n")
        for line2print in T2lines2print_5:
            to.write(line2print+"\n")
        to.write(" Checkbox 7:  \n")
        for line2print in T2lines2print_7:
            to.write(line2print+"\n")
        to.write(line3bisA+"\n")
        if "frac" in case:
            to.write(line3bisB+"\n")
            to.write(line3bisC+"\n")
            to.write(line3bisD+"\n")
            to.write(line3bisE+"\n")
            to.write(line3bisF+"\n")
            to.write(line3bisG+"\n")
        to.write(line4+"\n")
        to.write(line5+"\n")
    for i, st in enumerate(stars):
        st = int(st)
        if show_positions:
            line6 = "{:<5} {:<5} {:>20}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:>10}  {:<14} {:>18}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:<7}".format(
                        st, bg_value[i], 
                        T2_V2_3[i], T2_V3_3[i], T2_V2_5[i], T2_V3_5[i], T2_V2_7[i], T2_V3_7[i], 
                        T2bench_V2_list[i], T2bench_V3_list[i],
                        T2_diffV2_3[i], T2_diffV3_3[i], T2_diffV2_5[i], T2_diffV3_5[i], T2_diffV2_7[i], T2_diffV3_7[i], T2_min_diff[i])
        else:
            line6 = "{:<5} {:<5} {:>20}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:<7}".format(st, bg_value[i], 
                    T2_diffV2_3[i], T2_diffV3_3[i], T2_diffV2_5[i], T2_diffV3_5[i], T2_diffV2_7[i], T2_diffV3_7[i], T2_min_diff[i])            
        if "frac" in case:
            line6 += " {:<4} {:<5} {:<4} {:<5} {:<4} {:<5}".format(
                T2list_best_frbg_value_V2_3[i], T2list_best_frbg_value_V3_3[i],
                T2list_best_frbg_value_V2_5[i], T2list_best_frbg_value_V3_5[i],
                T2list_best_frbg_value_V2_7[i], T2list_best_frbg_value_V3_7[i])
        print (line6)
        if save_txt_file:
            to.write(line6+"\n")
    if save_txt_file:
        to.close()
        print (" * Results saved in file: ", txt_out)
    #raw_input(" * Press enter to continue... \n")

    # Text file 3
    line0 = "{}".format("Differences = True_Positions - Measured_Positions")
    if arcsecs:
        line0bis = "{}".format("***  units are arcsecs")
    else:
        line0bis = "{}".format("***  units are degrees")
    line1 = "{}".format("Test3: P1 and P2, transform to V2-V3 space individually, calculate differences position to position")
    if not single_case:    
        line2a = "std_dev_V2_P1_3 = {:<20}    std_dev_V3_P1_3 = {:<20}".format(T3stdev_V2_13, T3stdev_V3_13)
        line2b = "std_dev_V2_P1_5 = {:<20}    std_dev_V3_P1_5 = {:<20}".format(T3stdev_V2_15, T3stdev_V3_15)
        line2c = "std_dev_V2_P1_7 = {:<20}    std_dev_V3_P1_7 = {:<20}".format(T3stdev_V2_17, T3stdev_V3_17)
        line3a = "   mean_V2_P1_3 = {:<22}     mean_V3_P1_3 = {:<22}".format(T3mean_V2_13, T3mean_V3_13)
        line3b = "   mean_V2_P1_5 = {:<22}     mean_V3_P1_5 = {:<22}".format(T3mean_V2_15, T3mean_V3_15)
        line3c = "   mean_V2_P1_7 = {:<22}     mean_V3_P1_7 = {:<22}".format(T3mean_V2_17, T3mean_V3_17)
        line4a = "std_dev_V2_P2_3 = {:<20}    std_dev_V3_P2_3 = {:<20}".format(T3stdev_V2_23, T3stdev_V3_23)
        line4b = "std_dev_V2_P2_5 = {:<20}    std_dev_V3_P2_5 = {:<20}".format(T3stdev_V2_25, T3stdev_V3_25)
        line4c = "std_dev_V2_P2_7 = {:<20}    std_dev_V3_P2_7 = {:<20}".format(T3stdev_V2_27, T3stdev_V3_27)
        line5a = "   mean_V2_P2_3 = {:<22}     mean_V3_P2_3 = {:<22}".format(T3mean_V2_23, T3mean_V3_23)
        line5b = "   mean_V2_P2_5 = {:<22}     mean_V3_P2_5 = {:<22}".format(T3mean_V2_25, T3mean_V3_25)
        line5c = "   mean_V2_P2_7 = {:<22}     mean_V3_P2_7 = {:<22}".format(T3mean_V2_27, T3mean_V3_27)
        line5bisA = "Repetitions Diffs1: {}".format(T3_counter1)
        line5bisB = "Repetitions Diffs2: {}".format(T3_counter2)
    if "frac" in case:
        line5bisC = "Repetitions Fractional BG values Positions 1 and 2 ChBx3: V2 {}  {}".format(T3counterV2_13, T3counterV2_23)
        line5bisD = "                                                          V3 {}  {}".format(T3counterV3_13, T3counterV3_23)
        line5bisE = "                                                   ChBx5: V2 {}  {}".format(T3counterV2_15, T3counterV2_25)
        line5bisF = "                                                          V3 {}  {}".format(T3counterV3_15, T3counterV3_25)
        line5bisG = "                                                   ChBx7: V2 {}  {}".format(T3counterV2_17, T3counterV2_27)
        line5bisH = "                                                          V3 {}  {}".format(T3counterV3_17, T3counterV3_27)
    if show_positions:
        line6 = "{:<5} {:<15} {:<35} {:<39} {:<37} {:<38} {:<36} {:<38} {:<30} {:<40} {:<40} {:<40} {:<36} {:<38} {:<35} {:<25} {:<7}".format(
                    "Star", "BG_value", "Pos1_Checkbox_3", "Pos1_Checkbox_5", "Pos1_Checkbox_7",
                    "Pos2_Checkbox_3", "Pos2_Checkbox_5", "Pos2_Checkbox_7", 
                    "True_Pos1", "True_Pos2", "Diff1_Chbx_3", "Diff1_Chbx_5", "Diff1_Chbx_7", 
                    "Diff2_Chbx_3", "Diff2_Chbx_5", "Diff2_Chbx_7", "MinDiff1", "MinDiff2")
        line7 = "{:>22} {:>14} {:>22} {:>14} {:>22} {:>14} {:>22} {:>14} {:>22} {:>14} {:>22} {:>14} {:>22} {:>10} {:>22} {:>10} {:>22} {:>14} {:>22} {:>14} {:>22} {:>14} {:>28} {:>14} {:>22} {:>14} {:>22} {:>14}".format(
                        "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y",
                        "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y", 
                        "x", "y", "x", "y", "x", "y", "x", "y")
    else:
        line6 = "{:<5} {:<20} {:<40} {:<40} {:<40} {:<40} {:<40} {:<25} {:<9} {:<9}".format(
                    "Star", "BG_value", "Diff1_Chbx_3", "Diff1_Chbx_5", "Diff1_Chbx_7",
                    "Diff2_Chbx_3", "Diff2_Chbx_5", "Diff2_Chbx_7", "MinDiff1", "MinDiff2")
        line7 = "{:>25} {:>17} {:>22} {:>17} {:>22} {:>17} {:>22} {:>17} {:>22} {:>17} {:>22} {:>17}".format("x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y")        
    if "frac" in case:
        line6 += " {:>33}".format("BestFracVal")
        line7 += " {:>33} {:>10} {:>10} {:>11} {:>10} {:>10}".format("P13", "P15", "P17", "P23", "P25", "P27")
    print (line0)
    print (line0bis)
    print (line1)
    if not single_case:    
        print (line2a)
        print (line2b)
        print (line2c)
        print (line3a)
        print (line3b)
        print (line3c)
        print (line4a)
        print (line4b)
        print (line4b)
        print (line5a)
        print (line5b)
        print (line5c)
        print (line5bisA)
        print (line5bisB)
    if "frac" in case:
        print (line5bisC)
        print (line5bisD)
        print (line5bisE)
        print (line5bisF)
        print (line5bisG)
        print (line5bisH)
    print (line6)
    print (line7)
    if save_txt_file:
        txt_out = path4results+"Test3_results_"+case+".txt"
        to = open(txt_out, "w+")
        to.write(line0+"\n")
        to.write(line0bis+"\n")
        to.write(line1+"\n")
        to.write(line2a+"\n")
        to.write(line2b+"\n")
        to.write(line2c+"\n")
        to.write(line3a+"\n")
        to.write(line3b+"\n")
        to.write(line3c+"\n")
        to.write(line4a+"\n")
        to.write(line4b+"\n")
        to.write(line4c+"\n")
        to.write(line5a+"\n")
        to.write(line5b+"\n")
        to.write(line5c+"\n")
        # print standard deviations from least squares routine
        to.write("  * From least squares routine:  \n")
        to.write("     Position 1:  \n")
        to.write("       Checkbox 3:  \n")
        for line2print in T3LSlines2print_13:
            to.write(line2print+"\n")
        to.write("       Checkbox 5:  \n")
        for line2print in T3LSlines2print_15:
            to.write(line2print+"\n")
        to.write("       Checkbox 7:  \n")
        for line2print in T3LSlines2print_17:
            to.write(line2print+"\n")
        to.write("     Position 2:  \n")
        to.write("       Checkbox 3:  \n")
        for line2print in T3LSlines2print_23:
            to.write(line2print+"\n")
        to.write("       Checkbox 5:  \n")
        for line2print in T3LSlines2print_25:
            to.write(line2print+"\n")
        to.write("       Checkbox 7:  \n")
        for line2print in T3LSlines2print_27:
            to.write(line2print+"\n")
        # print standard deviations and means after n-sigma rejection
        to.write("     Position 1:  \n")
        to.write("       Checkbox 3:  \n")
        for l2p in T3lines2print_13:
            to.write(l2p+"\n")
        to.write("       Checkbox 5:  \n")
        for l2p in T3lines2print_15:
            to.write(l2p+"\n")
        to.write("       Checkbox 7:  \n")
        for l2p in T3lines2print_17:
            to.write(l2p+"\n")
        to.write("     Position 2:  \n")
        to.write("       Checkbox 3:  \n")
        for l2p in T3lines2print_23:
            to.write(l2p+"\n")
        to.write("       Checkbox 5:  \n")
        for l2p in T3lines2print_25:
            to.write(l2p+"\n")
        to.write("       Checkbox 7:  \n")
        for l2p in T3lines2print_27:
            to.write(l2p+"\n")
        to.write(line5bisA+"\n")
        to.write(line5bisB+"\n")
        if "frac" in case:
            to.write(line5bisC+"\n")
            to.write(line5bisD+"\n")
            to.write(line5bisE+"\n")
            to.write(line5bisF+"\n")
            to.write(line5bisG+"\n")
            to.write(line5bisH+"\n")
        to.write(line6+"\n")
        to.write(line7+"\n")
    for i, st in enumerate(stars):
        st = int(st)
        if show_positions:
            line8 = "{:<5} {:<5} {:>16}  {:<19} {:>16}  {:<19} {:>16}  {:<19} {:>16}  {:<19} {:>16}  {:<19} {:>16}  {:<19} {:>14}  {:<14}  {:>14}  {:<14} {:>18}  {:<19} {:>18}  {:<19} {:>18}  {:<19} {:>18}  {:<19} {:>18}  {:<19} {:>18}  {:<19} {:<7} {:<7}".format(
                        st, bg_value[i], 
                        T3_V2_13[i], T3_V3_13[i], T3_V2_15[i], T3_V3_15[i], T3_V2_17[i], T3_V3_17[i], 
                        T3_V2_23[i], T3_V3_23[i], T3_V2_25[i], T3_V3_25[i], T3_V2_27[i], T3_V3_27[i], 
                        T3bench_V2_listP1[i], T3bench_V3_listP1[i], T3bench_V2_listP2[i], T3bench_V3_listP2[i],
                        T3_diffV2_13[i], T3_diffV3_13[i], T3_diffV2_15[i], T3_diffV3_15[i], T3_diffV2_17[i], T3_diffV3_17[i],
                        T3_diffV2_23[i], T3_diffV3_23[i], T3_diffV2_25[i], T3_diffV3_25[i], T3_diffV2_27[i], T3_diffV3_27[i],
                        T3_min_diff1[i], T3_min_diff2[i])
        else:
            line8 = "{:<5} {:<5} {:>20}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:>18}  {:<20} {:<7} {:<7}".format(st, bg_value[i], 
                    T3_diffV2_13[i], T3_diffV3_13[i], T3_diffV2_15[i], T3_diffV3_15[i], T3_diffV2_17[i], T3_diffV3_17[i], 
                    T3_diffV2_23[i], T3_diffV3_23[i], T3_diffV2_25[i], T3_diffV3_25[i], T3_diffV2_27[i], T3_diffV3_27[i], 
                    T3_min_diff1[i], T3_min_diff2[i])            
        if "frac" in case:
            line8 += " {:<4} {:<5} {:<4} {:<5} {:<4} {:<6} {:<4} {:<5} {:<4} {:<5} {:<4} {:<5}".format(
                T3list_best_frbg_value_V2_13[i], T3list_best_frbg_value_V3_13[i],
                T3list_best_frbg_value_V2_15[i], T3list_best_frbg_value_V3_15[i],
                T3list_best_frbg_value_V2_17[i], T3list_best_frbg_value_V3_17[i],
                T3list_best_frbg_value_V2_23[i], T3list_best_frbg_value_V3_23[i],
                T3list_best_frbg_value_V2_25[i], T3list_best_frbg_value_V3_25[i],
                T3list_best_frbg_value_V2_27[i], T3list_best_frbg_value_V3_27[i])
        print (line8)
        if save_txt_file:
            to.write(line8+"\n")
    if save_txt_file:
        to.close()
        print (" * Results saved in file: ", txt_out)
    if single_case:
        exit()
    else:
        raw_input(" * Finished case. Press enter to continue... \n")


print ("\n Script 'comparison2sky.py' finished! ")
