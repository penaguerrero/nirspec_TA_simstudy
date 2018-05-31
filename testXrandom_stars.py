from __future__ import print_function, division
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
import copy

# other code
import TA_functions as TAf
import parse_rejected_stars as prs
import v2v3plots as vp

print("Modules correctly imported! \n")




# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Oct 2016 - Version 1.0: initial version completed


"""
DESCRIPTION:
    This script runs the centroid algorithm fully from beginning to end for the given initial
    conditions (which are set in the first part of the script). The script will choose X random
    stars from either detector and run the selected transformations for the specific test and
    case for the set of X stars:
    
     TEST1 - Average positions P1 and P2, transform to V2-V3 space, and compare to average 
             reference positions (V2-V3 space)
     TEST2 - Transform individual positions P1 and P2 to V2-V3 space, average V2-V3 space 
             positions, and compare to average reference positions.
     TEST3 - Transform P1 and P2 individually to V2-V3 space and compare star by star and 
             position by position.

NOTES:
    * Scenes are defined by 100 stars in each detector:
        Scene or scenario 1 = All stars with magnitude 23.0
        Scene or scenario 2 = Magnitude range from 18.0 to 23.0
    
    * case depends on scene, noise, background value, and shutter velocity; results in 36 files per scene.
    
    * The tests are performed on ESA data in the following directories on central sotrage:
        - /grp/jwst/wit4/nirspec/PFforMaria/Scene_1_AB23
            (which contains 200 closer to "real" stars of magnitude 23: cosmic rays, hot 
            pixels, noise, and some shutters closed; and an ideal case)
        - /grp/jwst/wit4/nirspec/PFforMaria/Scene_2_AB1823
            (which contains 200 stars of different magnitudes and also has the "real" data
            as well as the ideal case)
        ** There are 2 sub-folders in each scene: rapid and slow. This is the shutter speed. Both
        sub-cases will be tested.
        ** The simulated data is described in detail in the NIRSpec Technical Note NTN-2015-013, which
        is in /grp/jwst/wit4/nirspec/PFforMaria/Documentation.



OUTPUT:
    - display images with true and calculated centroid
    - text file for the test ran with standard deviations and means for centroid window sizes 3, 5, and 7,
      sigma-clipped standard deviations and means, iterative least squares standard deviations and means,
      and the list of stars, background value used, the differences (in arcsecs or degrees) with respect
      to true or benchmark sky positions, and the centroid window size that has the minimum difference with
      respect to the true value.
"""


#######################################################################################################################


def runXrandomstars(stars_detectors, primary_params, secondary_params, stars_sample,
                    path4results=None, extra_string=None):
    '''
    This function runs the full TA algorithm AND converts to sky for X random stars and performs the given test.

    Args:
        stars_detectors: list, pool of stars to select the random from
        primary_params: list, set of parameters specific to the case
        secondary_params: list, set of baseline parameters
        stars_sample: list, a list of stars must be provided if random_sample is set to False.

    Returns:
            case = string, string, for example 'Scene2_rapid_real_bgFrac'
            Tbench_Vs_list = list of benchmark V2 and V3s
            T_Vs = list of measured V2 and V3s
            T_diffVs = list of true-measured V2s and V3s
            LS_res = list, std deviations and means from least squared routine
            iterations = list, number of the last iteration of the least squared routine for 3, 5, and 7
    '''
    # Define the paths for results
    if path4results is None:
        path4results = "../resultsXrandomstars/"

    print (' Running TA algorithm to measure centroids...')

    # unfold variables
    primary_params1, primary_params2, primary_params3 = primary_params
    do_plots, save_plots, show_plots, detector, output_full_detector, show_onscreen_results, show_pixpos_and_v23_plots, save_text_file = primary_params1
    save_centroid_disp, keep_bad_stars, keep_ugly_stars, just_least_sqares, stars_in_sample, scene, background_method, background2use = primary_params2
    shutters, noise, filter_input, test2perform, Nsigma, abs_threshold, abs_threshold, min_elements, max_iters_Nsig = primary_params3
    secondary_params1, secondary_params2, secondary_params3 = secondary_params
    checkbox_size, xwidth_list, ywidth_list, vlim, threshold, max_iter, verbose = secondary_params1
    debug, arcsecs, determine_moments, display_master_img, show_centroids, show_disp = secondary_params2
    Pier_corr, tilt, backgnd_subtraction_method, random_sample = secondary_params3

    bg_choice, P1P2data, bench_starP1, benchmark_V2V3_sampleP1P2 = measure_centroidsP1P2(stars_detectors, primary_params,
                                                                                         secondary_params, stars_sample,
                                                                                         plot_pixpos=show_pixpos_and_v23_plots)
    print ('\n Transforming into V2 and V3, and running TEST...')
    case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, LS_info = transformAndRunTest(stars_sample, path4results,
                                                                                       primary_params, secondary_params,
                                                                                       bg_choice, P1P2data, bench_starP1,
                                                                                       benchmark_V2V3_sampleP1P2,
                                                                                       plot_v2v3pos=show_pixpos_and_v23_plots,
                                                                                       extra_string=extra_string)
    return case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, LS_info


def select_random_stars(scene, stars_in_sample, stars_detectors, keep_bad_stars, keep_ugly_stars, verbose):
    '''
    This function chooses a random set of stars
    Args:
        stars_in_sample: integer, number of stars to be studied
        stars_detectors: pool of stars to choose
        verbose: print a few debuging statements

    Returns:
        stars_sample = list of random stars to be studied

    '''
    stars_sample = []
    for i in range(stars_in_sample):
        random_star = random.choice(stars_detectors)
        stars_sample.append(random_star)
    #print ('before while: ', stars_sample)
    # make sure that there are no repetitions
    stars_sample = list(set(stars_sample))
    if not keep_bad_stars:
        stars_sample = TAf.remove_bad_stars(scene, stars_sample, keep_ugly_stars, verbose)
    while len(stars_sample) != stars_in_sample:
        random_star = random.choice(stars_detectors)
        stars_sample.append(random_star)
        stars_sample = list(set(stars_sample))
        # remove the bad stars
        if not keep_bad_stars:
            stars_sample = TAf.remove_bad_stars(scene, stars_sample, keep_ugly_stars, verbose)
    # order the star list from lowest to highest number
    stars_sample.sort(key=lambda xx: xx)
    print ("NEW stars_sample =", stars_sample)
    return stars_sample


def get_benchV2V3(scene, stars_sample, arcsecs):
    '''
    This function gets the benchmark (true) pixel position, V2s, V3s, and magnitudes for both the unshifted and
    shifted positions from the fits files given by Pierre.
    Args:
        scene: integer, 1 or 2

    Returns:
        bench_stars, benchP1P2, LoLeftCornersP1P2, benchmark_V2V3_sampleP1P2, magnitudes
        bench_stars = list of the star numbers in the sample for positions 1 and 2 (should be the same)
        benchP1P2 = list
        LoLeftCornersP1P2 = list
        benchmark_V2V3_sampleP1P2 = list
        magnitudes = np.array of simulated star magnitudes
    '''
    # Set the case to study according to the selected scene
    scene2study = "Scene"+str(scene)+"_"

    # get the benchmark data according to Scene selected
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
    bench_starP1, bench_xP1, bench_yP1, = np.array([]), np.array([]), np.array([])
    bench_V2P1, bench_V3P1, bench_xLP1, bench_yLP1 = np.array([]), np.array([]), np.array([]), np.array([])
    bench_starP2, bench_xP2, bench_yP2 = np.array([]),np.array([]), np.array([])
    bench_V2P2, bench_V3P2, bench_xLP2, bench_yLP2 = np.array([]), np.array([]), np.array([]), np.array([])
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
    # Compact variables
    bench_stars = [bench_starP1, bench_starP2]
    benchP1P2 = [bench_xP1, bench_yP1, bench_xP2, bench_yP2]
    LoLeftCornersP1P2 = [bench_xLP1, bench_yLP1, bench_xLP2, bench_yLP2]
    benchmark_V2V3_sampleP1P2 = [bench_V2P1, bench_V3P1,   bench_V2P2, bench_V3P2]
    return bench_stars, benchP1P2, LoLeftCornersP1P2, benchmark_V2V3_sampleP1P2, magnitudes


def measure_centroidsP1P2(stars_detectors, primary_params, secondary_params, stars_sample,
                          plot_pixpos=True):
    '''
    This function runs the full TA algorithm for X random stars and performs the given test.

    Args:
        stars_detectors: list, pool of stars to select the random from
        primary_params: list of baseline parameters
        secondary_params: list, set of baseline parameters
        stars_sample: list, star numbers of the sample

    Returns:
        bg_choice = string of the background method used
        P1P2data = list of pixel positions for centroid windows 3, 5, and 7 for both positions:
                        P1P2data = [x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27]
        bench_starP1 = list of the benchmark stars studied
        benchmark_V2V3_sampleP1P2 = list of benchmark V2s and V3s:
                        benchmark_V2V3_sampleP1P2 = [bench_V2P1, bench_V3P1,   bench_V2P2, bench_V3P2]
    '''


    # unfold variables
    # primary parameters
    primary_params1, primary_params2, primary_params3 = primary_params
    do_plots, save_plots, show_plots, detector, output_full_detector, show_onscreen_results, show_pixpos_and_v23_plots, save_text_file = primary_params1
    save_centroid_disp, keep_bad_stars, keep_ugly_stars, just_least_sqares, stars_in_sample, scene, background_method, background2use = primary_params2
    shutters, noise, filter_input, test2perform, Nsigma, abs_threshold, abs_threshold, min_elements, max_iters_Nsig = primary_params3
    # secondary parameters
    secondary_params1, secondary_params2, secondary_params3 = secondary_params
    checkbox_size, xwidth_list, ywidth_list, vlim, threshold, max_iter, verbose = secondary_params1
    debug, arcsecs, determine_moments, display_master_img, show_centroids, show_disp = secondary_params2
    Pier_corr, tilt, backgnd_subtraction_method, random_sample = secondary_params3

    if random_sample:
        stars_sample = select_random_stars(scene, stars_in_sample, stars_detectors, keep_bad_stars, keep_ugly_stars, verbose)
    else:
        # remove the bad stars
        if not keep_bad_stars:
            print ('sample before removing bad stars: ', stars_sample)
            # remove the bad stars
            stars_sample = TAf.remove_bad_stars(scene, stars_sample, keep_ugly_stars, verbose)
            # but keep the sample stars list with length 20
            while len(stars_sample) != stars_in_sample:
                random_star = random.choice(stars_detectors)
                stars_sample.append(random_star)
                stars_sample = list(set(stars_sample))
                # remove the bad stars
                stars_sample = TAf.remove_bad_stars(scene, stars_sample, keep_ugly_stars, verbose)
            if verbose:
                print (" * Removed bad stars, sample has %i stars left." % len(stars_sample))

    # order the star list
    stars_sample.sort(key=lambda xx: xx)
    str_stars_sample = ", ".join(str(st) for st in stars_sample)
    print ("stars_sample =", str_stars_sample)
    #raw_input("\nPress enter to continue...")

    # Background cases variable setting
    bg_frac, bg_value = None, None   # for the None case
    bg_choice = "_bgNone"
    if background_method is not None:
        if "frac" in background_method:
            bg_frac = background2use
            bg_choice = "_bgFrac"
        elif "fix" in background_method:
            bg_value = background2use
            bg_choice = "_bgFixed"
    else:
        background2use = 0.0

    # Paths to Scenes 1 and 2 local directories: /Users/pena/Documents/PyCharm/nirspec/TargetAcquisition/
    path4starfiles = "../PFforMaria/"

    # Set the case to study according to the selected scene
    str_thres = repr(threshold).replace('0.', '')
    thres = 'thres'+str_thres
    if detector == 'both':
        detectorScene_string = "2DetsScene"+str(scene)
    else:
        detectorScene_string = repr(detector)+"Scene"+str(scene)
    case = detectorScene_string+"_"+shutters+"_"+noise+bg_choice+repr(background2use)+'_'+thres+'_Nsigma'+repr(Nsigma)

    detectors = [491, 492]

    bench_stars, benchP1P2, LoLeftCornersP1P2, benchmark_V2V3_sampleP1P2, magnitudes = get_benchV2V3(scene,
                                                                                                     stars_sample,
                                                                                                     arcsecs)
    bench_starP1, bench_starP2 = bench_stars
    bench_xP1, bench_yP1, bench_xP2, bench_yP2 = benchP1P2
    bench_xLP1, bench_yLP1, bench_xLP2, bench_yLP2 = LoLeftCornersP1P2
    bench_V2P1, bench_V3P1,   bench_V2P2, bench_V3P2 = benchmark_V2V3_sampleP1P2


    ### Perform centroid algorithm for stars sample

    # start the text file with the measured centroids
    output_file_path = "../resultsXrandomstars/centroid_txt_files/"
    line0 = "Centroid indexing starting at 1 !"
    line0a = "{:<5} {:<15} {:<16} {:>23} {:>32} {:>40} {:>25} {:>15} {:>9}".format("Star", "Background",
                                                                      "Centroid width: 3", "5", "7",
                                                                      "TruePositions", "LoLeftCoords",
                                                                      "Magnitude",
                                                                      "MinDiff")
    line0b = "{:>25} {:>12} {:>16} {:>14} {:>16} {:>20} {:>16} {:>17} {:>11} {:>11} {:>16} {:>2}".format(
                                                                           "x", "y", "x", "y", "x", "y",
                                                                           "TrueX", "TrueY", "LoLeftX", "LoLeftY",
                                                                           "x", "y")
    lines4screenandfile = [line0, line0a, line0b]
    # write the file
    positions = ["_Position1", "_Position2"]
    if save_text_file:
        for pos in positions:
            output_file = os.path.join(output_file_path, "centroids_Scene"+repr(scene)+bg_choice+pos+".txt")
            f = open(output_file, "w+")
            f.write(line0+"\n")
            f.close()
        if keep_ugly_stars and not keep_bad_stars:
            for pos in positions:
                out_file_gduglies = os.path.join(output_file_path, "centroids_Scene"+repr(scene)+bg_choice+pos+"_GoodAndUglies.txt")
                f = open(out_file_gduglies, "w+")
                f.write(line0+"\n")
                f.close()

    # get the star files to run the TA algorithm on
    dir2test_list = TAf.get_raw_star_directory(path4starfiles, scene, shutters, noise)

    # run centroid algorithm on each position and save them into a text file
    x13, x15, x17 = np.array([]), np.array([]), np.array([])
    y13, y15, y17 = np.array([]), np.array([]), np.array([])
    x23, x25, x27 = np.array([]), np.array([]), np.array([])
    y23, y25, y27 = np.array([]), np.array([]), np.array([])
    min_diff_pixposX, min_diff_pixposY, mag_list = [], [], []
    loleftcoords_listX, loleftcoords_listY = [], []
    true_centerX, true_centerY = [], []

    for pos, dir2test in zip(positions, dir2test_list):
        dir_stars = glob(os.path.join(dir2test,"postageout_star_*.fits"))   # get all star fits files in that directory
        #print("does dir2test exist?", os.path.isdir(dir2test))
        for star in dir_stars:
            dir_star_number = int(os.path.basename(star).split()[1])
            # Test stars of detector of choice
            for st in stars_sample:
                if st == dir_star_number: #if str(st)+" quad_       " in star:
                    if verbose:
                        print ("Will test stars in directory: \n     ", dir2test)
                        print ("Star: ", os.path.basename(star))
                    # Make sure the file actually exists
                    star_exists = os.path.isfile(star)
                    if not star_exists:
                        print ("The file: ", star, "\n    does NOT exist. Exiting the script.")
                        exit()

                    # Obtain real star position and corresponding detector
                    if st <= 100:
                        detector = detectors[1]
                    else:
                        detector = detectors[0]
                    idx_star = stars_sample.index(st)
                    mag_i = magnitudes[idx_star]
                    true_center_fulldet = [bench_xP1[idx_star], bench_yP1[idx_star]]

                    #!!!  WE ARE NOT USING POSITIONS 2 (SHIFTED) BECAUSE WE ARE FIXING POSITION 1 AS
                    #    REFERENCE POINT TO BEST REPRODUCE OBSERVATION MODE
                    #if pos == "_Position2":
                    #    true_center_fulldet = [bench_xP2[idx_star], bench_yP2[idx_star]]

                    # Read FITS image
                    if verbose:
                        print ("Running centroid algorithm... ")
                    #hdr = fits.getheader(star, 0)
                    #print("** HEADER:", hdr)
                    master_img = fits.getdata(star, 0)
                    if verbose:
                        print ('Master image shape: ', np.shape(master_img))
                    # Obtain the combined FITS image that combines all frames into one image
                    # background subtraction is done here
                    psf = TAf.readimage(master_img, backgnd_subtraction_method, bg_method=background_method,
                                        bg_value=bg_value, bg_frac=bg_frac, verbose=verbose, debug=debug)
                    cb_centroid_list_in32x32pix = TAf.run_recursive_centroids(psf, bg_frac, xwidth_list, ywidth_list,
                                                               checkbox_size, max_iter, threshold,
                                                               determine_moments, verbose, debug)

                    corr_cb_centroid_list, loleftcoords, true_center32x32, differences_true_TA = TAf.centroid2fulldetector(cb_centroid_list_in32x32pix,
                                                                                                        true_center_fulldet, detector, perform_avgcorr=Pier_corr)
                    if not output_full_detector:
                        cb_centroid_list = cb_centroid_list_in32x32pix
                        true_center = true_center32x32
                    else:
                        true_center = true_center_fulldet
                    if show_centroids:
                        print ('***** Measured centroids for centroid window sizes 3, 5, and 7, respectively:')
                        print ('      cb_centroid_list = ', corr_cb_centroid_list)
                        print ('           True center = ', true_center)

                    # Show the display with the measured and true positions
                    fig_name = os.path.join("../resultsXrandomstars", "centroid_displays/Star"+repr(st)+"_Scene"+repr(scene)+bg_choice+pos+".jpg")
                    # Display the combined FITS image that combines all frames into one image
                    m_img = display_master_img
                    if display_master_img:
                        m_img = TAf.readimage(master_img, backgnd_subtraction_method=None, bg_method=None,
                                          bg_value=None, bg_frac=None, debug=False)
                    TAf.display_centroids(detector, st, case, psf, true_center32x32, cb_centroid_list_in32x32pix,
                                         show_disp, vlim, savefile=save_centroid_disp, fig_name=fig_name, display_master_img=m_img)
                    if pos == "_Position2":
                        true_center_fulldetP2 = [bench_xP2[idx_star], bench_yP2[idx_star]]
                        _, _, true_center32x32P2, _ = TAf.centroid2fulldetector(cb_centroid_list_in32x32pix,
                                                            true_center_fulldetP2, detector, perform_avgcorr=Pier_corr)
                        #print ('true_center32x32 P1:', true_center32x32)
                        #print ('true_center32x32 P2:', true_center32x32P2)
                        # the following correction is because the postage stamp is centered on position 1 even if the
                        # the star moved to position 2.
                        if st <= 100:
                            true_center32x32P2[0] = true_center32x32P2[0]+1.0
                            true_center32x32P2[1] = true_center32x32P2[1]+2.0
                        else:
                            true_center32x32P2[0] = true_center32x32P2[0]-1.0
                            true_center32x32P2[1] = true_center32x32P2[1]-2.0
                        #print ('true_center32x32 P2:', true_center32x32P2)
                        #print ('cb_centroid_list_in32x32pix:')
                        #print (cb_centroid_list_in32x32pix)
                        TAf.display_centroids(detector, st, case, psf, true_center32x32P2, cb_centroid_list_in32x32pix,
                                             show_disp, vlim, savefile=save_centroid_disp, fig_name=fig_name, display_master_img=m_img)
                    # Find the best centroid window size = minimum difference with true values
                    min_diff, _ = TAf.get_mindiff(differences_true_TA[0][0], differences_true_TA[0][1], differences_true_TA[0][2])
                    # Save output
                    true_centerX.append(true_center[0])
                    true_centerY.append(true_center[1])
                    loleftcoords_listX.append(loleftcoords[0])
                    loleftcoords_listY.append(loleftcoords[1])
                    mag_list.append(mag_i)
                    min_diff_pixposX.append(min_diff[0])
                    min_diff_pixposY.append(min_diff[1])
                    if pos == "_Position1":
                        x13 = np.append(x13, corr_cb_centroid_list[0][0])
                        x15 = np.append(x15, corr_cb_centroid_list[1][0])
                        x17 = np.append(x17, corr_cb_centroid_list[2][0])
                        y13 = np.append(y13, corr_cb_centroid_list[0][1])
                        y15 = np.append(y15, corr_cb_centroid_list[1][1])
                        y17 = np.append(y17, corr_cb_centroid_list[2][1])
                    if pos == "_Position2":
                        x23 = np.append(x23, corr_cb_centroid_list[0][0])
                        x25 = np.append(x25, corr_cb_centroid_list[1][0])
                        x27 = np.append(x27, corr_cb_centroid_list[2][0])
                        y23 = np.append(y23, corr_cb_centroid_list[0][1])
                        y25 = np.append(y25, corr_cb_centroid_list[1][1])
                        y27 = np.append(y27, corr_cb_centroid_list[2][1])
        # Write output into text file
        position = "_Position1"
        x_pixpos = [x13, x15, x17]
        y_pixpos = [y13, y15, y17]
        if pos == "_Position2":
            x_pixpos = [x23, x25, x27]
            y_pixpos = [y23, y25, y27]
            position = "_Position2"
        true_centers = [true_centerX, true_centerY]
        loleftcoords_list = [loleftcoords_listX, loleftcoords_listY]
        output_file = os.path.join(output_file_path, "centroids_Scene"+repr(scene)+bg_choice+position+".txt")
        data2write = [x_pixpos, y_pixpos, true_centers, loleftcoords_list, mag_list, min_diff_pixposX, min_diff_pixposY]
        TAf.writePixPos(save_text_file, show_centroids, output_file, lines4screenandfile, stars_sample, background2use, data2write)

    if debug:
        print ("Check that read BENCHMARK values correspond to expected for case: ", case)
        print ("Star, xP1, yP1, V2P1, V3P1, xLP1, yLP1")
        print (bench_starP1[0], bench_xP1[0], bench_yP1[0], bench_V2P1[0], bench_V3P1[0], bench_xLP1[0], bench_yLP1[0])
        print ("Star, xP2, yP2, V2P2, V3P2, xLP2, yLP2")
        print (bench_starP2[0], bench_xP2[0], bench_yP2[0], bench_V2P2[0], bench_V3P2[0], bench_xLP2[0], bench_yLP2[0])
        print ("Check that read MEASURED values correspond to expected for the same case: ", case)
        print ("   -> reading measured info from: ", case)
        print ("Star, BG, x13, y13, x15, y15, x17, y17, LoLeftP1 (x, y), TrueP1 (x, y)")
        print (stars_sample[0], bg_choice, x13[0], y13[0], x15[0], y15[0], x17[0], y17[0], bench_xLP1[0], bench_yLP1[0], bench_xP1[0], bench_yP1[0])
        print ("Star, BG, x23, y23, x25, y25, x27, y27, LoLeftP2 (x, y), TrueP2 (x, y)")
        print (stars_sample[0], bg_choice, x23[0], y23[0], x25[0], y25[0], x27[0], y27[0], bench_xLP2[0], bench_yLP2[0], bench_xP2[0], bench_yP2[0])
        raw_input(" * press enter to continue... \n")

    # show positions on screen
    line0 = "\n Centroid indexing starting at 1 !"
    line0a = "{:<5} {:<15} {:<16} {:>23} {:>30} {:>44} {:>17} {:>15}".format("Star", "Background",
                                                                      "Centroid windows: 3", "5", "7",
                                                                      "TruePositions", "LoLeftCoords",
                                                                      "Mag")
    line0b = "{:>25} {:>12} {:>16} {:>14} {:>16} {:>14} {:>16} {:>18} {:>12} {:>10}".format(
                                                                           "x", "y", "x", "y", "x", "y",
                                                                           "TrueX", "TrueY", "LoLeftX", "LoLeftY")
    print ("Analyzing case: ", case)
    print (line0)
    print (line0a)
    print (line0b)
    for i, st in enumerate(stars_sample):
        line1 = "{:<5} {:<10} {:<14} {:<16} {:<14} {:<16} {:<14} {:<16} {:<14} {:<16} {:<8} {:<12} {:<10.2f}".format(
                                                                    int(st), background2use,
                                                                    x13[i], y13[i], x15[i], y15[i], x17[i], y17[i],
                                                                    bench_xP1[i]-bench_xLP1[i], bench_yP1[i]-bench_yLP1[i],
                                                                    bench_xLP1[i], bench_yLP1[i],
                                                                    magnitudes[i])
        print (line1)

    # compact results for functions
    P1P2data = [x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27]

    #plot_pixpos = True
    if plot_pixpos:
        # plot of sample residual x and y for positions 1 and 2
        fig1 = plt.figure(1, figsize=(12, 10))
        ax1 = fig1.add_subplot(111)
        #plt.suptitle(plot_title, fontsize=18, y=0.96)
        plt.title(case)
        plt.xlabel('X Residuals [Pixels]')
        plt.ylabel('Y Residuals [Pixels]')
        arrx, arry = x17-bench_xP1, y17-bench_yP1
        xP1 = [min(arrx)+min(arrx)*0.5, max(arrx)+max(arrx)*0.5]
        yP1 = [min(arry)+min(arry)*0.5, max(arry)+max(arry)*0.5]
        arrx, arry = x27-bench_xP2, y27-bench_yP2
        xP2 = [min(arrx)+min(arrx)*0.5, max(arrx)+max(arrx)*0.5]
        yP2 = [min(arry)+min(arry)*0.5, max(arry)+max(arry)*0.5]
        # determine qhich limit is larger in P1
        if xP1[1] > yP1[1]:
            larP1 = xP1[1]
        else:
            larP1 = yP1[1]
        if xP2[1] > yP2[1]:
            larP2 = xP2[1]
        else:
            larP2 = yP2[1]
        if larP1 > larP2:
            uplim = larP1
            lolim = -1 * larP1
        else:
            uplim = larP2
            lolim = -1 * larP2
        plt.xlim(lolim, uplim)
        plt.ylim(lolim, uplim)
        plt.hlines(0.0, lolim, uplim, colors='k', linestyles='dashed')
        plt.vlines(0.0, lolim, uplim, colors='k', linestyles='dashed')
        # plot measured positions
        plt.plot(x13-bench_xP1, y13-bench_yP1, 'b^', ms=10, alpha=0.5, label='CentroidWindow3_P1')
        plt.plot(x15-bench_xP1, y15-bench_yP1, 'go', ms=10, alpha=0.5, label='CentroidWindow5_P1')
        plt.plot(x17-bench_xP1, y17-bench_yP1, 'r*', ms=13, alpha=0.5, label='CentroidWindow7_P1')
        plt.plot(x23-bench_xP2, y23-bench_yP2, 'c^', ms=10, alpha=0.5, label='CentroidWindow3_P2')
        plt.plot(x25-bench_xP2, y25-bench_yP2, 'yo', ms=10, alpha=0.5, label='CentroidWindow5_P2')
        plt.plot(x27-bench_xP2, y27-bench_yP2, 'm*', ms=13, alpha=0.5, label='CentroidWindow7_P2')
        # Shrink current axis by 20%
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))   # put legend out of the plot box
        y_reject = [-1.0, 1.0]
        x_reject = [-1.0, 1.0]
        for si, xi, yi in zip(stars_sample, x13-bench_xP1, y13-bench_yP1):
            #if yi >= y_reject[1] or yi <= y_reject[0] or xi >= x_reject[1] or xi <= x_reject[0]:
            si = int(si)
            subxcoord = 5
            subycoord = 0
            side = 'left'
            plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
        for si, xi, yi in zip(stars_sample, x23-bench_xP2, y23-bench_yP2):
            #if yi >= y_reject[1] or yi <= y_reject[0] or xi >= x_reject[1] or xi <= x_reject[0]:
            si = int(si)
            subxcoord = 5
            subycoord = 0
            side = 'left'
            plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
        plt.show()
    return bg_choice, P1P2data, bench_starP1, benchmark_V2V3_sampleP1P2


def transformAndRunTest(stars_sample, path4results, primary_params, secondary_params,
                        bg_choice, P1P2data, bench_starP1, benchmark_V2V3_sampleP1P2, plot_v2v3pos=True,
                        extra_string=None):
    '''
    This function converts to sky for the X random star sample, and performs the given test.

    Args:
        primary_params: list, set of parameters specific to the case
        secondary_params: list, set of generic parameters
        bg_choice: string of the background method used
        P1P2data: list of pixel positions for centroid windows 3, 5, and 7 for both positions,
                        P1P2data = [x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27]
        bench_starP1: list of the benchmark stars studied
        benchmark_V2V3_sampleP1P2: list of benchmark V2s and V3s,
                        benchmark_V2V3_sampleP1P2 = [bench_V2P1, bench_V3P1,   bench_V2P2, bench_V3P2]

    Returns:
            case = string, string, for example 'Scene2_rapid_real_bgFrac'
            Tbench_Vs_list = list of benchmark V2 and V3s
            T_Vs = list of measured V2 and V3s
            T_diffVs = list of true-measured V2s and V3s
            LS_res = list, std deviations and means from least squared routine
    '''

    # unfold variables
    primary_params1, primary_params2, primary_params3 = primary_params
    do_plots, save_plots, show_plots, detector, output_full_detector, show_onscreen_results, show_pixpos_and_v23_plots, save_text_file = primary_params1
    save_centroid_disp, keep_bad_stars, keep_ugly_stars, just_least_sqares, stars_in_sample, scene, background_method, background2use = primary_params2
    shutters, noise, filter_input, test2perform, Nsigma, abs_threshold, abs_threshold, min_elements, max_iters_Nsig = primary_params3
    secondary_params1, secondary_params2, secondary_params3 = secondary_params
    checkbox_size, xwidth_list, ywidth_list, vlim, threshold, max_iter, verbose = secondary_params1
    debug, arcsecs, determine_moments, display_master_img, show_centroids, show_disp = secondary_params2
    Pier_corr, tilt, backgnd_subtraction_method, random_sample = secondary_params3
    bench_V2P1, bench_V3P1,   bench_V2P2, bench_V3P2 = benchmark_V2V3_sampleP1P2
    trueVsP1 = [bench_V2P1, bench_V3P1]
    trueVsP2 = [bench_V2P2, bench_V3P2]

    # transform into sky coordinates
    #case2study = [scene, shutters, noise, bg_choice]
    if type(detector) is not str:
        det = repr(detector)
    else:
        det = '2Dets'
    case = det+"Scene"+str(scene)+"_"+shutters+"_"+noise+bg_choice+repr(background2use)+'_Nsigma'+repr(Nsigma)
    if extra_string is not None:
        case += extra_string

    # Now run the tests
    transf_direction = "forward"
    detectors = [491, 492]
    # TEST 1: (a) Avg P1 and P2, (b) transform to V2-V3, (c) compare to avg reference positions (V2-V3 space)
    if test2perform == "T1":
        resultsTEST1 = TAf.runTEST(test2perform, detectors, transf_direction, case, stars_sample, P1P2data, bench_starP1,
                                   trueVsP1, trueVsP2, filter_input, tilt, arcsecs, debug)
        T1P1P2data, T1_transformations, T1_diffs, T1_benchVs_list = resultsTEST1
        T1_V2_3, T1_V3_3, T1_V2_5, T1_V3_5, T1_V2_7, T1_V3_7 = T1_transformations
        T1_diffV2_3, T1_diffV3_3, T1_diffV2_5, T1_diffV3_5, T1_diffV2_7, T1_diffV3_7 = T1_diffs
        T1bench_V2_list, T1bench_V3_list = T1_benchVs_list
        # Get the statistics
        results_stats = TAf.get_stats(T1_transformations, T1_diffs, T1_benchVs_list, Nsigma, max_iters_Nsig,
                                      arcsecs, just_least_sqares, abs_threshold, min_elements)
        # unfold results
        T1_st_devsAndMeans, T1_diff_counter, T1_bench_values, T1_sigmas_deltas, T1_sigma_reject, rejected_elementsLS, rejected_eleNsig, iterations = results_stats
        T1stdev_V2_3, T1mean_V2_3, T1stdev_V2_5, T1mean_V2_5, T1stdev_V2_7, T1mean_V2_7, T1stdev_V3_3, T1mean_V3_3, T1stdev_V3_5, T1mean_V3_5, T1stdev_V3_7, T1mean_V3_7 = T1_st_devsAndMeans
        T1_min_diff, T1_counter = T1_diff_counter
        T1LSdeltas_3, T1LSsigmas_3, T1LSlines2print_3, T1LSdeltas_5, T1LSsigmas_5, T1LSlines2print_5, T1LSdeltas_7, T1LSsigmas_7, T1LSlines2print_7 = T1_sigmas_deltas
        T1sigmaV2_3, T1meanV2_3, T1sigmaV3_3, T1meanV3_3, T1newV2_3, T1newV3_3, T1niter_3, T1lines2print_3, T1sigmaV2_5, T1meanV2_5, T1sigmaV3_5, T1meanV3_5, T1newV2_5, T1newV3_5, T1niter_5, T1lines2print_5, T1sigmaV2_7, T1meanV2_7, T1sigmaV3_7, T1meanV3_7, T1newV2_7, T1newV3_7, T1niter_7, T1lines2print_7 = T1_sigma_reject

    # TEST 2: (a) Transform individual P1 and P2 to V2-V3, (b) avg V2-V3 space positions, (c) compare to avg reference positions
    if test2perform == "T2":
        resultsTEST2 = TAf.runTEST(test2perform, detectors, transf_direction, case, stars_sample, P1P2data, bench_starP1,
                                   trueVsP1, trueVsP2, filter_input, tilt, arcsecs, debug)
        T2P1P2data, T2_transformations, T2_diffs, T2_benchVs_list = resultsTEST2
        T2_V2_3, T2_V3_3, T2_V2_5, T2_V3_5, T2_V2_7, T2_V3_7 = T2_transformations
        T2_diffV2_3, T2_diffV3_3, T2_diffV2_5, T2_diffV3_5, T2_diffV2_7, T2_diffV3_7 = T2_diffs
        T2bench_V2_list, T2bench_V3_list = T2_benchVs_list
        # Get the statistics
        results_stats = TAf.get_stats(T2_transformations, T2_diffs, T2_benchVs_list, Nsigma, max_iters_Nsig,
                                      arcsecs, just_least_sqares, abs_threshold, min_elements)
        # unfold results
        T2_st_devsAndMeans, T2_diff_counter, T2_bench_values, T2_sigmas_deltas, T2_sigma_reject, rejected_elementsLS, rejected_eleNsig, iterations = results_stats
        T2stdev_V2_3, T2mean_V2_3, T2stdev_V2_5, T2mean_V2_5, T2stdev_V2_7, T2mean_V2_7, T2stdev_V3_3, T2mean_V3_3, T2stdev_V3_5, T2mean_V3_5, T2stdev_V3_7, T2mean_V3_7 = T2_st_devsAndMeans
        T2_min_diff, T2_counter = T2_diff_counter
        T2LSdeltas_3, T2LSsigmas_3, T2LSlines2print_3, T2LSdeltas_5, T2LSsigmas_5, T2LSlines2print_5, T2LSdeltas_7, T2LSsigmas_7, T2LSlines2print_7 = T2_sigmas_deltas
        T2sigmaV2_3, T2meanV2_3, T2sigmaV3_3, T2meanV3_3, T2newV2_3, T2newV3_3, T2niter_3, T2lines2print_3, T2sigmaV2_5, T2meanV2_5, T2sigmaV3_5, T2meanV3_5, T2newV2_5, T2newV3_5, T2niter_5, T2lines2print_5, T2sigmaV2_7, T2meanV2_7, T2sigmaV3_7, T2meanV3_7, T2newV2_7, T2newV3_7, T2niter_7, T2lines2print_7 = T2_sigma_reject

    # TEST 3: (a) Transform P1 and P2 individually to V2-V3 (b) compare star by star and position by position
    if test2perform == "T3":
        resultsTEST3 = TAf.runTEST(test2perform, detectors, transf_direction, case, stars_sample, P1P2data, bench_starP1,
                                   trueVsP1, trueVsP2, filter_input, tilt, arcsecs, debug)
        T3P1P2data, T3_transformations, T3_diffs, T3_benchVs_list = resultsTEST3
        x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27 = T3P1P2data
        T_V2_3, T_V3_3, T_V2_5, T_V3_5, T_V2_7, T_V3_7 = T3_transformations
        T3_V2_13, T3_V2_23 = T_V2_3
        T3_V3_13, T3_V3_23 = T_V3_3
        T3_V2_15, T3_V2_25 = T_V2_5
        T3_V3_15, T3_V3_25 = T_V3_5
        T3_V2_17, T3_V2_27 = T_V2_7
        T3_V3_17, T3_V3_27 = T_V3_7
        T_diffV2_3, T_diffV3_3, T_diffV2_5, T_diffV3_5, T_diffV2_7, T_diffV3_7 = T3_diffs
        T3_diffV2_13, T3_diffV2_23 = T_diffV2_3
        T3_diffV3_13, T3_diffV3_23 = T_diffV3_3
        T3_diffV2_15, T3_diffV2_25 = T_diffV2_5
        T3_diffV3_15, T3_diffV3_25 = T_diffV3_5
        T3_diffV2_17, T3_diffV2_27 = T_diffV2_7
        T3_diffV3_17, T3_diffV3_27 = T_diffV3_7
        T3bench_V2_list, T3bench_V3_list = T3_benchVs_list
        T3bench_V2_listP1, T3bench_V2_listP2 = T3bench_V2_list
        T3bench_V3_listP1, T3bench_V3_listP2 = T3bench_V3_list
        # combine the arrays (positions 1 and 2)
        T3_V2_3, T3_V2_5, T3_V2_7 = np.array([]), np.array([]), np.array([])
        T3_V2_3 = TAf.combine2arrays(T3_V2_13, T3_V2_23, T3_V2_3)
        T3_V2_5 = TAf.combine2arrays(T3_V2_15, T3_V2_25, T3_V2_5)
        T3_V2_7 = TAf.combine2arrays(T3_V2_17, T3_V2_27, T3_V2_7)
        T3_V3_3, T3_V3_5, T3_V3_7 = np.array([]), np.array([]), np.array([])
        T3_V3_3 = TAf.combine2arrays(T3_V3_13, T3_V3_23, T3_V3_3)
        T3_V3_5 = TAf.combine2arrays(T3_V3_15, T3_V3_25, T3_V3_5)
        T3_V3_7 = TAf.combine2arrays(T3_V3_17, T3_V3_27, T3_V3_7)
        T3_diffV2_3, T3_diffV2_5, T3_diffV2_7 = np.array([]), np.array([]), np.array([])
        T3_diffV2_3 = TAf.combine2arrays(T3_diffV2_13, T3_diffV2_23, T3_diffV2_3)
        T3_diffV2_5 = TAf.combine2arrays(T3_diffV2_15, T3_diffV2_25, T3_diffV2_5)
        T3_diffV2_7 = TAf.combine2arrays(T3_diffV2_17, T3_diffV2_27, T3_diffV2_7)
        T3_diffV3_3, T3_diffV3_5, T3_diffV3_7 = np.array([]), np.array([]), np.array([])
        T3_diffV3_3 = TAf.combine2arrays(T3_diffV3_13, T3_diffV3_23, T3_diffV3_3)
        T3_diffV3_5 = TAf.combine2arrays(T3_diffV3_15, T3_diffV3_25, T3_diffV3_5)
        T3_diffV3_7 = TAf.combine2arrays(T3_diffV3_17, T3_diffV3_27, T3_diffV3_7)
        T3bench_V2_list, T3bench_V3_list = np.array([]), np.array([])
        T3bench_V2_list = TAf.combine2arrays(np.array(T3bench_V2_listP1), np.array(T3bench_V2_listP2), T3bench_V2_list)
        T3bench_V3_list = TAf.combine2arrays(np.array(T3bench_V3_listP1), np.array(T3bench_V3_listP2), T3bench_V3_list)
        T3bench_V2_list.tolist()
        T3bench_V3_list.tolist()
        # Get the statistics
        T3_transformations = [T3_V2_3, T3_V3_3, T3_V2_5, T3_V3_5, T3_V2_7, T3_V3_7]
        T3_diffs = [T3_diffV2_3, T3_diffV3_3, T3_diffV2_5, T3_diffV3_5, T3_diffV2_7, T3_diffV3_7]
        T3_benchVs_list = [T3bench_V2_list, T3bench_V3_list]
        results_stats = TAf.get_stats(T3_transformations, T3_diffs, T3_benchVs_list, Nsigma, max_iters_Nsig,
                                      arcsecs, just_least_sqares, abs_threshold, min_elements)
        # unfold results
        T3_st_devsAndMeans, T3_diff_counter, T3_bench_values, T3_sigmas_deltas, T3_sigma_reject, rejected_elementsLS, rejected_eleNsig, iterations = results_stats
        T3stdev_V2_3, T3mean_V2_3, T3stdev_V2_5, T3mean_V2_5, T3stdev_V2_7, T3mean_V2_7, T3stdev_V3_3, T3mean_V3_3, T3stdev_V3_5, T3mean_V3_5, T3stdev_V3_7, T3mean_V3_7 = T3_st_devsAndMeans
        T3_min_diff, T3_counter = T3_diff_counter
        T3LSdeltas_3, T3LSsigmas_3, T3LSlines2print_3, T3LSdeltas_5, T3LSsigmas_5, T3LSlines2print_5, T3LSdeltas_7, T3LSsigmas_7, T3LSlines2print_7 = T3_sigmas_deltas
        T3sigmaV2_3, T3meanV2_3, T3sigmaV3_3, T3meanV3_3, T3newV2_3, T3newV3_3, T3niter_3, T3lines2print_3, T3sigmaV2_5, T3meanV2_5, T3sigmaV3_5, T3meanV3_5, T3newV2_5, T3newV3_5, T3niter_5, T3lines2print_5, T3sigmaV2_7, T3meanV2_7, T3sigmaV3_7, T3meanV3_7, T3newV2_7, T3newV3_7, T3niter_7, T3lines2print_7 = T3_sigma_reject

        #plot_v2v3pos = True
        if plot_v2v3pos:
            # plot of sample residual V2 and V3 for positions 1 and 2 for test 3
            fig1 = plt.figure(1, figsize=(12, 10))
            ax1 = fig1.add_subplot(111)
            #plt.suptitle(plot_title, fontsize=18, y=0.96)
            plt.title(case)
            plt.xlabel('V2 Residuals [arcsec]')
            plt.ylabel('V3 Residuals [arcsec]')
            #xlims = [-5.0, 5.0]
            #ylims = [-5.0, 5.0]
            #plt.xlim(xlims[0], xlims[1])
            #plt.ylim(ylims[0], ylims[1])
            #plt.hlines(0.0, xlims[0], xlims[1], colors='k', linestyles='dashed')
            #plt.vlines(0.0, ylims[0], ylims[1], colors='k', linestyles='dashed')
            arrx, arry = T3_diffV2_17, T3_diffV3_17
            xP1 = [min(arrx)+min(arrx)*0.5, max(arrx)+max(arrx)*0.5]
            yP1 = [min(arry)+min(arry)*0.5, max(arry)+max(arry)*0.5]
            arrx, arry = T3_diffV2_27, T3_diffV3_27
            xP2 = [min(arrx)+min(arrx)*0.5, max(arrx)+max(arrx)*0.5]
            yP2 = [min(arry)+min(arry)*0.5, max(arry)+max(arry)*0.5]
            # determine qhich limit is larger in P1
            if xP1[1] > yP1[1]:
                larP1 = xP1[1]
            else:
                larP1 = yP1[1]
            if xP2[1] > yP2[1]:
                larP2 = xP2[1]
            else:
                larP2 = yP2[1]
            if larP1 > larP2:
                uplim = larP1
                lolim = -1 * larP1
            else:
                uplim = larP2
                lolim = -1 * larP2
            plt.xlim(lolim, uplim)
            plt.ylim(lolim, uplim)
            plt.hlines(0.0, lolim, uplim, colors='k', linestyles='dashed')
            plt.vlines(0.0, lolim, uplim, colors='k', linestyles='dashed')
            # plot measured positions
            plt.plot(T3_diffV2_13, T3_diffV3_13, 'b^', ms=10, alpha=0.5, label='CentroidWindow3_P1')
            plt.plot(T3_diffV2_15, T3_diffV3_15, 'go', ms=10, alpha=0.5, label='CentroidWindow5_P1')
            plt.plot(T3_diffV2_17, T3_diffV3_17, 'r*', ms=13, alpha=0.5, label='CentroidWindow7_P1')
            plt.plot(T3_diffV2_23, T3_diffV3_23, 'c^', ms=10, alpha=0.5, label='CentroidWindow3_P2')
            plt.plot(T3_diffV2_25, T3_diffV3_25, 'yo', ms=10, alpha=0.5, label='CentroidWindow5_P2')
            plt.plot(T3_diffV2_27, T3_diffV3_27, 'm*', ms=13, alpha=0.5, label='CentroidWindow7_P2')
            # Shrink current axis by 20%
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))   # put legend out of the plot box
            x_reject, y_reject = [-1.0, 1.0], [-1.0, 1.0]
            for si, xi, yi in zip(stars_sample, T3_diffV2_13, T3_diffV3_13):
                #if yi >= y_reject[1] or yi <= y_reject[0] or xi >= x_reject[1] or xi <= x_reject[0]:
                si = int(si)
                subxcoord, subycoord = 5, 0
                side = 'left'
                plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
            for si, xi, yi in zip(stars_sample, T3_diffV2_23, T3_diffV3_23):
                #if yi >= y_reject[1] or yi <= y_reject[0] or xi >= x_reject[1] or xi <= x_reject[0]:
                si = int(si)
                subxcoord, subycoord = 5, 0
                side = 'left'
                plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
            plt.show()

    # Print results to screen and save into a text file if told so
    if test2perform == "T1":
        Tstdev_Vs = [T1stdev_V2_3, T1stdev_V3_3, T1stdev_V2_5, T1stdev_V3_5, T1stdev_V2_7, T1stdev_V3_7]
        Tmean_Vs = [T1mean_V2_3, T1mean_V3_3, T1mean_V2_5, T1mean_V3_5, T1mean_V2_7, T1mean_V3_7]
        T_diff_counter = [T1_min_diff, T1_counter]
        TLSlines2print = [T1LSlines2print_3, T1LSlines2print_5, T1LSlines2print_7]
        Tlines2print = [T1lines2print_3, T1lines2print_5, T1lines2print_7]
        Tbench_Vs_list = [T1bench_V2_list, T1bench_V3_list]
        T_Vs = [T1_V2_3, T1_V3_3, T1_V2_5, T1_V3_5, T1_V2_7, T1_V3_7]
        T_diffVs = [T1_diffV2_3, T1_diffV3_3, T1_diffV2_5, T1_diffV3_5, T1_diffV2_7, T1_diffV3_7]
        LS_res = [T1LSsigmas_3, T1LSsigmas_5, T1LSsigmas_7, T1LSdeltas_3, T1LSdeltas_5, T1LSdeltas_7]

    if test2perform == "T2":
        Tstdev_Vs = [T2stdev_V2_3, T2stdev_V3_3, T2stdev_V2_5, T2stdev_V3_5, T2stdev_V2_7, T2stdev_V3_7]
        Tmean_Vs = [T2mean_V2_3, T2mean_V3_3, T2mean_V2_5, T2mean_V3_5, T2mean_V2_7, T2mean_V3_7]
        T_diff_counter = [T2_min_diff, T2_counter]
        TLSlines2print = [T2LSlines2print_3, T2LSlines2print_5, T2LSlines2print_7]
        Tlines2print = [T2lines2print_3, T2lines2print_5, T2lines2print_7]
        Tbench_Vs_list = [T2bench_V2_list, T2bench_V3_list]
        T_Vs = [T2_V2_3, T2_V3_3, T2_V2_5, T2_V3_5, T2_V2_7, T2_V3_7]
        T_diffVs = [T2_diffV2_3, T2_diffV3_3, T2_diffV2_5, T2_diffV3_5, T2_diffV2_7, T2_diffV3_7]
        LS_res = [T2LSsigmas_3, T2LSsigmas_5, T2LSsigmas_7, T2LSdeltas_3, T2LSdeltas_5, T2LSdeltas_7]

    if test2perform == "T3":
        Tstdev_Vs = [T3stdev_V2_3, T3stdev_V3_3, T3stdev_V2_5, T3stdev_V3_5, T3stdev_V2_7, T3stdev_V3_7]
        Tmean_Vs = [T3mean_V2_3, T3mean_V3_3, T3mean_V2_5, T3mean_V3_5, T3mean_V2_7, T3mean_V3_7]
        T_diff_counter = [T3_min_diff, T3_counter]
        TLSlines2print = [T3LSlines2print_3, T3LSlines2print_5, T3LSlines2print_7]
        Tlines2print = [T3lines2print_3, T3lines2print_5, T3lines2print_7]
        Tbench_Vs_list = [T3bench_V2_list, T3bench_V3_list]
        T_Vs = [T3_V2_3, T3_V3_3, T3_V2_5, T3_V3_5, T3_V2_7, T3_V3_7]
        T_diffVs = [T3_diffV2_3, T3_diffV3_3, T3_diffV2_5, T3_diffV3_5, T3_diffV2_7, T3_diffV3_7]
        LS_res = [T3LSsigmas_3, T3LSsigmas_5, T3LSsigmas_7, T3LSdeltas_3, T3LSdeltas_5, T3LSdeltas_7]

    LS_info = [iterations, rejected_elementsLS]

    if show_onscreen_results or save_text_file:
        TAf.printTESTresults(stars_sample, case, test2perform, arcsecs, Tstdev_Vs, Tmean_Vs, T_diff_counter,
                      save_text_file, TLSlines2print, Tlines2print, Tbench_Vs_list, T_Vs, T_diffVs,
                      rejected_elementsLS, rejected_eleNsig, background_method, background2use, path4results)

    TV2_3, TV3_3, Tbench_V2_3, Tbench_V3_3 = rid_rejected_elements(rejected_elementsLS[0],
                                                                         T_Vs[0], T_Vs[1],
                                                                         Tbench_Vs_list[0], Tbench_Vs_list[1])
    TV2_5, TV3_5, Tbench_V2_5, Tbench_V3_5 = rid_rejected_elements(rejected_elementsLS[1],
                                                                         T_Vs[2], T_Vs[3],
                                                                         Tbench_Vs_list[0], Tbench_Vs_list[1])
    TV2_7, TV3_7, Tbench_V2_7, Tbench_V3_7 = rid_rejected_elements(rejected_elementsLS[2],
                                                                         T_Vs[4], T_Vs[5],
                                                                         Tbench_Vs_list[0], Tbench_Vs_list[1])
    TdiffV2_3, TdiffV3_3, _, _ = rid_rejected_elements(rejected_elementsLS[0],
                                                                         T_diffVs[0], T_diffVs[1],
                                                                         Tbench_Vs_list[0], Tbench_Vs_list[1])
    TdiffV2_5, TdiffV3_5, _, _ = rid_rejected_elements(rejected_elementsLS[1],
                                                                         T_diffVs[2], T_diffVs[3],
                                                                         Tbench_Vs_list[0], Tbench_Vs_list[1])
    TdiffV2_7, TdiffV3_7, _, _ = rid_rejected_elements(rejected_elementsLS[2],
                                                                         T_diffVs[4], T_diffVs[5],
                                                                         Tbench_Vs_list[0], Tbench_Vs_list[1])

    Tbench_Vs = [Tbench_V2_3, Tbench_V3_3, Tbench_V2_5, Tbench_V3_5, Tbench_V2_7, Tbench_V3_7]
    T_Vs = [TV2_3, TV3_3, TV2_5, TV3_5, TV2_7, TV3_7]
    T_diffVs = [TdiffV2_3, TdiffV3_3, TdiffV2_5, TdiffV3_5, TdiffV2_7, TdiffV3_7]
    new_stars_sample = ridstars_LSrejection(stars_sample, LS_info)

    return case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, LS_info


def rid_rejected_elements(rejected_elementsLS, TV2, TV3, TrueV2, TrueV3):
    TV2_cwin, TV3_cwin, TrueV2_cwin, TrueV3_cwin = [], [], [], []
    for idx, tv in enumerate(TV2):
        if idx in rejected_elementsLS:

            continue
        else:
            TV2_cwin.append(tv)
            TV3_cwin.append(TV3[idx])
            TrueV2_cwin.append(TrueV2[idx])
            TrueV3_cwin.append(TrueV3[idx])
    return TV2_cwin, TV3_cwin, TrueV2_cwin, TrueV3_cwin


def ridstars_LSrejection(stars_sample, LS_info):
    # unfold variables
    _, rejected_elementsLS = LS_info
    #  Create a new list with the elements not rejected by the least squares routine
    nw_stars_sample3, nw_stars_sample5, nw_stars_sample7 = [], [], []
    # append to the new lists for centroid window 3
    for i, st in enumerate(stars_sample):
        if i not in rejected_elementsLS[0]:
            nw_stars_sample3.append(st)
        if i not in rejected_elementsLS[1]:
            nw_stars_sample5.append(st)
        if i not in rejected_elementsLS[2]:
            nw_stars_sample7.append(st)
    new_stars_sample = [nw_stars_sample3, nw_stars_sample5, nw_stars_sample7]
    return new_stars_sample


def convert2milliarcsec(list2convert):
    for i, item in enumerate(list2convert):
        list2convert[i] = item * 1000.0
    return list2convert


def run_testXrandom_stars(stars_sample, primary_params, secondary_params, path4results, gen_path, extra_string):
    '''
    This is the function that coordinates all other functions within this script. It runs the script.
    Args:
        stars_sample: list of stars to analyze
        primary_params: list of 3 lists containing all variables in the primary parameters section
        secondary_params: list of 3 lists containing all variables in the secondary parameters section
        path4results: string of the path to place results
        gen_path: string of path to put plots and other resulting files
        extra_string: additional info added to name of text file with final V2 and V3

    Returns:
        results per window size = resulting V2 and V3, benchmark V2 and V3, the corresponding standard deviations,
                                the rotation angle, the total number of stars removed as well as the corresponding
                                star number, and the total number of iterations
        All of these results are in the following structure:
        results_of_test = [case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, LS_info] --> per TEST
        results_all_tests = [results_of_test, ...] --> can have only one list if only 1 test was ran
    '''

    # Unfold variables
    # primary parameters
    primary_params1, primary_params2, primary_params3 = primary_params
    do_plots, save_plots, show_plots, detector, output_full_detector, show_onscreen_results, show_pixpos_and_v23_plots, save_text_file = primary_params1
    save_centroid_disp, keep_bad_stars, keep_ugly_stars, just_least_sqares, stars_in_sample, scene, background_method, background2use = primary_params2
    shutters, noise, filter_input, test2perform, Nsigma, abs_threshold, abs_threshold, min_elements, max_iters_Nsig = primary_params3
    # secondary parameters
    secondary_params1, secondary_params2, secondary_params3 = secondary_params
    checkbox_size, xwidth_list, ywidth_list, vlim, threshold, max_iter, verbose = secondary_params1
    debug, arcsecs, determine_moments, display_master_img, show_centroids, show_disp = secondary_params2
    Pier_corr, tilt, backgnd_subtraction_method, random_sample = secondary_params3

    # Pool of stars to select from
    stars_detectors = range(1, 201)       # default is for both detectors
    if detector == 491:
        stars_detectors = range(101, 201) # only detector 491
    elif detector == 492:
        stars_detectors = range(1, 101)   # only detector 492

    # Loop over list_test2perform
    results_all_tests = []
    if test2perform == "all":
        list_test2perform = ["T1", "T2", "T3"]
        if not keep_bad_stars:
            # remove the bad stars and use the same sample for the 3 tests
            stars_sample = TAf.remove_bad_stars(scene, stars_sample, keep_ugly_stars, verbose)
            # but keep the sample stars list with length of desired number of stars
            while len(stars_sample) != stars_in_sample:
                random_star = random.choice(stars_detectors)
                stars_sample.append(random_star)
                stars_sample = list(set(stars_sample))
                # remove the bad stars
                stars_sample = TAf.remove_bad_stars(scene, stars_sample, keep_ugly_stars, verbose)
            keep_bad_stars = True
    else:
        list_test2perform = [test2perform]
    for test2perform in list_test2perform:
        print ('Starting analysis for TEST %s ...' % (test2perform))
        # RE-compact variables
        primary_params1 = [do_plots, save_plots, show_plots, detector, output_full_detector, show_onscreen_results,
                           show_pixpos_and_v23_plots, save_text_file]
        primary_params2 = [save_centroid_disp, keep_bad_stars, keep_ugly_stars, just_least_sqares, stars_in_sample,
                           scene, background_method, background2use]
        primary_params3 = [shutters, noise, filter_input, test2perform, Nsigma, abs_threshold, abs_threshold, min_elements,
                           max_iters_Nsig]
        primary_params = [primary_params1, primary_params2, primary_params3]
        secondary_params1 = [checkbox_size, xwidth_list, ywidth_list, vlim, threshold, max_iter, verbose]
        secondary_params2 = [debug, arcsecs, determine_moments, display_master_img, show_centroids, show_disp]
        secondary_params3 = [Pier_corr, tilt, backgnd_subtraction_method, random_sample]
        secondary_params = [secondary_params1, secondary_params2, secondary_params3]
        # Get centroids AND sky positions according to Test
        case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, LS_info = runXrandomstars(stars_detectors,
                                                                                       primary_params, secondary_params,
                                                                                       stars_sample,
                                                                                       path4results=path4results,
                                                                                       extra_string=extra_string)
        results_of_test = [case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, LS_info]
        results_all_tests.append(results_of_test)
        print ('TEST  %s  finished. \n' % (test2perform))


    if do_plots:
        print ('Generating plots...')

        # load the data fom the 3 tests
        for resTest in results_all_tests:
            # unfold variables per centroid window results_all_tests[0][5][s][width]
            case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, _ = resTest
            nw_stars_sample3, nw_stars_sample5, nw_stars_sample7 = new_stars_sample
            Tbench_V2_3, Tbench_V3_3, Tbench_V2_5, Tbench_V3_5, Tbench_V2_7, Tbench_V3_7 = Tbench_Vs
            TV2_3, TV3_3, TV2_5, TV3_5, TV2_7, TV3_7 = T_Vs
            TdiffV2_3, TdiffV3_3, TdiffV2_5, TdiffV3_5, TdiffV2_7, TdiffV3_7 = T_diffVs
            TLSsigmas_3, TLSsigmas_5, TLSsigmas_7, TLSdeltas_3, TLSdeltas_5, TLSdeltas_7 = LS_res

            milliarcsec = True
            if milliarcsec:
                TdiffV2_3 = convert2milliarcsec(TdiffV2_3)
                TdiffV3_3 = convert2milliarcsec(TdiffV3_3)
                TdiffV2_5 = convert2milliarcsec(TdiffV2_5)
                TdiffV3_5 = convert2milliarcsec(TdiffV3_5)
                TdiffV2_7 = convert2milliarcsec(TdiffV2_7)
                TdiffV3_7 = convert2milliarcsec(TdiffV3_7)
                TLSsigmas_3 = convert2milliarcsec(TLSsigmas_3)
                TLSsigmas_5 = convert2milliarcsec(TLSsigmas_5)
                TLSsigmas_7 = convert2milliarcsec(TLSsigmas_7)
                TLSdeltas_3 = convert2milliarcsec(TLSdeltas_3)
                TLSdeltas_5 = convert2milliarcsec(TLSdeltas_5)
                TLSdeltas_7 = convert2milliarcsec(TLSdeltas_7)

        # do the plots -> 2 plots per centroid window
        for cwin in xwidth_list:
            cwincase = case+'_CentroidWindow'+repr(cwin)

            # Plot to compare the mean values for the 3 tests -- plot only has 3 points
            plot_title = r'Residual Mean Values, $\mu$'
            xlabel = r'$\Delta$V2 [marcsec]'
            ylabel = r'$\Delta$V3 [marcsec]'
            destination = os.path.join(gen_path, 'plots/means_Cwin'+repr(cwin)+'.jpg')
            if cwin == 3:
                s, d, v = 0, 3, 0
            if cwin == 5:
                s, d, v = 1, 4, 2
            if cwin == 7:
                s, d, v = 2, 5, 4
            if len(list_test2perform) != 3:
                T1sigmaV2 = results_all_tests[0][5][s][0]   # Test ran sigma V2 value
                T1sigmaV3 = results_all_tests[0][5][s][1]   # Test ran sigma V3 value
                T1meanV3 = results_all_tests[0][5][d][1]    # Test ran mean V3 value
                T1meanV2 = results_all_tests[0][5][d][0]    # Test ran mean V2 value
                if test2perform == "T1":
                    labels_list = ['Avg in Pixel Space']
                if test2perform == "T2":
                    labels_list = ['Avg in Sky']
                if test2perform == "T3":
                    labels_list = ['No Avg']
                arrx = [T1meanV2]
                arry = [T1meanV3]
                print_side_values = [T1sigmaV2, T1meanV2, T1sigmaV3, T1meanV3]
            if len(list_test2perform) == 3:
                T1sigmaV2 = results_all_tests[0][5][s][0]   # Test 1 ran sigma V2 value
                T1sigmaV3 = results_all_tests[0][5][s][1]   # Test 1 ran sigma V3 value
                T1meanV3 = results_all_tests[0][5][d][1]    # Test 1 ran mean V3 value
                T1meanV2 = results_all_tests[0][5][d][0]    # Test 1 ran mean V2 value
                T2sigmaV2 = results_all_tests[1][5][s][0]   # Test 2
                T2sigmaV3 = results_all_tests[1][5][s][1]   # Test 2
                T2meanV2 = results_all_tests[1][5][d][0]    # Test 2
                T2meanV3 = results_all_tests[1][5][d][1]    # Test 2
                T3sigmaV2 = results_all_tests[2][5][s][0]   # Test 3
                T3sigmaV3 = results_all_tests[2][5][s][1]   # Test 3
                T3meanV2 = results_all_tests[2][5][d][0]    # Test 3
                T3meanV3 = results_all_tests[2][5][d][1]    # Test 3
                labels_list = ['Avg in Pixel Space', 'Avg in Sky', 'No Avg']
                arrx = [T1meanV2, T2meanV2, T3meanV2]
                arry = [T1meanV3, T2meanV3, T3meanV3]
                print_side_values = [T1sigmaV2, T1meanV2, T2sigmaV2, T2meanV2, T3sigmaV2, T3meanV2,
                                     T1sigmaV3, T1meanV3, T2sigmaV3, T2meanV3, T3sigmaV3, T3meanV3]
            print_side_string = ['V2$\mu$ [marcsec]', 'V3$\mu$ [marcsec]']
            # determine which one is larger
            if np.abs(T1meanV2) > np.abs(T1meanV3):
                largV = np.abs(T1meanV2)+np.abs(T1meanV2)*0.5
            else:
                largV = np.abs(T1meanV3)+np.abs(T1meanV3)*0.5
            xlims, ylims = [-1*largV, largV], [-1*largV, largV]

            vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title,
                      labels_list=labels_list, xlims=xlims, ylims=ylims,
                      print_side_string=print_side_string, print_side_values=print_side_values,
                      save_plot=save_plots, show_plot=show_plots, destination=destination)


            # Graphical display of the standard deviation
            plot_title = r'Graphical Display of the Standard Deviation, $\sigma$'
            destination = os.path.join(gen_path, 'plots/V2V3_Cwin'+repr(cwin)+'.jpg')
            if len(list_test2perform) == 3:
                arrx = [results_all_tests[0][4][v], results_all_tests[1][4][v], results_all_tests[2][4][v]]
                arry = [results_all_tests[0][4][v+1], results_all_tests[1][4][v+1], results_all_tests[2][4][v+1]]
                # determine which one is larger
                maxx = max(np.abs(results_all_tests[2][4][v]))
                maxy = max(np.abs(results_all_tests[2][4][v+1]))
                new_stars_sample = [results_all_tests[0][1][s], results_all_tests[1][1][s], results_all_tests[2][1][s]]
            else:
                arrx = [results_all_tests[0][4][v]]
                arry = [results_all_tests[0][4][v+1]]
                # determine which one is larger
                maxx = max(np.abs(results_all_tests[0][4][v]))
                maxy = max(np.abs(results_all_tests[0][4][v+1]))
                new_stars_sample = [results_all_tests[0][1][s]]
            if maxx > maxy:
                largsig = maxx + maxx*0.5
            else:
                largsig = maxy + maxy*0.5
            xlims, ylims = [-1*largsig, largsig], [-1*largsig, largsig]
            vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title,
                            labels_list=labels_list, xlims=xlims, ylims=ylims,
                            print_side_string=print_side_string, print_side_values=print_side_values,
                            save_plot=save_plots, show_plot=show_plots, destination=destination,
                            star_sample=new_stars_sample)

    return results_all_tests



#######################################################################################################################


### CODE

if __name__ == '__main__':


    # SET PRIMARY PARAMETERS
    do_plots = True                    # 1. Least squares plot in V2/V3 space showing the true position (0,0)
    #                                       and the mean of the three calculation cases:  Averaging in pixel space,
    #                                       averaging on sky, and no averaging : True or False
    #                                    2. Same plot but instead of the mean show all stars in one 20star calculation
    save_plots = False                 # Save the plots? True or False
    show_plots = True                  # Show the plots? True or False
    detector = 'both'                     # Integer (491 or 492) OR string, 'both' to select stars from both detectors
    output_full_detector = True        # Give resulting coordinates in terms of full detector: True or False
    show_onscreen_results = True       # Want to show on-screen resulting V2s, V3s and statistics? True or False
    show_pixpos_and_v23_plots = False  # Show the plots of x-y and v2-v3 residual positions?
    save_text_file = False             # Want to save the text file of comparison? True or False
    save_centroid_disp = False         # Save the display with measured and true positions?
    keep_bad_stars = False             # Keep the bad stars in the sample (both positions measured wrong)? True or False
    keep_ugly_stars = True             # Keep the ugly stars (one position measured wrong)? True or False
    perform_abs_threshold = False       # Perform abs_threshold routine (True) or only perform least squares routine (False)
    stars_in_sample = 5               # Number of stars in sample (165 for all good and uglies)
    scene = 1                          # Integer or string, scene=1 is constant Mag 23, scene=2 is stars with Mag 18-23
    background_method = 'frac'         # Select either 'fractional', 'fixed', or None
    background2use = 0.3               # Background to use for analysis: None or float
    shutters = "rapid"                 # Shutter velocity, string: "rapid" or "slow"
    noise = "real"                     # Noise level, string: "nonoise" or "real"
    filter_input = "F140X"             # Filter, string: for now only test case is "F140X"
    test2perform = "all"                # Test to perform, string: "all", "T1", "T2", "T3" for test 1, 2, and 3, respectively
    Nsigma = 2.5                         # N-sigma rejection of bad stars: integer or float
    abs_threshold = 0.32               # threshold to reject points after each iteration of least squares routine, default=0.32
    min_elements = 4                   # minimum number of elements in the absolute threshold least squares routine, default=4
    max_iters_Nsig = 10                # Max number of iterations for N-sigma function: integer

    # SET SECONDARY PARAMETERS THAT CAN BE ADJUSTED
    checkbox_size = 3                  # Real checkbox size
    xwidth_list = [3, 5, 7]            # Number of rows of the centroid region
    ywidth_list = [3, 5, 7]            # Number of columns of the centroid region
    vlim = (1, 100)                    # Sensitivity limits of image, i.e. (0.001, 0.1)
    threshold = 0.01                   # Convergence threshold of accepted difference between checkbox centroid and coarse location
    max_iter = 10                      # Maximum number of iterations for finding coarse location
    verbose = False                    # Show some debug messages (i.e. resulting calculations)
    debug = False                      # See all debug messages (i.e. values of variables and calculations)
    arcsecs = True                     # Final units in arcsecs? True or False (=degrees)
    determine_moments = False          # Want to determine 2nd and 3rd moments?
    display_master_img = False         # Want to see the combined ramped images for every star?
    show_centroids = False             # Print measured centroid on screen: True or False
    show_disp = False                  # Show display of resulting positions? (will show 2 figs, same but different contrast)
    Pier_corr = True                   # Include Pier's corrections to measured positions
    tilt = False                       # Tilt angle: True or False
    backgnd_subtraction_method = 1     # 1    = Do background subtraction on final image (after subtracting 3-2 and 2-1),
    #                                           before converting negative values into zeros
    #                                    2    = Do background subtraction on 3-2 and 2-1 individually
    #                                    None = Do not subtract background

    random_sample = False               # choose a random sample of stars from either detector: True or False
    # control samples to be used when random is set to False
    #stars_sample = [1, 10, 23, 29, 33, 47, 61, 67, 95, 100, 107, 128, 133, 139, 151, 171, 190, 194, 195, 198]
    #stars_sample = [9, 20, 32, 48, 65, 69, 82, 83, 93, 98, 99, 107, 111, 126, 128, 136, 172, 176, 196, 198] #all good stars
    #stars_sample = [3, 26, 32, 38, 46, 48, 51, 65, 75, 84, 92, 96, 121, 122, 132, 133, 160, 174, 186, 194]
    #stars_sample = [3, 8, 9, 32, 38, 65, 96, 128, 132, 133, 136, 143, 145, 147, 160, 175, 178, 191, 193, 194] #all good stars
    #stars_sample = [32, 41, 49, 64, 65, 68, 84, 96, 99, 104, 131, 167, 175, 182, 192, 194, 195, 196, 197, 198]# all good
    #stars_sample = [2, 4, 5, 6, 11, 32, 38, 47, 81, 127, 129, 136, 138, 141, 160, 163, 166, 171, 174, 179] #* all good
    #stars_sample = [6, 18, 41, 49, 66, 75, 84, 93, 97, 99, 108, 110, 134, 140, 151, 160, 164, 175, 186, 200]# VERY good!
    #stars_sample = [15, 20, 43, 46, 47, 57, 62, 69, 71, 83, 86, 87, 90, 106, 121, 168, 179, 182, 185, 194]
    #stars_sample = [4, 42, 44, 69, 76, 96, 97, 99, 102, 114, 116, 128, 129, 130, 132, 142, 167, 176, 193, 194] # good to show bads
    stars_sample = [1, 128, 130, 131, 196]
    #stars_sample = [1, 35, 128, 130, 164]
    #stars_sample = [3, 4, 8, 32, 139]
    #stars_sample = [32, 33, 104, 188, 199]
    #stars_sample = [3, 32, 33, 133, 162]
    #stars_sample = [16, 22, 29, 50, 108]
    #stars_sample = [2, 5, 15, 46, 154, 156, 163]
    #stars_sample = [5, 80, 116, 130, 135]#, 17, 31, 113, 182]
    #stars_sample = [8, 11, 27, 44, 90]
    #stars_sample = [12, 21, 32, 54, 77]
    ##stars_sample = [22, 90, 108, 126, 145]
    #stars_sample = [101, 110, 121, 133, 200]
    #stars_sample = [111, 120, 142, 173, 180]
    #stars_sample = [10, 32, 33, 35, 42, 47, 52, 70, 73, 77, 100, 128, 130, 135, 136, 137, 141, 147, 179, 192] # all good stars *
    #stars_sample = [8, 33, 37, 38, 44, 50, 51, 54, 63, 98, 99, 109, 114, 127, 138, 139, 162, 163, 171, 186]
    #stars_sample = [3, 16, 35, 36, 39, 64, 65, 70, 73, 90, 111, 122, 129, 134, 136, 154, 165, 183, 194, 196]
    #stars_sample = [2, 4, 6, 11, 36, 38, 43, 98, 102, 109, 110, 141, 149, 160, 161, 163, 165, 173, 174, 177]
    #stars_sample = [5, 7, 8, 12, 33, 37, 40, 101, 108, 109, 111, 151, 159, 162, 166, 167, 169, 170, 175, 187]
    # bad samples:
    #stars_sample = [7, 24, 51, 56, 66, 68, 71, 72, 74, 91, 106, 109, 120, 125, 127, 128, 138, 154, 187, 188]
    #stars_sample = [8, 9, 20, 21, 39, 40, 46, 54, 58, 76, 78, 87, 88, 121, 134, 146, 150, 167, 179, 180]
    # OLNY detector 491
    #stars_sample = [101, 105, 108, 109, 111, 113, 114, 133, 136, 147, 150, 157, 158, 161, 181, 184, 185, 186, 194, 199]
    #stars_sample = [101, 104, 105, 112, 117, 118, 133, 135, 136, 140, 145, 151, 152, 157, 159, 161, 174, 178, 184, 200]
    #stars_sample = [109, 114, 128, 135, 136, 145, 149, 153, 160, 166, 171, 174, 176, 177, 193, 194, 195, 198, 199, 200]
    #stars_sample = [101, 102, 104, 107, 117, 128, 130, 131, 132, 135, 136, 137, 141, 154, 167, 184, 185, 186, 187, 193]#*
    # ONLY detector 492
    ##stars_sample = [8, 11, 19, 24, 30, 37, 39, 41, 48, 51, 55, 65, 73, 85, 87, 88, 90, 91, 93, 98]
    #stars_sample = [2, 4, 8, 10, 11, 22, 25, 28, 33, 37, 54, 64, 68, 76, 80, 89, 96, 97, 99, 100]
    # all stars of one detector or both
    #stars_sample = [s+1 for s in range(200)]
    # Known bad stars in X and Y: 103, 105, 106, 112, 134, 152, 156, 170, 188
    #6, 23, 50, 55, 65, 67, 70, 71, 73, 90, 105, 108, 119, 124, 126, 127, 137, 153, 186, 187



    ######################################################

    ### CODE

    continue_code = True
    if not perform_abs_threshold and min_elements!=4:
        print ('***** You are running the code with  min_elements =', min_elements, ' and No absolute threshold, ')
        continue_code = raw_input('  Do you wish to continue?  y  [n]')
        if continue_code == 'y':
            raw_input('Ok, continuing... but the output files will not have a marker to know the number of minimum '
                      'elements allowed in the absolute threshold routine.  Press enter')
        else:
            exit()

    # start the timer to compute the whole running time
    start_time = time.time()

    # make sure that bad stars are gone if ugly stars are to be gone as well
    if not keep_ugly_stars:
        keep_bad_stars = False

    # Set variable as it appears defined in function
    if perform_abs_threshold:
        just_least_sqares = False  # Only perform least squares routine = True, perform abs_threshold routine = False
    else:
        just_least_sqares = True

    # set paths
    gen_path = os.path.abspath('../resultsXrandomstars')
    path4results =  "../resultsXrandomstars/"

    # Compact variables
    primary_params1 = [do_plots, save_plots, show_plots, detector, output_full_detector, show_onscreen_results,
                       show_pixpos_and_v23_plots, save_text_file]
    primary_params2 = [save_centroid_disp, keep_bad_stars, keep_ugly_stars, just_least_sqares, stars_in_sample,
                       scene, background_method, background2use]
    primary_params3 = [shutters, noise, filter_input, test2perform, Nsigma, abs_threshold, abs_threshold, min_elements,
                       max_iters_Nsig]
    primary_params = [primary_params1, primary_params2, primary_params3]
    secondary_params1 = [checkbox_size, xwidth_list, ywidth_list, vlim, threshold, max_iter, verbose]
    secondary_params2 = [debug, arcsecs, determine_moments, display_master_img, show_centroids, show_disp]
    secondary_params3 = [Pier_corr, tilt, backgnd_subtraction_method, random_sample]
    secondary_params = [secondary_params1, secondary_params2, secondary_params3]


    # Run main function of script
    extra_string = None
    results_all_tests = run_testXrandom_stars(stars_sample, primary_params, secondary_params, path4results,
                                              gen_path, extra_string)



    '''
        common3files = '_results_'+case+'.txt'
        test_fileT1 = os.path.join(gen_path, 'T1'+common3files)
        test_fileT2 = os.path.join(gen_path, 'T2'+common3files)
        test_fileT3 = os.path.join(gen_path, 'T3'+common3files)
        txt_files = [test_fileT1, test_fileT2, test_fileT3]
        T1V2_3, T1V3_3, T1V2_5, T1V3_5, T1V2_7, T1V3_7, T1TrueV2, T1TrueV3 = np.loadtxt(test_fileT1, comments='#',
                                                                    usecols=(2,3,4,5,6,7,8,9), unpack=True)
        T2V2_3, T2V3_3, T2V2_5, T2V3_5, T2V2_7, T2V3_7, T2TrueV2, T2TrueV3 = np.loadtxt(test_fileT2, comments='#',
                                                                    usecols=(2,3,4,5,6,7,8,9), unpack=True)
        T3V2_3, T3V3_3, T3V2_5, T3V3_5, T3V2_7, T3V3_7, T3TrueV2, T3TrueV3 = np.loadtxt(test_fileT3, comments='#',
                                                                    usecols=(2,3,4,5,6,7,8,9), unpack=True)
        # for test3 we only compare to position 1 because this is how the cutouts were made in order to see the shift

        ls_dataTESTS = []
        for i, Tfile in enumerate(txt_files):
            ls_data = prs.load_rejected_stars(Tfile)
            ls_dataTESTS.append(ls_data)   # ls_dataTESTS = list of 3 dictionaries, one per file
                                           #  (each dictionay contains 3 dictionaries, one per centroid window,
                                           #  the keyes per centroid window are: detla_theta, delta_x, delta_y,
                                           #  elements_left, iteration, sigma_theta, sigma_x, and sigma_y. For the
                                           #  dictionary of one of the text files, to access centroid 5, iterations
                                           #  type: ls_data['5']['iterations']


        # do the plots -> 2 plots per centroid window
        for cwin in xwidth_list:
            cwincase = case+'_CentroidWindow'+repr(cwin)

            # Plot to compare the mean values for the 3 tests -- plot only has 3 points
            plot_title = r'Residual Mean Values, $\mu$'
            xlabel = r'$\Delta$V2 [marcsec]'
            ylabel = r'$\Delta$V3 [marcsec]'
            destination = os.path.join(gen_path, 'plots/means_Cwin'+repr(cwin)+'.jpg')
            T1sigmaV2 = ls_dataTESTS[0][str(cwin)]['sigma_x']   # Test 1 sigma V2 value
            T2sigmaV2 = ls_dataTESTS[1][str(cwin)]['sigma_x']   # Test 2
            T3sigmaV2 = ls_dataTESTS[2][str(cwin)]['sigma_x']   # Test 3
            T1sigmaV3 = ls_dataTESTS[0][str(cwin)]['sigma_y']   # Test 1 sigma V3 value
            T2sigmaV3 = ls_dataTESTS[1][str(cwin)]['sigma_y']   # Test 2
            T3sigmaV3 = ls_dataTESTS[2][str(cwin)]['sigma_y']   # Test 3
            T1meanV2 = ls_dataTESTS[0][str(cwin)]['delta_x']   # Test 1 mean V2 value
            T2meanV2 = ls_dataTESTS[1][str(cwin)]['delta_x']   # Test 2
            T3meanV2 = ls_dataTESTS[2][str(cwin)]['delta_x']   # Test 3
            T1meanV3 = ls_dataTESTS[0][str(cwin)]['delta_y']   # Test 1 mean V3 value
            T2meanV3 = ls_dataTESTS[1][str(cwin)]['delta_y']   # Test 2
            T3meanV3 = ls_dataTESTS[2][str(cwin)]['delta_y']   # Test 3
            arrx = [T1meanV2*1000.0, T2meanV2*1000.0, T3meanV2*1000.0]
            arry = [T1meanV3*1000.0, T2meanV3*1000.0, T3meanV3*1000.0]
            labels_list = ['Avg in Pixel Space', 'Avg in Sky', 'No Avg']
            print_side_string = ['V2$\mu$ [marcsec]', 'V3$\mu$ [marcsec]']
            print_side_values = [T1sigmaV2*1000.0, T1sigmaV3*1000.0,
                                 T2sigmaV2*1000.0, T2sigmaV3*1000.0,
                                 T3sigmaV2*1000.0, T3sigmaV3*1000.0,
                                 T1meanV2*1000.0, T1meanV3*1000.0,
                                 T2meanV2*1000.0, T2meanV3*1000.0,
                                 T3meanV2*1000.0, T3meanV3*1000.0]
            xlims = [-5.0, 5.0]
            ylims = [-5.0, 5.0]
            vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title,
                      labels_list=labels_list, xlims=xlims, ylims=ylims,
                      print_side_string=print_side_string, print_side_values=print_side_values,
                      save_plot=save_plots, show_plot=show_plots, destination=destination)

            # Graphical display of the standard deviation
            plot_title = r'Graphical Display of the Standard Deviation, $\sigma$'
            destination = os.path.join(gen_path, 'plots/V2V3_Cwin'+repr(cwin)+'.jpg')
            if cwin == 3:
                T1V2, T2V2, T3V2 = T1V2_3-T1TrueV2, T2V2_3-T2TrueV2, T3V2_3-T3TrueV2
                T1V3, T2V3, T3V3 = T1V3_3-T1TrueV3, T2V3_3-T2TrueV3, T3V3_3-T3TrueV3
            elif cwin == 5:
                T1V2, T2V2, T3V2 = T1V2_5-T1TrueV2, T2V2_5-T2TrueV2, T3V2_5-T3TrueV2
                T1V3, T2V3, T3V3 = T1V3_5-T1TrueV3, T2V3_5-T2TrueV3, T3V3_5-T3TrueV3
            elif cwin == 7:
                T1V2, T2V2, T3V2 = T1V2_7-T1TrueV2, T2V2_7-T2TrueV2, T3V2_7-T3TrueV2
                T1V3, T2V3, T3V3 = T1V3_7-T1TrueV3, T2V3_7-T2TrueV3, T3V3_7-T3TrueV3
            arrx = [T1V2, T2V2, T3V2]
            arry = [T1V3, T2V3, T3V3]
            xlims = [-20., 20.]
            ylims = [-20., 20.]
            vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title,
                      labels_list=labels_list, xlims=xlims, ylims=ylims,
                      print_side_string=print_side_string, print_side_values=print_side_values,
                      save_plot=save_plots, show_plot=show_plots, destination=destination,
                      star_sample=stars_sample)
        '''

    print ("\n Script 'testXrandom_stars.py' finished! Took  %s  seconds to finish. \n" % (time.time() - start_time))
