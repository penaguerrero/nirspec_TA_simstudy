from __future__ import print_function, division
from glob import glob
import numpy as np
import os
import time
import string

# other code
import testXrandom_stars as tx

print("Modules correctly imported! \n")




# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Sept 2016 - Version 1.0: initial version completed


"""
DESCRIPTION:
    The script will choose X random stars from either detector and run the selected transformations
    for all 3 tests for the specific case given for the set of X stars, and repeat it Nktimes in order to produce
    3 plots.


NOTES:
     TEST1 - Average positions P1 and P2, transform to V2-V3 space, and compare to average
             reference positions (V2-V3 space)
     TEST2 - Transform individual positions P1 and P2 to V2-V3 space, average V2-V3 space
             positions, and compare to average reference positions.
     TEST3 - Transform P1 and P2 individually to V2-V3 space and compare star by star and
             position by position.
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

"""


#######################################################################################################################


### FUNCTIONS

def get_pixpositions(scenario, stars_sample):
    '''
    This function reads pixel positions from text files in Xrandomstars/centroid_txt_files/*All.txt

    Args:
        scenario: string, e.g. Scene1_rapid_real_bgFrac0.3_thres3
        stars_sample: list, stars to be studied

    Returns:
        P1P2data = list of pixel positions for centroid windows 3, 5, and 7 for both positions:
                        P1P2data = [x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27]
    '''
    # get pixel space data
    input_files_list = glob('../resultsXrandomstars/centroid_txt_files/*'+scenario+'*_ALL.txt')
    print ('\n Reading samples from:', input_files_list)
    data_P1 = np.loadtxt(input_files_list[0], skiprows=5, usecols=(0,1,2,3,4,5,6,7), unpack=True)
    stars, bg_value, x13, y13, x15, y15, x17, y17 = data_P1
    data_P2 = np.loadtxt(input_files_list[1], skiprows=5, usecols=(0,1,2,3,4,5,6,7), unpack=True)
    _, _, x23, y23, x25, y25, x27, y27 = data_P2

    # slice the arrays to only get the star sample data
    sampleX13, sampleY13 = np.array([]), np.array([])
    sampleX23, sampleY23 = np.array([]), np.array([])
    sampleX15, sampleY15 = np.array([]), np.array([])
    sampleX25, sampleY25 = np.array([]), np.array([])
    sampleX17, sampleY17 = np.array([]), np.array([])
    sampleX27, sampleY27 = np.array([]), np.array([])
    for i, st in enumerate(stars):
        if st in stars_sample:
            sampleX13 = np.append(sampleX13, x13[i])
            sampleY13 = np.append(sampleY13, y13[i])
            sampleX15 = np.append(sampleX15, x15[i])
            sampleY15 = np.append(sampleY15, y15[i])
            sampleX17 = np.append(sampleX17, x17[i])
            sampleY17 = np.append(sampleY17, y17[i])
            sampleX23 = np.append(sampleX23, x23[i])
            sampleY23 = np.append(sampleY23, y23[i])
            sampleX25 = np.append(sampleX25, x25[i])
            sampleY25 = np.append(sampleY25, y25[i])
            sampleX27 = np.append(sampleX27, x27[i])
            sampleY27 = np.append(sampleY27, y27[i])
    P1P2data = [sampleX13, sampleY13, sampleX23, sampleY23,
                sampleX15, sampleY15, sampleX25, sampleY25,
                sampleX17, sampleY17, sampleX27, sampleY27]
    return P1P2data


def save_sample_txt(counter, samples_txtfile, stars_sample):
    '''
    This function appends the star sample into the text file already created.
    Args:
        samples_txtfile: string, name of text file to append to
        stars_sample: list, star sample to append to file

    Returns:
        Nothing
    '''
    # convert the python list into a re-readable list (i.e. without [])
    str_stars_sample = ", ".join(str(st) for st in stars_sample)
    # save the sample into the file
    line2write = '{:<20} {:<20}'.format(counter+1, str_stars_sample)
    with open(samples_txtfile, "a") as txt:
        txt.write(line2write+'\n')


def read_sample_txt(samples_txtfile):
    '''
    This function reads thr samples_txtfile to obtain the samples to repeat the analysis for
    Args:
        samples_txtfile = string, name of text file to read from
    Returns:
        samples = list of lists (the star samples)
    '''
    samples = []
    f = open(samples_txtfile, 'r')
    for line in f.readlines():
        if '#' not in line:
            line_list = line.split()
            star_sample = []
            for i, st in enumerate(line_list):
                if i != 0:
                    st = string.replace(st, ',', '')
                    star_sample.append(int(st))
            samples.append(star_sample)
    f.close()
    return samples


def get_sample_numberN(counter, samples):
    '''
    This function returns the right number of sample from the list of samples.
    Args:
        counter: integer, the counter of the loop
        samples: list of lists (the star samples)

    Returns:
        stars_sample = list of stars to be analyzed
    '''
    for n, stars_sample in enumerate(samples):
        if n == counter:
            return stars_sample


def starttxtTESTfile(case, txtpaths, Nsigma, centroid_window, arcsecs):
    '''
    This function creates the test file to store the standard deviations and means from the 3 Tests
    Args:
        case: string, string, for example 'Scene2_rapid_real_bgFrac'
        txtpaths: list, base name of files are to be created
        Nsigma: sigmas to reject
        centroid_window: integer
        arcsecs: True or False, if False units are Degrees

    Returns:
        3 text files with headers and textfiles=list of the names of the files created
    '''
    txt1, txt2, txt3 = txtpaths
    txt1 += '_'+case+'_Nsigma'+repr(Nsigma)+'_cwin'+str(centroid_window)+'.txt'
    txt2 += '_'+case+'_Nsigma'+repr(Nsigma)+'_cwin'+str(centroid_window)+'.txt'
    txt3 += '_'+case+'_Nsigma'+repr(Nsigma)+'_cwin'+str(centroid_window)+'.txt'
    units = '# Units are: DEGREES'
    if arcsecs:
        units = '# Units are: ARCSEC'
    file_hdr0 = '{:<26} {:<65} {:<32} {:<17} {:<6}'.format('# Sample', 'Standard deviation', 'Mean values',
                                                         'Last iteration', 'Rejected stars')
    file_hdr1 = '{:<8} {:>12} {:>14} {:>19} {:>21} {:>19} {:>21}'.format('#', 'V2', 'V3', 'Theta',
                                                                         'V2', 'V3', 'Theta')
    f1 = open(txt1, 'w')
    f1.write(units+'\n')
    f1.write(file_hdr0+'\n')
    f1.write(file_hdr1+'\n')
    f1.close()
    f2 = open(txt2, 'w')
    f2.write(units+'\n')
    f2.write(file_hdr0+'\n')
    f2.write(file_hdr1+'\n')
    f2.close()
    f3 = open(txt3, 'w')
    f3.write(units+'\n')
    f3.write(file_hdr0+'\n')
    f3.write(file_hdr1+'\n')
    f3.close()
    textfiles = [txt1, txt2, txt3]
    return textfiles


def printTESTfile(test2perform, txtfiles, sigmas, means, iteration, rejected_stars):
    '''
    This function prints the standard deviations (sigmas), means, number of iterations ran, and
     stars rejected into the given text files.
    Args:
        test2perform: string, either 'T1', 'T2', or 'T3'
        txtfiles: list, names of the files to append to
        sigmas: list, standard deviations
        means: list, mean values
        iteration: integer
        rejected_stars: list, index of rejected stars

    Returns:
        Populated text files.
    '''
    line4file = '{:<5} {:>20}  {:<20}  {:<16}  {:<20} {:<20}  {:<20} {:<4} {:>10}'.format(counter+1,
                                                            sigmas[0], sigmas[1], sigmas[2],
                                                            means[0], means[1], means[2],
                                                            iteration, len(rejected_stars))
    # save the sample into the file
    [txt1, txt2, txt3] = txtfiles
    if test2perform == 'T1':
        txtfile = txt1
    elif test2perform == 'T2':
        txtfile = txt2
    elif test2perform == 'T3':
        txtfile = txt3
    with open(txtfile, "a") as txt:
        txt.write(line4file+'\n')


#######################################################################################################################


if __name__ == '__main__':


    # INITIAL CONDITIONS

    Nktimes = 5000                     # Integer number to repeat the entire analysis
    random_sample = False              # choose a random sample of stars from either detector: True or False
    save_Nktimes_text_files = False    # Want to save the text file with star samples used for later comparison? True or False
    show_onscreen_results = False      # Want to show on-screen resulting V2s, V3s and statistics? True or False
    output_full_detector = True        # Give resulting coordinates in terms of full detector: True or False
    keep_bad_stars = False             # Keep the bad stars in the sample? True or False
    keep_ugly_stars = True             # Keep the ugly stars (one position measured wrong)? True or False
    perform_abs_threshold = False       # Perform abs_threshold routine (True) or only perform least squares routine (False)
    detector = 'both'                  # Detector to analyze: 491, 491 or 'both'
    stars_in_sample = 20               # Number of stars in sample
    scene = 1                          # Integer or string, scene=1 is constant Mag 23, scene=2 is stars with Mag 18-23
    background_method = 'frac'         # Select either 'fractional', 'fixed', or None
    background2use = 0.3               # Background to use for analysis: None or float
    shutters = "rapid"                 # Shutter velocity, string: "rapid" or "slow"
    noise = "real"                     # Noise level, string: "nonoise" or "real"
    filter_input = "F140X"             # Filter, string: for now only test case is "F140X"
    Nsigma = 2.5                       # N-sigma rejection of bad stars: integer or float
    abs_threshold = 0.32               # threshold to reject points after each iteration of least squares routine, default=0.32
    min_elements = 4                   # minimum number of elements in the absolute threshold least squares routine, default=4
    max_iters_Nsig = 10                # Max number of iterations for N-sigma function: integer


    # SECONDARY PARAMETERS THAT CAN BE ADJUSTED

    checkbox_size = 3                  # Real checkbox size
    xwidth_list = [3, 5, 7]            # Number of rows of the centroid region
    ywidth_list = [3, 5, 7]            # Number of columns of the centroid region
    vlim = (1, 100)                    # Sensitivity limits of image, i.e. (0.001, 0.1)
    threshold = 0.01                   # Convergence threshold of accepted difference between checkbox centroid and coarse location
    max_iter = 10                      # Maximum number of iterations for finding coarse location
    verbose = False                    # Show some debug messages (i.e. resulting calculations)
    debug = False                      # See all debug messages (i.e. values of all calculations)
    arcsecs = True                     # Print the differences in arcsecs? True or False (=degrees)
    determine_moments = False          # Want to determine 2nd and 3rd moments?
    Pier_corr = True                   # Include Pier's corrections to measured positions
    tilt = False                       # Tilt angle: True or False
    backgnd_subtraction_method = 1     # 1    = Do background subtraction on final image (after subtracting 3-2 and 2-1),
    #                                           before converting negative values into zeros
    #                                    2    = Do background subtraction on 3-2 and 2-1 individually
    #                                    None = Do not subtract background


    ####################################################################################################################

    # by default set these variables to false in order to avoid having Nk text files and/or displays
    do_plots = False                   # 1. Least squares plot in V2/V3 space showing the true position (0,0)
    #                                       and the mean of the three calculation cases:  Averaging in pixel space,
    #                                       averaging on sky, and no averaging : True or False
    #                                    2. Same plot but instead of the mean show all stars in one 20star calculation
    save_plots = False                 # Save the plots? True or False
    show_plots = False                 # Show the plots? True or False
    show_pixpos_and_v23_plots = False  # Show the plots of x-y and v2-v3 residual positions?
    save_text_file = False             # Want to save the text file of comparison? True or False
    save_centroid_disp = False         # Save the display with measured and true positions?
    display_master_img = False         # Want to see the combined ramped images for every star?
    show_centroids = False             # Print measured centroid on screen: True or False
    show_disp = False                  # Show display of resulting positions? (will show 2 figs, same but different contrast)


    ######################################################

    ### CODE

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

    # tests to perform
    list_of_tests2perform = ['T1', 'T2', 'T3']

    # Pool of stars to select from
    stars_detectors = range(1, 201)       # default is for both detectors
    if detector == 491:
        stars_detectors = range(101, 201) # only detector 491
    elif detector == 492:
        stars_detectors = range(1, 101)   # only detector 492

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

    # set case to study
    case = repr(detector)+"Scene"+str(scene)+"_"+shutters+"_"+noise+bg_choice
    if detector == 'both':
        case = "2DetsScene"+str(scene)+"_"+shutters+"_"+noise+bg_choice
    if background_method is not None:
        case += repr(background2use)

    # start the text file to save the samples
    star_sample_dir = 'good_and_uglies/'+repr(stars_in_sample)+'_star_sample'
    if not keep_bad_stars and not keep_ugly_stars:
        star_sample_dir = 'only_good_stars/'+repr(stars_in_sample)+'_star_sample'
    if perform_abs_threshold and min_elements==4:
        star_sample_dir += '/abs_threshold'
    elif perform_abs_threshold and min_elements!=4:
        star_sample_dir += '/diff_min_elements_abs_threshold'
    elif not perform_abs_threshold and min_elements!=4:
        print ('***** You are running the code with  min_elements =', min_elements, ' and No absolute threshold... aborting.')
        exit()
    path_results_text_files = os.path.abspath('../resultsXrandomstars/'+star_sample_dir)
    detector_str = repr(detector)
    if not isinstance(detector, int):
        detector_str = '2Dets'
    samples_txtfile = os.path.join(path_results_text_files, detector_str+'samples_used.txt')
    if save_Nktimes_text_files:
        if random_sample:   # only create a new file when random is set to True
            fhdr = '{:<20} {:<10}'.format('# Star Sample', 'Stars in sample')
            f = open(samples_txtfile, 'w')
            f.write(fhdr+'\n')
            f.close()

    # start the text files to save the Test results
    bg = bg_choice+repr(background2use)
    str_thres = string.replace(repr(threshold), '0.', '')
    thres = 'thres'+str_thres
    scenario = "Scene"+str(scene)+"_"+shutters+"_"+noise+bg+'_'+thres
    case = case+'_'+thres
    if min_elements != 4:
        case += '_minele'+repr(min_elements)
    #print ('case = ', case)
    #print ('scenario =', scenario)
    #raw_input()
    if save_Nktimes_text_files:
        txt1 = os.path.join(path_results_text_files, 'TEST1results')
        txt2 = os.path.join(path_results_text_files, 'TEST2results')
        txt3 = os.path.join(path_results_text_files, 'TEST3results')
        txtpaths = [txt1, txt2, txt3]
        # centroid_windows are the same as xwidth_list
        textfiles357 = []
        for centroid_window in xwidth_list:
            textfiles = starttxtTESTfile(case, txtpaths, Nsigma, centroid_window, arcsecs)
            textfiles357.append(textfiles)

    # counter for the loop
    counter = 0

    # Repeat study with the same samples if random is set to False
    if not random_sample:
        samples = read_sample_txt(samples_txtfile)

    # loop over the number of random samples to use
    for n in range(Nktimes):
        # Select the sample of stars to study
        if random_sample:
            verbose = True
            stars_sample = tx.select_random_stars(scene, stars_in_sample, stars_detectors, keep_bad_stars,
                                                  keep_ugly_stars, verbose)
        else:
            stars_sample = get_sample_numberN(counter, samples)
        # Tell me how many sets you have ran
        print ('\n Set  # {}  of a total of  {}.\n'.format(counter+1, Nktimes))
        print ('stars_sample =', stars_sample)

        # Get benchmark (true) information
        bench_stars, benchP1P2, LoLeftCornersP1P2, benchmark_V2V3_sampleP1P2, magnitudes = tx.get_benchV2V3(scene,
                                                                                                            stars_sample,
                                                                                                            arcsecs)
        bench_star, _ = bench_stars   # list of benchmark stars used (is the same for both positions)
        bench_xP1, bench_yP1, _, _ = benchP1P2
        bench_xLP1, bench_yLP1, _, _ = LoLeftCornersP1P2

        # save the star sample into text file
        if save_Nktimes_text_files:
            if random_sample:   # only save into file when random is set to True
                save_sample_txt(counter, samples_txtfile, stars_sample)

        # loop over the tests to be performed
        for test2perform in list_of_tests2perform:
            # get the pixel positions
            P1P2data = get_pixpositions(scenario, stars_sample)

            # Create group of primary and secondary parameters
            primary_params1 = [do_plots, save_plots, show_plots, detector, output_full_detector, show_onscreen_results,
                               show_pixpos_and_v23_plots, save_text_file]
            primary_params2 = [save_centroid_disp, keep_bad_stars, keep_ugly_stars, just_least_sqares, stars_in_sample,
                               scene, background_method, background2use]
            primary_params3 = [shutters, noise, filter_input, test2perform, Nsigma, abs_threshold, abs_threshold,
                               min_elements, max_iters_Nsig]
            primary_params = [primary_params1, primary_params2, primary_params3]

            secondary_params1 = [checkbox_size, xwidth_list, ywidth_list, vlim, threshold, max_iter, verbose]
            secondary_params2 = [debug, arcsecs, determine_moments, display_master_img, show_centroids, show_disp]
            secondary_params3 = [Pier_corr, tilt, backgnd_subtraction_method, random_sample]
            secondary_params = [secondary_params1, secondary_params2, secondary_params3]
            # Run transformations for specific test
            print ('\n Transforming into V2 and V3, and running TEST...')
            path4results = ''   # we are not saving individual Test results so path does not matter
            case, new_stars_sample, Tbench_Vs, T_Vs, T_diffVs, LS_res, LS_info = tx.transformAndRunTest(stars_sample,
                                                                                path4results, primary_params,
                                                                                secondary_params, bg_choice, P1P2data,
                                                                                bench_star, benchmark_V2V3_sampleP1P2,
                                                                                plot_v2v3pos=False, extra_string=None)
            # unfold variables
            T3LSsigmas_3, T3LSsigmas_5, T3LSsigmas_7, T3LSdeltas_3, T3LSdeltas_5, T3LSdeltas_7 = LS_res
            iterations, rejected_elementsLS = LS_info
            nit3, nit5, nit7 = iterations
            rejel3, rejel5, rejel7 = rejected_elementsLS
            # Print results into the corresponding text file
            # Save text file for each Test and for each centroid window
            if save_Nktimes_text_files:
                printTESTfile(test2perform, textfiles357[0], T3LSsigmas_3, T3LSdeltas_3, nit3, rejel3)   # centroid window 3
                printTESTfile(test2perform, textfiles357[1], T3LSsigmas_5, T3LSdeltas_5, nit5, rejel5)   # centroid window 5
                printTESTfile(test2perform, textfiles357[2], T3LSsigmas_7, T3LSdeltas_7, nit7, rejel7)   # centroid window 7

                # Tell me how many sets you have ran
        print ('\n')
        counter += 1


    print ("\n Script 'XrandStarsNktimes_v2.py' finished! Took  %s  minutes to finish. \n" % ((time.time() - start_time)/60.))
    print ("    Main parameters used: ")
    print ("        stars_in_sample = ", stars_in_sample)
    print ("        perform_abs_threshold = ", perform_abs_threshold)
    print ("        Nsigma = ", Nsigma)
    print ("        detector = ", detector)
    print ("        threshold = ", threshold)

