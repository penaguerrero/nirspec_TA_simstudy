from __future__ import print_function, division
from astropy.io import fits
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

# modules used
import testing_functions as tf
import TA_functions as taf

print()



# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Feb 2016 - Version 1.0: initial version completed


"""

DESCRIPTION:
    This script runs 4 'zeroth' test cases for the Target Locate code jwst_targloc.py 
    written by T. Le Blanc, and modified by M. A. Pena-Guerrero. These fits files are
    a perfect MSA (all shutters open without hot pixels) over 200 stars randomly distributed, 
    100 over detector 491 (extension 1) and 100 over detector 492 (extension 2).
    Test cases are:
        1) All stars with magnitude 23.0
        2) Magnitude range from 18.0 to 23.0
    Tasks performed are:
        - Performing centroiding for widths of 3, 5, and 7 pixels
        - Fractional backgrounds from 0.0 to 1.0 in increments of 0.1, and a fixed 
        or None background=0 case
        - Moments are included in the code though not currently being used
        
"""

# Paths
perfect_s1 = "Scene_1_AB23/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB23_004/postage_redo"
perfect_s1_shift = "Scene_1_AB23/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB23-shifted_002/postage_redo"
perfect_s2 = "Scene_2_AB1823/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB18to23_002/postage_redo"
perfect_s2_shift = "Scene_2_AB1823/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB18to23-shifted_002/postage_redo"
flight_s1 = "Scene_1_AB23/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB23_005/postage_redo"
flight_s1_shift = "Scene_1_AB23/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB23-shifted_003/postage_redo"
flight_s2 = "Scene_2_AB1823/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB18to23_003/postage_redo"
flight_s2_shift = "Scene_2_AB1823/SKY-F140X-MIRROR_MOS_simuTA20150528-F140X-S50-K-AB18to23-shifted_003/postage_redo"

#                   0            1            2            3            
paths_list = [perfect_s1, perfect_s1_shift, perfect_s2, perfect_s2_shift, 
              flight_s1, flight_s1_shift, flight_s2, flight_s2_shift]
#                   4            5            6            7            

###########################################################################################################

# Set test parameters
path_number = 0                    # Select 0 through 7 from paths_list above
detector = 491                     # Which detector are we working with: 491 or 492
vlim = (0.001,10)                  # sensitivity limits of image, i.e. (0.001, 0.1) 
checkbox_size = 3                  # Real checkbox size
xwidth_list = [3, 5, 7]            # Number of rows of the centroid region
ywidth_list = [3, 5, 7]            # Number of columns of the centroid region
max_iter = 50
threshold = 1e-5
background_method = 'frac'         # Select either 'fractional', 'fixed', or None
backgnd_subtraction_method = 1     # 1    = Do background subtraction on final image (after subtracting 3-2 and 2-1), 
#                                           before converting negative values into zeros
#                                    2    = Do background subtraction on 3-2 and 2-1 individually
#                                    None = Do not subtract background
debug = False                      # see all debug messages (i.e. values of all calculations)
determine_moments = False          # Want to determine 2nd and 3rd moments?
display_master_img = False         # Want to see the combined ramped images for every star?
just_read_text_file = False        # skip the for loop to the plotting part
centroid_in_full_detector = False  # Give resulting coordinates in terms of full detector: True or False
show_disp = False                  # Show display of resulting positions: True or False
save_centroid_disp = False         # To modify go to lines 306 and 307
show_plot = True                   # Show plot(s) of x_offset vs y_offset and y_offset vs magnitude: True or False
zoom_plot = True                   # Perform zoom-in from -1 to +1 of the residuals plot: True or False
plot_type = '.jpg'                 # Type of image to be saved: pdf, jpg, eps (it is better to convert from jpg to eps)
save_plot = False                  # legend can be moved in line xxx
save_text_file = False             # Want to save the text file of comparison? True or False
perform_avgcorr = True             # Correct for average values given by Pierre? True or False
single_star = False                # If only want to test one star set to True and type the path for a single star
# NOTE: for the names of stars, single numbers before the .fits require 8 spaces after quad_star, 
#       while 2 numbers require 7 spaces.  
star_file_name = '/postageout_star_     134 quad_       3 quad_star       34.fits'


###########################################################################################################


main_path_infiles = "../PFforMaria/"
dir2test = main_path_infiles+paths_list[path_number]
print(dir2test)
if single_star:
    single_star_path = dir2test+star_file_name   
    save_text_file = False
    show_disp = True

# Background cases
bg_frac, bg_value = None, None   # for the None case
bg_choice = "_bgNone"
background = 0.0
if background_method is not None and "frac" in background_method:
    fractional_background_list = [x*0.1 for x in range(11)]
    bg_choice = "_bgFrac"
elif background_method is not None and "fix" in background_method:
    bg_value = 0.0
    bg_choice = "_bgFixed"
    background = bg_value

# start the timer to compute the whole running time
start_time = time.time()

# Get true positions from Pierre's position files
if "Scene_1" in dir2test:
    path2listfile = "../PFforMaria/Scene_1_AB23"
    list_file = "simuTA20150528-F140X-S50-K-AB23.list"
    positions_file = "simuTA20150528-F140X-S50-K-AB23_positions.fits" 
    if 'shifted' in dir2test: 
        list_file = "simuTA20150528-F140X-S50-K-AB23-shifted.list"
        positions_file = "simuTA20150528-F140X-S50-K-AB23-shifted_positions.fits"
        
if "Scene_2" in dir2test:
    # Read the text file just written to get the offsets from the "real" positions of the fake stars
    path2listfile = "../PFforMaria/Scene_2_AB1823"
    list_file = "simuTA20150528-F140X-S50-K-AB18to23.list"
    positions_file = "simuTA20150528-F140X-S50-K-AB18to23_positions.fits"
    if 'shifted' in dir2test: 
        list_file = "simuTA20150528-F140X-S50-K-AB18to23-shifted.list"
        positions_file = "simuTA20150528-F140X-S50-K-AB18to23-shifted_positions.fits"
lf = os.path.join(path2listfile,list_file)
pf = os.path.join(path2listfile,positions_file)
bench_star, xpos_arcsec, ypos_arcsec, factor, mag, bg_method = taf.read_listfile(lf, detector, background_method)
_, true_x, true_y, trueV2, trueV3 = taf.read_positionsfile(pf, detector)

"""
*** WE ARE NOT USING THIS PART RIGHT NOW BECAUSE THE star_parameters.txt FILES HAVE THE SAME DATA FOR
BOTH DETECTORS.

# Read fits table with benchmark data from the star parameters file to compare results
#     xL:  x-coordinate of the left edge of the postge stamp in the full image (range 0-2047)
#     xR: x-coord of right edge of the postage stamp
#     yL: y-coord of the lower edge of the postage stamp
#     yU:  y-coord of the upper edge of the postage stamp
star_param_txt = os.path.join(dir2test,"star parameters.txt")
if detector ==492:
    star_param_txt = os.path.join(dir2test,"star parameters_492.txt")
benchmark_data = np.loadtxt(star_param_txt, skiprows=3, unpack=True)
bench_star, quadrant, star_in_quad, x_491, y_491, x_492, y_492, V2, V3, xL, xR, yL, yU = benchmark_data
# Select appropriate set of stars to test according to chosen detector
if detector == 491:
    true_x = x_491
    true_y = y_491
elif detector == 492:
    true_x = x_492
    true_y = y_492
"""

# Set up the output file and path for plots and figures
main_path_outfiles = "../PFforMaria/electron_rate_maps/detector_"+str(detector)
output_file_path = main_path_outfiles+"/TAposition_text_files/"
paths_list_str = ["perfect_s1", "perfect_s1_shift", "perfect_s2", "perfect_s2_shift", 
                  "flight_s1", "flight_s1_shift", "flight_s2", "flight_s2_shift"]
case = paths_list_str[path_number]
output_file = os.path.join(output_file_path, case+bg_choice+".txt")
line0 = "Centroid indexing starting at 1 !"
line0a = "{:<5} {:<15} {:<16} {:>23} {:>32} {:>33} {:>26} {:>15} {:>35} {:>38} {:>43}".format("Star", "Background", 
                                                                  "Centroid width: 3", "5", "7", 
                                                                  "TruePositions", "LoLeftCoords",
                                                                  "Factor",
                                                                  "Difference with: centroid window 3", "centroid window 5", "centroid window 7")
line0b = "{:>25} {:>12} {:>16} {:>14} {:>16} {:>14} {:>16} {:>14} {:>12} {:>10} {:>26} {:>16} {:>26} {:>16} {:>28} {:>16}".format(
                                                                       "x", "y", "x", "y", "x", "y", 
                                                                       "TrueX", "TrueY", "LoLeftX", "LoLeftY",
                                                                       "x", "y", "x", "y", "x", "y")
lines4screenandfile = [line0, line0a, line0b]
if save_text_file:
    f = open(output_file, "w+")
    f.write(line0+"\n")
    f.write(line0a+"\n")
    f.write(line0b+"\n")
    f.close()
display_fig_name_path = main_path_outfiles+"/centroid_figs"

# offset lists
x3offst, x5offst, x7offst = [], [], []
y3offst, y5offst, y7offst = [], [], []

# run centroid algorithm on each position and save them into a text file
# Start the loop in the given directory
if not just_read_text_file:
    dir_stars = glob(os.path.join(dir2test,"postageout_star_*.fits"))   # get all star fits files in that directory
    for star in dir_stars:
        if single_star:
            star = single_star_path
        dir_star_number = int(os.path.basename(star).split()[1])
        # Test stars of detector of choice
        for st_idx, st in enumerate(bench_star):
            st = int(st)
            if st == dir_star_number: #if str(st)+" quad_       " in star:
                print ("Will test star in directory: \n     ", dir2test)
                print ("Star: ", os.path.basename(star))
                # Make sure the file actually exists
                star_exists = os.path.isfile(star)
                if not star_exists:
                    print ("The file: ", star, "\n    does NOT exist. Exiting the script.")
                    exit() 
                
                # Obtain real star position
                true_center = [true_x[st_idx], true_y[st_idx]]

                # define the magnitude (or factor from the list file)
                factor_i = factor[st_idx]

                # Obtain a fake set of 3 images
                ramp_im1 = fits.getdata(star, 0) * 0.0
                ramp_im2 = fits.getdata(star, 0)
                ramp_im3 = fits.getdata(star, 0) * 100.0
                master_img = np.array([ramp_im1, ramp_im2, ramp_im3])
                print ('Master image shape: ', np.shape(master_img))
                # Do background correction on each of 3 ramp images
                if background_method is not None and "frac" in background_method:
                    # If fractional method is selected, loop over backgrounds from 0.0 to 1.0 in increments of 0.1
                    for bg_frac in fractional_background_list:
                        print ("* Using fractional background value of: ", bg_frac)
                        # Obtain the combined FITS image that combines all frames into one image 
                        # background subtraction is done here
                        psf = taf.readimage(master_img, backgnd_subtraction_method, bg_method=background_method, 
                                            bg_value=bg_value, bg_frac=bg_frac, debug=debug)                
                        master_img_bgcorr_max = psf.max()
                        """
                        while master_img_bgcorr_max == 0.0:
                            print('  IMPORTANT WARNING!!! Combined ramped images have a max of 0.0 with bg_frac=', bg_frac)
                            bg_frac -= 0.02
                            if bg_frac < 0.0:   # prevent an infinite loop
                                print ('   ERROR - Cannot subtract from  bg_frac < 0.0 ...')
                                bg_frac = 0.0
                                break
                            print('       *** Setting  NEW  bg_frac = ', bg_frac)
                            psf = taf.readimage(master_img, backgnd_subtraction_method, bg_method=background_method, 
                                                bg_value=bg_value, bg_frac=bg_frac, debug=debug)                
                            master_img_bgcorr_max = psf.max()
                        """
                        cb_centroid_list_in32x32pix = taf.run_recursive_centroids(psf, background, xwidth_list, ywidth_list,
                                                                       checkbox_size, max_iter, threshold, determine_moments, debug)
                        corr_cb_centroid_list, loleftcoords, true_center32x32, differences_true_TA = taf.centroid2fulldetector(cb_centroid_list_in32x32pix,
                                                                                                    true_center, detector, perform_avgcorr=perform_avgcorr)
                        if not centroid_in_full_detector:
                            cb_centroid_list = cb_centroid_list_in32x32pix
                            true_center = true_center32x32
                        # Record offsets
                        x3offst.append(differences_true_TA[0][0][0])
                        y3offst.append(differences_true_TA[0][0][1])
                        x5offst.append(differences_true_TA[0][1][0])
                        y5offst.append(differences_true_TA[0][1][1])
                        x7offst.append(differences_true_TA[0][2][0])
                        y7offst.append(differences_true_TA[0][2][1])
                        # Write output into text file
                        bg = bg_frac
                        data2write = [save_text_file, output_file, st, bg, corr_cb_centroid_list, true_center, loleftcoords, factor_i, differences_true_TA]
                        tf.write2file(data2write, lines4screenandfile)
                else:
                    # Obtain the combined FITS image that combines all frames into one image AND
                    # check if all image is zeros, take the image that still has a max value
                    psf = taf.readimage(master_img, backgnd_subtraction_method, bg_method=background_method, 
                                                bg_value=bg_value, bg_frac=bg_frac, debug=debug)                   
                    cb_centroid_list_in32x32pix = taf.run_recursive_centroids(psf, background, xwidth_list, ywidth_list,
                                                                   checkbox_size, max_iter, threshold, determine_moments, debug)
                    corr_cb_centroid_list, loleftcoords, true_center32x32, differences_true_TA = taf.centroid2fulldetector(cb_centroid_list_in32x32pix,
                                                                                                true_center, detector, perform_avgcorr=perform_avgcorr)
                    if not centroid_in_full_detector:
                        cb_centroid_list = cb_centroid_list_in32x32pix
                        true_center = true_center32x32
                    # Record offsets
                    x3offst.append(differences_true_TA[0][0][0])
                    y3offst.append(differences_true_TA[0][0][1])
                    x5offst.append(differences_true_TA[0][1][0])
                    y5offst.append(differences_true_TA[0][1][1])
                    x7offst.append(differences_true_TA[0][2][0])
                    y7offst.append(differences_true_TA[0][2][1])

                    # Write output into text file
                    bg = background
                    data2write = [save_text_file, output_file, st, bg, corr_cb_centroid_list, true_center,
                                  loleftcoords, factor_i, differences_true_TA]
                    tf.write2file(data2write, lines4screenandfile)

                if bg_choice == "_bgFrac":
                    path2savefig = display_fig_name_path+"/bg_Fractional/"
                elif bg_choice == "_bgFixed":
                    path2savefig = display_fig_name_path+"/bg_Fixed/"
                elif bg_choice == "_bgNone":
                    path2savefig = display_fig_name_path+"/bg_None/"
                display_fig_name = path2savefig+"star_"+str(st)+"_"+case+bg_choice+plot_type
                taf.display_centroids(detector, st, case, psf, true_center, cb_centroid_list,
                                     show_disp, vlim, savefile=save_centroid_disp, fig_name=display_fig_name,
                                     display_master_img=show_disp)

                if single_star:
                    taf.display_centroids(detector, st, case, psf, true_center, cb_centroid_list_in32x32pix,
                                         show_disp, vlim, savefile=save_centroid_disp, fig_name=display_fig_name,
                                         display_master_img=show_disp)
                    print ("Recursive test script finished. \n")
                    exit()

### Obtain standard deviation from true star positions
# Read the text file just written to get the offsets from the "real" positions of the fake stars
if just_read_text_file:
    offsets = np.loadtxt(output_file, skiprows=3, usecols=(13,14,15,16,17,18), unpack=True)
else:
    offsets_list = [x3offst, y3offst, x5offst, y5offst, x7offst, y7offst]
    offsets = np.array(offsets_list)

# Plots if case is not fractional
if 'frac' not in bg_method:
    sig3, mean3 = taf.find_std(offsets[1])
    sig5, mean5 = taf.find_std(offsets[3])
    sig7, mean7 = taf.find_std(offsets[5])
    sigmas = [sig3, sig5, sig7]
    means = [mean3, mean5, mean7]
    if background_method is None:
        bg = 'None_'
    else:
        bg = 'fix_'
    #destination = os.path.abspath(main_path_outfiles+"/plots/XoffsetVsYoffset_"+bg+case)
    destination = os.path.abspath("../plots4presentationIST/det491_Sene1_"+bg+case+plot_type)
    plot_title = case+'_BG'+bg_method
    taf.plot_offsets(plot_title, offsets, sigmas, means, bench_star, destination,
                     plot_type='.jpg', save_plot=save_plot, show_plot=show_plot, xlims=None, ylims=None)
    # Do zoom-in
    if zoom_plot:
        #destination = os.path.abspath(main_path_outfiles+"/plots/XoffsetVsYoffset_zoomin_"+bg+case)
        destination = os.path.abspath("../plots4presentationIST/det491_Scene1_zoomin_"+bg+case+plot_type)
        minvalue = -0.50
        maxvalue = 0.50
        xlims, ylims = [minvalue, maxvalue], [minvalue, maxvalue]
        taf.plot_zoomin(plot_title, offsets_list, bench_star, destination,
                        plot_type='.jpg', save_plot=save_plot, show_plot=show_plot, xlims=xlims, ylims=ylims)
else:
    frac_data = tf.get_fracdata(offsets)
    sig3, mean3, sig5, mean5, sig7, mean7 = taf.get_frac_stdevs(frac_data)
    sigmas = [sig3, sig5, sig7]
    means = [mean3, mean5, mean7]
    frac_bgs = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5' ,'0.6' ,'0.7', '0.8', '0.9', '1.0']
    print ('\n{:<4} {:<20} {:>10}'.format('FrBG', 'Standard_deviation', 'Mean_y-offset'))
    print ('Centroid window sizes:')
    print ('{:<4} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}'.format('', '3', '5', '7', '3', '5', '7'))
    for fbg, s3, s5, s7, m3, m5, m7 in zip(frac_bgs, sig3, sig5, sig7, mean3, mean5, mean7):
        print('{:<4} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f}'.format(fbg, s3, s5, s7, m3, m5, m7))
    plot_title = case+"_BGfrac"
    #destination = os.path.abspath(main_path_outfiles+"/plots/XoffsetVsYoffset_frac_"+case+plot_type)
    destination = os.path.abspath("../plots4presentationIST/det491_Sene1_"+str(bg)+case+plot_type)
    taf.plot_offsets_frac(plot_title, frac_bgs, frac_data, sigmas, means, bench_star, destination,
                     save_plot=save_plot, show_plot=show_plot, xlims=None, ylims=None)
    # Do zoom-in
    if zoom_plot:
        #destination = os.path.abspath(main_path_outfiles+"/plots/XoffsetVsYoffset_frac__zoomin_"+case+plot_type)
        destination = os.path.abspath("../plots4presentationIST/det491_Scene1_zoomin_"+str(bg)+case+plot_type)
        print (destination)
        print (os.path.isdir("../plots4presentationIST"))
        minvalue = -0.50
        maxvalue = 0.50
        xlims, ylims = [minvalue, maxvalue], [minvalue, maxvalue]
        taf.plot_zoomin_frac(plot_title, frac_bgs, frac_data, bench_star, destination,
                             save_plot=save_plot, show_plot=show_plot, xlims=xlims, ylims=ylims)


# Make the plot of magnitude (in x) versus radial offset distance (in y) for Scene2
if "s2" in case:
    print ('For centroid window=3: ')
    sig3, mean3 = taf.find_std(offsets[1])
    print ('For centroid window=5: ')
    sig5, mean5 = taf.find_std(offsets[3])
    print ('For centroid window=7: ')
    sig7, mean7 = taf.find_std(offsets[5])
    if 'frac' not in bg_method:
        fig3 = plt.figure(1, figsize=(12, 10))
        ax1 = fig3.add_subplot(111)
        plt.title(case+'_BG'+bg_method)
        plt.xlabel('Magnitude')
        plt.ylabel('Radial offset in Y')
        plt.plot(mag, offsets[1], 'bo', ms=8, alpha=0.7, label='centroid window=3')
        plt.plot(mag, offsets[3], 'go', ms=8, alpha=0.7, label='centroid window=5')
        plt.plot(mag, offsets[5], 'ro', ms=8, alpha=0.7, label='centroid window=7')
        #plt.legend(loc='lower left')
        plt.legend(loc='upper right')
        for si,xi,yi in zip(bench_star, mag, offsets[1]): 
            if yi>=1.0 or yi<=-1.0:
                si = int(si)
                subxcoord = 5
                subycoord = 0
                side = 'left'
                plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
        textinfig3 = r'$\sigma3$ = %0.2f    $\mu3$ = %0.2f' % (sig3, mean3)
        textinfig5 = r'$\sigma5$ = %0.2f    $\mu5$ = %0.2f' % (sig5, mean5)
        textinfig7 = r'$\sigma7$ = %0.2f    $\mu7$ = %0.2f' % (sig7, mean7)
        ax1.annotate(textinfig3, xy=(0.75, 0.055), xycoords='axes fraction' )
        ax1.annotate(textinfig5, xy=(0.75, 0.03), xycoords='axes fraction' )
        ax1.annotate(textinfig7, xy=(0.75, 0.005), xycoords='axes fraction' )
        xmin, xmax = ax1.get_xlim()
        plt.hlines(0.0, xmin, xmax, colors='k', linestyles='dashed')
        if save_plot:
            if background_method is None:
                bg = 'None_'
            else:
                bg = 'fix_'
            destination = os.path.abspath(main_path_outfiles+"/plots/MagVsYoffset_"+bg+case+plot_type)
            fig3.savefig(destination)
            print ("\n Plot saved: ", destination)
        if show_plot:
            plt.show()
        else:
            plt.close('all')
    else:
        #print ("Reading text file: ",  output_file)
        frac00, frac01, frac02, frac03, frac04, frac05, frac06, frac07, frac08, frac09, frac10 = tf.get_fracdata(offsets)
        frac_data  = [frac00, frac01, frac02, frac03, frac04, frac05, frac06, frac07, frac08, frac09, frac10]
        sig3, mean3, sig5, mean5, sig7, mean7 = taf.get_frac_stdevs(frac_data)
        frac_bgs = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5' ,'0.6' ,'0.7', '0.8', '0.9', '1.0']
        print ('\n{:<4} {:<20} {:>10}'.format('FrBG', 'Standard_deviation', 'Mean_y-offset'))
        print ('centroid window sizes:')
        print ('{:<4} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}'.format('', '3', '5', '7', '3', '5', '7'))
        for fbg, s3, s5, s7, m3, m5, m7 in zip(frac_bgs, sig3, sig5, sig7, mean3, mean5, mean7):
            print('{:<4} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f}'.format(fbg, s3, s5, s7, m3, m5, m7))

        fig4 = plt.figure(1, figsize=(12, 10))
        fig4.subplots_adjust(hspace=0.10)
        ax1 = fig4.add_subplot(311)
        ax1.set_title(case+"_BGfrac")
        ax1.set_xlabel('Magnitude')
        ax1.set_ylabel('Radial offset in Y: centroid window=3')
        plt.hlines(0.0, 18.0, 23.0, colors='k', linestyles='dashed')
        ax1.plot(mag, frac00[1], 'bo', ms=8, alpha=0.7, label='bg_frac=0.0')
        ax1.plot(mag, frac01[1], 'ro', ms=8, alpha=0.7, label='bg_frac=0.1')
        ax1.plot(mag, frac02[1], 'mo', ms=8, alpha=0.7, label='bg_frac=0.2')
        ax1.plot(mag, frac03[1], 'go', ms=5, alpha=0.7, label='bg_frac=0.3')
        ax1.plot(mag, frac04[1], 'ko', ms=8, alpha=0.7, label='bg_frac=0.4')
        ax1.plot(mag, frac05[1], 'yo', ms=8, alpha=0.7, label='bg_frac=0.5')
        ax1.plot(mag, frac06[1], 'co', ms=8, alpha=0.7, label='bg_frac=0.6')
        ax1.plot(mag, frac07[1], 'b+', ms=10, alpha=0.7, label='bg_frac=0.7')
        ax1.plot(mag, frac08[1], 'r+', ms=8, alpha=0.7, label='bg_frac=0.8')
        ax1.plot(mag, frac09[1], 'm+', ms=5, alpha=0.7, label='bg_frac=0.9')
        ax1.plot(mag, frac10[1], 'k+', ms=5, alpha=0.7, label='bg_frac=1.0')
        #textinfig = r'$\sigma$ = %0.2f    $\mu$ = %0.2f' % (sig3, mean3)
        #ax1.annotate(textinfig, xy=(0.75, 0.05), xycoords='axes fraction' )
        # Shrink current axis by 10%
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
        ax2 = fig4.add_subplot(312)
        ax2.set_xlabel('Magnitude')
        ax2.set_ylabel('Radial offset in Y: centroid window=5')
        plt.hlines(0.0, 18.0, 23.0, colors='k', linestyles='dashed')
        ax2.plot(mag, frac00[3], 'bo', ms=8, alpha=0.7, label='bg_frac=0.0')
        ax2.plot(mag, frac01[3], 'ro', ms=8, alpha=0.7, label='bg_frac=0.1')
        ax2.plot(mag, frac02[3], 'mo', ms=8, alpha=0.7, label='bg_frac=0.2')
        ax2.plot(mag, frac03[3], 'go', ms=5, alpha=0.7, label='bg_frac=0.3')
        ax2.plot(mag, frac04[3], 'ko', ms=8, alpha=0.7, label='bg_frac=0.4')
        ax2.plot(mag, frac05[3], 'yo', ms=8, alpha=0.7, label='bg_frac=0.5')
        ax2.plot(mag, frac06[3], 'co', ms=8, alpha=0.7, label='bg_frac=0.6')
        ax2.plot(mag, frac07[3], 'b+', ms=10, alpha=0.7, label='bg_frac=0.7')
        ax2.plot(mag, frac08[3], 'r+', ms=8, alpha=0.7, label='bg_frac=0.8')
        ax2.plot(mag, frac09[3], 'm+', ms=5, alpha=0.7, label='bg_frac=0.9')
        ax2.plot(mag, frac10[3], 'k+', ms=5, alpha=0.7, label='bg_frac=1.0')
        textinfig = r'BG      $\sigma$3     $\sigma$5     $\sigma$7'
        ax2.annotate(textinfig, xy=(1.02, 0.88), xycoords='axes fraction' )
        sigx = 1.02
        sigy = 0.9
        for fbg, s3, s5, s7 in zip(frac_bgs, sig3, sig5, sig7):
            line = ('{:<7} {:<6.2f} {:<6.2f} {:<6.2f}'.format(fbg, s3, s5, s7))
            sigy -= 0.08
            ax2.annotate(line, xy=(sigx, sigy), xycoords='axes fraction' )
        # Shrink current axis by 10%
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # put legend out of the plot box
        #ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
        ax3 = fig4.add_subplot(313)
        ax3.set_xlabel('Magnitude')
        ax3.set_ylabel('Radial offset in Y: centroid window=7')
        plt.hlines(0.0, 18.0, 23.0, colors='k', linestyles='dashed')
        ax3.plot(mag, frac00[5], 'bo', ms=8, alpha=0.7, label='bg_frac=0.0')
        ax3.plot(mag, frac01[5], 'ro', ms=8, alpha=0.7, label='bg_frac=0.1')
        ax3.plot(mag, frac02[5], 'mo', ms=8, alpha=0.7, label='bg_frac=0.2')
        ax3.plot(mag, frac03[5], 'go', ms=5, alpha=0.7, label='bg_frac=0.3')
        ax3.plot(mag, frac04[5], 'ko', ms=8, alpha=0.7, label='bg_frac=0.4')
        ax3.plot(mag, frac05[5], 'yo', ms=8, alpha=0.7, label='bg_frac=0.5')
        ax3.plot(mag, frac06[5], 'co', ms=8, alpha=0.7, label='bg_frac=0.6')
        ax3.plot(mag, frac07[5], 'b+', ms=10, alpha=0.7, label='bg_frac=0.7')
        ax3.plot(mag, frac08[5], 'r+', ms=8, alpha=0.7, label='bg_frac=0.8')
        ax3.plot(mag, frac09[5], 'm+', ms=5, alpha=0.7, label='bg_frac=0.9')
        ax3.plot(mag, frac10[5], 'k+', ms=5, alpha=0.7, label='bg_frac=1.0')
        textinfig = r'BG      $\mu$3     $\mu$5     $\mu$7'
        ax3.annotate(textinfig, xy=(1.02, 0.90), xycoords='axes fraction' )
        sigx = 1.02
        sigy = 0.9
        for fbg, m3, m5, m7 in zip(frac_bgs, mean3, mean5, mean7):
            line = ('{:<7} {:<6.2f} {:<6.2f} {:<6.2f}'.format(fbg, m3, m5, m7))
            sigy -= 0.08
            ax3.annotate(line, xy=(sigx, sigy), xycoords='axes fraction' )
        # Shrink current axis by 10%
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        if save_plot:
            destination = os.path.abspath(main_path_outfiles+"/plots/MagVsYoffset_frac_"+case+plot_type)
            fig4.savefig(destination)
            print ("\n Plot saved: ", destination)
        if show_plot:
            plt.show()
        else:
            plt.close('all')
        
if not single_star:
    print ("\n Centroids and differences were written into: \n  {}".format(output_file))

print ("\n Controlled test script finished. Took  %s  seconds to finish. \n" % (time.time() - start_time))
