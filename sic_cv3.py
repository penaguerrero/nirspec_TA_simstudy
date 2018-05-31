from __future__ import print_function, division
from astropy.io import fits
from glob import glob
import numpy as np
#import matplotlib.pyplot as plt
import time
import os
import string

# extra coding used
import TA_functions as taf

print()

# Header
__author__ = "Maria A. Pena-Guerrero"
__version__ = "1.0"

"""

DESCRIPTION:
    This script runs the data from SIC test at CV3.
    
    Tasks performed are:
        - Performing centroiding for a width of 3 pixels
        - Fractional background = 0.0 
        - Moments are included in the code though not currently being used
        
"""

###########################################################################################################


# INITIAL CONDITIONS

output_full_detector = False       # Give resulting coordinates in terms of full detector: True or False
save_text_file = False             # Want to save the text file of comparison? True or False
save_centroid_disp = False         # Save the display with measured and true positions?
background_method = 'frac'         # Select either 'fractional', 'fixed', or None
analyze_all_frac_values = False    # Do you want to analyze fractional values from 0.0 through 1.0?
background2use = 0.3               # Background to use for analysis: None or float
backgnd_subtraction_method = 1     # 1    = Do background subtraction on final image (after subtracting 3-2 and 2-1),
#                                           before converting negative values into zeros (OSS)
#                                    2    = Do background subtraction on 3-2 and 2-1 individually
#                                    None = Do not subtract background
checkbox_size = 3                  # Real checkbox size
xwidth_list = [3, 5, 7]            # Number of rows of the centroid region
ywidth_list = [3, 5, 7]            # Number of columns of the centroid region
vlim = (1.0, 30)                   # Sensitivity limits of image, i.e. (0.001, 0.1)
threshold = 0.3                    # Convergence threshold of accepted difference between checkbox centroid and coarse location
max_iter = 10                      # Maximum number of iterations for finding coarse location
verbose = False                    # Show some debug messages (i.e. resulting calculations)
debug = False                      # See all debug messages (i.e. values of variables and calculations)
diffs_in_arcsecs = True            # Print the differences in arcsecs? True or False (=degrees) 
determine_moments = False          # Want to determine 2nd and 3rd moments?
display_master_img = False         # Want to see the combined ramped images for every star?
show_centroids = False             # Print measured centroid on screen: True or False
show_disp = False                  # Show display of resulting positions? (will show 2 figs, same but different contrast)


###########################################################################################################

# Paths
main_path = os.path.abspath("../SIC_CV3")
sic_cv3 = "/fits_files"

# Set up the output file and path for plots and figures
output_file_path = os.path.join(main_path, "results")

# other variables that need to be defined
true_center = [0.0, 0.0]
perform_PierCorr = False
case = "SICtest_CV3"

# start the timer to compute the whole running time
start_time = time.time()

#print("Looking for fits files in: ", main_path+sic_cv3)
dir2test = glob(main_path+sic_cv3+"/*.fits")

# Background cases variable setting
bg_frac, bg_value = None, None   # for the None case
bg_choice = "_bgNone"
if background_method is not None:
    if "frac" in background_method:
        bg_frac = background2use
        bg_choice = "_bgFrac"
        if analyze_all_frac_values:
            bg_frac = [x*0.1 for x in range(11)]
    elif "fix" in background_method:
        bg_value = background2use
        bg_choice = "_bgFixed"
else:
    background2use = 0.0

# Lists for later printing of results
fits_names = []
x_centroids3 = []
y_centroids3 = []
x_centroids5 = []
y_centroids5 = []
x_centroids7 = []
y_centroids7 = []

x_centroids = [x_centroids3, x_centroids5, x_centroids7]
y_centroids = [y_centroids3, y_centroids5, y_centroids7]

centroids_info = [true_center, output_full_detector, show_centroids, perform_PierCorr]

# start the loop over the fits files
for fits_file in dir2test:
    print ("Running centroid algorithm... ")
    
    bg_corr_info = [backgnd_subtraction_method, background_method, bg_value, bg_frac, debug]
    recursive_centroids_info = [xwidth_list, ywidth_list, checkbox_size, max_iter, threshold, 
                                determine_moments, display_master_img, vlim]
    display_centroids_info = [case, show_disp, save_centroid_disp]

    if analyze_all_frac_values:
        for bgf in bg_frac:
            bg_corr_info = [backgnd_subtraction_method, background_method, bg_value, bgf, debug]
            x_centroids, y_centroids = taf.find_centroid(fits_file, bg_corr_info, recursive_centroids_info,
                                                         display_centroids_info, x_centroids, y_centroids,
                                                         fits_names, output_file_path, centroids_info, verbose)
    else:
        x_centroids, y_centroids = taf.find_centroid(fits_file, bg_corr_info, recursive_centroids_info,
                                                     display_centroids_info, x_centroids, y_centroids,
                                                     fits_names, output_file_path, centroids_info, verbose)

# Write the results in a text file
output_file = os.path.join(output_file_path, case+bg_choice+".txt")

line0 = "Centroid indexing starting at 1 !"
if len(xwidth_list)==1:
    line0a = "{:<50} {:>16} {:>20}".format("Fits file name", "Background", "Centroid window = 5")
    line0b = "{:>63} {:>10} {:>16}".format(background_method, "x", "y")
else:
    line0a = "{:<50} {:>16} {:>20} {:>27} {:>33}".format("Fits file name", "Background", "Centroid window: 3", "5", "7")
    line0b = "{:>63} {:>10} {:>14} {:>18} {:>14} {:>18} {:>14}".format(background_method, "x", "y", "x", "y", "x", "y")
    
print (line0)
print (line0a)
print (line0b)

if save_text_file:
    f = open(output_file, "w+")
    f.write(line0+"\n")
    f.write(line0a+"\n")
    f.write(line0b+"\n")
    f.close()

x_centroids3, x_centroids5, x_centroids7 = x_centroids
y_centroids3, y_centroids5, y_centroids7 = y_centroids

# if analyzing all fractional values, create a list of background values with same length as centroids
if analyze_all_frac_values:
    background2use_list = []
    while len(background2use_list) != len(fits_names):
        for bgf in bg_frac:
            background2use_list.append(bgf)
        
for i, ff in enumerate(fits_names):
    if analyze_all_frac_values:
        background2use = background2use_list[i]
    taf.print_file_lines(output_file, save_text_file, xwidth_list, ff, background2use,
                             i, x_centroids, y_centroids)



print ("\n SIC test CV3 data script finished. Took  %s  seconds to finish. \n" % (time.time() - start_time))
