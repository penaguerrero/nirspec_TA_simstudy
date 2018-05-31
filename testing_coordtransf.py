from __future__ import print_function, division
import numpy as np
# other code
import coords_transform as ct



# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Feb 2016 - Version 1.0: initial version completed


"""
This script tests the coords_transform script. Given the set of detector and sky positions in 
file 491_F140X_notilt_test_answers_output_check_new_crud.txt, the code should give the other.

It outputs a text file with the same info for comparison.
"""

#######################################################################################################################

# general settings
detector = 491           # detector, integer: for now only 491 available
filter_input = "F140X"   # Filter, string: for now only test case is F140X
tilt = False             # tilt angle: True or False
debug = False            # See screen print statements for intermediate answers: True or False 
save_txt_file = False    # Save text file with resulting transformations: True or False
single_case = None       # test only a particular case: integer number of index from test_file, else set to None 

#######################################################################################################################

#  --> CODE

path2text_file = "../Coords_transforms/files_from_tony/"
test_file = "491_F140X_notilt_test_answers_output_check_new_crud.txt"
fname = path2text_file+test_file
# load the data
testnum, det_x, det_y, sky_x, sky_y = np.loadtxt(fname, skiprows=3, unpack=True)

# give the inputs of sky to transform to detector coordinates
transf_direction = "backward"
if single_case is not None:
    save_txt_file = False   # just in case, do not overwrite text file with only one case.
    xout_det, yout_det = ct.coords_transf(transf_direction, detector, filter_input, sky_x[single_case], sky_y[single_case], tilt, debug)
else:
    xout_det, yout_det = ct.coords_transf(transf_direction, detector, filter_input, sky_x, sky_y, tilt, debug)

# give the inputs of detector to transform to sky coordinates
transf_direction = "forward" 
if single_case is not None:
    xout_sky, yout_sky = ct.coords_transf(transf_direction, detector, filter_input, det_x[single_case], det_y[single_case], tilt, debug)
else:
    xout_sky, yout_sky = ct.coords_transf(transf_direction, detector, filter_input, det_x, det_y, tilt, debug)

# out file name
out_file = "../Coords_transforms/testing_coordstransf_"+str(detector)+"_"+filter_input+"_notilt.txt"
line0 = "{:<10} {:<16} {:<20} {:<16} {:<16}".format("\nTest_No.", "Detector_X", "Detector_Y", "Sky_X", "Sky_Y")
print (line0)
if single_case is not None:
    line1 = "{:<10} {:<16} {:<20} {:<16} {:<16}".format(single_case, xout_det, yout_det, xout_sky, yout_sky)
    print (line1) 
else:
    if save_txt_file:
        f = open(out_file, "w+")
        f.write(line0+"\n")
    for i, _ in enumerate(xout_det):
        line1 = "{:<10} {:<16} {:<20} {:<16} {:<16}".format(i, xout_det[i], yout_det[i], xout_sky[i], yout_sky[i])
        print (line1) 
        if save_txt_file:
            f.write(line1+"\n")
    if save_txt_file:
        f.close()
        print ("Results saved in file: ", out_file)

print ("\n Script finished!") 
