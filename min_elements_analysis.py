from __future__ import print_function, division
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import TA_functions as taf
import v2v3plots as vp




# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# June 2016 - Version 1.0: initial version completed


'''
This script creates a plot of the mean of the means and the standard deviation of the means for the 5000 runs for a
 star sample of 3, 5, 8, and 20 reference stars, versus the number of minimum elements in the array to do the
 absolute threshold routine.

 * All runs are expected to be in  PyCharm/nirspec/TargetAcquisition/resultsXrandomstars/good_and_uglies.
'''


############################################################################################################

# Set parameters

case = '2DetsScene1_rapid_real_bgFrac0.3_thres01'
save_plot = True
show_plot = False
milliarcsec = True


############################################################################################################


# Set additional variables
Nsigma2plot = 2.5
good_and_ugly_stars = True     # only plot good stars if set to False
centroid_windows = [3, 5, 7]
min_elements_list = [2, 3, 4, 5, 6, 7, 8]
stars_in_sample_list = [3, 5, 8, 20]
results_path = os.path.abspath('../good_and_uglies/')



def get_stdevmeans(muV2, muV3):
    V2stdevs, V2means, V3stdevs, V3means = [], [], [], []
    for mv2, mv3 in zip(muV2, muV3):
        sd2, m2 = taf.find_std(mv2)
        V2stdevs.append(sd2)
        V2means.append(m2)
        sd3, m3 = taf.find_std(mv3)
        V3stdevs.append(sd3)
        V3means.append(m3)
    stdevmeans_values = [V2stdevs[0], V2means[0], V2stdevs[1], V2means[1], V2stdevs[2], V2means[2],
                         V3stdevs[0], V3means[0], V3stdevs[1], V3means[1], V3stdevs[2], V3means[2]]
    return stdevmeans_values


# gather information and store it in corresponding dictionary - taken from the plots
refstars3, refstars5, refstars8, refstars20 = {}, {}, {}, {}
# 3 reference stars, minimum elements =     2      3      4      5      6      7      8
refstars3['stddev_of_meansV2_AvgPix'] = [79.68, 91.00, 91.00, 91.00, 91.00, 91.00, 91.00]
refstars3['stddev_of_meansV3_AvgPix'] = [66.54, 94.04, 94.04, 94.04, 94.04, 94.04, 94.04]
refstars3['mean_of_meansV2_AvgPix']   = [ 2.19, -9.60, -9.60, -9.60, -9.60, -9.60, -9.60]
refstars3['mean_of_meansV3_AvgPix']   = [-0.94,-15.49,-15.49,-15.49,-15.49,-15.49,-15.49]
refstars3['stddev_of_meansV2_AvgSky'] = [79.69, 91.01, 91.01, 91.01, 91.01, 91.01, 91.01]
refstars3['stddev_of_meansV3_AvgSky'] = [66.54, 94.04, 94.04, 94.04, 94.04, 94.04, 94.04]
refstars3['mean_of_meansV2_AvgSky']   = [ 2.18, -9.60, -9.60, -9.60, -9.60, -9.60, -9.60]
refstars3['mean_of_meansV3_AvgSky']   = [-0.95,-15.50,-15.50,-15.50,-15.50,-15.50,-15.50]
refstars3['stddev_of_meansV2_NoAvg']  = [10.85, 10.85, 11.45, 26.03, 91.03, 91.03, 91.03]
refstars3['stddev_of_meansV3_NoAvg']  = [ 7.98,  7.98,  7.99, 24.79, 94.02, 94.02, 94.02]
refstars3['mean_of_meansV2_NoAvg']    = [-3.09, -3.09, -3.04, -2.41, -9.60, -9.60, -9.60]
refstars3['mean_of_meansV3_NoAvg']    = [ 0.55,  0.55,  0.55, -0.67,-15.50,-15.50,-15.50]
# 5 reference stars, minimum elements =     2      3      4      5      6      7      8
refstars5['stddev_of_meansV2_AvgPix'] = [16.61, 17.87, 26.61, 51.75, 51.75, 51.75, 51.75]
refstars5['stddev_of_meansV3_AvgPix'] = [14.51, 13.42, 28.62, 70.49, 70.49, 70.49, 70.49]
refstars5['mean_of_meansV2_AvgPix']   = [ 1.01,  1.12, -0.20,-10.14,-10.14,-10.14,-10.14]
refstars5['mean_of_meansV3_AvgPix']   = [ 2.00,  1.75, -0.89,-13.17,-13.17,-13.17,-13.17]
refstars5['stddev_of_meansV2_AvgSky'] = [16.61, 17.87, 26.61, 51.75, 51.75, 51.75, 51.75]
refstars5['stddev_of_meansV3_AvgSky'] = [14.51, 13.42, 28.62, 70.49, 70.49, 70.49, 70.49]
refstars5['mean_of_meansV2_AvgSky']   = [ 1.01,  1.12, -0.20,-10.14,-10.14,-10.14,-10.14]
refstars5['mean_of_meansV3_AvgSky']   = [ 2.00,  1.75, -0.90,-13.17,-13.17,-13.17,-13.17]
refstars5['stddev_of_meansV2_NoAvg']  = [ 0.94,  0.94,  0.94,  0.94,  0.94,  0.94,  0.94]
refstars5['stddev_of_meansV3_NoAvg']  = [ 1.05,  1.05,  1.05,  1.05,  1.05,  1.05,  1.05]
refstars5['mean_of_meansV2_NoAvg']    = [-2.36, -2.36, -2.36, -2.36, -2.36, -2.36, -2.36]
refstars5['mean_of_meansV3_NoAvg']    = [-0.82, -0.82, -0.82, -0.82, -0.82, -0.82, -0.82]
# 8 reference stars, minimum elements =     2      3      4      5      6      7      8
refstars8['stddev_of_meansV2_AvgPix'] = [  8.21,  8.21,  8.21,  8.59, 11.47, 21.36, 39.60]
refstars8['stddev_of_meansV3_AvgPix'] = [  4.55,  4.55,  4.55,  4.81,  9.52, 26.06, 53.13]
refstars8['mean_of_meansV2_AvgPix']   = [ -1.33, -1.33, -1.33, -1.33, -1.44, -3.06,-10.48]
refstars8['mean_of_meansV3_AvgPix']   = [  0.02,  0.02,  0.02, -0.01, -0.52, -3.94,-13.04]
refstars8['stddev_of_meansV2_AvgSky'] = [  8.21,  8.21,  8.21,  8.58, 11.47, 21.36, 39.60]
refstars8['stddev_of_meansV3_AvgSky'] = [  4.55,  4.55,  4.55,  4.81,  9.52, 26.06, 53.13]
refstars8['mean_of_meansV2_AvgSky']   = [ -1.33, -1.33, -1.33, -1.33, -1.44, -3.11,-10.48]
refstars8['mean_of_meansV3_AvgSky']   = [  0.02,  0.02,  0.02, -0.01, -0.52, -3.94,-13.04]
refstars8['stddev_of_meansV2_NoAvg']  = [  0.69,  0.69,  0.69,  0.69,  0.69,  0.69,  0.69]
refstars8['stddev_of_meansV3_NoAvg']  = [  0.75,  0.75,  0.75,  0.75,  0.75,  0.75,  0.75]
refstars8['mean_of_meansV2_NoAvg']    = [ -2.38, -2.38, -2.38, -2.38, -2.38, -2.38, -2.38]
refstars8['mean_of_meansV3_NoAvg']    = [ -0.81, -0.81, -0.81, -0.81, -0.81, -0.81, -0.81]
# 20 reference stars, minimum elements=     2      3      4      5      6      7      8
refstars20['stddev_of_meansV2_AvgPix']= [  0.41,  0.41,  0.41,  0.41,  0.41,  0.41,  0.41]
refstars20['stddev_of_meansV3_AvgPix']= [  0.46,  0.46,  0.46,  0.46,  0.46,  0.46,  0.46]
refstars20['mean_of_meansV2_AvgPix']  = [ -2.41, -2.41, -2.41, -2.41, -2.41, -2.41, -2.41]
refstars20['mean_of_meansV3_AvgPix']  = [ -0.87, -0.87, -0.87, -0.87, -0.87, -0.87, -0.87]
refstars20['stddev_of_meansV2_AvgSky']= [  0.41,  0.41,  0.41,  0.41,  0.41,  0.41,  0.41]
refstars20['stddev_of_meansV3_AvgSky']= [  0.46,  0.46,  0.46,  0.46,  0.46,  0.46,  0.46]
refstars20['mean_of_meansV2_AvgSky']  = [ -2.41, -2.41, -2.41, -2.41, -2.41, -2.41, -2.41]
refstars20['mean_of_meansV3_AvgSky']  = [ -0.87, -0.87, -0.87, -0.87, -0.87, -0.87, -0.87]
refstars20['stddev_of_meansV2_NoAvg'] = [  0.42,  0.42,  0.42,  0.42,  0.42,  0.42,  0.42]
refstars20['stddev_of_meansV3_NoAvg'] = [  0.43,  0.43,  0.43,  0.43,  0.43,  0.43,  0.43]
refstars20['mean_of_meansV2_NoAvg']   = [ -2.41, -2.41, -2.41, -2.41, -2.41, -2.41, -2.41]
refstars20['mean_of_meansV3_NoAvg']   = [ -0.80, -0.80, -0.80, -0.80, -0.80, -0.80, -0.80]

# create list of dictionaries
number_of_ref_stars = [refstars3, refstars5, refstars8, refstars20]

'''
# To fill up dictionaries loop over reference stars and minimum elements
# loop over list of number of reference stars: 3, 5, 8, 20
for stars_in_sample in stars_in_sample_list:
    # loop over number of minimum elements per number of reference stars: 2, 3, 4, 5, 6, 7, 8
    for min_elements in min_elements_list:
        # general path to text files
        star_sample_dir = repr(stars_in_sample)+'_star_sample'
        type_of_stars = 'only_good_stars'
        if good_and_ugly_stars:
            type_of_stars = 'good_and_uglies'
        gen_path = os.path.abspath('../resultsXrandomstars/'+type_of_stars+'/'+star_sample_dir+
                                   '/diff_min_elements_abs_threshold')

        # Loop over centroid_windows
        case += '_minele'+repr(min_elements)
        for cwin in centroid_windows:
            # load the data fom the 3 tests
            test_files_list = glob(os.path.join(gen_path, 'TEST*'+case+'*_Nsigma'+repr(Nsigma2plot)+'*'+repr(cwin)+'.txt'))
            #          0        1        2          3        4       5         6         7         8
            # data = sample, sigmaV2, sigmaV3, sigmaTheta, meanV2, meanV3, meanTheta, LastIter, RejStars
            dataT1 = np.loadtxt(test_files_list[0], comments='#', unpack=True)
            dataT2 = np.loadtxt(test_files_list[1], comments='#', unpack=True)
            dataT3 = np.loadtxt(test_files_list[2], comments='#', unpack=True)

            # compact variables and convert to milli arcsec
            conversion = 1.0
            if milliarcsec:
                conversion = 1000.0
            muV2 = [dataT1[4]*conversion, dataT2[4]*conversion, dataT3[4]*conversion]
            muV3 = [dataT1[5]*conversion, dataT2[5]*conversion, dataT3[5]*conversion]
            sigmaV2 = [dataT1[1]*conversion, dataT2[1]*conversion, dataT3[1]*conversion]
            sigmaV3 = [dataT1[2]*conversion, dataT2[2]*conversion, dataT3[2]*conversion]
            theta = [dataT1[6]*conversion, dataT2[6]*conversion, dataT3[6]*conversion]
            cwincase = case+'_CentroidWin'+repr(cwin)+'_'+str(stars_in_sample)+'star'+str(len(dataT1[0]))+'samples_withAbsThres'

            # calculate mean of the means and standard deviation of the means
            stdevmeans_values = get_stdevmeans(muV2, muV3)
            cwincase += '_Nsigma'+repr(Nsigma2plot)

            # store values
            #refstars3['stddev_of_meansV2_AvgPix'] =
'''


# Plots
cwincase = '2DetsScene1_rapid_real_bgFrac0.3_thres01_CentroidWin3_withAbsThres_Nsigma2.5'
for nrs, refstars in zip(stars_in_sample_list, number_of_ref_stars):
    plot_title = repr(nrs)+' reference Stars'
    labels_list = ['Avg in Pixel Space', 'Avg in Sky', 'No Avg']
    arrx = [min_elements_list, min_elements_list, min_elements_list]
    arry = [refstars3['stddev_of_meansV2_AvgPix'], refstars3['stddev_of_meansV2_AvgSky'], refstars3['stddev_of_meansV2_NoAvg']]
    xlabel, ylabel = 'Number of minimum elements', r'$\sigma$V2 [marcsecs]'
    xlims = [0, 10]
    ylims = [0, max(arry[0])+max(arry[0])*0.2]
    vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title, labels_list=labels_list, xlims=xlims, ylims=ylims,
                 print_side_string = None, print_side_values=None,
                 save_plot=False, show_plot=True, destination=None, star_sample=None, square=False)
    arry = [refstars['stddev_of_meansV3_AvgPix'], refstars['stddev_of_meansV3_AvgSky'], refstars['stddev_of_meansV3_NoAvg']]
    xlabel, ylabel = 'Number of minimum elements', r'$\sigma$V3 [marcsecs]'
    vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title, labels_list=labels_list, xlims=xlims, ylims=ylims,
                  print_side_string = None, print_side_values=None,
                  save_plot=False, show_plot=True, destination=None, star_sample=None, square=False)
    arry = [refstars['mean_of_meansV2_AvgPix'], refstars['mean_of_meansV2_AvgSky'], refstars['mean_of_meansV2_NoAvg']]
    xlabel, ylabel = 'Number of minimum elements', r'$\mu$V2 [marcsecs]'
    xlims = [0, 10]
    ylims = None
    vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title, labels_list=labels_list, xlims=xlims, ylims=ylims,
                  print_side_string = None, print_side_values=None,
                  save_plot=False, show_plot=True, destination=None, star_sample=None, square=False)
    arry = [refstars['mean_of_meansV3_AvgPix'], refstars['mean_of_meansV3_AvgSky'], refstars['mean_of_meansV3_NoAvg']]
    xlabel, ylabel = 'Number of minimum elements', r'$\mu$V3 [marcsecs]'
    xlims = [0, 10]
    ylims = None
    vp.make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=plot_title, labels_list=labels_list, xlims=xlims, ylims=ylims,
                  print_side_string = None, print_side_values=None,
                  save_plot=False, show_plot=True, destination=None, star_sample=None, square=False)
