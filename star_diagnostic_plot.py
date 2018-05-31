from __future__ import print_function, division
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import string




# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# April 2016 - Version 1.0: initial version completed


'''
This script crates a plot of the star number (x-axis) versus several residuals, R (where R=|True-Measured|):
    Rx, Ry, RV2, RV3, Rr ( r = square_root(V2^2 + V3^2) ).

Definitions of tests ran:
     TEST1 - Average positions P1 and P2, transform to V2-V3 space, and compare to average
             reference positions (V2-V3 space)
     TEST2 - Transform individual positions P1 and P2 to V2-V3 space, average V2-V3 space
             positions, and compare to average reference positions.
     TEST3 - Transform P1 and P2 individually to V2-V3 space and compare star by star and
             position by position.

'''


######################################################################################################################

### Set variables

case = '2DetsScene1_rapid_real_bgFrac0.3_thres01'
centroid_windows = [3, 5, 7]       # Centroid windows to plot: list of integers
Nsigma = 2.5                       # N-sigma rejection of bad stars: integer or float
abs_threshold = True               # Go to abs_threshold directory? True or False (results from least squares routine)
save_plot = False                 # Save the plots? True or False
show_plot = True                  # Show the plots? True or False


######################################################################################################################

### FUNCTIONS

def residuals_plot(plot_title, cwincase, arrx, arry, labels_list, destination,
                   save_plot=False, show_plot=True,
                   xlims=None, ylims=None):
    fig1 = plt.figure(1, figsize=(12, 10))
    ax1 = fig1.add_subplot(111)
    plt.suptitle(plot_title, fontsize=18, y=0.96)
    plt.title(cwincase)
    plt.xlabel('Star number')
    plt.ylabel('Residuals')
    '''
    if xlims is None:
        xmax = np.abs(max(arrx[0])+max(arrx[0])*0.2)
        xlims = [-1*xmax, xmax]
    if ylims is None:
        ymax = np.abs(max(arry[0])+max(arry[0])*0.2)
        ylims = [-1*ymax, ymax]
    # Compare which one is larger and use that one
    if xlims[1] > ylims[1]:
        ylims = xlims
    else:
        xlims = ylims
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.hlines(0.0, xlims[0], xlims[1], colors='k', linestyles='dashed')
    plt.vlines(0.0, ylims[0], ylims[1], colors='k', linestyles='dashed')
    '''
    #plt.plot(arrx[0], arry[0], 'm>', ms=6, alpha=0.6, label=labels_list[0])
    #plt.plot(arrx[1], arry[1], 'go', ms=5, alpha=0.6, label=labels_list[1])
    plt.plot(arrx[2], arry[2], 'r*', ms=8, alpha=0.6, label=labels_list[2])
    plt.plot(arrx[3], arry[3], 'bs', ms=4, alpha=0.6, label=labels_list[3])
    plt.plot(arrx[4], arry[4], 'g^', ms=6, alpha=0.6, label=labels_list[4])
    y_reject = [-0.010, 0.010]
    for xi, yi in zip(arrx[2], arry[2]):
        if yi >= y_reject[1] or yi <= y_reject[0]:
            si = int(xi)
            subxcoord = 5
            subycoord = 0
            side = 'left'
            plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
    for xi, yi in zip(arrx[0], arry[3]):
        if yi >= y_reject[1] or yi <= y_reject[0]:
            si = int(xi)
            subxcoord = 5
            subycoord = 0
            side = 'left'
            plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
    # Shrink current axis by 10%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.7))   # put legend out of the plot box
    if save_plot:
        fig1.savefig(destination)
        print ("\n Plot saved: ", destination)
    if show_plot:
        plt.show()
    else:
        plt.close('all')



### CODE

# Define the paths
path_randXstars = os.getcwd().split('src')[0]+'resultsXrandomstars'
path_good_stars = os.path.join(path_randXstars, 'T123_only_good_stars')
path_gs_abs_thres = os.path.join(path_good_stars, 'abs_threshold')
path2stars = path_good_stars
if abs_threshold:
    path2stars = path_gs_abs_thres

# Get measured pixel residuals
pixpos1 = os.path.join(path_good_stars, 'centroids_Scene1_bgFrac_Position1.txt')
pixpos2 = os.path.join(path_good_stars, 'centroids_Scene1_bgFrac_Position2.txt')
stars, bg, x31, y31, x51, y51, x71, y71, Tx, Ty = np.loadtxt(pixpos1, skiprows=5, usecols=(0,1,2,3,4,5,6,7,8,9), unpack=True)
x32, y32, x52, y52, x72, y72 = np.loadtxt(pixpos1, skiprows=5, usecols=(2,3,4,5,6,7), unpack=True)
Rx31, Rx51, Rx71 = np.abs(Tx - x31), np.abs(Tx - x51), np.abs(Tx - x71)
Ry31, Ry51, Ry71 = np.abs(Ty - y31), np.abs(Ty - y51), np.abs(Ty - y71)
Rx32, Rx52, Rx72 = np.abs(Tx - x32), np.abs(Tx - x52), np.abs(Tx - x72)
Ry32, Ry52, Ry72 = np.abs(Ty - y32), np.abs(Ty - y52), np.abs(Ty - y72)
# combine the 2 positions into 1 array
RX3, RY3 = [], []
sign_change = len(Rx31)
for xi, yi in zip(Rx31, Ry31):
    RX3.append(xi)
    RY3.append(yi)
for xi, yi in zip(Rx32, Ry32):
    RX3.append(xi)
    RY3.append(yi)
for i, _ in enumerate(RX3):
    if i >= sign_change:
        RX3[i] = -1.0*RX3[i]
        RY3[i] = -1.0*RY3[i]
    else:
        RX3[i] = RX3[i]
        RY3[i] = RY3[i]

# Get V2 and V3 residuals
test_files_list = glob(os.path.join(path2stars, 'T*'+case+'_Nsigma'+repr(Nsigma)+'.txt'))
print (case)
# data = star, bg, V2_3, V3_3, V2_5, V3_5, V2_7, V3_7,  V2_True, V3_True
dataT1 = np.loadtxt(test_files_list[0], comments='#', usecols=(0,1,2,3,4,5,6,7,8,9), unpack=True)
dataT2 = np.loadtxt(test_files_list[1], comments='#', usecols=(0,1,2,3,4,5,6,7,8,9), unpack=True)
dataT3 = np.loadtxt(test_files_list[2], comments='#', usecols=(0,1,2,3,4,5,6,7,8,9), unpack=True)
RavgpixV2_3, RavgpixV2_5, RavgpixV2_7 = np.abs(dataT1[8]-dataT1[2]), np.abs(dataT1[8]-dataT1[4]), np.abs(dataT1[8]-dataT1[6])
RavgpixV3_3, RavgpixV3_5, RavgpixV3_7 = np.abs(dataT1[9]-dataT1[3]), np.abs(dataT1[9]-dataT1[5]), np.abs(dataT1[9]-dataT1[7])
RavgpixR_3 = np.sqrt(RavgpixV2_3**2 + RavgpixV3_3**2)
RavgpixR_5 = np.sqrt(RavgpixV2_5**2 + RavgpixV3_5**2)
RavgpixR_7 = np.sqrt(RavgpixV2_7**2 + RavgpixV3_7**2)
RavgskyV2_3, RavgskyV2_5, RavgskyV2_7 = np.abs(dataT2[8]-dataT2[2]), np.abs(dataT2[8]-dataT2[4]), np.abs(dataT2[8]-dataT2[6])
RavgskyV3_3, RavgskyV3_5, RavgskyV3_7 = np.abs(dataT2[9]-dataT2[3]), np.abs(dataT2[9]-dataT2[5]), np.abs(dataT2[9]-dataT2[7])
RavgskyR_3 = np.sqrt(RavgskyV2_3**2 + RavgskyV3_3**2)
RavgskyR_5 = np.sqrt(RavgskyV2_5**2 + RavgskyV3_5**2)
RavgskyR_7 = np.sqrt(RavgskyV2_7**2 + RavgskyV3_7**2)
RV2_3, RV2_5, RV2_7 = np.abs(dataT3[8]-dataT3[2]), np.abs(dataT3[8]-dataT3[4]), np.abs(dataT3[8]-dataT3[6])
RV3_3, RV3_5, RV3_7 = np.abs(dataT3[9]-dataT3[3]), np.abs(dataT3[9]-dataT3[5]), np.abs(dataT3[9]-dataT3[7])
Rr_3 = np.sqrt(RV2_3**2 + RV3_3**2)
Rr_5 = np.sqrt(RV2_5**2 + RV3_5**2)
Rr_7 = np.sqrt(RV2_7**2 + RV3_7**2)
sign_change = len(RV2_3)/2
for i, _ in enumerate(RV2_3):
    if i >= sign_change:
        RV2_3[i], RV2_5[i], RV2_7[i] = -1.0*RV2_3[i], -1.0*RV2_5[i], -1.0*RV2_7[i]
        RV3_3[i], RV3_5[i], RV3_7[i] = -1.0*RV3_3[i], -1.0*RV3_5[i], -1.0*RV3_7[i]
        Rr_3[i], Rr_5[i], Rr_7[i] = -1*Rr_3[i], -1*Rr_5[i], -1*Rr_7[i]

for cwin in centroid_windows:
    # Create plots
    cwincase = case+'_CentroidWindow'+repr(cwin)
    if abs_threshold:
        cwincase += '_withAbsThres'
    cwincase += '_Nsigma'+repr(Nsigma)
    plot_title = 'Stars Diagnostic'
    destination = os.path.join(path2stars, 'StarsDiag_NoAvg_'+cwincase+'.jpg')
    arry = [RX3, RY3, RV2_3, RV3_3, Rr_3]
    arrx = [dataT3[0], dataT3[0], dataT3[0], dataT3[0], dataT3[0]]
    labels_list = ['RX', 'RY', 'RV2', 'RV3', 'RR']
    residuals_plot(plot_title, cwincase, arrx, arry, labels_list, destination,
                   save_plot=save_plot, show_plot=show_plot,
                   xlims=None, ylims=None)
