from __future__ import print_function, division
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pylab as P
import os
import copy
import string

# other code
import testing_functions as tf

# Header
__author__ = "Maria A. Pena-Guerrero"
__version__ = "1.0"


"""

DESCRIPTION:
This script calculates the average centroids determined for the modeled ESA reference stars from 
the first and shifted positions. It creates plots of the histogram of TruePosition-AvgTAPosition
versus number of stars. It finds the standard deviations and means for checkbox size 3, 5, and 7.

"""

###########################################################################################################


# Set script parameters
detector = 492                     # Which detector are we working with: 491 or 492
centroid_in_full_detector = False  # Give resulting coordinates in terms of full detector: True or False
save_text_file = False             # Want to save the text file of comparison? True or False
save_figs = False                  # Save the distribution figures: True or False
show_plot = False                  # Show plots: True or False


###########################################################################################################

# --> FUNCTIONS

def read_centroid_txt(fname):
    data = np.loadtxt(fname, skiprows=3, usecols=(0,1,2,3,4,5,6,7,8,9,10,11), unpack=True)
    star_num, bg, x3, y3, x5, y5, x7, y7, trueX, trueY, LoLeftX, LoLeftY = data
    return star_num, bg, x3, y3, x5, y5, x7, y7, trueX, trueY, LoLeftX, LoLeftY
    
    
def write_txt_file(Scenario_all_centroids_list, Scenario_centroid_names, TrueXpositions, TrueYpositions, save_text_file):
    # Read the files and calculate average positions for comparison with true positions
    for case in Scenario_all_centroids_list:
        print ("now comparing: ")
        print (case[0])
        print (case[1])
        #raw_input()
        # prepare text file
        case_idx = Scenario_all_centroids_list.index(case)
        name_txt_file_path = "../PFforMaria/detector_"+str(detector)+"_comparison_txt_positions"
        name_txt_file = os.path.join(name_txt_file_path, Scenario_centroid_names[case_idx]+".txt")
        line0 = '{:<5} {:<6} {:<42} {:<30} {:<20} {:<40} {:<30} {:<20} {:<40} {:<30} {:<20} {:<40} {:<30} {:<26} {:<22} {:<21} {:<43} {:<30} {:<20}'.format(
                                              'Star', 'BkGd',
                                              'TA_pos1:   ChBx3', 'ChBx5', 'ChBx7', 'TA_pos2: ChBx3', 'ChBx5', 'ChBx7', 
                                              'Pos2-Pos1: ChBx3', 'ChBx5', 'ChBx7', 'AvgTA_pos: ChBx3', 'ChBx5', 'ChBx7', 
                                              'TruePositions', 'TrueLoLeftCorner',
                                              'True-AvgTA_pos: ChBx3', 'ChBx5', 'ChBx7')
        line1 = '{:>21} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13} {:>12} {:>10} {:>16} {:>13} {:>16} {:>13} {:>16} {:>13}'.format(
                'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y')
        if save_text_file:
            f = open(name_txt_file, "w+")
            f.write(line0+"\n")
            f.write(line1+"\n")
            f.close()
        
        # first position
        star_num0, bg0, x30, y30, x50, y50, x70, y70, trueX0, trueY0, LoLeftX0, LoLeftY0 = read_centroid_txt(case[0])
        # shifted position
        star_num1, bg1, x31, y31, x51, y51, x71, y71, trueX1, trueY1, LoLeftX1, LoLeftY1 = read_centroid_txt(case[1])

        # difference between positions 1 and 2
        diffP2P1_x3, diffP2P1_y3 = x31-x30, y31-y30
        diffP2P1_x5, diffP2P1_y5 = x51-x50, y51-y50 
        diffP2P1_x7, diffP2P1_y7 = x71-x70, y71-y70
        # average positions
        avgx3, avgx5, avgx7 = (x30+x31)/2., (x50+x51)/2., (x70+x71)/2.  
        avgy3, avgy5, avgy7 = (y30+y31)/2., (y50+y51)/2., (y70+y71)/2.  
        
        # compare to true positions
        for i, _ in enumerate(star_num0):
            # transform positions to be in 32x32 pixel format
            ESA_center = [0,0]
            avg_centroid_list = [[avgx3[i], avgy3[i]], [avgx5[i], avgy5[i]], [avgx7[i], avgy7[i]]]
            # the position: (TrueXpositions[i], TrueYpositions[i]) = (LoLeftX0[i]+trueX0[i], LoLeftY0[i]+trueY0[i])
            #print (TrueXpositions[i], TrueYpositions[i])
            #print (LoLeftX0[i]+trueX0[i], LoLeftY0[i]+trueY0[i])
            true_center = [LoLeftX0[i]+trueX0[i], LoLeftY0[i]+trueY0[i]]
            true_positions, AvgTAcentroids, loleftcoords, differences_true_TA = tf.transform2fulldetector(
                                                                                    detector, 
                                                                                    centroid_in_full_detector,
                                                                                    avg_centroid_list, ESA_center, 
                                                                                    true_center, perform_avgcorr=False)

            line2 = '{:<5} {:<6} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<16} {:<13} {:<18} {:<8} {:<10} {:<13.10f} {:<16.10f} {:<13.10f} {:<16.10f} {:<13.10f} {:<16.10f}'.format(
                              int(star_num0[i]), bg0[i],
                              x30[i], y30[i], x50[i], y50[i], x70[i], y70[i],
                              x31[i], y31[i], x51[i], y51[i], x71[i], y71[i],
                              diffP2P1_x3[i], diffP2P1_y3[i], diffP2P1_x5[i], diffP2P1_y5[i], diffP2P1_x7[i], diffP2P1_y7[i],
                              AvgTAcentroids[0][0], AvgTAcentroids[0][1], AvgTAcentroids[1][0], AvgTAcentroids[1][1], AvgTAcentroids[2][0], AvgTAcentroids[2][1],
                              true_positions[0], true_positions[1], int(loleftcoords[0]), int(loleftcoords[1]),
                              differences_true_TA[0][0][0], differences_true_TA[0][0][1], 
                              differences_true_TA[0][1][0], differences_true_TA[0][1][1],
                              differences_true_TA[0][2][0], differences_true_TA[0][2][1]) 
            if save_text_file:
                f = open(name_txt_file, "a")
                f.write(line2+"\n")
                f.close()
                print ("Results saved in: ", name_txt_file)
            print (line0)
            print (line1)   
            print (line2)
            print ("\n Finished case: ", name_txt_file)
        #raw_input(" PRESS ENTER TO CONTINUE")


def gaus(a, mu, sigma, x):
    return a * np.exp(-0.5*((x-mu)/sigma)**2)
        
        
def get_distriution(txt_file, save_figs):
    # Read the text files created in order to find a distribution with a mean and a standard deviation
    data = np.loadtxt(txt_file, skiprows=2, unpack=True)
    star,bg,x30,y30,x50,y50,x70,y70,x31,y31,x51,y51,x71,y71,dp1p2x3,dp1p2y3,dp1p2x5,dp1p2y5,dp1p2x7,dp1p2y7,avx3,avy3,avx5,avy5,avx7,avy7,trueX,trueY,loleftX,loleftY,diffX3,diffY3,diffX5,diffY5,diffX7,diffY7 = data
    
    # Find standard deviation and mean of:  TrueY - AvgTAyCheckbox
    Ystd_dev3, Ymean3 = tf.find_std(diffY3)
    Ystd_dev5, Ymean5 = tf.find_std(diffY5)
    Ystd_dev7, Ymean7 = tf.find_std(diffY7)
    
    # the histogram of the data with histtype='step'
    n3, bins3, patches3 = P.hist(diffY3, 60, histtype='stepfilled')
    n5, bins5, patches5 = P.hist(diffY5, 60, histtype='stepfilled')
    n7, bins7, patches7 = P.hist(diffY7, 60, histtype='stepfilled')
    a3, a5, a7 = max(n3), max(n5), max(n7)
    P.setp(patches3, 'facecolor', 'b', 'alpha', 0.5)
    P.setp(patches5, 'facecolor', 'g', 'alpha', 0.5)
    P.setp(patches7, 'facecolor', 'r', 'alpha', 0.5)
    x3 = copy.deepcopy(diffY3)
    x5 = copy.deepcopy(diffY5)
    x7 = copy.deepcopy(diffY7)
    y3 = gaus(a3, Ymean3, Ystd_dev3, x3)
    y5 = gaus(a5, Ymean5, Ystd_dev5, x5)
    y7 = gaus(a7, Ymean7, Ystd_dev7, x7)
    plt.plot(x3, y3,'bo',label='Checkbox=3')
    plt.plot(x5, y5,'go',label='Checkbox=5')
    plt.plot(x7, y7,'ro',label='Checkbox=7')
    list_x3, list_x5, list_x7 = x3.tolist(), x5.tolist(), x7.tolist()
    list_y3, list_y5, list_y7 = y3.tolist(), y5.tolist(), y7.tolist()
    s3, s5, s7 = [], [], []
    for i, _ in enumerate(list_x3):
        s3i = [list_x3[i], list_y3[i]]
        s5i = [list_x5[i], list_y5[i]]
        s7i = [list_x7[i], list_y7[i]]
        s3.append(s3i)
        s5.append(s5i)
        s7.append(s7i)
    s3.sort(key=lambda xx: xx[0])
    s5.sort(key=lambda xx: xx[0])
    s7.sort(key=lambda xx: xx[0])
    sx3, sy3, sx5, sy5, sx7, sy7 = [], [], [], [], [], []
    for i,_ in enumerate(s3):
        sx3.append(s3[i][0])
        sy3.append(s3[i][1])
        sx5.append(s5[i][0])
        sy5.append(s5[i][1])
        sx7.append(s7[i][0])
        sy7.append(s7[i][1])
    fig = plt.figure(1, figsize=(12, 10))
    #fig.subplots_adjust(hspace=0.30)
    ax = fig.add_subplot(111)
    plt.plot(sx3, sy3,'b:')
    plt.plot(sx5, sy5,'g:')
    plt.plot(sx7, sy7,'r:')
    plt.xlabel('TruePosition - AvgTA')
    plt.ylabel('Number of stars')
    textinfig3 = r'$\sigma3$ = %0.2f    $\mu3$ = %0.2f' % (Ystd_dev3, Ymean3)
    textinfig5 = r'$\sigma5$ = %0.2f    $\mu5$ = %0.2f' % (Ystd_dev5, Ymean5)
    textinfig7 = r'$\sigma7$ = %0.2f    $\mu7$ = %0.2f' % (Ystd_dev7, Ymean7)
    # Shrink current axis by 10%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # put legend out of the plot box
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
    ax.legend(loc='upper left')           
    # obtain title for figure
    fig_name_list = string.rsplit(txt_file, sep='/')
    fig_name = fig_name_list[3].replace('.txt', '') 
    plt.title(fig_name)
    if fig_name == 'Scene2_centroids_slow_nonoise_fixed':
        ax.annotate(textinfig3, xy=(0.03, 0.7), xycoords='axes fraction' )
        ax.annotate(textinfig5, xy=(0.03, 0.65), xycoords='axes fraction' )
        ax.annotate(textinfig7, xy=(0.03, 0.6), xycoords='axes fraction' )
    else:
        ax.annotate(textinfig3, xy=(0.68, 0.93), xycoords='axes fraction' )
        ax.annotate(textinfig5, xy=(0.68, 0.88), xycoords='axes fraction' )
        ax.annotate(textinfig7, xy=(0.68, 0.83), xycoords='axes fraction' )

    if save_figs:
        destination = os.path.abspath("../PFforMaria/detector_"+str(detector)+"_plots_comparison/"+fig_name+'.jpg')
        fig.savefig(destination)
        print ("\n Plot saved: ", destination) 
    if show_plot:
        plt.show()
    else:
        plt.close('all')
    return Ystd_dev3, Ymean3, Ystd_dev5, Ymean5, Ystd_dev7, Ymean7
    

###########################################################################################################

# ---> CODE

# Set the path for comparison files and store them accordingly
all_files = glob("../PFforMaria/detector_"+str(detector)+"_resulting_centroid_txt_files_redo/*")
Scene1_centroids_rapid_nonoise_None, Scene2_centroids_rapid_nonoise_None  = [], []
Scene1_centroids_rapid_nonoise_fixed,Scene2_centroids_rapid_nonoise_fixed = [], []
Scene1_centroids_rapid_nonoise_frac, Scene2_centroids_rapid_nonoise_frac  = [], []
Scene1_centroids_rapid_real_None,    Scene2_centroids_rapid_real_None     = [], []
Scene1_centroids_rapid_real_fixed,   Scene2_centroids_rapid_real_fixed    = [], []
Scene1_centroids_rapid_real_frac,    Scene2_centroids_rapid_real_frac     = [], []
Scene1_centroids_slow_nonoise_None,  Scene2_centroids_slow_nonoise_None   = [], []
Scene1_centroids_slow_nonoise_fixed, Scene2_centroids_slow_nonoise_fixed  = [], []
Scene1_centroids_slow_nonoise_frac,  Scene2_centroids_slow_nonoise_frac   = [], []
Scene1_centroids_slow_real_None,     Scene2_centroids_slow_real_None      = [], []
Scene1_centroids_slow_real_fixed,    Scene2_centroids_slow_real_fixed     = [], []
Scene1_centroids_slow_real_frac,     Scene2_centroids_slow_real_frac      = [], []

for f in all_files:
    #print (f)
    if "scene1" in f:
        if "rapid" in f:
            if "_bgNone_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene1_centroids_rapid_nonoise_None.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene1_centroids_rapid_real_None.append(f)
            if "_bgFixed_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene1_centroids_rapid_nonoise_fixed.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene1_centroids_rapid_real_fixed.append(f)
            if "_bgFrac_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene1_centroids_rapid_nonoise_frac.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene1_centroids_rapid_real_frac.append(f)
        if "slow" in f:
            if "_bgNone_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene1_centroids_slow_nonoise_None.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene1_centroids_slow_real_None.append(f)
            if "_bgFixed_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene1_centroids_slow_nonoise_fixed.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene1_centroids_slow_real_fixed.append(f)
            if"_bgFrac_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene1_centroids_slow_nonoise_frac.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene1_centroids_slow_real_frac.append(f)
    if "scene2"in f:
        if "rapid" in f:
            if "_bgNone_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene2_centroids_rapid_nonoise_None.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene2_centroids_rapid_real_None.append(f)
            if "_bgFixed_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene2_centroids_rapid_nonoise_fixed.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene2_centroids_rapid_real_fixed.append(f)
            if "_bgFrac_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene2_centroids_rapid_nonoise_frac.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene2_centroids_rapid_real_frac.append(f)
        if "slow" in f:#_real_shifted_redo_bgFixed
            if "_bgNone_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene2_centroids_slow_nonoise_None.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene2_centroids_slow_real_None.append(f)
            if "_bgFixed_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene2_centroids_slow_nonoise_fixed.append(f)
                elif "_real" in f or "_real_shifted" in f:
                    Scene2_centroids_slow_real_fixed.append(f)
            if "_bgFrac_redo" in f:
                if "_nonoise" in f or "_nonoise_shifted" in f:
                    Scene2_centroids_slow_nonoise_frac.append(f) 
                elif "_real" in f or "_real_shifted" in f:
                    Scene2_centroids_slow_real_frac.append(f)

Scene1_all_centroids = [Scene1_centroids_rapid_nonoise_None, Scene1_centroids_rapid_nonoise_fixed, Scene1_centroids_rapid_nonoise_frac,
                        Scene1_centroids_rapid_real_None, Scene1_centroids_rapid_real_fixed, Scene1_centroids_rapid_real_frac,
                        Scene1_centroids_slow_nonoise_None, Scene1_centroids_slow_nonoise_fixed, Scene1_centroids_slow_nonoise_frac,
                        Scene1_centroids_slow_real_None, Scene1_centroids_slow_real_fixed, Scene1_centroids_slow_real_frac]

Scene2_all_centroids = [Scene2_centroids_rapid_nonoise_None, Scene2_centroids_rapid_nonoise_fixed, Scene2_centroids_rapid_nonoise_frac,
                        Scene2_centroids_rapid_real_None, Scene2_centroids_rapid_real_fixed, Scene2_centroids_rapid_real_frac,
                        Scene2_centroids_slow_nonoise_None, Scene2_centroids_slow_nonoise_fixed, Scene2_centroids_slow_nonoise_frac,
                        Scene2_centroids_slow_real_None, Scene2_centroids_slow_real_fixed, Scene2_centroids_slow_real_frac]

# List of names for text files
Scene1_centroid_names = ['Scene1_centroids_rapid_nonoise_None', 'Scene1_centroids_rapid_nonoise_fixed', 'Scene1_centroids_rapid_nonoise_frac',
                        'Scene1_centroids_rapid_real_None', 'Scene1_centroids_rapid_real_fixed', 'Scene1_centroids_rapid_real_frac',
                        'Scene1_centroids_slow_nonoise_None', 'Scene1_centroids_slow_nonoise_fixed', 'Scene1_centroids_slow_nonoise_frac',
                        'Scene1_centroids_slow_real_None', 'Scene1_centroids_slow_real_fixed', 'Scene1_centroids_slow_real_frac']

Scene2_centroid_names = ['Scene2_centroids_rapid_nonoise_None', 'Scene2_centroids_rapid_nonoise_fixed', 'Scene2_centroids_rapid_nonoise_frac',
                        'Scene2_centroids_rapid_real_None', 'Scene2_centroids_rapid_real_fixed', 'Scene2_centroids_rapid_real_frac',
                        'Scene2_centroids_slow_nonoise_None', 'Scene2_centroids_slow_nonoise_fixed', 'Scene2_centroids_slow_nonoise_frac',
                        'Scene2_centroids_slow_real_None', 'Scene2_centroids_slow_real_fixed', 'Scene2_centroids_slow_real_frac']

# Get true positions from Pierre's position files
main_path_infiles = "../PFforMaria/"
path2listfileScene1 = main_path_infiles+"Scene_1_AB23"
positions_file = "simuTA20150528-F140X-S50-K-AB23_positions.fits" 
shifted_positions_file = "simuTA20150528-F140X-S50-K-AB23-shifted_positions.fits"
pf = os.path.join(path2listfileScene1, positions_file)
psf = os.path.join(path2listfileScene1, shifted_positions_file)
S1star_number, S1TrueXpos, S1TrueYpos = tf.read_positionsfile(pf, detector)
_, S1ShiftTrueXpos, S1ShiftTrueYpos, S1ShiftTrueV2, S1ShiftTrueV3 = tf.read_positionsfile(psf, detector)

path2listfileScene2 = main_path_infiles+"Scene_2_AB1823"
positions_file = "simuTA20150528-F140X-S50-K-AB18to23_positions.fits"
shifted_positions_file = "simuTA20150528-F140X-S50-K-AB18to23-shifted_positions.fits"
pf = os.path.join(path2listfileScene2, positions_file)
psf = os.path.join(path2listfileScene2, shifted_positions_file)
S2star_number, S2TrueXpos, S2TrueYpos = tf.read_positionsfile(pf, detector)
_, S2ShiftTrueXpos, S2ShiftTrueYpos, S2ShiftTrueV2, S2ShiftTrueV3 = tf.read_positionsfile(psf, detector)

# Do Scenario 1
write_txt_file(Scene1_all_centroids, Scene1_centroid_names, S1TrueXpos, S1TrueYpos, save_text_file)

# Do Scenario 2
write_txt_file(Scene2_all_centroids, Scene2_centroid_names, S2TrueXpos, S2TrueYpos, save_text_file)

# Get the distribution, standard deviations, and means
name_txt_file_path = "../PFforMaria/detector_"+str(detector)+"_comparison_txt_positions"
#txt_file = name_txt_file_path+'/Scene1_centroids_rapid_nonoise_fixed.txt'
txt_files_list = glob(name_txt_file_path+'/Scene*.txt')
for txt_file in txt_files_list:
    get_distriution(txt_file, save_figs)
    #raw_input()

print ("\n Script finished!")        
