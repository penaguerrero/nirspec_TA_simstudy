from __future__ import print_function, division
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from glob import glob
import collections

# Tommy's code
import tautils as tu
import jwst_targloc as jtl



# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Feb 2016 - Version 1.0: initial version completed


# FUNCTIONS

def get_raw_star_directory(path4starfiles, scene, shutters, noise, redo=True):
    """
    This function returns a list of the directories (positions 1 and 2) to be studied.
    # Paths to Scenes 1 and 2 local directories: /Users/pena/Documents/AptanaStudio3/NIRSpec/TargetAcquisition/PFforMaria
    path_scene1_slow = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRS/postage"
    path_scene1_slow_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRS no_noise/postage"
    path_scene1_rapid = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRSRAPID/postage"
    path_scene1_rapid_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 first NRS no_noise/postage"
    path_scene1_slow_shifted = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRS/postage"
    path_scene1_slow_shifted_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRS no_noise/postage"
    path_scene1_rapid_shifted = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRSRAPID/postage"
    path_scene1_rapid_shifted_nonoise = "Scene_1_AB23/NIRSpec_TA_Sim_AB23 shifted NRS no_noise/postage"
    path_scene2_slow = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRS/postage"
    path_scene2_slow_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRS no_noise/postage"
    path_scene2_rapid = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRSRAPID/postage"
    path_scene2_rapid_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 first NRSRAPID no_noise/postage"
    path_scene2_slow_shifted = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRS/postage"
    path_scene2_slow_shifted_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRS no_noise/postage"
    path_scene2_rapid_shifted = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRSRAPID/postage"
    path_scene2_rapid_shifted_nonoise = "Scene_2_AB1823/NIRSpec_TA_Sim_AB1823 shifted NRSRAPID no_noise/postage"
    """
    # define shutter velocity to be used
    shutter_vel = "NRS"   # for slow case
    if shutters == "rapid":
        shutter_vel = "NRSRAPID"
    # define noise level
    noise_level = " no_noise"
    if noise == "real":
        noise_level = ""
    # define directory path for scenario 1
    position1 = path4starfiles+"Scene_"+repr(scene)+"_AB23/NIRSpec_TA_Sim_AB23 first "+shutter_vel+noise_level+"/postage"
    position2 = path4starfiles+"Scene_"+repr(scene)+"_AB23/NIRSpec_TA_Sim_AB23 shifted "+shutter_vel+noise_level+"/postage"
    if scene == 2:        
        position1 = path4starfiles+"Scene_"+repr(scene)+"_AB1823/NIRSpec_TA_Sim_AB1823 first "+shutter_vel+noise_level+"/postage"
        position2 = path4starfiles+"Scene_"+repr(scene)+"_AB1823/NIRSpec_TA_Sim_AB1823 shifted "+shutter_vel+noise_level+"/postage"
    if redo:
        position1 += "_redo"
        position2 += "_redo"
    dir2test_list = [position1, position2]
    return dir2test_list    
    
    
def run_recursive_centroids(psf, background, xwidth_list, ywidth_list, checkbox_size, max_iter, 
                            threshold, determine_moments, debug, display_master_img, vlim=()):   
    """
    Determine the centroid location given the that the background is already subtracted. 
    """ 
    # Display the combined FITS image that combines all frames into one image
    if display_master_img: 
        tu.display_ns_psf(psf, vlim=vlim)
    # Test checkbox piece
    cb_centroid_list = []
    for xwidth, ywidth in zip(xwidth_list, ywidth_list):
        print ("Testing centroid width: ", checkbox_size)
        print ("     xwidth = ", xwidth, "  ywidth = ", ywidth)
        cb_cen, cb_hw = jtl.checkbox_2D(psf, checkbox_size, xwidth, ywidth, debug=debug)
        print ('Got coarse location for checkbox_size {} \n'.format(checkbox_size))
        # Checkbox center, in base 1
        print('Checkbox Output:')
        print('Checkbox center: [{}, {}]'.format(cb_cen[0], cb_cen[1]))
        print('Checkbox halfwidths: xhw: {}, yhw: {}'.format(cb_hw[0], cb_hw[1]))
        print()
        # Calculate the centroid based on the checkbox region calculated above
        cb_centroid, cb_sum = jtl.centroid_2D(psf, cb_cen, cb_hw, max_iter=max_iter, threshold=threshold, debug=debug)
        cb_centroid_list.append(cb_centroid)
        print('Final sum: ', cb_sum)
        print('cb_centroid: ', cb_centroid)
        print()
        #raw_input()
    # Find the 2nd and 3rd moments
    if determine_moments:
        x_mom, y_mom = jtl.find2D_higher_moments(psf, cb_centroid, cb_hw, cb_sum)
        print('Higher moments(2nd, 3rd):')
        print('x_moments: ', x_mom)
        print('y moments: ', y_mom)
        print('---------------------------------------------------------------')
        print()
    return cb_centroid_list


def do_Piers_correction(detector, cb_centroid_list):
    xy3, xy5, xy7 = cb_centroid_list
    xy3corr = Pier_correction(detector, xy3)
    xy5corr = Pier_correction(detector, xy5)
    xy7corr = Pier_correction(detector, xy7)
    corr_cb_centroid_list = [xy3corr, xy5corr, xy7corr]
    return corr_cb_centroid_list 
    
    
def Pier_correction(detector, XandYarr):
    """
    KEYWORD ARGUMENTS:
        Pier_corr                  -- Perform average correction suggested by Pier: True or False
        
    OUTPUT:
    cb_centroid_list               -- Values corrected for Pier's values 
    """
    # Corrections for offsets in positions (see section 2.5 of Technical Notes in Documentation directory)
    offset_491 = (-0.086, -0.077)
    offset_492 = (0.086, 0.077)
    corrected_x = XandYarr[0]
    corrected_y = XandYarr[1]
    if detector == 491:
        corrected_x = XandYarr[0] + offset_491[0]
        corrected_y = XandYarr[1] + offset_491[1]
    elif detector == 492:
        corrected_x = XandYarr[0] + offset_492[0]
        corrected_y = XandYarr[1] + offset_492[1]
    corr_XandYarr = [corrected_x, corrected_y]
    return corr_XandYarr


def centroid2fulldetector(cb_centroid_list, true_center):
    """
    Transform centroid coordinates into full detector coordinates.
    
    KEYWORD ARGUMENTS:
    cb_centroid_list           -- Checkbox based centroid determined by target acquisition (TA) algorithm in 
                                  terms of 32 by 32 pixels for checkbox sizes 3, 5, and 7
    true_center                -- Actual (true) position of star in terms of full detector  
    
    OUTPUT:
    cb_centroid_list_fulldetector  -- List of centroid locations determined with the TA algorithm in 
                                      terms of full detector. List is for positions determined with
                                      3, 5, and 7 checkbox sizes. 
    loleftcoords                   -- Coordinates of the lower left corner of the 32x32 pixel box
    true_center32x32               -- True center given in coordinates of 32x32 pix
    differences_true_TA            -- Difference of true-observed positions       
    """
        
    # Get the lower left corner coordinates in terms of full detector. We subtract 16.0 because indexing
    # from centroid function starts with 1
    corrected_x = true_center[0]
    corrected_y = true_center[1]
    loleft_x = np.floor(corrected_x) - 16.0
    loleft_y = np.floor(corrected_y) - 16.0
    loleftcoords = [loleft_x, loleft_y]
    #print(loleft_x, loleft_y)
    
    # get center in term of 32x32 checkbox
    true_center32x32 = [corrected_x-loleft_x, corrected_y-loleft_y]
    
    # Add lower left corner to centroid location to get it in terms of full detector
    cb_centroid_list_fulldetector = []
    for centroid_location in cb_centroid_list:
        centroid_fulldetector_x = centroid_location[0] + loleft_x
        centroid_fulldetector_y = centroid_location[1] + loleft_y
        centroid_fulldetector = [centroid_fulldetector_x, centroid_fulldetector_y]
        cb_centroid_list_fulldetector.append(centroid_fulldetector)
    corr_cb_centroid_list = cb_centroid_list_fulldetector
    
    # Determine difference between center locations
    differences_true_TA = []
    d3_x = true_center[0] - corr_cb_centroid_list[0][0]
    d3_y = true_center[1] - corr_cb_centroid_list[0][1]
    d3 = [d3_x, d3_y]
    if len(corr_cb_centroid_list) != 1:   # make sure this function works even for one checkbox
        d5_x = true_center[0] - corr_cb_centroid_list[1][0]
        d5_y = true_center[1] - corr_cb_centroid_list[1][1]
        d7_x = true_center[0] - corr_cb_centroid_list[2][0]
        d7_y = true_center[1] - corr_cb_centroid_list[2][1]
        d5 = [d5_x, d5_y]
        d7 = [d7_x, d7_y]
        diffs = [d3, d5, d7]
    else:
        diffs = d3
    differences_true_TA.append(diffs)
    return corr_cb_centroid_list, loleftcoords, true_center32x32, differences_true_TA


def get_mindiff(d1, d2, d3):
    """ This function determines the minimum difference from checkboxes 3, 5, and 7,
    and counts the number of repetitions.  """
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


def transform2fulldetector(detector, centroid_in_full_detector, cb_centroid_list, 
                           ESA_center, true_center, perform_avgcorr=True):
    """
    Transform centroid coordinates into full detector coordinates.
    
    Keyword arguments:
    detector                   -- Either 491 or 492
    centroid_in_full_detector  -- Resulting coordinates in terms of full detector? True or False
    cb_centroid_list           -- Checkbox based centroid determined by target acquisition (TA) algorithm in 
                                  terms of 32 by 32 pixels for checkbox sizes 3, 5, and 7
    ESA_center                 -- Centroid determined with the ESA version of TA algorithm in terms of full detector
    true_center                -- Actual (true) position of star in terms of full detector  
    perform_avgcorr            -- Add the average correction given by Pierre
    
    Output(s):
    corr_true_center_centroid      -- List of true positions either in full detector terms or 32x32 pix
    cb_centroid_list_fulldetector  -- List of centroid locations determined with the TA algorithm in 
                                      terms of full detector. List is for positions determined with
                                      3, 5, and 7 checkbox sizes. 
    loleftcoords                   -- Coordinates of the lower left corner of the 32x32 pixel box
    differences_true_TA            -- Difference of true-observed positions
    
    """
    # Corrections for offsets in positions (see section 2.5 of Technical Notes in Documentation directory)    
    corr_cb_centroid_list = []

    if perform_avgcorr:
        offset_491 = (-0.086, -0.077)
        offset_492 = (0.086, 0.077)
        
        correction = offset_491
        if detector == 492:
            correction = offset_492
        for centroid_location in cb_centroid_list:
            print (centroid_location[0], correction[0])
            corrected_cb_centroid_x = centroid_location[0] + correction[0]
            corrected_cb_centroid_y = centroid_location[1] + correction[1]
            corrected_cb_centroid = [corrected_cb_centroid_x, corrected_cb_centroid_y]
            corr_cb_centroid_list.append(corrected_cb_centroid)

    # Get the lower left corner coordinates in terms of full detector. We subtract 16.0 because indexing
    # from centroid function starts with 1
    loleft_x = np.floor(true_center[0]) - 16.0
    loleft_y = np.floor(true_center[1]) - 16.0
    loleftcoords = [loleft_x, loleft_y]
    #print(loleft_x, loleft_y)

    if centroid_in_full_detector:    
        # Add lower left corner to centroid location to get it in terms of full detector
        cb_centroid_list_fulldetector = []
        for centroid_location in cb_centroid_list:
            centroid_fulldetector_x = centroid_location[0] + loleft_x
            centroid_fulldetector_y = centroid_location[1] + loleft_y
            centroid_fulldetector = [centroid_fulldetector_x, centroid_fulldetector_y]
            cb_centroid_list_fulldetector.append(centroid_fulldetector)
        corr_cb_centroid_list = cb_centroid_list_fulldetector
        corr_true_center_centroid = true_center
    else:
        
        # Subtract lower left corner to from true center to get it in terms of 32x32 pixels
        corr_true_center_x = true_center[0] - loleft_x
        corr_true_center_y = true_center[1] - loleft_y
        true_center_centroid_32x32 = [corr_true_center_x, corr_true_center_y]
        corr_true_center_centroid = true_center_centroid_32x32

    # Determine difference between center locations
    differences_true_TA = []
    d3_x = corr_true_center_centroid[0] - corr_cb_centroid_list[0][0]
    d5_x = corr_true_center_centroid[0] - corr_cb_centroid_list[1][0]
    d7_x = corr_true_center_centroid[0] - corr_cb_centroid_list[2][0]
    d3_y = corr_true_center_centroid[1] - corr_cb_centroid_list[0][1]
    d5_y = corr_true_center_centroid[1] - corr_cb_centroid_list[1][1]
    d7_y = corr_true_center_centroid[1] - corr_cb_centroid_list[2][1]
    d3 = [d3_x, d3_y]
    d5 = [d5_x, d5_y]
    d7 = [d7_x, d7_y]
    diffs = [d3, d5, d7]
    print(corr_true_center_centroid[0], corr_cb_centroid_list[0][0])
    differences_true_TA.append(diffs)
    return corr_true_center_centroid, corr_cb_centroid_list, loleftcoords, differences_true_TA


def write2file(data2write, lines4screenandfile):
    line0, line0a, line0b = lines4screenandfile
    save_text_file, output_file, st, bg, corr_cb_centroid_list, corr_true_center_centroid, loleftcoords, factor, differences_true_TA = data2write
    line1 = "{:<5} {:<10} {:<14} {:<16} {:<14} {:<16} {:<14} {:<16} {:<12} {:<14} {:<10} {:<14} {:<10.2f} {:<20} {:<22} {:<20} {:<22} {:<20} {:<22}\n".format(
                                                    st, bg, 
                                                    corr_cb_centroid_list[0][0], corr_cb_centroid_list[0][1],
                                                    corr_cb_centroid_list[1][0], corr_cb_centroid_list[1][1],
                                                    corr_cb_centroid_list[2][0], corr_cb_centroid_list[2][1],
                                                    corr_true_center_centroid[0], corr_true_center_centroid[1],
                                                    loleftcoords[0], loleftcoords[1],
                                                    factor,
                                                    differences_true_TA[0][0][0], differences_true_TA[0][0][1],
                                                    differences_true_TA[0][1][0], differences_true_TA[0][1][1],
                                                    differences_true_TA[0][2][0], differences_true_TA[0][2][1])
    if save_text_file:
        f = open(output_file, "a")
        f.write(line1)
        f.close()
    print(line0)
    print(line0a)
    print(line0b)
    print(line1) 


def read_listfile(list_file_name, detector=None, background_method=None):    
    """ This function reads the fits table that contains the flux and converts to magnitude for the 
    simulated stars. """
    listfiledata = fits.getdata(list_file_name)
    star_number, xpos, ypos, orient, factor = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])  
    for row in listfiledata:
        #print ("row: ", row)
        star_number = np.append(star_number, row[0]) 
        xpos = np.append(xpos, row[1]) 
        ypos = np.append(ypos, row[2])
        orient = np.append(orient, row[3])
        factor = np.append(factor, row[4])
    # convert the flux into magnitude (factor=1.0 is equivalent to magnitude=23.0,
    #  and factor=100.0 is equivalent to magnitude=18.0)
    mag = -2.5*np.log10(factor) + 23.0
    #mag = 2.5*np.log10(factor) + 18.0   # --> this conversion is wrong!
    # Get the correct slices according to detector or return all if no detector was chosen
    if detector is not None:
        if detector == 491:   # slice from star 101 to 200
            star_number, xpos, ypos, factor, mag = star_number[100:], xpos[100:], ypos[100:], factor[100:], mag[100:]
        elif detector == 492:   # slice from star 1 to 100
            star_number, xpos, ypos, factor, mag = star_number[:100], xpos[:100], ypos[:100:], factor[:100], mag[:100]
    bg_method = background_method
    if background_method is None:   # convert the None value to string
        bg_method = 'None'
    return star_number, xpos, ypos, factor, mag, bg_method


def read_positionsfile(positions_file_name, detector=None):
    """ This function reads the fits table that contains the true full detector positions of all simulated stars. """
    posfiledata = fits.getdata(positions_file_name)
    star_number, xpos491, ypos491, xpos492, ypos492 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])  
    trueV2, trueV3 = np.array([]), np.array([])
    for row in posfiledata:
        star_number = np.append(star_number, row[0]) 
        xpos491 = np.append(xpos491, row[3]) 
        ypos491 = np.append(ypos491, row[4])
        xpos492 = np.append(xpos492, row[5]) 
        ypos492 = np.append(ypos492, row[6])
        trueV2 = np.append(trueV2, row[13])
        trueV3 = np.append(trueV3, row[14])
    # Get the correct slices according to detector or return all if no detector was chosen
    xpos, ypos = np.array([]), np.array([])
    if detector is not None:
        if detector == 491:   # slice from star 101 to 200
            star_number, xpos, ypos, trueV2, trueV3 = star_number[100:], xpos491[100:], ypos491[100:], trueV2[100:], trueV3[100:]
        elif detector == 492:   # slice from star 1 to 100
            star_number, xpos, ypos, trueV2, trueV3 = star_number[:100], xpos492[:100], ypos492[:100], trueV2[:100], trueV3[:100]
    else:
        # return the 200 values
        xpos = np.append(xpos, xpos492[:100])
        xpos = np.append(xpos, xpos491[100:])
        ypos = np.append(ypos, ypos492[:100])
        ypos = np.append(ypos, ypos491[100:])
    #for s, x, y in zip(star_number, xpos, ypos):
    #    print(s, x, y)
    #    raw_input()
    return star_number, xpos, ypos, trueV2, trueV3

def get_fracdata(offsets):
    """ This function gets arrays for each fractional background for the same star from the text file array
    given by numpy.loadtxt. """
    frac003x, frac005x, frac007x = [], [], []
    frac003y, frac005y, frac007y = [], [], []
    frac013x, frac015x, frac017x = [], [], []
    frac013y, frac015y, frac017y = [], [], []
    frac023x, frac025x, frac027x = [], [], []
    frac023y, frac025y, frac027y = [], [], []
    frac033x, frac035x, frac037x = [], [], []
    frac033y, frac035y, frac037y = [], [], []
    frac043x, frac045x, frac047x = [], [], []
    frac043y, frac045y, frac047y = [], [], []
    frac053x, frac055x, frac057x = [], [], []
    frac053y, frac055y, frac057y = [], [], []
    frac063x, frac065x, frac067x = [], [], []
    frac063y, frac065y, frac067y = [], [], []
    frac073x, frac075x, frac077x = [], [], []
    frac073y, frac075y, frac077y = [], [], []
    frac083x, frac085x, frac087x = [], [], []
    frac083y, frac085y, frac087y = [], [], []
    frac093x, frac095x, frac097x = [], [], []
    frac093y, frac095y, frac097y = [], [], []
    frac103x, frac105x, frac107x = [], [], []
    frac103y, frac105y, frac107y = [], [], []
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10 = 0,1,2,3,4,5,6,7,8,9,10
    for i, _ in enumerate(offsets[0]):
        #row = [offsets[0][i], offsets[1][i], offsets[2][i], offsets[3][i], offsets[4][i], offsets[5][i]]
        #row = np.array(row).reshape(1,6)
        if i == i0:
            frac003x.append(offsets[0][i])
            frac003y.append(offsets[1][i])
            frac005x.append(offsets[2][i])
            frac005y.append(offsets[3][i])
            frac007x.append(offsets[4][i])
            frac007y.append(offsets[5][i])
            i0 += 11
        if i == i1:
            frac013x.append(offsets[0][i])
            frac013y.append(offsets[1][i])
            frac015x.append(offsets[2][i])
            frac015y.append(offsets[3][i])
            frac017x.append(offsets[4][i])
            frac017y.append(offsets[5][i])
            i1 += 11
        if i == i2:
            frac023x.append(offsets[0][i])
            frac023y.append(offsets[1][i])
            frac025x.append(offsets[2][i])
            frac025y.append(offsets[3][i])
            frac027x.append(offsets[4][i])
            frac027y.append(offsets[5][i])
            i2 += 11
        if i == i3:
            frac033x.append(offsets[0][i])
            frac033y.append(offsets[1][i])
            frac035x.append(offsets[2][i])
            frac035y.append(offsets[3][i])
            frac037x.append(offsets[4][i])
            frac037y.append(offsets[5][i])
            i3 += 11
        if i == i4:
            frac043x.append(offsets[0][i])
            frac043y.append(offsets[1][i])
            frac045x.append(offsets[2][i])
            frac045y.append(offsets[3][i])
            frac047x.append(offsets[4][i])
            frac047y.append(offsets[5][i])
            i4 += 11
        if i == i5:
            frac053x.append(offsets[0][i])
            frac053y.append(offsets[1][i])
            frac055x.append(offsets[2][i])
            frac055y.append(offsets[3][i])
            frac057x.append(offsets[4][i])
            frac057y.append(offsets[5][i])
            i5 += 11
        if i == i6:
            frac063x.append(offsets[0][i])
            frac063y.append(offsets[1][i])
            frac065x.append(offsets[2][i])
            frac065y.append(offsets[3][i])
            frac067x.append(offsets[4][i])
            frac067y.append(offsets[5][i])
            i6 += 11
        if i == i7:
            frac073x.append(offsets[0][i])
            frac073y.append(offsets[1][i])
            frac075x.append(offsets[2][i])
            frac075y.append(offsets[3][i])
            frac077x.append(offsets[4][i])
            frac077y.append(offsets[5][i])
            i7 += 11
        if i == i8:
            frac083x.append(offsets[0][i])
            frac083y.append(offsets[1][i])
            frac085x.append(offsets[2][i])
            frac085y.append(offsets[3][i])
            frac087x.append(offsets[4][i])
            frac087y.append(offsets[5][i])
            i8 += 11
        if i == i9:
            frac093x.append(offsets[0][i])
            frac093y.append(offsets[1][i])
            frac095x.append(offsets[2][i])
            frac095y.append(offsets[3][i])
            frac097x.append(offsets[4][i])
            frac097y.append(offsets[5][i])
            i9 += 11
        if i == i10:
            frac103x.append(offsets[0][i])
            frac103y.append(offsets[1][i])
            frac105x.append(offsets[2][i])
            frac105y.append(offsets[3][i])
            frac107x.append(offsets[4][i])
            frac107y.append(offsets[5][i])
            i10 += 11
    frac00 = np.array([frac003x, frac003y, frac005x, frac005y, frac007x, frac007y])
    frac01 = np.array([frac013x, frac013y, frac015x, frac015y, frac017x, frac017y])
    frac02 = np.array([frac023x, frac023y, frac025x, frac025y, frac027x, frac027y])
    frac03 = np.array([frac033x, frac033y, frac035x, frac035y, frac037x, frac037y])
    frac04 = np.array([frac043x, frac043y, frac045x, frac045y, frac047x, frac047y])
    frac05 = np.array([frac053x, frac053y, frac055x, frac055y, frac057x, frac057y])
    frac06 = np.array([frac063x, frac063y, frac065x, frac065y, frac067x, frac067y])
    frac07 = np.array([frac073x, frac073y, frac075x, frac075y, frac077x, frac077y])
    frac08 = np.array([frac083x, frac083y, frac085x, frac085y, frac087x, frac087y])
    frac09 = np.array([frac093x, frac093y, frac095x, frac095y, frac097x, frac097y])
    frac10 = np.array([frac103x, frac103y, frac105x, frac105y, frac107x, frac107y])
    return frac00, frac01, frac02, frac03, frac04, frac05, frac06, frac07, frac08, frac09, frac10


def find_std(arr):
    """ This function determines the standard deviation of the given array. """
    N = float(len(arr))
    mean = sum(arr) / N
    diff2meansq_list = []
    for a in arr:
        diff = a - mean
        diffsq = diff * diff
        diff2meansq_list.append(diffsq)
    std = (1.0 / N * sum(diff2meansq_list)) ** 0.5
    #print ('sigma = ', std, '    mean = ', mean)
    return std, mean


def get_frac_stdevs(frac_data):
    sig3, mean3 = [], []
    sig5, mean5 = [], []
    sig7, mean7 = [], []
    for f in frac_data:
        s3, m3 = find_std(f[1])
        s5, m5 = find_std(f[3])
        s7, m7 = find_std(f[5])
        sig3.append(s3)
        sig5.append(s5)
        sig7.append(s7)
        mean3.append(m3)
        mean5.append(m5)
        mean7.append(m7)
    return sig3, mean3, sig5, mean5, sig7, mean7


def display_centroids(detector, st, case, psf, corr_true_center_centroid, 
                      corr_cb_centroid_list, show_disp, vlims=None, savefile=False, 
                      fig_name=None, redos=False):  
    if isinstance(st, int): 
        fig_title = "star_"+str(st)+"_"+case
    else:
        fig_title = st
    # Display both centroids for comparison.
    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(fig_title)
    ax.autoscale(enable=False, axis='both')
    ax.imshow(psf, cmap='gray', interpolation='nearest')
    ax.set_ylim(1.0, np.shape(psf)[0])
    ax.set_xlim(1.0, np.shape(psf)[1])
    ax.plot(corr_cb_centroid_list[0][0], corr_cb_centroid_list[0][1], marker='*', ms=20, mec='black', mfc='blue', ls='', label='Checkbox=3')
    if len(corr_cb_centroid_list) != 1:
        ax.plot(corr_cb_centroid_list[1][0], corr_cb_centroid_list[1][1], marker='*', ms=17, mec='black', mfc='green', ls='', label='Checkbox=5')
        ax.plot(corr_cb_centroid_list[2][0], corr_cb_centroid_list[2][1], marker='*', ms=15, mec='black', mfc='red', ls='', label='Checkbox=7')
        ax.plot(corr_true_center_centroid[0], corr_true_center_centroid[1], marker='o', ms=8, mec='black', mfc='yellow', ls='', label='True Centroid')
    # Shrink current axis by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc='upper right', bbox_to_anchor=(1.26, 1.0), prop={"size":"small"})   # put legend out of the plot box   
    # Add plot with different sensitivity limits
    if vlims is None:
        vlims = (1, 10)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(fig_title)
    ax.autoscale(enable=False, axis='both')
    ax.imshow(psf, cmap='gray', interpolation='nearest')
    ax.set_ylim(1.0, np.shape(psf)[0])
    ax.set_xlim(1.0, np.shape(psf)[1])
    ax.plot(corr_cb_centroid_list[0][0], corr_cb_centroid_list[0][1], marker='*', ms=20, mec='black', mfc='blue', ls='', label='Checkbox=3')
    if len(corr_cb_centroid_list) != 1:
        ax.plot(corr_cb_centroid_list[1][0], corr_cb_centroid_list[1][1], marker='*', ms=17, mec='black', mfc='green', ls='', label='Checkbox=5')
        ax.plot(corr_cb_centroid_list[2][0], corr_cb_centroid_list[2][1], marker='*', ms=15, mec='black', mfc='red', ls='', label='Checkbox=7')
        ax.plot(corr_true_center_centroid[0], corr_true_center_centroid[1], marker='o', ms=8, mec='black', mfc='yellow', ls='', label='True Centroid')
    # Shrink current axis by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc='upper right', bbox_to_anchor=(1.26, 1.0), prop={"size":"small"})   # put legend out of the plot box   
    ax.imshow(psf, cmap='gray', interpolation='nearest', vmin=vlims[0], vmax=vlims[1])
    if show_disp:
        plt.show()
    else:
        plt.close('all')
    if savefile:
        path4fig = "../PFforMaria/detector_"+str(detector)+"_centroid_figs"
        if "scene1" in fig_title:
            if "slow" in fig_title:
                if "real" in fig_title:
                    in_dir = "Scene1_slow_real"
                else:
                    in_dir = "Scene1_slow_nonoise"
            elif "rapid" in fig_title:
                if "real" in fig_title:
                    in_dir = "Scene1_rapid_real"
                else:
                    in_dir = "Scene1_rapid_nonoise"
        if "scene2" in fig_title:
            if "slow" in fig_title:
                if "real" in fig_title:
                    in_dir = "Scene2_slow_real"
                else:
                    in_dir = "Scene2_slow_nonoise"
            elif "rapid" in fig_title:
                if "real" in fig_title:
                    in_dir = "Scene2_rapid_real"
                else:
                    in_dir = "Scene2_rapid_nonoise"
        if fig_name is None:
            fig_name = path4fig+in_dir+"/"+fig_title+".jpg"
        if redos:
            fig_name = path4fig+"_redo/"+in_dir+"_redo/"+fig_title+"_redo.jpg"
        fig.savefig(fig_name)
        print ("Figure ", fig_name, " was saved!")
    
    
def get_cutouts(ref_star_position, detector_x, detector_y):
    """
    This function does the cutouts for the reference star position.
    Inputs:
        - ref_star_position = [true_x_center, true_y_center]
        - detector_x = array of x detector values
        - detector_y = array of y detector values
    """
    xc, yc = ref_star_position[0], ref_star_position[1]
    # Get the box coordinates in terms of full detector (subtract 16.0 because indexing
    # from centroid function starts with 1) 
    lo_x = np.floor(xc) - 16.0
    lo_y = np.floor(yc) - 16.0
    up_x = lo_x + 32.0
    up_y = lo_y + 32.0
    cutout_x = detector_x[(detector_x >= lo_x) & (detector_x <= up_x)]
    cutout_y = detector_y[(detector_y >= lo_y) & (detector_y <= up_y)]
    print (len(cutout_x), len(cutout_y))
    
    
def Nsigma_rejection(N, x, y, max_iterations=10):
    """ This function will reject any residuals that are not within N*sigma in EITHER coordinate. 
        Input: 
                 - x and y must be the arrays of the differences with respect to true values: True-Measured 
                 - N is the factor (integer or float) by which sigma will be multiplied
                 - max_iterations is the maximum integer allowed iterations 
        Output:
                 - sigma_x = the standard deviation of the new array x 
                 - mean_x  = the mean of the new array x 
                 - sigma_y = the standard deviation of the new array y
                 - mean_y  = the mean of the new array y
                 - x_new   = the new array x (with rejections) 
                 - y_new   = the new array y (with rejections) 
                 - niter   = the number of iterations to reach a convergence (no more rejections)
        Usage:
             import testing_finctions as tf
             sigma_x, mean_x, sigma_y, mean_y, x_new, y_new, niter = tf.Nsigma_rejection(N, x, y, max_iterations=10)
    """
    N = float(N)
    or_sigma_x, or_mean_x = find_std(x)
    or_sigma_y, or_mean_y = find_std(y)
    x_new = copy.deepcopy(x)
    y_new = copy.deepcopy(y)
    original_diffs = copy.deepcopy(x)

    for nit in range(max_iterations):
        # Determine the standard deviation for each array
        sigma_x, mean_x = find_std(x_new)
        sigma_y, mean_y = find_std(y_new)
        thres_x = N*sigma_x
        thres_y = N*sigma_y
        xdiff = np.abs(x_new - mean_x) 
        ydiff = np.abs(y_new - mean_y)
        xn = x_new[(np.where((xdiff<=thres_x) & (ydiff<=thres_y)))]
        yn = y_new[(np.where((xdiff<=thres_x) & (ydiff<=thres_y)))]
        if len(xn) == len(x_new): 
            niter = nit
            break   # exit the loop since no additional rejections on this iteration
        else:
            x_new, y_new = xn, yn
            niter = nit
    line0 = "N-sigma rejection function:  values calculated from differences"
    line1 = "                             - stopped at {} iterations".format(niter)
    line2 = "                             - arrays have {} elements left out of {} initial".format(len(x_new), len(x))
    line3 = "                             - original sigma and mean in X = {}  {}".format(or_sigma_x, or_mean_x)
    line4 = "                             - original sigma and mean in Y = {}  {}".format(or_sigma_y, or_mean_y)
    line5 = "                             - new sigma and mean in X = {}  {}".format(sigma_x, mean_x)
    line6 = "                             - new sigma and mean in Y {}  {}".format(sigma_y, mean_y)
    lines2print = [line0, line1, line2, line3, line4, line5, line6]
    print (line0)
    print (line1)
    print (line2)
    print (line3)
    print (line4)
    print (line5)
    print (line6)
    # find what elements got rejected            
    rejected_elements_idx = []
    for i, centroid in enumerate(original_diffs):
        if centroid not in x:
            rejected_elements_idx.append(i)
    return sigma_x, mean_x, sigma_y, mean_y, x_new, y_new, niter, lines2print, rejected_elements_idx


def read_star_param_files(test_case, detector=None, path4starfiles=None, paths_list=None):
    """ This function reads the corresponding star parameters file and returns the data for P1 and P2. """
    cases_list = ["Scene1_slow_real", "Scene1_slow_nonoise", "Scene1_rapid_real", "Scene1_rapid_nonoise",
                  "Scene2_slow_real", "Scene2_slow_nonoise", "Scene2_rapid_real", "Scene2_rapid_nonoise"]
    if path4starfiles is not None and paths_list is not None:
        bench_dirs = [[path4starfiles+paths_list[0], path4starfiles+paths_list[4]],
                      [path4starfiles+paths_list[1], path4starfiles+paths_list[5]],
                      [path4starfiles+paths_list[2], path4starfiles+paths_list[6]],
                      [path4starfiles+paths_list[3], path4starfiles+paths_list[7]],
                      [path4starfiles+paths_list[8], path4starfiles+paths_list[12]],
                      [path4starfiles+paths_list[9], path4starfiles+paths_list[13]],
                      [path4starfiles+paths_list[10], path4starfiles+paths_list[14]],
                      [path4starfiles+paths_list[11], path4starfiles+paths_list[15]]]
        for i, case in enumerate(cases_list):
            if case in test_case:
                dirs4test = bench_dirs[i]
            
        """
        *** WE ARE NOT USING THIS PART RIGHT NOW BECAUSE THE star_parameters FILES HAVE THE SAME DATA FOR 
        BOTH DETECTORS.
        #    xL:  x-coordinate of the left edge of the postge stamp in the full image (range 0-2047)
        #    xR: x-coord of right edge of the postage stamp
        #    yL: y-coord of the lower edge of the postage stamp
        #    yU:  y-coord of the upper edge of the postage stamp
        
        # Load parameters of Position 1
        star_param_txt = os.path.join(dirs4test[0],"star parameters.txt")
        if detector == 492:
            star_param_txt = os.path.join(dirs4test[0],"star parameters_492.txt")
        benchmark_dataP1 = np.loadtxt(star_param_txt, skiprows=3, unpack=True)
        # benchmark_data is list of: bench_star, quadrant, star_in_quad, x_491, y_491, x_492, y_492, V2, V3, xL, xR, yL, yU 
        bench_starP1, _, _, x_491P1, y_491P1, x_492P1, y_492P1, V2P1, V3P1, xLP1, _, yLP1, _ = benchmark_dataP1
        if detector == 491:
            bench_P1 = [bench_starP1, x_491P1, y_491P1, V2P1, V3P1, xLP1, yLP1]
        elif detector == 492:
            bench_P1 = [bench_starP1, x_492P1, y_492P1, V2P1, V3P1, xLP1, yLP1]        
        # Load parameters of Position 2
        star_param_txt = os.path.join(dirs4test[1],"star parameters.txt")
        if detector == 492:
            star_param_txt = os.path.join(dirs4test[1],"star parameters_492.txt")
        benchmark_dataP2 = np.loadtxt(star_param_txt, skiprows=3, unpack=True)
        #bench_star, quadrant, star_in_quad, x_491, y_491, x_492, y_492, V2, V3, xL, xR, yL, yU = benchmark_data
        bench_starP2, _, _, x_491P2, y_491P2, x_492P2, y_492P2, V2P2, V3P2, xLP2, _, yLP2, _ = benchmark_dataP2
        if detector == 491:
            bench_P2 = [bench_starP2, x_491P2, y_491P2, V2P2, V3P2, xLP2, yLP2]
        elif detector == 492:
            bench_P2 = [bench_starP2, x_492P2, y_492P2, V2P2, V3P2, xLP2, yLP2]        
        """
    #else:
    # Read fits table with benchmark data
    main_path_infiles = "../PFforMaria/"
    S1path2listfile = main_path_infiles+"Scene_1_AB23"
    S1list_file1 = "simuTA20150528-F140X-S50-K-AB23.list"
    S1positions_file1 = "simuTA20150528-F140X-S50-K-AB23_positions.fits" 
    S1list_file2 = "simuTA20150528-F140X-S50-K-AB23-shifted.list"
    S1positions_file2 = "simuTA20150528-F140X-S50-K-AB23-shifted_positions.fits"
    S2path2listfile = main_path_infiles+"Scene_2_AB1823"
    S2list_file1 = "simuTA20150528-F140X-S50-K-AB18to23.list"
    S2positions_file1 = "simuTA20150528-F140X-S50-K-AB18to23_positions.fits"
    S2list_file2 = "simuTA20150528-F140X-S50-K-AB18to23-shifted.list"
    S2positions_file2 = "simuTA20150528-F140X-S50-K-AB18to23-shifted_positions.fits"
    if "Scene1" in test_case:
        benchmark_data, magnitudes = read_TruePosFromFits(S1path2listfile, S1list_file1, S1positions_file1, S1list_file2, S1positions_file2)
    if "Scene2" in test_case:
        benchmark_data, magnitudes = read_TruePosFromFits(S2path2listfile, S2list_file1, S2positions_file1, S2list_file2, S2positions_file2)
    return benchmark_data, magnitudes

def read_TruePosFromFits(path2listfile, list_file1, positions_file1, list_file2, positions_file2, test_case=None, detector=None):
    # Read the text file just written to get the offsets from the "real" positions of the fake stars
    lf1 = os.path.join(path2listfile, list_file1)
    pf1 = os.path.join(path2listfile, positions_file1)
    lf2 = os.path.join(path2listfile, list_file2)
    pf2 = os.path.join(path2listfile, positions_file2)
    if test_case is not None:
        if "None" in test_case:
            background_method = None
        elif "fix" in test_case:
            background_method = "fix"
        elif "frac" in test_case:
            background_method = "frac"
    else:
        background_method = None
    bench_starP1, xpos_arcsecP1, ypos_arcsecP1, factorP1, magP1, bg_methodP1 = read_listfile(lf1, detector, background_method)
    _, true_xP1, true_yP1, trueV2P1, trueV3P1 = read_positionsfile(pf1, detector)
    bench_starP2, xpos_arcsecP2, ypos_arcsecP2, factorP2, magP2, bg_methodP2 = read_listfile(lf2, detector, background_method)
    _, true_xP2, true_yP2, trueV2P2, trueV3P2 = read_positionsfile(pf2, detector)
    # Get the lower left corner coordinates in terms of full detector. We subtract 15.0 because indexing
    # starts with 0
    xLP1 = np.floor(true_xP1) - 16.0
    yLP1 = np.floor(true_yP1) - 16.0
    xLP2 = np.floor(true_xP2) - 16.0
    yLP2 = np.floor(true_yP2) - 16.0
    #for i, _ in enumerate(trueV2P1):
    #    print(bench_starP1[i], true_xP1[i], true_yP1[i], trueV2P1[i], trueV3P1[i], xLP1[i], yLP1[i])
    #    print(bench_starP2[i], true_xP2[i], true_yP2[i], trueV2P2[i], trueV3P2[i], xLP2[i], yLP2[i])
        #raw_input()
    # Organize elements of positions 1 and 2
    bench_P1 = [bench_starP1, true_xP1, true_yP1, trueV2P1, trueV3P1, xLP1, yLP1]
    bench_P2 = [bench_starP2, true_xP2, true_yP2, trueV2P2, trueV3P2, xLP2, yLP2]
    benchmark_data = [bench_P1, bench_P2]
    return benchmark_data, magP1


def compare2ref(case, bench_stars, benchV2, benchV3, stars, V2in, V3in):
    """ This function obtains the differences of the input arrays with the reference or benchmark data. """
    # calculate the differences with respect to the benchmark data
    if len(stars) == len(bench_stars):   # for the fixed and None background case
        diffV2 = (benchV2 - V2in)
        diffV3 = (benchV3 - V3in)
        bench_V2_list = benchV2.tolist()
        bench_V3_list = benchV3.tolist()
    else:                               # for the fractional background case
        bench_V2_list, bench_V3_list = [], []
        diffV2, diffV3 = [], []
        for i, s in enumerate(stars):
            if s in bench_stars:
                j = bench_stars.tolist().index(s)
                dsV2 = (benchV2[j] - V2in[i])
                dsV3 = (benchV3[j] - V3in[i])
                diffV2.append(dsV2)
                diffV3.append(dsV3)
                bench_V2_list.append(benchV2[j])
                bench_V3_list.append(benchV3[j])
        diffV2 = np.array(diffV2)
        diffV3 = np.array(diffV3)
    return diffV2, diffV3, bench_V2_list, bench_V3_list


def convert2fulldetector(detector, stars, P1P2data, bench_stars, benchmark_xLyL_P1, benchmark_xLyL_P2, Pier_corr=False):
    """ This function simply converts from 32x32 pixel to full detector coordinates according to 
    background method - lengths are different for the fractional case. """
    x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27 = P1P2data
    benchxL_P1, benchyL_P1 = benchmark_xLyL_P1
    benchxL_P2, benchyL_P2 = benchmark_xLyL_P2
    if len(stars) == len(bench_stars):   # for the fixed and None background case
        x13, y13 = x13+benchxL_P1, y13+benchyL_P1
        x23, y23 = x23+benchxL_P2, y23+benchyL_P2
        x15, y15 = x15+benchxL_P1, y15+benchyL_P1
        x25, y25 = x25+benchxL_P2, y25+benchyL_P2
        x17, y17 = x17+benchxL_P1, y17+benchyL_P1
        x27, y27 = x27+benchxL_P2, y27+benchyL_P2
    else:                               # for the fractional background case
        for i, s in enumerate(stars):
            if s in bench_stars:
                j = bench_stars.tolist().index(s)
                x13[i], y13[i] = x13[i]+benchxL_P1[j], y13[i]+benchyL_P1[j]
                x23[i], y23[i] = x23[i]+benchxL_P2[j], y23[i]+benchyL_P2[j]
                x15[i], y15[i] = x15[i]+benchxL_P1[j], y15[i]+benchyL_P1[j]
                x25[i], y25[i] = x25[i]+benchxL_P2[j], y25[i]+benchyL_P2[j]
                x17[i], y17[i] = x17[i]+benchxL_P1[j], y17[i]+benchyL_P1[j]
                x27[i], y27[i] = x27[i]+benchxL_P2[j], y27[i]+benchyL_P2[j]
    # Include Pier's corrections
    x_corr = 0.086
    y_corr = 0.077
    if detector == 491:
        x13 -= x_corr
        x15 -= x_corr
        x17 -= x_corr
        y13 -= y_corr
        y15 -= y_corr
        y17 -= y_corr
        x23 -= x_corr
        x25 -= x_corr
        x27 -= x_corr
        y23 -= y_corr
        y25 -= y_corr
        y27 -= y_corr
    elif detector == 492:
        x13 += x_corr
        x15 += x_corr
        x17 += x_corr
        y13 += y_corr
        y15 += y_corr
        y17 += y_corr
        x23 += x_corr
        x25 += x_corr
        x27 += x_corr
        y23 += y_corr
        y25 += y_corr
        y27 += y_corr
    return x13,y13, x23,y23, x15,y15, x25,y25, x17,y17, x27,y27
        
    

# Print diagnostic load message
print("(testing_functions): testing functions script Version {} loaded!".format(__version__))

