# Import necessary modules
from __future__ import division     
from __future__ import print_function
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Tommy's code
import tautils as tu
import jwst_targloc as jtl




# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Feb 2016 - Version 1.0: initial version completed


"""

This script tests selected synthetic data from Pierre Ferruit in order to compare it to the centroids
calulated by him and by IDL code written by Tony Keyes.

"""

if __name__ == '__main__':
    # Benchmark values:
    star101_pierre = [16.865, 16.857]
    star111_pierre = [16.262, 16.697]
    star_BOTA_2 = [16.816, 15.196]
    star_BOTA_3 = [18.816, 12.196]


    """ Running Tommy's code """
    #star = '../Tommys_testData/WEBB_Test_00_7.fits'
    # Read input image: 3-ram integration master image
    star = '../PierreFerruitData/TA_cutouts/postageout_star_     101 quad_       3 quad_star        1.fits'
    #star = '../PierreFerruitData/TA_cutouts/postageout_star_     111 quad_       3 quad_star       11.fits'
    #star = '../BOTA_testData/c491_nid8543_TA_2_BOTA.fits'
    #star = '../BOTA_testData/c491_nid8543_TA_3_BOTA.fits'

    # Set parameters
    checkbox_size = 3
    xwidth, ywidth = 5, 5
    max_iter = 3
    threshold = 0.3
    debug = False

    #img = fits.open(star)
    #img.info()
    #hdr = img[0].header
    #for item in hdr:
    #    print (item)
    #master_img = img[0].data
    if 'Tommy' not in star:
        master_img = fits.getdata(star, 0)
        print ('Master image shape: ', np.shape(master_img))
        #tu.display_ns_psf(master_img[2,:,:])   # to show the individual images

    # Obtain and display the combined FITS image that combines all frames into one image.
    if 'Tommy' not in star:
        psf = tu.readimage(master_img)
        tu.display_ns_psf(psf)
    else:
        psf = fits.getdata(star, 1)
        # Create a 32x32 blank image to place the 16x16 psf...
        full_psf = np.zeros([32,32])
        #tu.display_ns_psf(full_psf, vlim=(0.001, 0.01))
        # ...and place psf into blank for the synthetic 32x32 cutout
        full_psf[8:24, 8:24] = psf
        tu.display_ns_psf(full_psf, vlim=(0.001, 0.01))
        psf = full_psf

    # Test checkbox piece
    cb_cen, cb_hw = jtl.checkbox_2D(psf, checkbox_size, xwidth, ywidth, debug=debug)
    print('Got coarse location. \n')
    #raw_input()

    # Checkbox center, in base 1
    print('Checkbox Output:')
    print('Checkbox center: [{}, {}]'.format(cb_cen[0], cb_cen[1]))
    print('Checkbox halfwidths: xhw: {}, yhw: {}'.format(cb_hw[0], cb_hw[1]))
    print()

    # Now calculate the centroid based on the checkbox region calculated above
    cb_centroid, cb_sum = jtl.centroid_2D(psf, cb_cen, cb_hw, max_iter=max_iter, threshold=threshold, debug=debug)
    print('Final sum: ', cb_sum)
    print('cb_centroid: ', cb_centroid)
    print()

    # Compare with Pierre's results
    if '101' in star:
        diff = np.abs(cb_centroid - star101_pierre)
        print ("Difference from Pierre's values for star 101: ", diff)
    if '111' in star:
        diff = np.abs(cb_centroid - star111_pierre)
        print ("Difference from Pierre's values for star 111: ", diff)

    # Compare with Bob's results
    if '2_BOTA' in star:
        diff = np.abs(cb_centroid - star_BOTA_2)
        print ("Difference from Bob's values for star BOTA_2: ", diff)
    if '3_BOTA' in star:
        diff = np.abs(cb_centroid - star_BOTA_3)
        print ("Difference from Bob's values for star BOTA_3: ", diff)
