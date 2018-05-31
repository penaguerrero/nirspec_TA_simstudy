# Import necessary modules
#from __future__ import division     # Added by Maria Pena-Guerrero
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from os import path

import logging
import pywcs 

# Header
__author__ = "T. Le Blanc & M. A. Pena-Guerrero"
__version__ = "2.0"

# HISTORY
#    1. Jan 2013 - Vr. 1.0: Added initial versions of centroid and bytescl
#       Aug 2013   Vr. 1.1: Changed/added features to centroid function
#                          - Optimized computation by summing over row/column 
#                            in one go.
#                          - Modified to include a smaller centroiding 
#                            computation. An initial guess is computed over 
#                            the entire NxN input grid, and then a tighter 
#                            computation over a smaller nxn pixel grid is conducted.
#                          - Included centroiding window edge detection when 
#                            determining new computation window.
#                          - Added a debug keyword for returning degub figures 
#                            back to the main program for saving/viewing.
#                          - Updated the debug portion of scripts by adding plot 
#                            and figure titles.
#                          - Included a "verbose" flag to turn off diagnostic 
#                            output to screen.
#    2. Sep 2013 - Vr. 1.2: 
#                          - Added readimage function
#                          - Added diff_image function
#                          - Added a1600_pixtosky function
#                          - Added Rotate2D function
#                          - Added ta_transform function
#                          - Changed the print function to the Python 3.0 format
#                            for future compatibility.
#    3. Sep 2013 - Vr. 1.2.1:
#                          - Updated readimage to read from ext=1
#    4. Oct 2013 - Vr. 1.2.2:
#                          - Minor updates to doctrings for individual functions
#                          - Updated readimage function to include a ext keyword
#                            to choose which extension to load.
#    5. Nov 2013 - Vr. 1.2.3:
#                          - Added get_filenames function
#                          - Added zero_correct function
#                          - Added display_ns_image function
#                          - Added e_dist function
#    6. Dec 2013 - Vr. 1.3:
#                          - Minor update to pyfits; using astropy.io.fits instead
#                            of pyfits
#                          - Changed the readimage() function to accept a 3-frame
#                            image instead of an input file and extension. Makes it
#                            easier to use by not requiring a read and then analysis
#                            within the function.
#    7. Dec 2013 - Vr. 1.4:
#                          - New function: gen_superdark
#    8. Dec 2013 - Vr. 1.4.1:
#                          - Minor updates to Rotate2D
#    9. Jan 2014 - Vr. 1.4.2:
#                          - Minor update to centroid function. Initial search 
#                            coords introduced a bias to final centroid (by the 
#                            initial coord fraction). Updated to change init coords 
#                            to whole pixel number with np.floor().
#   10. Jan 2014 - Vr. 1.4.3:
#                          - Minor indexing error for centroiding (grid) window fixed.
#   11. Feb 2014 - Vr. 1.4.4:
#                          - Changed save input parameter to savefile in display_psf.
#                            Added the ability to pass filename (including dir 
#                            structure) for saving the resulting plot. Still defaults
#                            to False if parameter not passed.
#   12. Feb 2014 - Vr. 1.4.5:
#                          - Added the ability to make axes actual pixel values using
#                            the extent keyword. Shifts axes .5 pixel to make pixel 
#                            center origin.
#   13. Jul 2015 - Vr. 2.0: 
#                          - Code modified by Maria Pena-Guerrero for non-Notebook 
#                            environment. 
#                          - Corrected x and y width values in checkbox functions,
#                            as well as xwh and ywh values. 
#
# Utility definitions
# *********************** centroid ***********************
def centroid(grid, gw, initial=None, debug=False, verbose=False):
    """
    Perform a centroid on a given image.
    This function takes the input image grid, as well as a grid width 
    (in pixel units) over which to do a final centroiding. An initial guess 
    is executed over the entire input image, and then another centroiding is 
    iterated given the initial computed guess. The final output is a rounded 
    centroid coordinate (rounded to account for any loss of accuracy due to 
    using floors throughout computation).
    Keyword arguments:    
    grid    -- Image for which centrod is to be computed. For now, an NxN image
               is required.
    gw      -- The grid width (in pixels) over which to compute the final centroid.
    initial -- An initial guess for targeted centroiding, in the form (x, y).
               If none supplied, approximate image center is used. (default None).
    debug   -- Toggle debugging mode. Defaults to False, in which case no 
               diagnostic image is produced or passed.
    Output(s):
    centroid -- The computed centroid coordinates. The final coordinates are 
                relative to grid dimensions.
    fig      -- Diagnostic figure generated to troubleshoot the centroiding 
                calculation. Retured along with centroid if debug flag set.
    Example uses:
        Can be used in one of two ways:
        1. Debug mode:
             
            >>  center, debugfig = centroid(img, gw, debug=True)
             
             In this mode, the centroiding algorithm will estimate an initial centroid
             of img, given a centroiding window cgw. Setting debug=True produces a 
             diagnostic figure that is passed (along with the centroid) to the calling 
             script; this figure can be captured and printed for analysis.
        
        2. Centroid calculation mode:
             
            >> center = centroid(img, gw, initial=init)
             In this mode, you can calculate a targeted centroid of img, given a 
             centroiding window cgw (of a few pixels), and an initial centroid guess. 
             The result is the calculated centroid for img in the form (x, y).
    """
        
    # Set the initial guess coordinates
    if initial is None:
        if verbose: print("Initial Guess (x,y)...")
        init_guess = ((grid.shape[1]/2) - 1, (grid.shape[0]/2) - 1)
    else:
        if verbose: print("Centroid...")
        init_guess = initial

    # Ensure that initial guess coordinates are whooe pixel numbers
    init_guess = np.floor(init_guess)
    if verbose: print("(centroid) Centroid Search Position: ({:.2f}, {:.2f})".format(init_guess[0], init_guess[1]))    
    

    # ************* Centroiding *************
    
    # Define centroiding limits based on input grid width
    # specifications (grid width x/y lower and upper limits)
    gw_xl = init_guess[0] - (gw/2) + 1
    gw_xu = gw_xl + gw - 1
    gw_yl = init_guess[1] - (gw/2) + 1
    gw_yu = gw_yl + gw - 1
    
    # Check wether the new centroid window passes the edges, and if so, 
    # reset window to edges
    if gw_xl < 0:
        gw_xl = 0.
        gw_xu = gw_xl + gw - 1. 
    elif gw_xu >= grid.shape[1]:
        gw_xu = grid.shape[1] - 1.
        gw_xl = grid.shape[1] - gw 

    if gw_yl < 0:
        gw_yl = 0.
        gw_yu = gw_yl + gw - 1.
    elif gw_yu >= grid.shape[0]:
        gw_yu = grid.shape[0] - 1.
        gw_yl = grid.shape[0] - gw
        
    # Define optimized centroiding grid
    newGrid = grid[gw_yl:gw_yu+1, gw_xl:gw_xu+1]

    
    # ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ******
    if debug:
        fig = plt.figure(figsize=(8,8))
        axWeightX = plt.subplot(221)
    
        axImgScaled = plt.subplot(222)
        axImg = plt.subplot(223)
        axWeightY = plt.subplot(224)
    
        axWeightX.set_xlim(0, newGrid.shape[1]-1)
        axWeightY.set_ylim(0, newGrid.shape[0]-1)

        # Set display min/max
        if grid.shape[0] > 16:
            va, vb = 0, 1.e-4
        else:
            va, vb = 0, 0.15

        axImg.imshow(newGrid, cmap='gray', interpolation='nearest', vmin=va, vmax=vb)
        axImg.autoscale(axis='both', enable=False)
    
        axImgScaled.set_title('Centroid window: {}'.format(newGrid.shape))
        cax=axImgScaled.imshow(newGrid, cmap='jet', interpolation='nearest', vmin=va, vmax=vb)
        axImgScaled.autoscale(axis='both', enable=False)
    # ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ******
  
    
    #Diagnostic output
    if verbose: print("(centroid) Centroiding window: ({:.2f}, {:.2f}), ({:.2f}, {:.2f})".format(gw_xl, gw_yl, gw_xu, gw_yu))
    if verbose: print("(centroid) Total Pixel Count:", np.sum(newGrid))
    if verbose: print("(centroid) Average Pixel Count:", np.average(newGrid))
    if verbose: print("(centroid) Grid Shape:", newGrid.shape)

    # Weights and weighted averages
    weight_i = np.sum(newGrid, 0)
    weight_j = np.sum(newGrid, 1)
    weight_i = np.where(weight_i < 0, 0, weight_i)
    weight_j = np.where(weight_j < 0, 0, weight_j)
    
    weight_avg_i = weight_i * np.arange(len(newGrid))
    weight_avg_j = weight_j * np.arange(len(newGrid))
    
        
    # X and Y position of the centroid    
    x = np.sum(weight_avg_i)/np.sum(weight_i)
    y = np.sum(weight_avg_j)/np.sum(weight_j)
    centroid = (x+gw_xl, y+gw_yl)
    if verbose: print("(centroid) Raw X/Y and centroid: ({:.2f}, {:.2f})".format(x, y))
    # ************* Centroiding *************
  
    
    
    # ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ******
    if debug:
        temp_i = list(enumerate(weight_i))
        px, py = zip(*temp_i)
        axWeightX.plot(px,py, marker='o', mfc='blue', mec='blue', ms=4, alpha=0.5)

        temp_j = list(enumerate(weight_j))
        px, py = zip(*temp_j)
        axWeightY.plot(py,px, marker='o', mfc='blue', mec='blue', ms=4, alpha=0.5)

        temp_ai = list(enumerate(weight_avg_i))
        px, py = zip(*temp_ai)
        axWeightX.plot(px,py, marker='o', mfc='green', mec='green', ms=4, alpha=0.5)

        temp_aj = list(enumerate(weight_avg_j))
        px, py = zip(*temp_aj)
        axWeightY.plot(py,px, marker='o', mfc='green', mec='green', ms=4, alpha=0.5)
    
        axImg.plot(x, y, marker='o', mfc='red', mec='red', alpha=0.5)
        axImgScaled.plot(x, y, marker='o', mfc='red', mec='white', alpha=0.5)
    
        fig.tight_layout()
        fig.colorbar(cax, ax=axImgScaled, shrink=0.75, ticks=[va, 0, vb])
    # ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ****** DEBUG! ******
 
    
    
    if verbose: print("(centroid) New centroid: ({:.2f}, {:.2f})".format(centroid[0], centroid[1]))
    
    # If in debugging mode, return the figure as well, else just return the centroid
    if debug: return centroid, fig
    else: return centroid
# *********************** centroid ***********************


# *********************** bytescl ***********************
def bytescl(img, bottom, top):
    """
    Scale a pixel image to limits (0, 1).
    Keyword arguments:
    img    -- Original pixel image.
    bottom -- Lower limit of img.
    top    -- Upper limit of img.
    Output(s):
    scl_img -- Scaled image with new limits 0(min) - 1(max).
    """

    scl_img = (((top - bottom) * (img - img.min())) / (img.max() - img.min())) + bottom
    
    return scl_img
# *********************** bytescl ***********************


# *********************** readimage ***********************
# Extract an image from a multi-ramp integration FITS file
def readimage(master_img, debug=False):
    """
    Extract am image from a multi-ramp integration FITS file.
    Currently, JWST NIRSpec FITS images consists of a 3-ramp integration, 
    with each succesive image containing more photon counts than the next. 
    Uses a cube-differencing calculation to eliminate random measurements
    such as cosmic rays.
    Keyword arguments:
    master_img -- 3-frame image (as per NIRSpec output images)
    Output(s):
    omega -- A combined FITS image that combines all frames into one image.
    """
    
    # Read in input file, and generate the alpha and beta images
    # (for subtraction)
    #alpha = master_img[1, :, :] - master_img[0, :, :]
    #beta = master_img[2, :, :] - master_img[1, :, :]
    alpha = master_img[1] - master_img[0]
    beta = master_img[2] - master_img[1]
    #fits.writeto("/Users/pena/Documents/AptanaStudio3/NIRSpec/TargetAcquisition/alpha.fits", alpha)
    #fits.writeto("/Users/pena/Documents/AptanaStudio3/NIRSpec/TargetAcquisition/beta.fits", beta)
    
    # Generate a final image by doing a pixel to pixel check 
    # between alpha and beta images, storing lower value
    omega = np.where(alpha < beta, alpha, beta)
    
    # Convert negative pixel values to zero
    negative_idx = np.where(omega < 0.0)
    omega[negative_idx] = 0.0

    # show on screen the values of rows and column for debugging other functions of the code
    if debug:
        image = omega   # Type image to be displayed
        if image is omega:
            print ('Combined ramped images:  ')
            print ('   AFTER zeroing negative pixels')
        else:
            print ('   BEFORE zeroing negative pixels')
        print ('max_image = ', image.max())
        print ('min_image = ', image.min())
        for j in range(np.shape(image)[0]):
            print (j, image[j, :])#, alpha[j, :], beta[j, :])
    

    print('(readimage): Image processed!')
        
    # Return the extracted image
    return omega
# *********************** readimage ***********************


# *********************** diff_image ***********************
# New method for conducting image differencing
def diff_image(im1, im2):
    """
    Subtract im2 from im1, returning an image with the lowest pixel count 
    between im1 and im2.
    This function takes as input the images to be subtracted,
    and output the final subtracted image. The order in which
    the input images are input is important; the 2nd will be 
    subtracted from the first.
    
    Keyword arguments:
    im1, im2 -- The images for which differencing is to be perfomred.
    
    Output(s): 
    output_im -- Final difference image.
    """
    
    # Subtract the min of im1 and im2 for im1
    output_im = im1 - np.where(im1 < im2, im1, im2)
    
    return output_im
# *********************** diff_image ***********************


# *********************** get_filenames ***********************
def get_filenames(search_str, path):
    """ 
    Return a list of filenames in a certain criteria given
    a path.
    """

    import os, re
        
    file_list = []
    
    filenames = os.listdir(path)
    
    for ii in xrange(len(filenames)):
        step = re.search(search_str, filenames[ii])
        if step is not None:
            file_list.append(path+step.group(0))

    return file_list
# *********************** get_filenames ***********************


# *********************** zero_correct ***********************
def zero_correct(dim_over, dim_detc):
    """
    This short function calculates the correction for the change 
    in the location of the origin pixel (the very first, or "0"), 
    which is applied to the calculation of centroid computed for 
    a grid that has been downsampled.
    """
    
    factor = dim_over / dim_detc
    corr = factor / 2. - 0.5
    
    return corr/factor
# *********************** zero_correct ***********************


# *********************** display_ns_psf ***********************
def display_ns_psf(image, vlim=(), fsize=(8, 8), interp='nearest', \
    title='', cmap='gray', extent=None, savefile=None, cb=False):
    """
    Custom display a PSF generated with WEBBPSF or similar tool.
    A quick tool for displaying NIRSpec images in native size 
    (2048x2048) with additional options for display.
    Keyword arguments:
    image    --  A 2D image to display
    vlim     --  The image range (in terms of image counts) to display.
                 Defaults to empty (), displaying full spectrum.
    fsize    --  Figure image size (in cm?)
    interp   --  Interpolation type. Defaults to 'nearest'.
    title    --  Title for plot. Defaults to ''.
    cmap     --  Color map for plot. Defaults to 'gray'.
    cb       --  Color bar toggle. Defaults to 'False'.
    savefile --  Figure save toggle. Defaults to 'None'. A filename
                 (with directory structure) can be entered to save to
                 given filename.
    """

    # Display PSF (oversampled and detector levels)
    fig, ax = plt.subplots(figsize=fsize)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(0.0, np.shape(image)[0])

    if vlim == ():
        vlim = (image.min(), image.max())
    
    if extent is not None:     
        cax = ax.imshow(image, cmap=cmap, interpolation=interp, vmin=vlim[0], \
              extent=extent-.5, vmax=vlim[1])
    else:
        cax = ax.imshow(image, cmap=cmap, interpolation=interp, vmin=vlim[0], vmax=vlim[1])       
    
    if cb: fig.colorbar(cax, ax=ax, shrink=0.8)
    
    # Added by Maria to see plots when not in Notebook environment
    plt.show()
    
    if savefile is not None:
        fig.savefig(savefile)
# *********************** display_ns_psf ***********************


# *********************** e_dist ***********************
def e_dist(xa, xb):
    """
    Calculate the euclidian distance between two cartesian points
    """
    
    distance = np.sqrt((xa[0]-xb[0])**2 + (xa[1]-xb[1])**2)

    return distance
# *********************** e_dist ***********************


# *********************** gen_superdark ***********************
def gen_superdark(inlist, fname='./Superdark.fits'):
    """
    Generate a median image from a list of input image files.

    The purpose of this script is to generate a "superdark";
    a median dark multi-frame image to mimic the noise present
    in NIRSpec-read images.

    The median (average) calculation is done across the 3-frame
    images expected from NIRSpec, and is more accurate as more
    images are used.

    Keyword arguments:
    inlist -- The list of image files to be used for generating
              the median image.
    fname  -- The name of the generated meadian image. Defaults
              to 'Superdark'.
    """
    
    stacked_darks = 0
    
    # Read in the multiframe images and create a stack
    for ii in xrange(len(inlist)):
        fileBase = path.basename(inlist[ii])
        print('(gen_superdark): Reading/processing {} ...'.format(fileBase))
        
        stacked_darks = stacked_darks + fits.getdata(inlist[ii], 0)
    
    # Create a superdark frame initiazed with zeros
    superdark_frame = stacked_darks / len(inlist)
    
    # Save the above information to a FITS file for later use
    hdu = fits.PrimaryHDU()
    print('(gen_superdark): Writing {} to file ...'.format(fname))
    hdu.header.append('FILENAME', fname, '')
    hdu.writeto(fname, clobber=True)

    fits.append(fname, superdark_frame)
        
    return superdark_frame
# *********************** gen_superdark ***********************


# *********************** a1600_pixtosky ***********************
def a1600_pixtosky(angle_d, pixcrd, origin):
    """
    Convert input pixel coordinates to sky coordinates for NIRSpec 
    A1600 aperture.
    Keyword arguments:
    angle_d -- Degree angle rotation between the aperture grid and 
               the sky coordinates.
    pixcrd  -- Pixel coordinates to be converted. Can be a single 
               coordinate in the form (x_pix, y_pix), or a numpy 
               array of coordinates.
    origin  -- Origin of the target coordinate system.
    Output(s):
    world - Computed sky coordinates, rounded to nearest 1. The 
            final coordinates are relative to grid dimensions. 
            Output dimensions match those of input. 
    """
        
    # Change angle from degrees to radians
    angle_r = np.radians(angle_d)
    
    # Create a new WCS object of two axes
    wcs = pywcs.WCS(naxis=2)

    # Setup a pixel projection
    wcs.wcs.crpix = [7.5, 7.5]
    wcs.wcs.crval = [321.536442, 5.689362]
    wcs.wcs.cdelt = np.array([0.106841547, 0.108564798])
    wcs.wcs.pc = ([np.sin(angle_r),np.cos(angle_r)],
                  [np.cos(angle_r),-np.sin(angle_r)])

    # Calculate new coordinates for each pair of input coordinates
    sky_coords = wcs.wcs_pix2sky(pixcrd, origin)
    
    return sky_coords
# *********************** a1600_pixtosky ***********************


# *********************** ta_transform ***********************
def ta_transform(pnts, mirror=False):
    """
    Transforms an array of pixel positions into sky positions in a 
    2D plane.
    
    Keyword arguments:
    pnts   -- Array of pixel coordinates.
    mirror -- Flag to allow mirroring along a specific plane (default=False).
    Output(s):
    transformed_pnts - Transformed pixel coordinates, in sky coordinates.
    """

    # Function constants
    
    # Transformation derivatives (as per pixel)
    dxdx, dydx = .076839, -0.068522
    dxdy, dydy = .069291, .079071
    
    # Reference point in pixels and sky coordinates, respectively
    Xo_pix, Yo_pix = 989.204158, 1415.929442
    Xo_sky, Yo_sky = 321.536442, 5.689362
    
    
    # Split the entered tuples into seperate variables
    #print('pnts LENGTH!!! = {}'.format(len(pnts)))
    if len(pnts) > 1:
        X_pix, Y_pix = np.array(zip(*pnts))
    else:
        X_pix, Y_pix = pnts[0][0], pnts[0][1]

        
    print('(ta_tranform): X_pix {}'.format(X_pix))
    print('(ta_tranform): Y_pix {}'.format(Y_pix))
    print('(ta_tranform): type {}'.format(type(X_pix)))
    
    ## Convert angle to radians
    #ang = np.deg2rad(angle)
    
    # Calculate the coordinate transforms
    X_temp = Xo_sky + (dxdx*(X_pix-Xo_pix)) + (dxdy*(Y_pix-Yo_pix))
    Y_temp = Yo_sky + (dydx*(X_pix-Xo_pix)) + (dydy*(Y_pix-Yo_pix))
    
    
    if mirror:
        # Rotate/Mirror the tranformed coordinated
        if len(pnts) > 1:
            temp = np.array(zip(X_temp, Y_temp))
        else:
            temp = np.array((X_temp, Y_temp))
        
        cnt = np.array((321.536442, 5.689362))
        
        a, b = 0.992682, 0.120757
        transformed_pnts = np.dot(temp-cnt,np.array([[a,b],[b,-a]]))+cnt
        
        print('(ta_tranform): Mirroring at y=x included!\n')
    
    else:
        print('(ta_tranform): No mirroring included')
        if len(pnts) > 1:
            transformed_pnts = np.array(zip(X_temp, Y_temp))
        else:
            transformed_pnts = np.array((X_temp, Y_temp))
    
    return transformed_pnts
# *********************** ta_transform ***********************


# *********************** Rotate2D ***********************
def Rotate2D(pts,cnt,angle):
    """
    Rotates a set of points in 2D space around a center point.
    
    pts   -- Array of points to be rotated.
    cnt   -- Center of rotation. Coordinate in the form of (y, x) expected.
    angle -- Radian angle of rotation.
    Output(s):
    rot_pts -- Rotated coordinate pairs.
    """
    
    ang = np.deg2rad(angle)
    
    #Includes rotation and reflection along x=y
    rot_pts = np.dot(pts-cnt,np.array([[np.sin(ang),np.cos(ang)],[np.cos(ang),np.sin(-ang)]]))+cnt
    
    #rot_pts = np.dot(pts-cnt,np.array([[np.cos(ang),np.sin(ang)],[np.sin(-ang),np.cos(ang)]]))+cnt

    return rot_pts
# *********************** Rotate2D ***********************



# Print diagnostic load message
print("(tautils): TA Utilities Version {} loaded!".format(__version__))