""" 1. MODULES """
# Import necessary modules
from __future__ import division     # Added by Maria Pena-Guerrero
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from __future__ import print_function

import tautils as tu
import jwst_targloc as jtl

#################################################################### Added by Maria Pena-Guerrero

''' Options to run all or individual parts of the script. '''

# Run only part 3-Testing Ground: checkbox with 2D and 1D algorithms
run_part3 = False

# Run only part 4-Recursive Testing
run_part4 = True

# Run all the script:
run_all = False

#################################################################### Added by Maria Pena-Guerrero

''' 2. FUNCTIONS '''
def checkbox_2D(image, checkbox, debug=False):
    """
    Find the course location of an input psf by finding the 
    brightest checkbox.
    
    This function uses a 2 dimensional image as input, and
    finds the the brightest checkbox of given size in the
    image.
    
    Keyword arguments:
    image    -- 2 dimensional psf image
    checkbox -- A sliding partial filter that equal the sum
                of values in an n x n region centered on the
                current pixel, where n is an odd integer.
    
    Output(s):
    checkbox_ctr -- A tuple containing the brightest checkbox 
                    location.
    checkbox_hfw -- A tuple containing the checkbox halfwidth.
    
    Example usage:
    
        >> cb_cen, cb_hw = checkbox_2D(psf, 5)
        
        Find the location of the brightest checkbox, given a
        checkbox size of 5. Returns the brightest checkbox 
        center and halfwidths.
    """
    
    # Calculate the checkbox half-width
    chw = (checkbox - 1) / 2
    
    # Calculate the image size
    xsize, ysize = image.shape[1], image.shape[0]
    
    # Calculate the x and y widths of checkbox region
    xwidth, ywidth = xsize - checkbox + 1, ysize - checkbox + 1
    
    # If the checkbox size is not equal to both the X and Y sizes, 
    # find the pixel with the brightest checkbox
    if checkbox != xsize and checkbox != ysize:
        xpeak = 0
        ypeak = 0
        sumpeak = 0
        for ii in xrange(xsize - checkbox):
            for jj in xrange(ysize - checkbox):
                t = np.sum(image[jj:jj+checkbox, ii:ii+checkbox])
                if t > sumpeak:
                    xpeak = ii + chw + 1
                    ypeak = jj + chw + 1
                    sumpeak = t
        
        print('(checkbox_2D): Checkbox not equal to both x/ysize.')
        print()        

    
    # If the checkbox size is equal to both the X and Y sizes
    if checkbox == xsize and checkbox == ysize:
        xpeak = xsize / 2
        ypeak = ysize / 2
        sumpeak = np.sum(image, axis=None)
        
        print('(checkbox_2D): Checkbox equal to x/ysize.')
        print()
        
    # Print calculated checkbox center, and sum within checkbox centroid

    # Find the checkbox region half-width in x and y
    xhw = xwidth / 2
    yhw = ywidth / 2
        
    if xpeak < xhw or xpeak > xsize - xhw or ypeak < yhw or ypeak > ysize - yhw:
        print('(checkbox_2D): WARNING - Peak too close to edge of image.')
        print()
        
#    NOTE: Use this section of the input image is a subset of a larger image
#          Not currently needed for this analysis
#    # Determine the center of the brightest checkbox, in extracted
#    # image coordinates
#    xpeak = xpeak + xhw
#    ypeak = ypeak + yhw

    # Debug messages
    if debug:
        print('(checkbox_2D): chw = ', chw)
        print('(checkbox_2D): xsize, ysize = {}, {}'.format(xsize, ysize))
        print('(checkbox_2D): xwidth, ywidth = {}, {}'.format(xwidth, ywidth))
        print('(checkbox_2D): xpeak, ypeak = {}, {}'.format(xpeak, ypeak))
        print('(checkbox_2D): sumpeak = ', sumpeak)
        print('(checkbox_2D): xhw, yhw = {}, {}'.format(xhw, yhw))
        print()
        
    checkbox_ctr = np.array((xpeak, ypeak))
    checkbox_hfw = np.array((xhw, yhw))

    return checkbox_ctr, checkbox_hfw

def checkbox_1D(image, checkbox, debug=False):
    """
    Find the course location of an flattened input psf by 
    finding the brightest checkbox.
    
    This function uses an image as input, flattens it into a 
    vector and finds the the brightest checkbox of given size 
    in the image.
    
    Keyword arguments:
    image    -- 2 dimensional psf image
    checkbox -- A sliding partial filter that equal the sum
                of values in an n x n region centered on the
                current pixel, where n is an odd integer.
    
    Output(s):
    xpeak -- The brightest checkbox location.
    xhw   -- Checkbox halfwidth.

    
    Example usage:
    
        >> cb_center, cb_hw = checkbox_1D(vector, 5)
        
        Find the location of the brightest checkbox in a vector, 
        given a checkbox size of 5. Returns the brightest checkbox 
        center and halfwidth.
    """
    
    # Collapse input image, currently onto X axis
    # Reshape to reflect collapse onto x axis
    vector = np.sum(image, axis=0)
    print('(checkbox_1D): Image collapsed into 1D vector.')
    print()
    
    # Calculate the checkbox half-width
    chw = (checkbox - 1) / 2

    
    # Calculate the image size
    xsize, ysize = image.shape[1], image.shape[0]
    
    # Calculate the x and y widths of checkbox region
    xwidth = xsize - checkbox + 1

    # If the checkbox size is not equal to both the X and Y sizes, 
    # find the pixel with the brightest checkbox
    if checkbox != xsize and checkbox != ysize:
        xpeak = 0
        ypeak = 1
        sumpeak = 0
        for ii in xrange(xsize - checkbox):
            t = np.sum(vector[ii:ii+checkbox])
            if t > sumpeak:
                xpeak = ii + 1
                sumpeak = t

        print('(checkbox_1D): Checkbox not equal to xsize.')
                
                
    # If the checkbox size is equal to both the X and Y sizes
    if checkbox == xsize:
        xpeak = xsize / 2
        sumpeak = np.sum(vector, axis=None)
        
        print('(checkbox_1D): Checkbox equal to xsize.')
        
    # Print checkbox center and peak around centroid region

    # Find the checkbox region half-width in x and y
    xhw = xwidth / 2
        
    if xpeak < xhw or xpeak > xsize - xhw:
        print('(checkbox_1D): WARNING - Peak too close to edge of image.')
        
    
    # Debug messages
    if debug:
        print('(checkbox_1D): chw = ', chw)
        print('(checkbox_1D): xhw = ', xhw)
        print('(checkbox_1D): xsize = ', xsize)
        print('(checkbox_1D): xwidth = ', xwidth)
        print('(checkbox_1D): xpeak = ', xpeak)
        print('(checkbox_1D): sumpeak = ', sumpeak)
        print()    
        
#    NOTE: Use this section of the input image is a subset of a larger image
#          Not currently needed for this analysis
#    # Determine the center of the brightest checkbox, in extracted
#    # image coordinates
#    xpeak = xpeak + xhw
    
    return xpeak, xhw
def centroid_2D(image, checkbox_center, checkbox_halfwidth, max_iter=0, threshold=0, debug=False):
    """
    Fine location of the target by calculating the centroid for 
    the region centered on the brightest checkbox.
    
    Performs the centroid calculation on the checkbox region 
    calculated using the function checkbox_2D().
    
    Keyword arguments:
    image              -- 2 dimensional psf image
    checkbox_center    -- The location of the brightest checkox, 
                          in x and y, in the input image. (pix)
    checkbox_halfwidth -- The halfwidths in both x and y of the 
                          checkbox from centroid_2D(). (pix)
    max_iter           -- Max number of iterations for fine 
                          centroiding. Ignored if convergence is
                          reached beforehand.
    threshold          -- Threshold for successful convergence.
    
    
    Output(s):
    centroid  -- Tuple containing the location of the target, 
                 in the format [x, y].
    c_cum     -- The calculated flux sum within the checkbox 
                 region.
    
    Example usage:
    
        >> cb_centroid, cb_sum = centroid_2D(psf, cb_cen, cb_hw, 
                        max_iter=5, threshold=0.0001, debug=True)
        
        Find the fine centroid location of the target in psf, given
        both the checkbox center and halfwidths. Do a fine centroid 
        iteration of a maximum of 5, unless an threshold of 0.0001 
        is reached first
    """
    
    # First calculate centroid to use for the first iteration
    c_sum = 0
    xsum = 0
    ysum = 0
    
    convergence_flag = 'N/A'
    
    # Unpack the checkbox_center and checkbox_halfwidth into 
    # their appropriate variables
    xpeak, ypeak = checkbox_center
    xhw, yhw = checkbox_halfwidth    
    
    for ii in xrange(xpeak - xhw - 1, xpeak + xhw - 1):
        for jj in xrange(ypeak - yhw - 1, ypeak + yhw - 1):
            xloc = ii + 1
            yloc = jj + 1
            c_sum = c_sum + image[jj, ii]
            xsum += xloc * image[jj, ii]
            ysum += yloc * image[jj, ii]
            
    if debug:
        # Initial sum calculation (before iterations)
        print('(centroid_2D): Init. Sum (before iterations) = ', c_sum)
        print()

    if c_sum == 0:
        print('(centroid_2D): ERROR - divide by zero.')
        print()
        exit
    else:
        xcen = xsum / c_sum
        ycen = ysum / c_sum
            
    # Iteratively calculate centroid until solution converges, using 
    # neighboring pixels to apply weighting...
    old_xcen = xcen
    old_ycen = ycen
    num_iter = 0
    
    for kk in xrange(max_iter):
        num_iter += 1
        c_sum = 0
        xsum = 0
        ysum = 0
        
        # Set up x and y centroid scanning ranges
        x_range = np.array((np.floor(old_xcen - xhw) - 1, np.ceil(old_xcen + xhw) - 1))
        y_range = np.array((np.floor(old_ycen - yhw) - 1, np.ceil(old_ycen + yhw) - 1))
                
        for ii in xrange(np.int(x_range[0]), np.int(x_range[1])):
            for jj in xrange(np.int(y_range[0]), np.int(y_range[1])):
                
                # Initalize weights to zero
                xweight = 0
                yweight = 0
                
                # Adjust weights given distance from current centroid
                xoff = np.abs((ii + 1) - old_xcen)
                yoff = np.abs((jj + 1) - old_ycen)
                
                # If within the original centroid box, set weight to 1
                # for both x and y.
                # If on the border, the scale weight
                if xoff <= xhw:
                    xweight = 1
                elif xhw < xoff < (xhw + 1):
                    xweight = xhw + 1 - xoff
                    
                if yoff <= yhw:
                    yweight = 1
                elif yhw < yoff < (yhw + 1):
                    yweight = yhw + 1 - yoff
                    
                # Compute cummulative weight
                weight = xweight * yweight
                
                # Calculate centroid
                xloc = ii + 1
                yloc = jj + 1

                c_sum += image[jj, ii] * weight
                xsum += xloc * image[jj, ii] * weight
                ysum += yloc * image[jj, ii] * weight
                
        if c_sum == 0:
            print('(centroid_2D): ERROR - Divide by zero.')
            print()
        else:
            xcen = xsum / c_sum
            ycen = ysum / c_sum
                    
            # Check for convergence
            if np.abs(xcen - old_xcen) <= threshold and np.abs(ycen - old_ycen) <= threshold:
                convergence_flag = 'Success'
                break
            elif kk == max_iter:
                convergence_flag = 'Fail'
                break
            else:
                old_xcen = xcen
                old_ycen = ycen
    
    # Debug messages
    if debug:
        print('(centroid_2D): xpeak, ypeak = {}, {}'.format(xpeak, ypeak))
        print('(centroid_2D): xhw, yhw = {}, {}'.format(xhw, yhw))
        print('(centroid_2D): xcen, ycen = {}, {} '.format(xcen, ycen))        
        print()
        
                        
    print('(centroid_2D): Centroid = [{}, {}] for num_iter = {}.'.format(xcen-1, ycen-1, num_iter))
    print('(centroid_2D): Converged? ', convergence_flag)
    print()
          
    # -1 on both axes, as Python is 0 major    
    centroid = np.array((xcen-1, ycen-1))
    return centroid, c_sum

def find2D_higher_moments(image, centroid, halfwidths, c_sum):
    """
    Calculate the higher moments of the object in the image.
    
    Find the normalized squared and cubed moments with reference 
    to an origin at the centroid, using the centroid and and sum 
    values calculatated previously.
    
    Keyword arguments:
    image      -- 2 dimensional psf image
    centroid   -- Tuple containing centroid of the object in 
                  image ([x, y] in pixels).
    halfwidths -- The halfwidths in both x and y of the 
                  checkbox from centroid_2D(). (pix)
    c_sum      -- The calculated flux sum within the checkbox 
                 region.
                 
    
    Output(s):
    x_mom, y_mom -- Tuples containing the x and y higher moments 
                    (normalized squared and cubed).
    
    Example usage:
    
        >> x_mom, y_mom = find2D_higher_moments(psf, cb_centroid, 
                          cb_hw, cb_sum)
        
        Find the second and third moments of psf, given the 
        centroid, checkbox halfwidths, and calculated sum.
    """
    
    # Unpack centroid to seperate values
    xcen, ycen = np.floor(centroid)
    xhw, yhw = halfwidths
    
    xmoment2 = 0
    xmoment3 = 0
    ymoment2 = 0
    ymoment3 = 0
        
    # Set up x and y centroid scanning ranges
    x_range = np.array((np.floor(xcen - xhw) - 1, np.ceil(xcen + xhw) - 1))
    y_range = np.array((np.floor(ycen - yhw) - 1, np.ceil(ycen + yhw) - 1))
    
    
    for ii in xrange(np.int(x_range[0]), np.int(x_range[1])):
        for jj in xrange(np.int(y_range[0]), np.int(y_range[1])):
            
            xloc = ii - np.floor(xcen)
            yloc = jj - np.floor(ycen)
            
            xweight = 0
            yweight = 0
            
            xoff = np.abs(ii - xcen)
            yoff = np.abs(jj - ycen)
            
            if xoff <= xhw:
                xweight = 1
            elif xhw < xoff < (xhw + 1):
                xweight = xhw + 1 - xoff
                
            if yoff <= yhw:
                yweight = 1
            elif yhw < yoff < (yhw + 1):
                yweight = yhw + 1 - yoff
                
            weight = xweight * yweight

            xmoment2 += xloc ** 2 * image[jj, ii] * weight
            xmoment3 += xloc ** 3 * image[jj, ii] * weight
            ymoment2 += yloc ** 2 * image[jj, ii] * weight
            ymoment3 += yloc ** 3 * image[jj, ii] * weight
            
    xmoment2 = xmoment2 / c_sum
    xmoment3 = xmoment3 / c_sum
    ymoment2 = ymoment2 / c_sum
    ymoment3 = ymoment3 / c_sum

    # Pack the x and y moments to return to main program
    x_moment = np.array((xmoment2, xmoment3))
    y_moment = np.array((ymoment2, ymoment3))
    
    return x_moment, y_moment   

def centroid_1D(image, xpeak, xhw, debug=False):
    """
    Fine location of the target by calculating the centroid for 
    the region centered on the brightest checkbox.
    
    Performs the centroid calculation on the checkbox region 
    calculated using the function checkbox_1D().
    
    Keyword arguments:
    image -- 2 dimensional psf image
    xpeak -- The location of the brightest checkox in the 
             flattened vector.
    xhw   -- The halfwidth of the checkbox region calculated in
             checkbox_1D.
             
    
    Output(s):
    x_cen -- Target centroid location.
    c_cum -- The calculated flux sum within the checkbox 
             region.
    
    Example usage:
    
        >> cb_centroid, cb_sum = centroid_1D(psf, cb_center, 
                                 cb_hw)
        
        Find the vector centroid given the checkbox center and 
        halfwidth.
    """
    
    # Collapse input image unto x axis
    vector = np.sum(image, axis=0)
    
    c_sum = 0.0
    xcen = 0.0
        
    for ii in xrange(int(xpeak - xhw - 1), int(xpeak + xhw - 1)):
        c_sum = c_sum + vector[ii]
        xloc = ii + 1
        xcen += xloc * vector[ii]
    
    print('(centroid_1D): Sum = ', c_sum)
    
    
    if c_sum == 0:
        print('(centroid_1D): ERROR - divide by zero')
    else:
        xcen /= c_sum
        
    print('(centroid_1D): Centroid = ', xcen-1)
        
    # -1 on both axes, as Python is 0 major    
    return xcen-1, c_sum  

def find1D_higher_moments(image, xcen, xhw, c_sum):
    """
    Calculate the higher moments of the object in the image.
    
    Find the normalized squared and cubed moments with reference 
    to an origin at the centroid, using the centroid and and sum 
    values calculatated previously.
    
    Keyword arguments:
    image -- 2 dimensional psf image
    x_cen -- Target cebtroid location.
    xhw   -- The checkbox halfwidth.
    c_sum -- The calculated flux sum within the checkbox region.
                 
    
    Output(s):
    x_mom -- Vector higher moments.
    
    Example usage:
    
        >> x_mom = find1D_higher_moments(psf, cb_centroid, cb_hw, 
                   cb_sum)
        
        Find the second and third moments of psf, given the 
        vector centroid, checkbox halfwidth, and calculated sum.
    """
    
    # Collapse input image unto x axis
    vector = np.sum(image, axis=0)
    
    xmoment2 = 0.0
    xmoment3 = 0.0
    
    # Set up x and y centroid scanning ranges
    x_range = np.array((np.floor(xcen - xhw) - 1, np.ceil(xcen + xhw) - 1))

    for ii in xrange(np.int(x_range[0]), np.int(x_range[1])):
        xloc = (ii + 1) - np.floor(xcen)
        
        xweight = 0
        xoff = np.abs(ii - xcen)
        
        if xoff <= xhw:
            xweight = 0
        elif xhw < xoff < xhw + 1:
            xweight = xhw + 1 - xoff

        xmoment2 += xloc ** 2 * vector[ii] * xweight
        xmoment3 += xloc ** 3 * vector[ii] * xweight
        
    xmoment2 = xmoment2 / c_sum
    xmoment3 = xmoment3 / c_sum
    
    # Pack moments for return to main program
    x_mom = np.array((xmoment2, xmoment3))
    
    return x_mom


if run_part3 or run_all:   # Added by Maria Pena-Guerrero
    ''' 3. TESITNG GROUND 
    
    3.1 - Checkbox calculation on 2D image'''
    
    # Test input image, and display to verify location
    # these test files were downloaded from:  /grp/jwst/wit/nirspec/Data_LeBlanc/TA/WEBB_psfs
    #psf = fits.getdata('../Tommys_testData/WEBB_Test_00_3.fits', 1)
    psf = fits.getdata('../Tommys_testData/WEBB_Test_00_7.fits', 1)
    #psf = fits.getdata('../Tommys_testData/WEBB_Test_00_13.fits', 1)
    #psf = fits.getdata('../Tommys_testData/WEBB_Test_90_10.fits', 1)
    
    fig_name = '../initial_fig.jpg'
    tu.display_ns_psf(psf, vlim=(0.001, 0.01), savefile=fig_name)
    
    ''' 
    Calculate the checkbox region and centroid. Currently testing a checkbox = 5, 
    with max_iterations = 5 and a threshold = 0.0001 pixels.
    
    For this exercise, perform the following checkbox calculation: 
    '''
    # Test checkbox piece
    cb_cen, cb_hw = jtl.checkbox_2D(psf, 5, debug=True)
    
    # Checkbox center, in base 1
    print('Checkbox Output:')
    print('Checkbox center: [{}, {}]'.format(cb_cen[0], cb_cen[1]))
    print('Checkbox halfwidths: xhw: {}, yhw: {}'.format(cb_hw[0], cb_hw[1]))
    print()
    
    # Now calculate the centroid based on the checkbox region calculated above
    cb_centroid, cb_sum = jtl.centroid_2D(psf, cb_cen, cb_hw, max_iter=5, threshold=0.0001, debug=True)
    print('Final sum: ', cb_sum)
    print()
    
    # ... find the 2nd and 3rd moments...
    x_mom, y_mom = jtl.find2D_higher_moments(psf, cb_centroid, cb_hw, cb_sum)
    print('Higher moments(2nd, 3rd):')
    print('x_moments: ', x_mom)
    print('y moments: ', y_mom)
    print()
    
    # and verify with previously written (tautils.centroid())
    init_cen = tu.centroid(psf, psf.shape[0])
    check_centroid = tu.centroid(psf, 7, initial=init_cen)
    print('My centroid: ', check_centroid)
    
    # Display both centroids for comparison.
    # Plot results for verification
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.imshow(psf, vmin=0.001, vmax=0.01, cmap='gray', interpolation='nearest')
    ax.autoscale(enable=False, axis='both')
    
    ax.plot(cb_centroid[0], cb_centroid[1], marker='s', mec='black', mfc='red', ls='', label='JWST Target Locate')
    ax.plot(check_centroid[0], check_centroid[1], marker='o', mec='black', mfc='blue', ls='', label='Homebrew Centroid')
    ax.legend(numpoints=1, loc=1, prop={"size":"small"})
    
    # Save figure for later use
    fig.savefig('../TargLocate.pdf')
    
    ''' 
    3.2 - Checkbox calculation on a vector (1D):
    Same as above, expect on a 1D image. Using initial psf, we flatten over the y-axis.
    '''
    # Test input image
    tu.display_ns_psf(psf, vlim=(0.001, 0.01))
    
    vector = np.sum(psf, axis=0).reshape(1, 16)
    tu.display_ns_psf(vector, vlim=(0.001, 0.1))
    # 1D Checkbox calculation
    cb_center, cb_hw = checkbox_1D(psf, 5)
    
    # Print output
    print('Checkbox Output:')
    print('Checkbox center: {}'.format(cb_center))
    print('Checkbox Halfwidth: {}'.format(cb_hw))
    print()
    
    # Calculate centroid...
    cb_centroid, cb_sum = centroid_1D(vector, cb_center, cb_hw)
    print('Final sum: ', cb_sum)
    print()
    
    # ... find the 2nd and 3rd moments...
    x_mom = find1D_higher_moments(psf, cb_centroid, cb_hw, cb_sum)
    print('Higher moments(2nd, 3rd):')
    print('x_moments: ', x_mom)
    print()


if run_part4 or run_all:   # Added by Maria Pena-Guerrero
    ''' 4. RECURSIVE TESTING 
    Here, we will perform some recursive testing of the scripts, in particular to determine the
    optimal values for some of the inputs (see JWST-STScI-001117-SM2):
     - checkboxSize
     - convergenceThrees
    
    Input PSF
    Note: The current psf, provided by WebbPSF is 16x16 px; this analysis requires 32x32 px (see extractCols
    and extractRows keywords in JWST Cross-Instrument Target Locate document, JWST-STScT-001117, SM-12).
    
    The following cells below will take the 16x16 psf and implant it into an empty 32x32 black frame. Please
    note that this is an approximation, some light will be lost, ~ 7%.
    
    The expected center of this psf is (15.5, 15.5) (0.0 is the center of the pixel in Python).
    '''
    # Read input image, and display to verify location
    img = fits.open('../Tommys_testData/WEBB_Test_00_7.fits')
    img.info()
    psf = fits.getdata('../Tommys_testData/WEBB_Test_00_7.fits', 1)
    print (np.shape(psf))
    #print(psf)
    tu.display_ns_psf(psf, vlim=(0.001, 0.01))
    
    # Create a 32x32 blank image to place the 16x16 psf ...
    full_psf = np.zeros([32,32])
    tu.display_ns_psf(full_psf, vlim=(0.001, 0.01))
    
    # ...and place psf into blank for the synthetic 32x32 cutout
    full_psf[8:24, 8:24] = psf
    tu.display_ns_psf(full_psf, vlim=(0.001, 0.01))
    '''Note: I wanted to verify the shape of the psf; since the expected centroid turns out to be
    intrapixel (15.5, 15.5), I plotted the rows/columns sorrounding the expected centroid...
    '''
    
    # Plot the rows/columns encompasing the centroid to verify psf shape
    x_radial_1 = full_psf[:, 15]
    x_radial_2 = full_psf[:, 16]
    y_radial_1 = full_psf[15, :]
    y_radial_2 = full_psf[16, :]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim(0, 31)
    ax.set_xlabel('Pixel Location')
    ax.set_ylabel('Counts (of total 1)')
    
    ax.plot(xrange(32), x_radial_1, ls='-', lw=2., label='Column 15')
    ax.plot(xrange(32), x_radial_2, ls='-', lw=2., label='Column 16')
    ax.plot(xrange(32), y_radial_1, ls='-', lw=2., label='Row 15')
    ax.plot(xrange(32), y_radial_2, ls='-', lw=2., color='purple', label='Row 16')
    
    ax.legend(numpoints=1, loc=1, prop={"size":"medium"})
    plt.show()
    
    ''' 4.1 - CheckboxSize
    The size of the initial checkbox (in pixels) over which to do an initial (course) centroid. The larger the
    checkbox, the smaller the resulting area over which to do the centroid.
    
    Now that the 32x32 pixels psf is verified, recursively calculate the centroid, changing the checkbox size in each iteration...
    
    Note: The output of each iteration is printed, including the calculation of the higher moments.
    '''
    # Make a list of checkboxes based on width of input image,
    # and create an empty list to store centroids based on input 
    # checkboxes
    psf_width = full_psf.shape[0]
    checkbox_list = [x for x in xrange(psf_width) if x % 2 == 1]
    num_centroids = len(checkbox_list)
    centroids = np.zeros(num_centroids * 2).reshape(num_centroids, 2)
    
    print(full_psf.shape)
    
    # Recursively calculate centroids with input checkbox list
    for num, ii in enumerate(checkbox_list[0:-1]):
    
        # Calculate checkbox and checkbox halfwidth
        cb_cen, cb_hw = jtl.checkbox_2D(full_psf, ii)
    
        # Calculate centroid and the flux sum (within centroid)...
        cb_centroid, cb_sum = jtl.centroid_2D(full_psf, cb_cen, cb_hw, max_iter=3, threshold=0.3, debug=True)
        centroids[num, 0], centroids[num, 1] = cb_centroid
        
        print('Final sum: ', cb_sum)
        print('cb_centroid: ', cb_centroid)
        
        if ii==3:
            raw_input()
        '''
        # ... find the 2nd and 3rd moments...
        x_mom, y_mom = jtl.find2D_higher_moments(full_psf, cb_centroid, cb_hw, cb_sum)
        print('Higher moments(2nd, 3rd):')
        print('x_moments: ', x_mom)
        print('y moments: ', y_mom)
        print('---------------------------------------------------------------')
        print()
        '''
    
