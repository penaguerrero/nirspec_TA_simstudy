from __future__ import print_function, division
import numpy as np
import copy   # Added by Maria Pena-Guerrero

# Header
__author__ = "T. Le Blanc & M. A. Pena-Guerrero"
__version__ = "2.0"

# HISTORY
#    1. Feb 2014 - Vr. 1.0: Added initial versions the JWST target locate
#                           algorithm. 
#    2. Jul 2015 - Vr. 2.0: Modifications made by Maria Pena-Guerrero


# Utility definitions


# *************************** checkbox_2D ***************************
def checkbox_2D(image, checkbox, xwidth=0, ywidth=0, verbose=True, debug=False):
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
    xwidth   -- Number of rows for the centroid region (positive
                odd integer, or 0).
    ywidth   -- Number of columns for the centroid region (positive
                odd integer, or 0).
    
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
    # Note: the x and y are opposite to intuitive values in the images.  - Added by M. Pena-Guerrero
    xsize, ysize = image.shape[1], image.shape[0]
        
    # If the checkbox size is not equal to both the X and Y sizes, 
    # find the pixel with the brightest checkbox
    if checkbox != xsize and checkbox != ysize:
        xpeak = 0
        ypeak = 0
        sumpeak = 0
        sumpeak_list = []
        xy_peak_list = []
        for ii in xrange(xsize - checkbox +1):  # the +1 is because python stops the loop at idx=n
            for jj in xrange(ysize - checkbox +1):  # the +1 is because python stops the loop at idx=n
                t = np.sum(image[jj:jj+checkbox, ii:ii+checkbox])
                if t > sumpeak:
                    xpeak = ii + chw + 1   
                    ypeak = jj + chw + 1
                    sumpeak = t
                    xy_peak = [xpeak, ypeak]
                    xy_peak_list.append(xy_peak)
                    sumpeak_list.append(sumpeak)
        if verbose:
            print('(checkbox_2D): Checkbox not equal to both x/ysize.')
    
    # If the checkbox size is equal to both the X and Y sizes
    if checkbox == xsize and checkbox == ysize:
        xpeak = xsize / 2
        ypeak = ysize / 2
        sumpeak = np.sum(image, axis=None)
        if verbose:
            print('(checkbox_2D): Checkbox equal to x/ysize.')
        
    # Find the centroid region half-width in x and y
    xhw = (xwidth - 1) / 2
    yhw = (ywidth - 1) / 2
    if verbose:
        if xpeak < xhw or xpeak > xsize - xhw or ypeak < yhw or ypeak > ysize - yhw:
            print('(checkbox_2D): WARNING - Peak too close to edge of image.')
        
#    NOTE: Use this section if the input image is a subset of a larger image.
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
        
    checkbox_ctr = np.array((xpeak, ypeak))
    checkbox_hfw = np.array((xhw, yhw))
    
    if verbose:
        print('(checkbox_2D): Checkbox centroid is given in indexing: starting at 1')
    
    return checkbox_ctr, checkbox_hfw
# *************************** checkbox_2D ***************************

# *************************** checkbox_1D ***************************
def checkbox_1D(image, checkbox, xwidth=0, debug=False):
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
    xwidth   -- Number of rows for the centroid region (positive
                odd integer, or 0).
    
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
    
    # Calculate the checkbox half-width
    chw = (checkbox - 1) / 2

    # Calculate the image size
    xsize, ysize = image.shape[1], image.shape[0]
    
    # If the checkbox size is not equal to both the X and Y sizes, 
    # find the pixel with the brightest checkbox
    if checkbox != xsize and checkbox != ysize:
        xpeak = 0
        sumpeak = 0
        for ii in xrange(xsize - checkbox +1):  # the +1 is because python stops the loop at idx=n
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
        
    # Find the checkbox region half-width in x and y
    xhw = (xwidth - 1) / 2
        
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
        
#    NOTE: Use this section of the input image is a subset of a larger image
#          Not currently needed for this analysis
#    # Determine the center of the brightest checkbox, in extracted
#    # image coordinates
#    xpeak = xpeak + xhw
    
    print('(checkbox_1D): Checkbox centroid is given in indexing: starting at 1')
    
    return xpeak, xhw
# *************************** checkbox_1D ***************************

# *************************** centroid_2D ***************************
def centroid_2D(image, checkbox_center, checkbox_halfwidth, max_iter=0, threshold=0, 
                verbose=True, debug=False):
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
    c_sum     -- The calculated flux sum within the checkbox 
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

    # Added by M. Pena-Guerrero   ->   Remove the -1 if centroid is given in Python indexing (starting at 0)
    lolim_x = int(xpeak - xhw - 1)
    uplim_x = int(xpeak + xhw - 1)
    lolim_y = int(ypeak - yhw - 1)
    uplim_y = int(ypeak + yhw - 1)
    
    if debug:
        print ('\n xpeak, ypeak, xhw, yhw', xpeak, ypeak, xhw, yhw)
        print ('lolim_x, uplim_x', lolim_x, uplim_x)
        print ('lolim_y, uplim_y', lolim_y, uplim_y, '\n')

    # Make sure that the limits are within the data   - Added by M. Pena-Guerrero
    if lolim_x < 0:
        lolim_x = 0
        if verbose:
            print ('(centroid_2D): WARNING - lower limit in x is out of data, setting to 0.')
    if uplim_x > 32:
        uplim_x = 31
        if verbose:
            print ('(centroid_2D): WARNING - upper limit in x is out of data, setting to 31.')
    if lolim_y < 0:
        lolim_y = 0
        if verbose:
            print ('(centroid_2D): WARNING - lower limit in y is out of data, setting to 0.')
    if uplim_y > 32:
        uplim_y = 31
        if verbose:
            print ('(centroid_2D): WARNING - upper limit in y is out of data, setting to 31.')

    for ii in xrange(lolim_x, uplim_x+1):  # the +1 is because python stops the loop at idx=n
        for jj in xrange(lolim_y, uplim_y+1):  # the +1 is because python stops the loop at idx=n
            xloc = ii + 1
            yloc = jj + 1

            # Make sure that the limits are within the data   - Added by M. Pena-Guerrero
            if xloc >= 32:
                xloc = 31
                ii, jj = 31, 31
                if verbose:
                    print ('(centroid_2D): WARNING - Upper limit in x is out of data, setting to 31.')
            if yloc >= 32:
                yloc = 31
                ii, jj = 31, 31
                if verbose:
                    print ('(centroid_2D): WARNING - Upper limit in y is out of data, setting to 31.')

            c_sum = c_sum + image[jj, ii]
            xsum += xloc * image[jj, ii]
            ysum += yloc * image[jj, ii]
            if debug:
                print('xloc, yloc', xloc, yloc)
                print ('ii, jj, image[jj, ii]: ', ii, jj, image[jj, ii])
                print ('xsum, ysum, c_sum: ', xsum, ysum, c_sum)
                print ('xsum, ysum, c_sum: ', xsum, ysum, c_sum)
            
    if debug:
        # Initial sum calculation (before iterations)
        print('(centroid_2D): Init. Sum (before iterations) = ', c_sum)

    if c_sum == 0:
        if verbose:
            print('(centroid_2D): WARNING - Dividing by zero: c_sum=0. Not going into the for loop! ')
            print('               Keeping checkbox center.')
        xcen, ycen = xpeak, ypeak
    else:
        xcen = xsum / c_sum
        ycen = ysum / c_sum
            
    # Iteratively calculate centroid until solution converges, using 
    # neighboring pixels to apply weighting...
    old_xcen = copy.deepcopy(xcen)   # Modified by Maria Pena-Guerrero
    old_ycen = copy.deepcopy(ycen)   # Modified by Maria Pena-Guerrero
    num_iter = 0
    
    if verbose:
        print ('(centroid_2D): Maximum iterations = ', max_iter)   # Added by M. Pena-Guerrero
    
    for kk in xrange(max_iter +1):  # the +1 is because python stops the loop at idx=n
        num_iter += 1
        c_sum = 0
        xsum = 0
        ysum = 0
        
        # Set up x and y centroid scanning ranges
        x_range = np.array((np.floor(old_xcen - xhw) - 1, np.ceil(old_xcen + xhw) - 1))
        y_range = np.array((np.floor(old_ycen - yhw) - 1, np.ceil(old_ycen + yhw) - 1))
        
        # Debug messages  - Added by M. Pena-Guerrero
        #if debug:
        #    print ('x_range=', x_range)
        #    print ('y_range=', y_range)
        #    print ('old_xcen, old_ycen, xhw, yhw', old_xcen, old_ycen, xhw, yhw)
        #    print ('(np.floor(old_xcen - xhw) - 1, np.ceil(old_xcen + xhw) - 1): ', np.floor(old_xcen - xhw) - 1, np.ceil(old_xcen + xhw) - 1) 
        #    print ('(np.floor(old_ycen - yhw) - 1, np.ceil(old_ycen + yhw) - 1): ', np.floor(old_ycen - yhw) - 1, np.ceil(old_ycen + yhw) - 1) 
        #    print ('iteration number: ', num_iter)
                
        for ii in xrange(np.int(x_range[0]), np.int(x_range[1]) +1):  # the +1 is because python stops the loop at idx=n
            for jj in xrange(np.int(y_range[0]), np.int(y_range[1]) +1):  # the +1 is because python stops the loop at idx=n

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
                
                # Make sure that the limits are within the data   - Added by M. Pena-Guerrero
                if ii >= 32:
                    ii = 31
                    if verbose:
                        print ('(centroid_2D): WARNING - X index is out of data, setting to 31.')
                if jj >= 32:
                    jj = 31
                    if verbose:
                        print ('(centroid_2D): WARNING - Y index is out of data, setting to 31.')

                c_sum += image[jj, ii] * weight
                xsum += xloc * image[jj, ii] * weight
                ysum += yloc * image[jj, ii] * weight
        
        if c_sum == 0:
            if verbose:
                print('(centroid_2D): WARNING - Still dividing by zero --> c_sum is not being updated!')
                print('               Keeping checkbox center.')
            # If the centering box was too small, keep checkbox center
            xcen, ycen = checkbox_center
            break
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
    
    # Now subtract 1 on both axes, since Python starts counting from 0   - Modified by M. Pena-Guerrero
    start_from0 = False
    if start_from0:
        centroid = np.array((xcen-1, ycen-1))   # but we are not correcting because JWST will start with 1
    else:
        centroid = np.array((xcen, ycen))   # Leave results starting from 1 (as in OS for JWST)
        
    # Debug messages
    if debug:
        print('(centroid_2D): Starting count for columns and rows from 0 set to: ', str(start_from0))  # Added by M. Pena-Guerrero
        print('(centroid_2D): xpeak, ypeak = {}, {}'.format(xpeak, ypeak))
        print('(centroid_2D): xhw, yhw = {}, {}'.format(xhw, yhw))
        print('(centroid_2D): xcen, ycen = {}, {} '.format(xcen, ycen))        
                                
    if start_from0:
        starting_point = 0
    else:
        starting_point = 1

    if verbose:
        print('(centroid_2D): Centroid indexing starts with: ', starting_point)
        print('(centroid_2D): Centroid = [{}, {}] for num_iter = {}.'.format(centroid[0], centroid[1], num_iter))
        print('(centroid_2D): Converged? ', convergence_flag)
          
    return centroid, c_sum
# *************************** centroid_2D ***************************



# *************************** centroid_1D ***************************
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
        
    for ii in xrange(xpeak - xhw - 1, xpeak + xhw - 1 +1):  # the +1 is because python stops the loop at idx=n
        c_sum = c_sum + vector[ii]
        xloc = ii + 1
        xcen += xloc * vector[ii]
    
    print('(centroid_1D): Sum = ', c_sum)
    
    
    if c_sum == 0:
        print('(centroid_1D): ERROR - divide by zero')
    else:
        xcen = xcen / c_sum
        
    print('(centroid_1D): Centroid = ', xcen-1)
        
    # -1 on both axes, as Python is 0 major    
    return xcen-1, c_sum  
# *************************** centroid_1D ***************************


# *************************** find2D_higher_moments ***************************
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
    
    
    for ii in xrange(np.int(x_range[0]), np.int(x_range[1]) +1):  # the +1 is because python stops the loop at idx=n
        for jj in xrange(np.int(y_range[0]), np.int(y_range[1]) +1):  # the +1 is because python stops the loop at idx=n
            
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
# *************************** find2D_higher_moments ***************************

# *************************** find1D_higher_moments ***************************
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

    for ii in xrange(np.int(x_range[0]), np.int(x_range[1]) +1):  # the +1 is because python stops the loop at idx=n
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
# *************************** find1D_higher_moments ***************************



# Print diagnostic load message
print("(jwst_targloc): JWST Target Locate Utilities Version {} loaded!".format(__version__))