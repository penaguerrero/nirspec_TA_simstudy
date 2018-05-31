from __future__ import print_function, division
import numpy as np
import copy 

# Header
__author__ = "Maria A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
#    1. Jul 2015 - Vr. 1.0: Initial Python translation of IDL code.

"""
This script is a Python translation from the ta_lsfitdbl.pro by Tony Keyes, which 
in turn is a sample code from the fitting algorithm with roll correction 
(Jakobsen, 2006) draft coding for spatial offset sigmas added.
     -- needs verification and does not contain calculation for roll sigma

Keyword arguments:
    niter       -- number of maximum iterations
    x_input     -- input measured x-centroid converted to V2 
    y_input     -- input measured y-centroid converted to V3 
        xcentroids  -- input measured x-centroid converted to V2 
        ycentroids  -- input measured y-centroid converted to V3 
    xtrue       -- numpy array of true V2 positions
    ytrue       -- numpy array of true V3 positions

Output(s):
    deltas = [delta_x, delta_y, delta_theta]
    sigmas = [sigma_x, sigma_y, sigma_theta]

Example usage:
    import least_squares_iterate as lsi
    deltas, sigmas, lines2print, rejected_elements_idx = lsi.ls_fit_iter(niter, x_input, y_input, xtrue, ytrue, Nsigma)


*** Testing suite of the script at the bottom

"""

def ls_fit_iter(niter, xt, yt, x, y, Nsigma, arcsec=True, verbose=True):
    """
    This funciton finds the standard deviation and mean from all points, then subtracts that mean from
    each point and compares it with the true values to reject all the points that are Nsigma away. It
    iterates until no more points are being rejected.
    Args:
         niter  = number of max iterations
         xt     = input measured x-centroid converted to V2
         yt     = input measured y-centroid converted to V3
         x      = numpy array of true V2 positions
         y      = numpy array of true V3 positions
         Nsigma = sigma limit to reject stars
         arcsec = True or False, give delta theta in arcsecs?
    Returns:
        deltas, sigmas, lines2print, rejected_elements_idx
        deltas = list of means for x, y, and theta
        sigmas = list of standard deviations for x, y, and theta
        lines2print = list of lines to pretty print results on screen and/or in a file
        rejected_elements_idx = list of the index of the rejected points
        nit = integer, number of iterations
    """

    # do up to niter iterations of sigma-clipping (first time through is 
    # initial calculation, then up to niter iterations)
    original_elements = len(x)
    original_true_centroids = copy.deepcopy(x)
        
    for nit in range(niter):
        n = len(x)
        # Initialize the sums
        sum_tot = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_xt = 0.0
        sum_yt = 0.0
        sum_xt2 = 0.0
        sum_yt2 = 0.0
        sum_xyt = 0.0
        sum_xty = 0.0
        
        for i in range(n):
            sum_tot += 1.0
            sum_x = sum_x + x[i]
            sum_y = sum_y + y[i]
            sum_xt = sum_xt + xt[i]
            sum_yt = sum_yt + yt[i]
            sum_xt2 = sum_xt2 + xt[i]*xt[i]
            sum_yt2 = sum_yt2 + yt[i]*yt[i]
            sum_xyt = sum_xyt + x[i]*yt[i]
            sum_xty = sum_xty + xt[i]*y[i]
        
        det = sum_tot*sum_tot*(sum_xt2 + sum_yt2) - sum_tot*(sum_xt*sum_xt + sum_yt*sum_yt)
        
        delta_x = (sum_tot*(sum_xt2 + sum_yt2) - sum_xt*sum_xt) * (sum_x - sum_xt)
        delta_x = delta_x - (sum_y-sum_yt)*sum_xt*sum_yt - sum_tot*sum_yt*(sum_xyt-sum_xty)
        delta_x /= det  # offset in x-coordinate

        delta_y = (sum_tot*(sum_xt2+sum_yt2) - sum_yt*sum_yt) * (sum_y-sum_yt)
        delta_y = delta_y - (sum_x-sum_xt)*sum_xt*sum_yt - sum_tot*sum_xt*(sum_xty-sum_xyt)
        delta_y /= det  # offset in y-coordinate

        delta_theta = sum_tot*((sum_xt*sum_y-sum_xt*sum_yt) + sum_tot*(sum_xyt-sum_xty))
        delta_theta /= det  # roll angle correction  (what units??)

        # outputs:  delta_x, delta_y, delta_theta, sigma_x, sigma_y, sigma_theta
        line1 = '(least_squares_iterate):  iteration number: {}'.format(nit)
        if arcsec:
            delta_theta = delta_theta * (180.0/np.pi) * 3600.0
            line2 = '(least_squares_iterate):  delta_x = {}   delta_y = {}   delta_theta = {} arcsec'.format(delta_x, delta_y,
                                                                                                            delta_theta)
        else:
            line2 = '(least_squares_iterate):  delta_x = {}   delta_y = {}   delta_theta = {} radians'.format(delta_x, delta_y,
                                                                                                             delta_theta)
        deltas = [delta_x, delta_y, delta_theta]
        
        # verify this coding for sigma_xtrue and sigma_ytrue
        sum_delta_x2 = 0.0
        sum_delta_y2 = 0.0
        sum_delta_theta2 = 0.0
        for i in range(n):
            sum_delta_x2 += (-xt[i] + x[i] - delta_x) * (-xt[i] + x[i] - delta_x)
            sum_delta_y2 += (-yt[i] + y[i] - delta_y) * (-yt[i] + y[i] - delta_y)
        
        sigma_x = np.sqrt(sum_delta_x2/n)   # sigma for xtrue-offset  -- is this right?
        sigma_y = np.sqrt(sum_delta_y2/n)   # sigma for ytrue-offset  -- is this right?
        
        # for now set sigma_thera to bogus value  (we don't presently know how to calculate it)
        sigma_theta = -999.0
        
        line3 = '(least_squares_iterate):  sigma_x = {}   sigma_y = {}   sigma_theta = {}'.format(sigma_x, sigma_y, sigma_theta)
        sigmas = [sigma_x, sigma_y, sigma_theta]
        
        # calculate new best position and residuals for each coordinate (neglect any roll correction for now)
        xnewpos = x - delta_x
        xdiff = xnewpos - xt
        ynewpos = y - delta_y
        ydiff = ynewpos - yt
        
        # Rejection sequence:
        # reject any residuals that are not within 3*sigma in EITHER coordinate
        # (this is slightly more restrictive than doing this for the vector sum, 
        # but easier to implement right now)
        #Nsigma = 2.0
        thres_x = Nsigma*sigma_x
        thres_y = Nsigma*sigma_y
        var_clip = xdiff[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        xcentroids_new = xt[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        ycentroids_new = yt[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        x_new = x[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        y_new = y[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        
        # This commented following section would do rejection for 3.0*sigma AND a specified
        # physical distance threshold zerop7, a 0.7 arcsec threshold
        """
        zerop7 = 0.7
        var_clip = xdiff[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y) & (np.abs(xdiff)<=zerop7) & (np.abs(ydiff)<=zerop7)))]
        xcentroids_new = xt[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y) & (np.abs(xdiff)<=zerop7) & (np.abs(ydiff)<=zerop7)))]
        ycentroids_new = yt[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y) & (np.abs(xdiff)<=zerop7) & (np.abs(ydiff)<=zerop7)))]
        x_new = x[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y) & (np.abs(xdiff)<=zerop7) & (np.abs(ydiff)<=zerop7)))]
        y_new = y[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y) & (np.abs(xdiff)<=zerop7) & (np.abs(ydiff)<=zerop7)))]
        """
        elements_left = len(xcentroids_new)
        line4 = '(least_squares_iterate):  elements_left={} out of original_elements={}'.format(elements_left, original_elements)
                
        if len(xcentroids_new) == len(xt):
            break   # exit the loop since no additional rejections on this iteration
        else:
            xt = xcentroids_new
            yt = ycentroids_new
            x = x_new
            y = y_new
    if verbose:
        print (line1)
        print (line2)
        print (line3)
        print (line4)
    lines2print = [line1, line2, line3, line4]

    # find what elements got rejected            
    rejected_elements_idx = []
    for i, centroid in enumerate(original_true_centroids):
        if centroid not in x:
            rejected_elements_idx.append(i)
            #print("len(original_true_centroids): ", len(original_true_centroids), "   appending index: ", i)
            #raw_input()
    
    return deltas, sigmas, lines2print, rejected_elements_idx, nit
    
    # Still do not know how to do delta_theta sigma  -- is this calculation needed?


if __name__ == '__main__':

    # Print diagnostic load message
    print("(least_squares_iterate): Least squares iteration algorithm Version {} loaded!".format(__version__))

    testing = True
    if testing:
        # Set test values  for arrays
        n = 10            # max number of iterations
        xtrue = np.array(range(10))     # true x-coordinate of each reference star: from 0 to 9
        ytrue = np.array(range(1, 11))  # true y-coordinate of each reference star: from 1 to 10
        xinput = xtrue + 0.02     # measured centroid x-coordinate of each reference star
        yinput = ytrue + 0.01     # measured centroid y-coordinate of each reference star
        Nsigma = 3.0
        deltas, sigmas, _, _, _ = ls_fit_iter(n, xinput, yinput, xtrue, ytrue, Nsigma)
        """
        With these parameters output should be:
            (least_squares_iterate): Least squares iteration algorithm Version 1.0 loaded!
            (least_squares_iterate):  elements_left=10 out of original_elements=10
            (least_squares_iterate):  iteration number: 0
            (least_squares_iterate):  delta_x = -0.02   delta_y = -0.01   delta_theta = -0.00667878787879 radians
            (least_squares_iterate):  sigma_x = 4.564982887e-15   sigma_y = 3.69348008392e-15   sigma_theta = -999.0
        """
    

