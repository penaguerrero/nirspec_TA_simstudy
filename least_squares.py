from __future__ import print_function, division
import numpy as np

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
    n  -- Number of positions
    x  -- numpy array of true x-coordinates of each reference star
    y  -- numpy array of true y-coordinates of each reference star
    xt -- measured centroid x-coordinate of each reference star
    yt -- measured centroid y-coordinate of each reference star

Output(s):
    deltas = [delta_x, delta_y, delta_theta]
    sigmas = [sigma_x, sigma_y, sigma_theta]

Example usage:
    import least_squares as ls
    deltas, sigmas = ls.ls_fit(n, x, y, xt, yt)


*** Testing suite of the script at the bottom

"""

def ls_fit(n, x, y, xt, yt):
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

    delta_theta = sum_tot*((sum_xt*sum_y-sum_x*sum_yt) + sum_tot*(sum_xyt-sum_xty))
    delta_theta /= det  # roll angle correction  (what units??)

    # outputs:  delta_x, delta_y, delta_theta, sigma_x, sigma_y, sigma_theta
    print ('delta_x = {}   delta_y = {}   delta_theta = {}'.format(delta_x, delta_y, delta_theta))
    deltas = [delta_x, delta_y, delta_theta]
    
    # verify this coding for sigma_x and sigma_y
    sum_delta_x2 = 0.0
    sum_delta_y2 = 0.0
    #sum_delta_theta2 = 0.0
    for i in range(n):
        sum_delta_x2 += (-xt[i] + x[i] - delta_x) * (-xt[i] + x[i] - delta_x)
        sum_delta_y2 += (-yt[i] + y[i] - delta_y) * (-yt[i] + y[i] - delta_y)
    
    sigma_x = np.sqrt(sum_delta_x2/n)   # sigma for x-offset  -- is this right?
    sigma_y = np.sqrt(sum_delta_y2/n)   # sigma for y-offset  -- is this right?
    
    # for now set sigma_thera to bogus value  (we don't presently know how to calculate it)
    sigma_theta = -999.0
    
    print ('sigma_x = {}   sigma_y = {}   sigma_theta = {}'.format(sigma_x, sigma_y, sigma_theta))
    sigmas = [sigma_x, sigma_y, sigma_theta]
    
    return deltas, sigmas
    
    # Still do not know how to do delta_theta sigma  -- is this calculation needed?


# Print diagnostic load message
print("(least_squares): Least squares algorithm Version {} loaded!".format(__version__))


testing = False
if testing: 
    # Set test values  for arrays
    n = 10            # Number of positions
    x = np.array(range(10))     # true x-coordinate of each reference star: from 0 to 9
    y = np.array(range(1, 11))  # true y-coordinate of each reference star: from 1 to 10
    xt = x + 0.02     # measured centroid x-coordinate of each reference star
    yt = y + 0.01     # measured centroid y-coordinate of each reference star
    deltas, sigmas = ls_fit(n, x, y, xt, yt)
    """
    With these parameters output should be:
       >> (least_squares): Least squares algorithm Version 1.0 loaded!
       >> delta_x = -0.02   delta_y = -0.01   delta_theta = 7.57912251477e-16
       >> sigma_x = 4.564982887e-15   sigma_y = 3.69348008392e-15   sigma_theta = -999.0    
    """
    

