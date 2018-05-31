from __future__ import print_function, division
import numpy as np
import copy



# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# May 2016 - Version 1.0: initial version completed



def ls_fit_iter(niter, x_input, y_input, xtrue, ytrue, Nsigma, arcsec=True, verbose=True, min_elements=4):
    """
    This function finds the standard deviation and mean from all points, then subtracts that mean from
    each point and compares it with the true values to reject all the points that are Nsigma away. It
    iterates until no more points are being rejected.
    Args:
         niter  = number of max iterations
         x_input= input measured x-centroid converted to V2
         y_input= input measured y-centroid converted to V3
         xtrue  = numpy array of true V2 positions
         ytrue  = numpy array of true V3 positions
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
    xt, yt = copy.deepcopy(x_input), copy.deepcopy(y_input)
    x, y = xtrue, ytrue

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


        line1 = '(abs_thres_least_squares_iterate):  iteration number: {}'.format(nit)
        if arcsec:
            line2 = '(abs_thres_least_squares_iterate):  delta_x = {}   delta_y = {}   delta_theta = {} arcsec'.format(delta_x, delta_y,
                                                                                                            delta_theta * (180.0/np.pi) * 3600.0)
        else:
            line2 = '(abs_thres_least_squares_iterate):  delta_x = {}   delta_y = {}   delta_theta = {} radians'.format(delta_x, delta_y,
                                                                                                             delta_theta)
        deltas = [delta_x, delta_y, delta_theta]

        # verify this coding for sigma_xtrue and sigma_ytrue
        sum_delta_x2 = 0.0
        sum_delta_y2 = 0.0
        sum_delta_theta2 = 0.0
        for i in range(n):
            sum_delta_x2 += (-xt[i] + x[i] - delta_x) * (-xt[i] + x[i] - delta_x)
            sum_delta_y2 += (-yt[i] + y[i] - delta_y) * (-yt[i] + y[i] - delta_y)

        sigma_x = np.sqrt(sum_delta_x2/n)   # sigma for xtrue-offset
        sigma_y = np.sqrt(sum_delta_y2/n)   # sigma for ytrue-offset

        # for now set sigma_thera to bogus value  (we don't presently know how to calculate it)
        sigma_theta = -999.0

        line3 = '(abs_thres_least_squares_iterate):  sigma_x = {}   sigma_y = {}   sigma_theta = {}'.format(sigma_x, sigma_y, sigma_theta)
        sigmas = [sigma_x, sigma_y, sigma_theta]

        # calculate new best position and residuals for each coordinate (neglect any roll correction for now)
        xnewpos = x - delta_x
        xdiff = xnewpos - xt
        ynewpos = y - delta_y
        ydiff = ynewpos - yt

        # Rejection sequence:
        # reject any residuals that are not within N*sigma in EITHER coordinate
        # (this is slightly more restrictive than doing this for the vector sum,
        # but easier to implement right now)
        thres_x = Nsigma*sigma_x
        thres_y = Nsigma*sigma_y
        var_clip = xdiff[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        xcentroids_new = xt[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        ycentroids_new = yt[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        x_new = x[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]
        y_new = y[(np.where((np.abs(xdiff)<=thres_x) & (np.abs(ydiff)<=thres_y)))]

        # This commented following section would do rejection for N*sigma AND a specified
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
        line4 = '(abs_thres_least_squares_iterate):  elements_left={} out of original_elements={}'.format(elements_left, original_elements)

        if len(xcentroids_new) == len(xt) or len(xcentroids_new) < min_elements:
            xcentroids_new = xt
            ycentroids_new = yt
            x_new = x
            y_new = y
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

    # compact results
    centroids_new = [xcentroids_new, ycentroids_new]
    truexy = [x_new, y_new]

    return deltas, sigmas, lines2print, rejected_elements_idx, nit, centroids_new, truexy


def remove_largest_res(abs_threshold, centroids_new, truexy):
    # since the arrays have more elements, remove ONLY the farthest point than the threshold
    # start by finding the farthest point in either direction
    x_greater_than_thres, y_greater_than_thres = [], []
    idx_in_arrays = []
    rejected_element_idx = 0
    for i, xi in enumerate(centroids_new[0]):
        yi = centroids_new[1][i]
        residual_x = np.abs(xi - truexy[0][i])
        residual_y = np.abs(yi - truexy[1][i])
        #print (residual_x, residual_y)
        if residual_x > abs_threshold or residual_y > abs_threshold:
            x_greater_than_thres.append(residual_x)
            y_greater_than_thres.append(residual_y)
            idx_in_arrays.append(i)
    # now get which residual is the largest and remove that point from the arrays
    if len(x_greater_than_thres) == 0:
        print ('No points are farther than threshold.')
        return centroids_new, truexy, rejected_element_idx
    largest_Xresidual = max(x_greater_than_thres)
    largest_Yresidual = max(y_greater_than_thres)
    if largest_Xresidual > largest_Yresidual:
        idx_x = x_greater_than_thres.index(largest_Xresidual)
        idx = idx_in_arrays[idx_x]
    else:
        idx_y = y_greater_than_thres.index(largest_Yresidual)
        idx = idx_in_arrays[idx_y]
    new_centroids_x = np.delete(centroids_new[0], [idx])
    new_centroids_y = np.delete(centroids_new[1], [idx])
    new_true_x = np.delete(truexy[0], [idx])
    new_true_y = np.delete(truexy[1], [idx])
    new_centroids = [new_centroids_x, new_centroids_y]
    new_trues = [new_true_x, new_true_y]
    # find out which element was rejected
    for i, tx in enumerate(truexy[0]):
        if tx in new_true_x:
            continue
        else:
            rejected_element_idx = i
    return new_centroids, new_trues, rejected_element_idx


def LS_and_minelement_check(niter, Nsigma, arcsec, min_elements, verbose, LS_results):
    # unfold variables
    _, _, total_rejected_elements_idx, total_iterations, centroids, trues = LS_results

    # do least squares routine
    deltas, sigmas, lines2print, rejected_elements_idx, nit, new_centroids, new_trues = ls_fit_iter(niter,
                                                                                    centroids[0], centroids[1],
                                                                                    trues[0], trues[1],
                                                                                    Nsigma, arcsec=arcsec,
                                                                                    verbose=verbose,
                                                                                    min_elements=min_elements)
    # keep track of the rejected elements
    for rejel in rejected_elements_idx:
        total_rejected_elements_idx.append(rejel)

    # keep track of the iterations
    total_iterations += nit

    new_LS_results = [deltas, sigmas, total_rejected_elements_idx, total_iterations, new_centroids, new_trues]
    return new_LS_results



def abs_threshold_rejection(abs_threshold, niter, x_input, y_input, xtrue, ytrue, Nsigma, arcsec=True,
                            just_least_sqares=False, min_elements=4):
    # perform least squared rejection algorithm
    total_rejected_elements_idx  = []
    total_iterations = 0
    verbose = True
    deltas, sigmas, lines2print, rejected_elements_idx, nit, centroids, trues = ls_fit_iter(niter, x_input, y_input,
                                                                                    xtrue, ytrue,
                                                                                    Nsigma, arcsec=arcsec,
                                                                                    verbose=verbose,
                                                                                    min_elements=min_elements)
    # start the iteration count at 1 instead of 0
    nit += 1

    # skip the rest if just_least_sqares=True
    if just_least_sqares:
        return deltas, sigmas, lines2print, rejected_elements_idx, nit, centroids, trues
    else:
        # keep track of the rejected elements
        for rejel in rejected_elements_idx:
            total_rejected_elements_idx.append(rejel)

        # keep track of the iterations
        total_iterations += nit

        # check if the length of the new arrays has 4 elements and if so do not do anything else
        new_length = len(trues[0])

        if new_length <= min_elements:
            print ('(abs_threshold:) Finished abs_threshold routine. \n Arrays have {} or less elements! '.format(min_elements))
            return deltas, sigmas, lines2print, rejected_elements_idx, nit, centroids, trues

        # once no more elements are rejected from least squares routine do absolute threshold
        new_centroids, new_trues, rejected_element_idx = remove_largest_res(abs_threshold, centroids, trues)

        # update rejected elements
        total_rejected_elements_idx.append(rejected_element_idx)
        LS_results = [deltas, sigmas, total_rejected_elements_idx, total_iterations, new_centroids, new_trues]
        continue_main_while = True
        while continue_main_while:
            # do LS
            LS_results = LS_and_minelement_check(niter, Nsigma, arcsec, min_elements, verbose, LS_results)
            deltas, sigmas, total_rejected_elements_idx, total_iterations, centroids, trues = LS_results
            if len(trues[0]) == min_elements:
                print ('Finished abs_threshold routine. Minimum elements reached! ')
                return deltas, sigmas, lines2print, rejected_elements_idx, nit, centroids, trues
            elif len(trues[0]) > min_elements:
                # do abs threshold
                new_centroids, new_trues, rejected_element_idx = remove_largest_res(abs_threshold, centroids, trues)
                # update rejected elements
                total_rejected_elements_idx.append(rejected_element_idx)
                # length check
                new_length = len(new_trues[0])
                old_length = len(trues[0])
                if new_length == old_length:
                    print ('Finished abs_threshold routine. No more elements to be removed.')
                    return deltas, sigmas, lines2print, rejected_elements_idx, nit, new_centroids, new_trues
                else:
                    LS_results = [deltas, sigmas, total_rejected_elements_idx, total_iterations, new_centroids, new_trues]



if __name__ == '__main__':

    # Print diagnostic load message
    import TA_functions as taf
    print("(abs_thres_least_squares_iterate): Absolute threshold algorithm loaded!")

    testing = True
    example = 1
    if testing:
        print (" * Testing code... ")
        niter = 10            # max number of iterations
        Nsigma = 2.5
        abs_threshold = 0.32
        just_least_sqares = False
        arcsec = False
        min_elements = 4

        if example == 1:
            # EXAMPLE 1 -  mock data
            # Set test values  for arrays
            xtrue = np.array(range(10))     # true x-coordinate of each reference star: from 0 to 9
            ytrue = np.array(range(1, 11))  # true y-coordinate of each reference star: from 1 to 10
            xinput = xtrue + 0.02     # measured centroid x-coordinate of each reference star
            yinput = ytrue + 0.01     # measured centroid y-coordinate of each reference star
            """
            With these parameters output should be:
                (abs_thres_least_squares_iterate): Absolute threshold algorithm loaded!
                 * Testing code...
                These are the initial arrays:
                 xinput =  [ 0.02  1.02  2.02  3.02  4.02  5.02  6.02  7.02  8.02  9.02]
                 yinput =  [  1.01   2.01   3.01   4.01   5.01   6.01   7.01   8.01   9.01  10.01]
                (abs_thres_least_squares_iterate):  iteration number: 0
                (abs_thres_least_squares_iterate):  delta_x = -0.02   delta_y = -0.01   delta_theta = -0.00667878787879 radians
                (abs_thres_least_squares_iterate):  sigma_x = 4.564982887e-15   sigma_y = 3.69348008392e-15   sigma_theta = -999.0
                (abs_thres_least_squares_iterate):  elements_left=10 out of original_elements=10
                new_length =  10
                No points are farther than threshold.
                * before while loop
                deltas:  [-0.020000000000004309, -0.0099999999999961162, -0.0066787878787873825]
                sigmas:  [4.5649828869982097e-15, 3.6934800839225164e-15, -999.0]
                total_rejected_elements_idx:  [0]
                total_iterations:  1
                new_centroids:
                [ 0.02  1.02  2.02  3.02  4.02  5.02  6.02  7.02  8.02  9.02]
                [  1.01   2.01   3.01   4.01   5.01   6.01   7.01   8.01   9.01  10.01]
                new_trues:
                [0 1 2 3 4 5 6 7 8 9]
                [ 1  2  3  4  5  6  7  8  9 10]
                * in while loop
                (abs_thres_least_squares_iterate):  iteration number: 0
                (abs_thres_least_squares_iterate):  delta_x = -0.02   delta_y = -0.01   delta_theta = -0.00667878787879 radians
                (abs_thres_least_squares_iterate):  sigma_x = 4.564982887e-15   sigma_y = 3.69348008392e-15   sigma_theta = -999.0
                (abs_thres_least_squares_iterate):  elements_left=10 out of original_elements=10
                len(trues[0]) =  10
                No points are farther than threshold.
                No more elements to be removed
                    TLSdeltas:  [-0.020000000000004309, -0.0099999999999961162, -0.0066787878787873825]
                    TLSsigmas:  [4.5649828869982097e-15, 3.6934800839225164e-15, -999.0]
                    TLSlines2print:
                (abs_thres_least_squares_iterate):  iteration number: 0
                (abs_thres_least_squares_iterate):  delta_x = -0.02   delta_y = -0.01   delta_theta = -0.00667878787879 radians
                (abs_thres_least_squares_iterate):  sigma_x = 4.564982887e-15   sigma_y = 3.69348008392e-15   sigma_theta = -999.0
                (abs_thres_least_squares_iterate):  elements_left=10 out of original_elements=10
                    rejected_elements:  []
                    nit:  1
                    new_centroids:
                [ 0.02  1.02  2.02  3.02  4.02  5.02  6.02  7.02  8.02  9.02]
                [  1.01   2.01   3.01   4.01   5.01   6.01   7.01   8.01   9.01  10.01]
                    new_trues:
                [0 1 2 3 4 5 6 7 8 9]
                [ 1  2  3  4  5  6  7  8  9 10]

            """

        # EXAMPLE 2 -  mock data
        if example == 2:
            # after convert2MSAcenter:     Position 1                                 Position 2
            xinput = np.array([ 0.13015361,  0.10034588,  0.11215926,     0.13007643,  0.1002607,   0.11206002])
            yinput = np.array([ -0.00305732, -0.03733928, 0.01302006,    -0.00308166, -0.03736481,  0.0129836])
            xtrue =  np.array([ 0.13015443,  0.10033719,  0.11215901,     0.13007863,  0.10026139,  0.11208321])
            ytrue =  np.array([ -0.00305743, -0.03733417,  0.01302122,   -0.00308733, -0.03736407,  0.01299132])
            if arcsec:
                xinput, yinput = xinput*3600.0, yinput*3600.0
                xtrue, ytrue = xtrue*3600.0, ytrue*3600.0
                #xinput = np.array([ 468.44835572, 361.14052769, 403.66867649,   468.17048236, 360.83387397, 403.31142531])
                #yinput = np.array([ -10.9953641, -134.41040845,  46.88318515,   -11.082975,   -134.50232193,  46.7519372])
                #xtrue = np.array([ 468.45129306, 361.10923464,  403.66779144,   468.17840512, 360.8363467, 403.39490349])
                #ytrue = np.array([ -10.99576619, -134.39204083,  46.88737556,   -11.10340575, -134.49967787, 46.77973516])



        # Determine the standard deviation for each array
        diffx = xtrue - xinput
        diffy = ytrue - yinput
        sigma_x, mean_x = taf.find_std(diffx)
        sigma_y, mean_y = taf.find_std(diffy)
        # debug
        print ('These are the initial arrays: ')
        print (' xinput = ', xinput)
        print (' yinput = ', yinput)
        print (' original_means:     deltaX = ', mean_x, '       deltaY = ', mean_y)
        print (' original_std_devs:  sigmaX = ', sigma_x, '   sigmaY = ', sigma_y)
        abs_thres_data = abs_threshold_rejection(abs_threshold, niter, xinput, yinput, xtrue, ytrue, Nsigma,
                                    arcsec=arcsec, just_least_sqares=just_least_sqares, min_elements=min_elements)
        TLSdeltas, TLSsigmas, TLSlines2print, rejected_elements, nit, new_centroids, new_trues = abs_thres_data
        print ('    TLSdeltas: ', TLSdeltas)
        print ('    TLSsigmas: ', TLSsigmas)
        print ('    TLSlines2print: ' )
        for item in TLSlines2print:
            print (item)
        print ('    rejected_elements: ', rejected_elements)
        print ('    nit: ', nit)
        print ('    new_centroids: ')
        for item in new_centroids:
            print (item)
        print ('    new_trues: ')
        for item in new_trues:
            print (item)

