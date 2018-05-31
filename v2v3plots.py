from __future__ import print_function, division
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import TA_functions as taf




# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Oct 2016 - Version 1.0: initial version completed


'''
This script crates the plots in DeltaV2-DeltaV3 space, that compare the 3 tests ran.
     TEST1 - Average positions P1 and P2, transform to V2-V3 space, and compare to average
             reference positions (V2-V3 space)
     TEST2 - Transform individual positions P1 and P2 to V2-V3 space, average V2-V3 space
             positions, and compare to average reference positions.
     TEST3 - Transform P1 and P2 individually to V2-V3 space and compare star by star and
             position by position.
'''


def v2theta_plot(case, meanV2, theta, save_plot=False, show_plot=False, destination=None):
    """
    This function creates the plot in V2-theta space of the 3 tests: averaging in pixel space, averaging on sky,
     and no averaging.
    Args:
        case               -- string, for example '491Scene1_rapid_real_bgFrac0.3'
        meanV2             -- list of 3 numpy array of theta values of V2 for Tests 1, 2, and 3
        theta              -- list of 3 numpy array of theta values for Tests 1, 2, and 3
        save_plot          -- True or False
        show_plot          -- True or False
        destination        -- string, destination directory
    Returns:

    """
    # Set the paths
    results_path = os.path.abspath('../plots4presentationIST')

    # check if the plot is for an Nk set
    basename = case
    if not isinstance(meanV2, float):
        basename = case+'_'+str(len(meanV2[0]))+'samples'

    # Make the plot of V2-THETA
    plot_title = r'Residual Mean Calculated Angle, $\theta$'
    fig1 = plt.figure(1, figsize=(12, 10))
    ax1 = fig1.add_subplot(111)
    plt.suptitle(plot_title, fontsize=18, y=0.96)
    plt.title(basename)
    plt.xlabel(r'$\Delta$V2')
    plt.ylabel(r'$\theta$')
    xmin, xmax = -0.01, 0.01
    ymin, ymax = -40.0, 40.0
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.hlines(0.0, xmin, xmax*2, colors='k', linestyles='dashed')
    plt.vlines(0.0, ymin, ymax*2, colors='k', linestyles='dashed')
    plt.plot(meanV2[0], theta[0], 'b^', ms=10, alpha=0.7, label='Avg in Pixel Space')
    plt.plot(meanV2[1], theta[1], 'go', ms=10, alpha=0.7, label='Avg in Sky')
    plt.plot(meanV2[2], theta[2], 'r*', ms=13, alpha=0.7, label='No Avg')
    # Shrink current axis by 10%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))   # put legend out of the plot box
    if isinstance(meanV2, float):
        textinfig = r'V2$\mean$ = %0.2f    $\theta$ = %0.2f' % (meanV2, theta)
        ax1.annotate(textinfig, xy=(1.02, 0.35), xycoords='axes fraction' )
    if save_plot:
        if destination is not None:
            fig_name = os.path.join(destination, 'thetaV2_'+basename+'.jpg')
        else:
            fig_name = os.path.join(results_path, 'thetaV2_'+basename+'.jpg')
        fig1.savefig(fig_name)
        print ("\n Plot saved: ", fig_name)
    if show_plot:
        plt.show()
    else:
        plt.close('all')


def v3theta_plot(case, meanV3, theta, save_plot=False, show_plot=False, destination=None):
    """
    This function creates the plot in V3-theta space of the 3 tests: averaging in pixel space, averaging on sky,
     and no averaging.
    Args:
        case               -- string, for example '491Scene1_rapid_real_bgFrac0.3'
        meanV3             -- list of 3 numpy array of theta values of V3 for Tests 1, 2, and 3
        theta              -- list of 3 numpy array of theta for Tests 1, 2, and 3
        save_plot          -- True or False
        show_plot          -- True or False
        destination        -- string, destination directory
    Returns:

    """
    # Set the paths
    results_path = os.path.abspath('../plots4presentationIST')

    # check if the plot is for an Nk set
    basename = case
    if not isinstance(meanV3, float):
        basename = case+'_'+str(len(meanV3[0]))+'samples'

    # Make the plot of V3-THETA
    plot_title = r'Residual Mean Calculated Angle, $\theta$'
    fig1 = plt.figure(1, figsize=(12, 10))
    ax1 = fig1.add_subplot(111)
    plt.suptitle(plot_title, fontsize=18, y=0.96)
    plt.title(basename)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\Delta$V3')
    xmin, xmax = -40.0, 40.0
    ymin, ymax = -0.02, 0.02
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.hlines(0.0, xmin, xmax*2, colors='k', linestyles='dashed')
    plt.vlines(0.0, ymin, ymax*2, colors='k', linestyles='dashed')
    plt.plot(theta[0], meanV3[0], 'b^', ms=10, alpha=0.7, label='Avg in Pixel Space')
    plt.plot(theta[1], meanV3[1], 'go', ms=10, alpha=0.7, label='Avg in Sky')
    plt.plot(theta[2], meanV3[2], 'r*', ms=13, alpha=0.7, label='No Avg')
    # Shrink current axis by 10%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))   # put legend out of the plot box
    if isinstance(meanV2, float):
        textinfig = r'V3$\mean$ = %0.2f    $\theta$ = %0.2f' % (meanV3, theta)
        ax1.annotate(textinfig, xy=(1.02, 0.35), xycoords='axes fraction' )
    if save_plot:
        if destination is not None:
            fig_name = os.path.join(destination, 'thetaV3_'+basename+'.jpg')
        else:
            fig_name = os.path.join(results_path, 'thetaV3_'+basename+'.jpg')
        fig1.savefig(fig_name)
        print ("\n Plot saved: ", fig_name)
    if show_plot:
        plt.show()
    else:
        plt.close('all')


def theta_plot(case, theta, save_plot=False, show_plot=False, destination=None, print_side_values=None):
    """
    This function creates the plot of theta for the 3 tests: averaging in pixel space, averaging on sky,
     and no averaging.
    Args:
        case               -- string, for example '491Scene1_rapid_real_bgFrac0.3'
        theta              -- list of 3 numpy array of theta for Tests 1, 2, and 3
        save_plot          -- True or False
        show_plot          -- True or False
        destination        -- string, destination directory
    Returns:

    """
    # Set the paths
    results_path = os.path.abspath('../plots4presentationIST')

    # check if the plot is for an Nk set
    basename = case
    #if not isinstance(theta, float):
    #    basename = case+'_'+str(len(theta[0]))+'samples'

    # Make the plot of THETA
    plot_title = r'Residual Mean Calculated Angle, $\theta$'
    fig1 = plt.figure(1, figsize=(12, 10))
    ax1 = fig1.add_subplot(111)
    plt.suptitle(plot_title, fontsize=18, y=0.96)
    plt.title(basename)
    plt.xlabel('Sample Number')
    plt.ylabel(r'$\theta$  [marcsec]')
    xmin, xmax = -500.0, 5500.0
    #ymin, ymax = -40.0, 40.0
    ymin, ymax = min(theta[2])+min(theta[2])*0.2, max(theta[2])+max(theta[2])*0.2
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    #ax = plt.gca()
    # recompute the ax.dataLim
    #ax.relim()
    # update ax.viewLim using the new dataLim
    #ax.autoscale_view()
    plt.hlines(0.0, xmin, xmax, colors='k', linestyles='dashed')
    plt.vlines(0.0, ymin, ymax, colors='k', linestyles='dashed')
    plt.plot(theta[0], 'b^', ms=10, alpha=0.7, label='Avg in Pixel Space')
    plt.plot(theta[1], 'go', ms=10, alpha=0.7, label='Avg in Sky')
    plt.plot(theta[2], 'r*', ms=13, alpha=0.7, label='No Avg')
    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.6))   # put legend out of the plot box
    if print_side_values is not None:
        # standard deviations and means of theta-axis for the 3 tests
        textinfig0 = '{:<13}'.format('Theta Standard Deviations and Means')
        textinfig1 = r'$\sigma(AvgPix)$={:<6.2f} $\mu(AvgPix)$={:<6.2f}'.format(print_side_values[0], print_side_values[1])
        textinfig2 = r'$\sigma(AvgSky)$={:<6.2f} $\mu(AvgSky)$={:<6.2f}'.format(print_side_values[2], print_side_values[3])
        textinfig3 = r'$ \sigma(NoAvg)$={:<6.2f}  $\mu(NoAvg)$={:<6.2f}'.format(print_side_values[4], print_side_values[5])
        ax1.annotate(textinfig0, xy=(1.02, 0.48), xycoords='axes fraction' )
        ax1.annotate(textinfig1, xy=(1.02, 0.45), xycoords='axes fraction' )
        ax1.annotate(textinfig2, xy=(1.02, 0.42), xycoords='axes fraction' )
        ax1.annotate(textinfig3, xy=(1.02, 0.39), xycoords='axes fraction' )
    if save_plot:
        if destination is not None:
            fig_name = os.path.join(destination, basename+'_thetas.jpg')
        else:
            fig_name = os.path.join(results_path, 'theta_'+basename+'.jpg')
        fig1.savefig(fig_name)
        print ("\n Plot saved: ", fig_name)
    if show_plot:
        plt.show()
    else:
        plt.close('all')


def make_plot(cwincase, arrx, arry, xlabel, ylabel, plot_title=None, labels_list=None, xlims=None, ylims=None,
              print_side_string = None, print_side_values=None,
              save_plot=False, show_plot=True, destination=None, star_sample=None, square=True):
    '''
    This function creates a plot of the given arrays for the 3 tests.
    Args:
        cwincase: string, for example '491Scene1_rapid_real_bgFrac0.3_Nsigma2' (this will be the subtitle)
        arrx: list of 3 numpy arrays
        arry: list of 3 numpy arrays
        xlabel: string, name of x-axis
        ylabel: string, name of y-axis
        plot_title: string, title of the plot
        labels_list: list of 3 strings
        xlims: list, limits of x-axis
        ylims: list, limits of y-axis
        print_side_string: list, strings to print on the side (sigma or mu)
        print_side_values: list, values to print on the side (standard deviations or means)
        save_plot: True or False
        show_plot: True or False
        destination: path and name of the resulting plot

    Returns:
        Nothing
    '''
    fig1 = plt.figure(1, figsize=(12, 10))
    ax1 = fig1.add_subplot(111)
    plt.suptitle(plot_title, fontsize=18, y=0.96)
    plt.title(cwincase)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlims is None:
        x = np.abs(arrx[0])
        xmax = max(x)+max(x)*0.2
        xlims = [-1*xmax, xmax]
    if ylims is None:
        y = np.abs(arry[0])
        ymax = max(y)+max(y)*0.2
        ylims = [-1*ymax, ymax]
    # Compare which one is larger and use that one
    if square:
        if xlims[1] > ylims[1]:
            ylims = xlims
        else:
            xlims = ylims
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.hlines(0.0, xlims[0], xlims[1], colors='k', linestyles='dashed')
    plt.vlines(0.0, ylims[0], ylims[1], colors='k', linestyles='dashed')
    plt.plot(arrx[0], arry[0], 'b^', ms=10, alpha=0.5, label=labels_list[0])
    if len(arrx) != 1:
        plt.plot(arrx[1], arry[1], 'go', ms=10, alpha=0.5, label=labels_list[1])
        plt.plot(arrx[2], arry[2], 'r*', ms=13, alpha=0.5, label=labels_list[2])
    if star_sample is not None:
        if len(arrx) == 3:
            stars_sample1, stars_sample2, stars_sample3 = star_sample
            # double the lenght of the list for test 3 because position 2 after position 1
            new_star_sample3 = []
            for position in range(2):
                for st in stars_sample3:
                    new_star_sample3.append(st)
        else:
            stars_sample1 = star_sample[0]
        x_reject, y_reject = [-0.05, 0.05], [-0.05, 0.05]
        # for test1 and 2
        for si, xi, yi in zip(stars_sample1, arrx[0], arry[0]):
            if yi >= y_reject[1] or yi <= y_reject[0] or xi >= x_reject[1] or xi <= x_reject[0]:
                si = int(si)
                subxcoord = 5
                subycoord = 0
                side = 'left'
                plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
        if len(arrx) == 1:
            if len(arrx) == 2*len(stars_sample1):   # then we are dealing with TEST3 data
                new_star_sample3 = []
                for position in range(2):
                    for st in stars_sample3:
                        new_star_sample3.append(st)
                for si, xi, yi in zip(new_star_sample3, arrx[0], arry[0]):
                    if yi >= y_reject[1] or yi <= y_reject[0] or xi >= x_reject[1] or xi <= x_reject[0]:
                        si = int(si)
                        subxcoord = 5
                        subycoord = 0
                        side = 'left'
                        plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
        else:
            # for test3
            for si, xi, yi in zip(new_star_sample3, arrx[2], arry[2]):
                if yi >= y_reject[1] or yi <= y_reject[0] or xi >= x_reject[1] or xi <= x_reject[0]:
                    si = int(si)
                    subxcoord = 5
                    subycoord = 0
                    side = 'left'
                    plt.annotate('{}'.format(si), xy=(xi,yi), xytext=(subxcoord, subycoord), ha=side, textcoords='offset points')
    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.7))   # put legend out of the plot box
    if print_side_values is not None:
        # standard deviations and means of x-axis for the 3 tests
        textinfig0 = '{:<13}'.format(print_side_string[0])
        ax1.annotate(textinfig0, xy=(1.02, 0.58), xycoords='axes fraction' )
        # standard deviations and means of y-axis for the 3 tests
        textinfig0 = r'{:<13}'.format(print_side_string[1])
        ax1.annotate(textinfig0, xy=(1.02, 0.43), xycoords='axes fraction' )
        if len(arrx) == 3:
            #print_side_values = [0 T1sigmaV2, 1 T1meanV2, 2 T2sigmaV2, 3 T2meanV2, 4 T3sigmaV2, 5 T3meanV2,
            #                     6 T1sigmaV3, 7 T1meanV3, 8 T2sigmaV3, 9 T2meanV3, 10 T3sigmaV3, 11 T3meanV3]
            # standard deviations and means of x-axis for the 3 tests
            textinfig1 = r'$\sigma(AvgPix)$={:<6.2f} $\mu(AvgPix)$={:<6.2f}'.format(print_side_values[0], print_side_values[1])
            textinfig2 = r'$\sigma(AvgSky)$={:<6.2f} $\mu(AvgSky)$={:<6.2f}'.format(print_side_values[2], print_side_values[3])
            textinfig3 = r'$ \sigma(NoAvg)$={:<6.2f}  $\mu(NoAvg)$={:<6.2f}'.format(print_side_values[4], print_side_values[5])
            ax1.annotate(textinfig1, xy=(1.02, 0.55), xycoords='axes fraction' )
            ax1.annotate(textinfig2, xy=(1.02, 0.52), xycoords='axes fraction' )
            ax1.annotate(textinfig3, xy=(1.02, 0.49), xycoords='axes fraction' )
            # standard deviations and means of y-axis for the 3 tests
            textinfig1 = r'$\sigma(AvgPix)$={:<6.2f} $\mu(AvgPix)$={:<6.2f}'.format(print_side_values[6], print_side_values[7])
            textinfig2 = r'$\sigma(AvgSky)$={:<6.2f} $\mu(AvgSky)$={:<6.2f}'.format(print_side_values[8], print_side_values[9])
            textinfig3 = r' $\sigma(NoAvg)$={:<6.2f}  $\mu(NoAvg)$={:<6.2f}'.format(print_side_values[10], print_side_values[11])
            ax1.annotate(textinfig1, xy=(1.02, 0.40), xycoords='axes fraction' )
            ax1.annotate(textinfig2, xy=(1.02, 0.37), xycoords='axes fraction' )
            ax1.annotate(textinfig3, xy=(1.02, 0.34), xycoords='axes fraction' )
        else:
            # standard deviations and means of x-axis
            textinfig1 = r'$\sigma$={:<6.2f} $\mu$={:<6.2f}'.format(print_side_values[0], print_side_values[1])
            ax1.annotate(textinfig1, xy=(1.02, 0.55), xycoords='axes fraction' )
            # standard deviations and means of y-axis
            textinfig1 = r'$\sigma$={:<6.2f} $\mu$={:<6.2f}'.format(print_side_values[2], print_side_values[3])
            ax1.annotate(textinfig1, xy=(1.02, 0.40), xycoords='axes fraction' )
    if save_plot:
        fig1.savefig(destination)
        print ("\n Plot saved: ", destination)
    if show_plot:
        plt.show()
    else:
        plt.close('all')


def get_stdevmeans4print_side_values(muV2, muV3):
    V2stdevs, V2means, V3stdevs, V3means = [], [], [], []
    for mv2, mv3 in zip(muV2, muV3):
        sd2, m2 = taf.find_std(mv2)
        V2stdevs.append(sd2)
        V2means.append(m2)
        sd3, m3 = taf.find_std(mv3)
        V3stdevs.append(sd3)
        V3means.append(m3)
    print_side_values = [V2stdevs[0], V2means[0], V2stdevs[1], V2means[1], V2stdevs[2], V2means[2],
                         V3stdevs[0], V3means[0], V3stdevs[1], V3means[1], V3stdevs[2], V3means[2]]
    return print_side_values



#######################################################################################################################


if __name__ == '__main__':


    #### Set parameters

    centroid_windows = [3, 5, 7]
    min_elements = 4
    Nsigma2plot = 2.5
    stars_in_sample = 5
    case = '2DetsScene1_rapid_real_bgFrac0.3_thres01'
    save_plot = False
    show_plot = True
    milliarcsec = True              # arcsec if set to False
    used_abs_threshold = True      # only plot least squares routine results if set to False
    good_and_ugly_stars = True     # only plot good stars if set to False


    ######################################################

    # general path to text files
    star_sample_dir = repr(stars_in_sample)+'_star_sample'
    type_of_stars = 'only_good_stars'
    if good_and_ugly_stars:
        type_of_stars = 'good_and_uglies'
    gen_path = os.path.abspath('../resultsXrandomstars/'+type_of_stars+'/'+star_sample_dir)
    if used_abs_threshold and min_elements==4:
        gen_path += '/abs_threshold'
    elif used_abs_threshold and min_elements !=4:
        gen_path += '/diff_min_elements_abs_threshold'
    #results_path = os.path.abspath('../plots4presentationIST')
    results_path = gen_path
    if good_and_ugly_stars:
        results_path = gen_path
    print (gen_path)

    # Loop over centroid_windows
    if min_elements != 4:
        case += '_minele'+repr(min_elements)
    for cwin in centroid_windows:
        # load the data fom the 3 tests
        test_files_list = glob(os.path.join(gen_path, 'TEST*'+case+'*_Nsigma'+repr(Nsigma2plot)+'*'+repr(cwin)+'.txt'))
        #          0        1        2          3        4       5         6         7         8
        # data = sample, sigmaV2, sigmaV3, sigmaTheta, meanV2, meanV3, meanTheta, LastIter, RejStars
        dataT1 = np.loadtxt(test_files_list[0], comments='#', unpack=True)
        dataT2 = np.loadtxt(test_files_list[1], comments='#', unpack=True)
        dataT3 = np.loadtxt(test_files_list[2], comments='#', unpack=True)

        # compact variables and convert to milli arcsec
        conversion = 1.0
        if milliarcsec:
            conversion = 1000.0
        muV2 = [dataT1[4]*conversion, dataT2[4]*conversion, dataT3[4]*conversion]
        muV3 = [dataT1[5]*conversion, dataT2[5]*conversion, dataT3[5]*conversion]
        sigmaV2 = [dataT1[1]*conversion, dataT2[1]*conversion, dataT3[1]*conversion]
        sigmaV3 = [dataT1[2]*conversion, dataT2[2]*conversion, dataT3[2]*conversion]
        theta = [dataT1[6]*conversion, dataT2[6]*conversion, dataT3[6]*conversion]
        cwincase = case+'_CentroidWin'+repr(cwin)+'_'+str(stars_in_sample)+'star'+str(len(dataT1[0]))+'samples'
        if used_abs_threshold:
            cwincase += '_withAbsThres'

        # calculate mean of the means and standard deviation of the means
        print_side_values = get_stdevmeans4print_side_values(muV2, muV3)
        cwincase += '_Nsigma'+repr(Nsigma2plot)

        # Means plot
        xlabel, ylabel = r'$\Delta$V2 [marcsecs]', r'$\Delta$V3 [marcsecs]'
        plot_title = 'Mean Residual Values'# for '+repr(len(sigmaV2[0]))+' samples of '+repr(stars_in_sample)+' stars'
        labels_list = ['Avg in Pixel Space', 'Avg in Sky', 'No Avg']
        #xlims, ylims = [-10.0, 10.0], [-10.0, 10.0]
        #xlims, ylims = [-1100.0, 1100.0], [-1100.0, 1100.0]
        xlims, ylims = None, None
        print_side_string = [r'$\Delta$V2', r'$\Delta$V3']
        destination = os.path.join(results_path, cwincase+'_means.jpg')
        make_plot(cwincase, muV2, muV3, xlabel, ylabel, plot_title=plot_title, labels_list=labels_list,
                  xlims=xlims, ylims=ylims, print_side_string = print_side_string, print_side_values=print_side_values,
                  save_plot=save_plot, show_plot=show_plot, destination=destination)


        # calculate mean of the sigmas and standard deviation of the sigmas
        print_side_values = get_stdevmeans4print_side_values(sigmaV2, sigmaV3)
        # Standard deviations plot
        xlabel, ylabel = r'$\Delta$V2 [marcsecs]', r'$\Delta$V3 [marcsecs]'
        plot_title = 'Standard Deviations'
        labels_list = ['Avg in Pixel Space', 'Avg in Sky', 'No Avg']
        #xlims, ylims = [-5.0, 50.0], [-5.0, 50.0]
        #xlims, ylims = None, None
        print_side_string = [r'$\Delta$V2', r'$\Delta$V3']
        destination = os.path.join(results_path, cwincase+'_stdevs.jpg')
        make_plot(cwincase, sigmaV2, sigmaV3, xlabel, ylabel, plot_title=plot_title, labels_list=labels_list,
                  xlims=xlims, ylims=ylims, print_side_string = print_side_string, print_side_values=print_side_values,
                  save_plot=save_plot, show_plot=show_plot, destination=destination)


        # calculate mean of the thetas and standard deviation of the thetas
        theta_stdevs, theta_means = [], []
        for th in theta:
            sd, m = taf.find_std(th)
            theta_stdevs.append(sd)
            theta_means.append(m)
        print_side_values = [theta_stdevs[0], theta_stdevs[1], theta_stdevs[2],
                             theta_means[0], theta_means[1], theta_means[2]]
        # Thetas plot
        #v2theta_plot(cwincase, muV2, theta, save_plot=save_plot, show_plot=show_plot, destination=None)
        #v3theta_plot(cwincase, muV3, theta, save_plot=save_plot, show_plot=show_plot, destination=None)
        destination = results_path
        theta_plot(cwincase, theta, save_plot=save_plot, show_plot=show_plot,
                   destination=destination, print_side_values=print_side_values)

