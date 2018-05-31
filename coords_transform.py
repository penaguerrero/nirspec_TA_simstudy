from __future__ import print_function, division
import numpy as np


# Header
__author__ = "Maria A. Pena-Guerrero"
__version__ = "1.0"


"""
This script can transform from sky to detector coordinates (backward direction), and from
detector to sky (forward direction).

Keyword arguments:
    transf_direction  -- Direction of transformation, string: forward or backward
    detector          -- Which detector are we working with, integer: 491 or 492
    filter_input      -- Filter being used, string: F140X, CLEAR, or F110W
    x_input           -- Depending on transform direction, float: either X or V2 centroid
    y_input           -- Depending on transform direction, float: either X or V2 centroid
    tilt              -- Use tilt: True or False
    arcsecs           -- Units of V2 and V3, either arcsecs (=True) or degrees (=False)
    debug             -- Print diagnostics to screen: True or False

Output(s):
    x_out = transformed X position
    y_out = transformed Y position

Example usage:
    import coords_transform as ct
    x_out, y_out = ct.coords_transf(transf_direction, detector, filter_input, x_input, y_input, tilt, arcsecs, debug)


*** Testing suite of the script at the bottom
"""


# HISTORY
#    1. Sept 2015 - Vr. 1.0: Initial Python translation of IDL code.


###########################################################################################################


def coords_transf(transf_direction, detector, filter_input, x_input, y_input, tilt=False, arcsecs=False, debug=False):
    if transf_direction == "forward":
        x_out, y_out = ct_forward(transf_direction, detector, filter_input, x_input, y_input, tilt, debug)
    elif transf_direction == "backward":
        x_out, y_out = ct_backward(transf_direction, detector, filter_input, x_input, y_input, tilt, debug)
    if arcsecs:
        x_out, y_out = x_out*3600.0, y_out * 3600.0
    return x_out, y_out
 
    
def set_params(transf_direction, detector, filter_input, tilt, debug):
    path2text_files = "../Coords_transforms/files_from_tony/"
    gwa_xtil = 0.0  
    gwa_ytil = 0.0
    if not tilt:
        # Constants used for the no tilt case
        gwa_xtil = 0.343027200  
        gwa_ytil = 0.197058170
        
    # Read the tilt correction file
    tilt_file = path2text_files+"tilt_correction.txt"
    AX, AY, rx0, ry0 = np.loadtxt(tilt_file, skiprows=1, unpack=True)
    if debug:
        print ("(coords_transform - set_params):  slopeAX=", AX, " SlopeAY=", AY, " intercept_rx0=", rx0, " intercept_ry0=", ry0)
        print ("                               :  gwa_xtil=", gwa_xtil, "gwa_ytil=", gwa_xtil)
    delta_theta_x = 0.5 * AX * (gwa_ytil - rx0)    
    delta_theta_y = 0.5 * AY * (gwa_xtil - ry0)   
     
    # Read the detector correction file
    detector_gwa_txt = path2text_files+str(detector)+"_GWA.txt"
    detidx, detxfor, detyfor, detxback, detyback = np.loadtxt(detector_gwa_txt, skiprows=2, unpack=True)
    FitOrder_det = 5
    if debug:
        print ("(coords_transform - set_params):  For detector: ", detector, " and transf_direction: ", transf_direction)
        print ("                                  delta_theta_x=", delta_theta_x, " delta_theta_y=",delta_theta_y)
        if transf_direction == "backward":
            print ("(coords_transform - set_params):  lenght_index", len(detidx),
                   " lenght_xBackwardCoefficients_ote=", len(detxback), " lenght_yBackwardCoefficients_ote=", len(detyback))
            #print ("                                test_index    xBackwardCoefficients_ote   yBackwardCoefficients_ote    ")
            #for i in detidx:
            #    print ("                                ", i, detxback[i], detyback[i])
        elif transf_direction == "forward":
            print ("(coords_transform - set_params):  lenght_index", len(detidx), 
                   " lenght_xForwardCoefficients_ote=", len(detxfor), " lenght_yForwardCoefficients_ote=", len(detyfor))
            #print ("                                test_index    xForwardCoefficients_ote   yForwardCoefficients_ote    ")
            #for i in detidx:
            #    print ("                                ", i, detxfor[i], detyfor[i])
        
    # Read the filter correction file
    filter_gwa_txt = path2text_files+filter_input+"_GWA_OTE.txt"
    filidx, filxfor, filyfor, filxback, filyback = np.loadtxt(filter_gwa_txt, skiprows=2, unpack=True)
    if debug:
        print ("(coords_transform - set_params):  For filter: ", filter_input, " and transf_direction: ", transf_direction)
        if transf_direction == "backward":
            print ("(coords_transform - set_params):  length_index", len(filidx), 
                    " length_xBackwardCoefficients_det=", len(filxback), " lenght_yBackwardCoefficients_det=", len(filyback))
            #print ("                                test_index, xBackwardCoefficients_det   yBackwardCoefficients_det    ")
            #for i in filidx:
            #    print ("                                ", i, filxback[i], filyback[i])
        elif transf_direction == "forward":
            print ("(coords_transform - set_params):  lenght_index", len(filidx), 
                   " length_xForwardCoefficients_det=", len(filxfor), " length_yForwardCoefficients_det=", len(filyfor))
            #print ("                                test_index, xForwardCoefficients_det   yForwardCoefficients_det    ")
            #for i in filidx:
            #    print ("                                ", i, filxfor[i], filyfor[i])
    FitOrder_ote = 5
    if transf_direction == "backward":
        direction_data = [delta_theta_x, delta_theta_y, detxback, detyback, FitOrder_det, filxback, filyback, FitOrder_ote]
    elif transf_direction == "forward":
        direction_data = [delta_theta_x, delta_theta_y, detxfor, detyfor, FitOrder_det, filxfor, filyfor, FitOrder_ote]
    return  direction_data


def ct_backward(transf_direction, detector, filter_input, x_input, y_input, tilt, debug):
    """ Perform coordinates transform in the BACKWARD direction (sky to detector) """
    direction_data = set_params(transf_direction, detector, filter_input, tilt, debug)
    delta_theta_x, delta_theta_y, detxback, detyback, FitOrder_det, filxback, filyback, FitOrder_ote = direction_data
    # Coord transformation to go from OTE to GWA
    ict = -1
    xt, yt = 0.0, 0.0
    for i in range(FitOrder_ote+1):
        for j in range(FitOrder_ote+1-i):
            ict += 1
            xt += filxback[ict] * x_input ** i * y_input ** j
            yt += filyback[ict] * x_input ** i * y_input ** j
            #print ("ict, i, j, xt, yt", ict, i, j, xt, yt)
    if debug:
        print ("(coords_transform - ct_backward):  x_input=", x_input, " y_input=", y_input)
        print ("                                   GWA_x=",xt,"   GWA_y=",yt)
        #raw_input("*** press enter to continue... ")
    # Now that we have angular position at GWA of xy SCA pixels in xt, yt, do tilt-correction
    w = 180.0 * 3600.0    # arcsec in pi radians, so 1 arcsec is pi/w, get pi from cos(-1.0)
    w1 = np.arccos(-1.0)  # gives value of pi
    w2 = w1 / w           # 1 arcsec expressed in radians
    delta_theta_xrad = delta_theta_x * w2   # now calculated for the general case
    delta_theta_yrad = delta_theta_y * w2
    if debug:
        print ("w, w1, w2: ", w, w1, w2)
        print ("(coords_transform - ct_backward):  delta_theta_x=", delta_theta_x, " delta_theta_y=", delta_theta_y)
        print ("                                   delta_theta_xrad=", delta_theta_xrad, "delta_theta_yrad=", delta_theta_yrad)
    # Do backward rotation
    # calculate direction cosines of xt, yt, (xgwa, ygwa)
    v = np.abs(np.sqrt(1.0 + xt*xt + yt*yt))
    x3 = xt / v
    y3 = yt / v
    z3 = 1.0 / v
    # do inverse rotation around the x-axis
    x2 = x3
    y2 = y3 + delta_theta_xrad*z3
    z2 = np.sqrt(1.0 - x2*x2 - y2*y2)
    # rotate to mirror reference system with small angle approx. and perform rotation
    x1 = x2 - delta_theta_yrad*z2
    y1 = y2
    z1 = np.sqrt(1.0 - x1*x1 - y1*y1)
    # rotate reflected ray back to reference GWA coordinate system (use small angle approx.),
    # first with an inverse rotation around the y-axis:
    x0 = -1.0*x1 + delta_theta_yrad*np.sqrt(1.0 - x1*x1 - (y1+delta_theta_xrad*z1)*(y1+delta_theta_xrad*z1))
    y0 = -1.0*y1 - delta_theta_xrad*z1
    z0 = np.sqrt(1.0 - x0*x0 - y0*y0)
    
    xt_corr = x0/z0
    yt_corr = y0/z0
    
    if debug:
        print ("(coords_transform - ct_backward):  Checking tilt rotation")
        print ("                                   v=", v)
        print ("                                   x0=", x0, " y0=", y0, " z0=", z0)
        print ("                                   x1=", x1, " y1=", y1, " z1=", z1)
        print ("                                   x2=", x2, " y2=", y2, " z2=", z2)
        print ("                                   x3=", x3, " y3=", y3, " z3=", z3)
        print ("                                   xt_corr=", xt_corr, " yt_corr", yt_corr)
        #raw_input("*** press enter to continue... ")
    
    # coord transform to go from tilt-corrected GWA to detector - 5th order polynomial in backward coefficients: 
    # detxback, detyback
    ict_det = -1
    xt_det, yt_det = 0.0, 0.0
    for i in range(FitOrder_det+1):
        for j in range(FitOrder_det+1-i):
            ict_det += 1
            xt_det += detxback[ict_det] * xt_corr ** i * yt_corr ** j
            yt_det += detyback[ict_det] * xt_corr ** i * yt_corr ** j
            #print ("ict, i, j, xt, yt", ict_det, i, j, xt_det, yt_det)
    if debug:
        print ("(coords_transform - ct_backward):  x_input=", x_input, " y_input=", y_input, " OTE_x=", xt_det, "OTE_y=", yt_det) 
        #raw_input("*** press enter to continue... ")
    
    # Final backward output
    x_out, y_out =  xt_det, yt_det
    return x_out, y_out


def ct_forward(transf_direction, detector, filter_input, x_input, y_input, tilt, debug):
    # Perform coordinates transform in the FORWARD direction (detector to sky)
    direction_data = set_params(transf_direction, detector, filter_input, tilt, debug)
    delta_theta_x, delta_theta_y, detxfor, detyfor, FitOrder_det, filxfor, filyfor, FitOrder_ote = direction_data
    # Coord transformation to go from OTE to GWA
    ict = -1
    xt, yt = 0.0, 0.0
    for i in range(FitOrder_det+1):
        for j in range(FitOrder_det+1-i):
            ict += 1
            xt += detxfor[ict] * x_input ** i * y_input ** j
            yt += detyfor[ict] * x_input ** i * y_input ** j
    if debug:
        print ("(coords_transform - ct_forward):  x_input=", x_input, " y_input=", x_input) 
        print ("                                  GWA_x=", xt, " GWA_y=", yt)
        #raw_input("*** press enter to continue... ")
    # Now that we have angular position at GWA of xy SCA pixels in xt, yt, do tilt-correction
    w = 180. * 3600.    # arcsec in pi radians, so 1 arcsec is pi/w, get pi from cos(-1.0)
    w1 = np.arccos(-1.0)   # gives value of pi
    w2 = w1 / w         # 1 arcsec expressed in radians
    delta_theta_xrad = delta_theta_x * w2   # now calculated for the general case
    delta_theta_yrad = delta_theta_y * w2
    if debug:
        print ("(coords_transform - ct_forward):  delta_theta_x=", delta_theta_x, " delta_theta_y=", delta_theta_y)
        print ("                                  delta_theta_xrad=", delta_theta_xrad, "delta_theta_yrad=", delta_theta_yrad)
        #raw_input("*** press enter to continue... ")
    
    # Do forward rotation
    # calculate direction cosines of xt, yt, (xgwa, ygwa)
    v = np.abs(np.sqrt(1.0 + xt*xt + yt*yt))
    x0 = xt / v
    y0 = yt / v
    z0 = 1.0 / v
    # rotate to mirror reference system with small angle approx. and perform rotation
    x1 = -1 * (x0 - delta_theta_yrad * np.sqrt(1.0 - x0**2 - (y0+delta_theta_xrad*z0)**2))
    y1 = -1 * (y0 + delta_theta_xrad * z0)
    z1 = np.sqrt(1.0 - x1**2 - y1**2)
    # rotate reflected ray back to ref GWA coord system with small angle approx., 
    # but first with an inverse rotation around the y-axis
    x2 = x1 + delta_theta_yrad *z1
    y2 = y1
    z2 = np.sqrt(1.0 - x2**2 - y2**2)
    # now do an inverse rotation around the x-axis
    x3 = x2
    y3 = y2 - delta_theta_xrad*2
    z3 = np.sqrt(1.0 - x3**2 - y3**2)
    # compute the cosines from direction cosines
    xt_corr = x3 / z3
    yt_corr = y3 / z3
    
    if debug:
        print ("(coords_transform - ct_forward):  Checking tilt rotation")
        print ("                                  v=", v)
        print ("                                  x0=", x0, " y0=", y0, " z0=", z0)
        print ("                                  x1=", x1, " y1=", y1, " z1=", z1)
        print ("                                  x2=", x2, " y2=", y2, " z2=", z2)
        print ("                                  x3=", x3, " y3=", y3, " z3=", z3)
        print ("                                  xt_corr=", xt_corr, " yt_corr", yt_corr)
        #raw_input("*** press enter to continue... ")
    
    # coord transform to go from tilt-corrected GWA to detector - 5th order polynomial in forward coefficients: 
    # detxfor, detyfor
    ict_ote = -1
    xt_ote, yt_ote = 0.0, 0.0
    for i in range(FitOrder_ote+1):
        for j in range(FitOrder_ote+1-i):
            ict_ote += 1
            xt_ote += filxfor[ict_ote] * xt_corr ** i * yt_corr ** j
            yt_ote += filyfor[ict_ote] * xt_corr ** i * yt_corr ** j
    if debug:
        print ("(coords_transform - ct_forward):  x_input=", x_input, " y_input=", y_input, " OTE_x=", xt_ote, "OTE_y=", yt_ote) 
    
    # Final forward output
    x_out, y_out =  xt_ote, yt_ote
    return x_out, y_out    
    

if __name__ == '__main__':

    # Print diagnostic load message
    print("(coords_transform): Coordinate transform algorithm Version {} loaded!".format(__version__))


    ###########################################################################################################


    # Test functions

    testing = False
    if testing:
        # Set test script parameters
        transf_direction = "forward"   # Direction of transformation, string: forward or backward
        detector = 491                 # Which detector are we working with: 491 or 492
        filter_input = "F140X"         # Filter being used, string: F140X, CLEAR, or F110W
        x_input = 1542.5               # Depending on transform direction, either X or V2 centroid
        y_input = 1542.5               # Depending on transform direction, either X or V2 centroid
        tilt = False                   # Use tilt: True or False
        debug = False                  # Print diagnostics to screen: True or False

        # Run transformation
        x_out, y_out = coords_transf(transf_direction, detector, filter_input, x_input, y_input, tilt, debug)
        print ("Final results: ")
        if transf_direction=="forward":
            print ("* For direction=", transf_direction, "   \n coordinates are=", x_out, y_out, " arcsec")
        if transf_direction=="backward":
            print ("* For direction=", transf_direction, "   \n coordinates are=", x_out, y_out, " pixels")
