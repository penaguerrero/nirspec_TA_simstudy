import string
import os
from glob import glob



# HEADER
__author__ = "M. A. Pena-Guerrero"
__version__ = "1.0"

# HISTORY
# Feb 2016 - Version 1.0: initial version completed


"""
modifies the names of my files for detector to make them all
consistent.
"""

def change_fname(fol, fname):
    f = os.path.basename(fname)
    fname_list = string.split(f, sep=".")
    if "_redo" in fname_list[-2]:
        # remove all previous _redo
        kk = fname_list[-2].replace("_redo", "")
        # add the dots and the new file termination
        newfname = fol+"/"+kk+"_redo."+fname_list[-1]
    else:
        newfname = fol+"/"+fname_list[-2]+"_redo."+fname_list[-1]
    print "new fname: ", newfname
    # change the name
    os.system("mv "+fname+" "+newfname)


detector1 = "../PFforMaria/detector_491_"
detector2 = "../PFforMaria/detector_492_"

# folder names to modify for detector 491
detector_491_centroid_figs_redo = detector1+"centroid_figs_redo"
detector_491_resulting_centroid_txt_files_redo = detector1+"resulting_centroid_txt_files_redo"

# folder names to modify for detector 492
detector_492_plots_redo = detector2+"plots_redo"

# list of all file names to modify to *_redo.*
folders2modify = [detector_491_centroid_figs_redo, detector_491_resulting_centroid_txt_files_redo,
                  detector_492_plots_redo]

for fol in folders2modify:
    fol_list = glob(fol+"/*")
    #print "folder: ", fol
    for f in fol_list:
        #print os.path.isdir(f)
        if os.path.isdir(f):
            f_list = glob(f+"/*") 
            for fname in f_list:
                #print "fname: ", fname
                change_fname(fol, fname)
        else:
            #print "fname: ", f
            change_fname(fol, f)


print "Script finished! "
