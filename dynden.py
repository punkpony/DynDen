#! /usr/bin/python

from random import random
import argparse
import sys, os
from copy import deepcopy
import logging

import numpy as np
import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import MDAnalysis as MDA
from MDAnalysis.analysis.lineardensity import LinearDensity
from MDAnalysis.coordinates.base import Timestep
from MDAnalysis.analysis.rms import RMSD
from MDAnalysis import *


def print_dinosaur():
    '''
    most useful function in the code... rawrrr!
    '''
    logger.debug("")
    n=random()
    if n<0.2:
        head = "                  ___" 
    elif n>=0.2 and n<0.4:    
        head = "                  ##_"
    elif n>=0.4 and n<0.8:
        head = "                  _\\\\"
    else:
        head = "                  _|||"

    n=random()
    if n<0.2:
        face = "                 / '_)"
    elif n>=0.2 and n<0.4:    
        face = "                 / ^_)"
    elif n>=0.4 and n<0.8:
        face = "                 / *_)"
    else:
        face = "                 / 0 )"

    n=random()
    if n<0.2:
        back = "       _.----._/ /"
    elif n>=0.2 and n<0.4:    
        back = "       _.^^^^._/ /"
    elif n>=0.4 and n<0.8:
        back = "       _.||||._/ /"
    else:
        back = "       _.||||._/ /"

    n=random()
    if n<0.5:
        feet = "/_/    |_|  |_|"
    else:   
        feet = "/_/    | |  | |\n       o-o  o-o"

    logger.debug(head)
    logger.debug(face)
    logger.debug("                / /")
    logger.debug(back)
    logger.debug("     _/         /\n   _/ _( |  (  /\n _/_/  | |--| |")
    logger.debug(feet)


def get_partial(u, box_dims, sel, bins=100, start=0, stop=-1):
        '''        
        params: universe
        params box_dims measuring box dimensions along z [frame, min, max, dim]
        params: selection name of residue of interest
        params: number of bins
        params: start first frame to study
        params: stop last frame to study
        returns: density of each timepoint
        '''
        mol = u.select_atoms(sel)

        res = [] #density collector
        cnt = 0
        for ts in u.trajectory[start:stop]:

            curr_box = box_dims[cnt]
            binning = np.linspace(curr_box[1]-1, curr_box[2]+1, bins)

            result = np.histogram(mol.atoms.positions[:, 2], weights=mol.masses, bins=binning)[0]            
            res.append(result)

            logger.debug(">> frame %s density..."%ts.frame)
            cnt += 1

        result = np.array(res)
        return result


def block_average(result, tsteps):
        '''        
        params: result density profile for each timestep
        params: tsteps time measuring points
        returns: density of each average slice, timesteps, z-box binning points
        '''
        tm = []
        res2 = []
        for i in range(0, len(tsteps)-1):

            try:
                avg_dens = np.mean(result[tsteps[i]:tsteps[i+1]], axis=0)
                if len(avg_dens) > 0:
                    res2.append(avg_dens)
                    tm.append(tsteps[i])
            except:
                continue

        # prepare data for densities, associated time frames, and density evaluation points
        dens = np.array(res2)
        time = np.array(tm)

        return dens, time


def sliding_window(result, tsteps):
        '''        
        params: result density profile for each timestep
        params: tsteps time measuring points
        returns: density of each average slice, timesteps, z-box binning points
        '''        
        tm = []
        res2 = []
        for i in range(1, len(tsteps)-1):

            try:
                avg_dens = np.mean(result[tsteps[i-1]:tsteps[i+1]], axis=0)
                if len(avg_dens) > 0:
                    res2.append(avg_dens)
                    tm.append(tsteps[i])
            except:
                continue

        # prepare data for densities, associated time frames, and density evaluation points
        dens = np.array(res2)
        time = np.array(tm)

        return dens, time


def get_rmsd(u, start, stop, all_select, timestep):
        '''        
        params: universe
        params: start first frame
        params: stop last frame
        params: all_select list of all residue selections
        params: timestep timestep of simulation
        returns: RMSD numpy array with respect of start frame
        '''

        #R = RMSD(u, ref_frame=start, groupselections=all_select)
        #R.run(start=start, stop=stop)

        ##R.rmsd contains [frame, time, rmsd_1, rmsd_2, ...]
        ##exclude column 1 (guessed simulation timestep) and column 2 ("all" selection from the system)
        #keep_cols = np.arange(0, R.rmsd.shape[1])
        #rmsd = R.rmsd[:, keep_cols[np.logical_and(keep_cols!=1, keep_cols!=2)]]
        #rmsd[:, 0] *= (timestep/1000.0)

        app = True
        for currselect in all_select:
            logger.debug(">> %s"%currselect)
            R = RMSD(u, ref_frame=start, selection=currselect, groupselections=[currselect])
            R.run(start=start, stop=stop)

            if app:
                result = R.rmsd[:, 0] 
                result *= (timestep/1000.0)

            result = np.concatenate(([result], [R.rmsd[:, 3]]))
            app = False

        #save file, with added headers like in density outputs
        header="time\t"
        header+= "\t".join(all_labels)
        np.savetxt("bkp_RMSD_all_traces.dat", result, header=header, fmt="%5.3f", delimiter="\t")

        return result


def leave(message):
    '''
    report error message, close logger, and exit
    '''
    logger = logging.getLogger("dynden")
    logger.critical(message)
    for h in logger.handlers:
        h.close()
        logger.removeHandler(h)

    sys.exit()


#########################################################

### PARAMETERS ###

#create arguments parser
parser = argparse.ArgumentParser(description='Parse input parameters')
parser.add_argument('-t', metavar="trajectory", help='trajectory file')
parser.add_argument('-s', metavar="topology", help='topology file')
parser.add_argument('-b', metavar="bins", default=100, help='number of bins for partial density')
parser.add_argument('-nf', metavar="frames", default=50, help='number of frames to include in time averaging')
parser.add_argument('-f', metavar="start", default=0, help='starting frame to account for in analysis')
parser.add_argument('-l', metavar="stop", default=-1, help='last frame to account for in analysis')
parser.add_argument('-am', metavar="avg_method", choices=["block", "slide", "none"], default="block", help='method used for averaging')
parser.add_argument('-ts', metavar="timestep", default=10, help='timestep of input simulation in ps')
parser.add_argument('-disp', metavar="display", choices=["yes", "no"], default="yes", help='display matplotlib plots on screen')
parser.add_argument('-string', metavar="atomselects", default="", help='list of atom selections to analyse, delimited by quotes and with comma-separated items')
parser.add_argument('-v', action="store_false", help='tell me more!')

args = vars(parser.parse_args())

trajectory = args["t"] # "ex: run3-0-200.trr"
topology = args["s"] #"ex: run3-0-200.tpr"
bins = int(args["b"]) # e.g. 100.  z-axis number of bins
frames = int(args["nf"]) #e.g. 50 number of frames to average for partial density
start = int(args["f"]) #e.g. 0, start time (frame)
stop = int(args["l"]) #e.g. 20000, end time (frame)
timestep = float(args["ts"]) #e.g. 10, simulation timestep in ps
display= args["disp"] #"no or "yes", show matplotlib on screen?
avg_method = args["am"] #how to average? "block" (default), "slide", "none"

#splitting string selection, if the user wants to provide a custom selection
if args["string"] == "":
    selection = []
else:
    selection = args["string"].split(",")

# create logger
logname = "dynden.log"
if os.path.isfile(logname):
    os.remove(logname)

logger = logging.getLogger("dynden")
fh = logging.FileHandler(logname)
ch = logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(ch)
if not args["v"]:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.info(" ".join(sys.argv))
logger.info("")
        

#########################################################

### SETUP DATA ###
replot = False
if topology == None or trajectory == None:
    replot = True
    try:
        box_dims = np.loadtxt("bkp_box_dims.dat")
        trajlen = len(box_dims)
        timestep = (box_dims[1, 0] - box_dims[0, 0])*1000.0 #extract timestep from box size file
    except:
        leave("ERROR: topology/trajectory not provided, and backup information available insufficient for replotting!")
    
    logger.info("> no topology/trajectory provided, plotting data in bkp files (%s steps)"%trajlen)
        
else:
    
    # test input makes sense
    if not os.path.isfile(trajectory):
        leave("ERROR: trajectory file %s not found!"%trajectory)
    
    if not os.path.isfile(topology):
        leave("ERROR: trajectory file %s not found!"%topology)

    try:
        u = MDA.Universe(topology, trajectory)
    except:
        leave("ERROR: could not load topology/trajectory!")
    
    trajlen = len(u.trajectory)
    logger.debug("> loaded trajectory with %s steps"%trajlen)

# check smoothing window is not too large
if frames > trajlen/3.0 and avg_method != "none":
        leave("ERROR: number of frames used for smoothing (%s) too large for trajectory length (%s)!"%(frames, trajlen))

#checking time boundary conditions
if stop >= trajlen:
    logger.warning("WARNING: desired last frame is beyond simulation length, ignoring...")
    stop = trajlen-1 
if stop == -1:
    stop = trajlen-1

#definition of measuring timesteps ("none" means that any timestep should be ignored)
if avg_method == "none":
    frames = 1

tsteps = np.arange(0, stop-start, frames).astype(int)

# to avoid confusion: "all" means "every aminoacid individually, "system" means every atom allthogether.
# for MDAnalysis, however, the "all" selection means "every atom all together", so some renaming is taking place below.
if replot:
    # if no topology/trajectory are provided, load labels pointing to existing bkp files
    import glob
    all_select = []
    all_labels = []
    files = glob.glob("bkp_result_*npy")
    if len(files) == 0:
        leave("ERROR: no bkp density trace files found!")
        
    for f in files:
        l = f.split("_")[-1].split(".")[0]
        all_select.append("resname %s"%l)
        all_labels.append(l)

elif len(selection) == 0:
    #select all molecule types in the system
    #create selection strings, and add "all" to list of selections (to study whole system)
    all_labels = np.unique(u.residues.resnames)
    all_select = []
    for l in all_labels:
        all_select.append("resname %s"%l)

    all_labels = np.concatenate((all_labels, ["system"]))
    all_select = np.concatenate((all_select, ["all"]))

else:
    all_labels = []
    all_select = []
    for s in selection:
        s2 = s.strip()

        #test a user-defined selection selection
        try:
            sel_len = len(u.select_atoms(s2))
        except:
            leave('ERROR: selection "%s" is invalid!'%s2)

        if sel_len == 0:
            leave('ERROR: selection "%s" is empty!'%s2)
        
        all_labels.append(s2)
        all_select.append(s2)

    all_labels.append("system")
    all_select.append("all")


### GET BOX SIZE INFORMATION ###
# TODO: z-axis size derived from box information
try:
    if not replot:
        box_dims = np.loadtxt("bkp_box_dims.dat")
        logger.info("> Loaded precalculated box information")

except:
        #get box volume (if loading of precalculated box sizes failed)
        logger.info("> getting simulation box dimensions...")
        v = []
        for ts in u.trajectory[start:stop]:

            ptmp = u.atoms.positions[:, 2]
            minpos = np.min(ptmp)
            maxpos = np.max(ptmp)
            dim = np.max(ptmp) - np.min(ptmp)
            v.append([ts.frame, minpos, maxpos, dim])

            logger.debug(">> frame %s: z = %5.2f A..."%(ts.frame, dim))

        box_dims = np.array(v)
        box_dims[:, 0] *= timestep/1000.0 
        np.savetxt("bkp_box_dims.dat", box_dims)

print_dinosaur() # rawrrr

### GET RMSD INFORMATION ###
#
#if os.path.isfile("bkp_RMSD_all_traces.dat"):
#        rmsd = np.loadtxt("bkp_RMSD_all_traces.dat")
#        if rmsd.shape[1]-1 != len(all_select):
#                logger.warning("> Column count mismatch between RMSD backup file (%s), and residue select (%s)."%(rmsd.shape[1]-1, len(all_select)))
#                logger.warning("> Recalculating!")
#                rmsd = get_rmsd(u, start, stop, all_select, timestep)
#        else:
#                logger.info("> Loaded precalculated RMSD information")
#
#else:
#        logger.info("> Gathering RMSD information...")
#        rmsd = get_rmsd(u, start, stop, all_select, timestep)


### GET DENSITY INFORMATION ###

all_dens = [] #densities time evolutions
all_ccc = [] # pairwise CCCs
all_ccc_traces = [] #CCC traces
testbins = []
for k, sel in enumerate(all_select):

    # get partial density information for residue of interest (if none available for loading)
    # if in replotting mode, the try section will be always successful
    try:
        result = np.load("bkp_result_%s.npy"%str(all_labels[k]))
        logger.debug("> Loaded pre-calculated density for selection %s (%s bins)..."%(all_labels[k], result.shape[1]))
        testbins.append(result.shape[1])
    except:
        atomcount = len(u.select_atoms(sel))
        logger.info("> analyzing density of selection %s of %s: %s (%s atoms)"%(k+1, len(all_labels), all_labels[k], atomcount))
        result = get_partial(u, box_dims, sel, bins, start, stop)
        np.save("bkp_result_%s"%str(all_labels[k]), result)
        testbins.append(bins)
        
    # perform running average
    # obtain average densities and times where average was effectively taken
    if avg_method == "block":
        logger.debug(">> running block average...")
        dens, time = block_average(result, tsteps)
    elif avg_method == "slide":
        logger.debug(">> running sliding window...")
        dens, time = sliding_window(result, tsteps)
    elif avg_method == "none":
        dens = result.copy()
        time = tsteps.copy()
    else:
        leave("ERROR: %s averaging method not recognised, should be block, slide or none"%avg_method)

    logger.debug(">> calculating pairwise cross-correlation...")
    #calculate correlation all vs all (lower triangular matrix only)
    conv_pairwise = np.ones((len(dens), len(dens)))*(-1)
    for i in range(0, len(dens)):
        for j in range(0, len(dens)):
            if i<j:
                pred = dens[i]
                curr = dens[j]
                v1 = np.corrcoef(pred, curr)[0, 1]
                conv_pairwise[i, j] = v1

    #get cross-correlation of consecutive frames and ignore diagonal elements of pairwise CCC
    conv = np.diagonal(conv_pairwise, offset=1).copy()
    np.fill_diagonal(conv_pairwise, -1)

    #store all data of current residue
    all_dens.append(dens)
    all_ccc_traces.append(conv)    
    all_ccc.append(conv_pairwise)

    print_dinosaur() # rawrrr

#test whether all loaded and recalculated datasets have same bin size
b = np.unique(np.array(testbins))
if len(b) != 1:
    leave("ERROR: result files feature different bin sizes.\nrelaunch dynden from scratch or remove datasets with different bin size.")
axes = np.arange(b[0])

#time series for z-box evolution as well as its running average
# (over 100 frames, unless simulation has very little frames)
thistime = tsteps*timestep/1000.0

N = len(box_dims[:, 2])/100
if N%2 != 0:
    N-=1
if N<2 and len(box_dims)>20:
    N = 10
else:
    N = 2

running_avg = np.convolve(box_dims[start:stop, 3], np.repeat(1.0, N)/N, mode='valid')
thistime2 = (np.arange(len(running_avg))+N/2)*timestep/1000.0

#time axis for running averages
sampling_time = time*timestep/1000.0

#save file with all data generated for plot
header="time\t"
header+= "\t".join(all_labels)
all_traces = np.hstack((sampling_time[:-1][:, np.newaxis], np.array(all_ccc_traces).T)) 
np.savetxt("bkp_CCC_all_traces_%s_%s.dat"%(bins, frames), all_traces, header=header, fmt="%10.6f", delimiter="\t")


### PLOT DATA ###

logger.info("> Plotting all data...")

#1. z-box volume
logger.debug(">> z-box size...")
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(box_dims[start:stop, 0], box_dims[start:stop, 3], "-", color="gray")
ax.plot(thistime2, running_avg, "-", color=(126/255.0, 49/255.0, 123/255.0))
ax.set_xlabel("time (ns)")
ax.set_ylabel("z box ($\AA$)")
#ax.set_ylim([np.min(box_dims[:, 3]), np.max(box_dims[:, 3])])
ax.set_xlim([0, np.max(thistime)])
logger.debug(">>> boundaries: %s, %s"%(np.min(box_dims[:, 3]), np.max(box_dims[:, 3])))

fig.savefig("fig_z_box.png")


#2. compare CCC traces of each residue
logger.debug(">> aggregated CCC evolution per residue...")
fig = plt.figure(dpi=120, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
c = cm.get_cmap("viridis")
mycycle = [c(i) for i in np.linspace(0.0, 1.0, len(all_labels))]
plt.gca().set_prop_cycle("color", mycycle)

for i in range(len(all_labels)):
        ax.plot(sampling_time[:-1], all_ccc_traces[i], "-", label="%s"%all_labels[i])

ax.set_xlabel("time (ns)")
ax.set_ylabel("CCC")
ax.set_xlim([0, np.max(sampling_time)])

# > decide whether a multi-column legend is needed
if len(all_labels)>3:
    cols = 2
else:
    cols = 1

ax.legend(loc="lower right", ncol=cols, frameon=False)
fig.savefig("fig_CCC_all_traces_%s_%s.png"%(bins, frames))


# iterate over each residue selection, for individual plots
for i in range(len(all_labels)):

        label = all_labels[i]
        dens = all_dens[i]
        conv_all = all_ccc[i]
        conv_trace = all_ccc_traces[i]

        logger.debug(">> %s time evolution..."%label)

        #3. density vs time surface and cross-correlation
        fig = plt.figure(dpi=120, figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(label)
        X, Y = np.meshgrid(axes, sampling_time)
        c = cm.get_cmap("viridis")
        ax.pcolormesh(X, Y, dens, cmap=c)
        ax.set_xlabel("Box z-binning")
        ax.set_ylabel("time (ns)")
        ax.set_xlim([np.min(axes), np.max(axes)])
        ax.set_ylim([np.min(sampling_time), np.max(sampling_time)])

        # commented below: old routine comparing stepwise CCC with RMSD convergence
        ## plot pairwise CCC
        #N = int(len(conv_trace)/100)
        #if N%2 != 0:
        #    N-=1
        #
        #CCC_running_avg = np.convolve(conv_trace, np.repeat(1.0, N)/N, mode='valid')
        #startpos = int(N/2)
        #thistime3 = sampling_time[startpos:startpos+len(CCC_running_avg)]
        #
        #ax2 = fig.add_subplot(1, 2, 2)
        #ax2.plot(conv_trace, sampling_time[:-1], "gray")
        #ax2.plot(CCC_running_avg, thistime3, "black")
        #ax2.set_ylim([np.min(sampling_time[:-1]), np.max(sampling_time[:-1])])
        #ax2.set_xlim([np.min(conv_trace)*0.995, 1])
        #ax2.set_yticklabels([])
        #ax2.set_yticks([])
        #ax2.set_xlabel("CCC")
        #
        ## plot RMSD
        #ax3 = ax2.twiny()
        #ax3.plot(rmsd[:, i+1], rmsd[:, 0], color='firebrick')
        #ax3.set_yticklabels([])
        #ax3.set_yticks([])
        #ax3.tick_params('x', colors='firebrick')
        #ax3.set_xlabel("RMSD ($\AA$)", color='firebrick')
        #
        #plt.subplots_adjust(wspace=0)
        fig.savefig("fig_density_%s_%s_%s.png"%(label, bins, frames))

        #4. pairwise cross-correlation
        Zm = ma.masked_where(conv_all==-1, conv_all)
        fig = plt.figure(dpi=120, figsize=(8, 8))
        ax.set_title(label)
        ax = fig.add_subplot(1, 1, 1)
        X, Y = np.meshgrid(sampling_time, sampling_time)
        c = cm.get_cmap("viridis")
        minccc = max(0.9, np.min(Zm))
        plt.pcolormesh(X, Y, Zm, cmap=c, vmax=1, vmin=minccc)
        plt.xlabel("time (ns)")
        plt.ylabel("time (ns)")
        plt.colorbar()

        plt.savefig("fig_CCC_pairwise_%s_%s_%s.png"%(label, bins, frames))

if display=="yes" or display==True:
    plt.show()

logger.info("> All done. Thank you for having used dynden!")
for h in logger.handlers:
    h.close()
    logger.removeHandler(h)

