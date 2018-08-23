#usage:
# python plot.py *Simulation* *Plotsection*

from ConfigParser import SafeConfigParser
import time
import glob
import sys
import argparse
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndimage
from matplotlib.colors import LinearSegmentedColormap

def load_known_size(fname, ncol, nrow):
	x = np.empty((nrow, ncol), dtype = np.double)
	with open(fname) as f:
		for irow, line in enumerate(f):
			x[irow, :] = line.split()
	return x

#commandline arguments
parser = argparse.ArgumentParser(description='Plot graphics.')
parser.add_argument("inputDir", help="input File")
args = parser.parse_args()

#read Parameters from Simulation
parser = SafeConfigParser()
parser.read(os.path.join(args.inputDir,'param.ini'))
nx = parser.getint('main','N')
Length = parser.getfloat('main','L')
cellNumber = parser.getint('main','CellNumber')
SaveTime = parser.getfloat('main','SaveTime')

folder = 'Combined'
DivideDim = 1

#create directories
if not os.path.exists(os.path.join(args.inputDir, folder)):
	os.makedirs(os.path.join(args.inputDir, folder))

Backround = np.loadtxt(os.path.join(args.inputDir, 'Micropattern.dat'))

ReadPhiFileName = glob.glob(os.path.join(args.inputDir, 'Phase*.dat'))
PhiCon = np.loadtxt(ReadPhiFileName[0])
os.remove(ReadPhiFileName[0])
PhiCont = ndimage.zoom(PhiCon, 2)

ReadRhoFileName = glob.glob(os.path.join(args.inputDir, 'Pol*.dat'))
Rho = np.loadtxt(ReadRhoFileName[0])
os.remove(ReadRhoFileName[0])
#~ os.rename(ReadRhoFileName[0],os.path.join(args.inputDir,folder, ReadRhoFileName[0]) )

ReadInhFileName = glob.glob(os.path.join(args.inputDir, 'Inh*.dat'))
Inh = np.loadtxt(ReadInhFileName[0])
os.remove(ReadInhFileName[0])
#~ os.rename(ReadRhoFileName[0],os.path.join(args.inputDir,folder, ReadRhoFileName[0]) )

PlotFileName = os.path.splitext(os.path.basename(ReadPhiFileName[0]))[0]
CurrentStep = PlotFileName.split('_')[1]
CurrentTime = float(CurrentStep) * SaveTime

#create custom colormap for rho (only pos values), Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0.5, 1, cmap.N / 2))
cmapUpper = LinearSegmentedColormap.from_list('Upper Half', colors)

#plot the file
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.suptitle('Time: '+str(CurrentTime)+'s',verticalalignment='bottom', fontsize='25',y=0.825)

ax1.set_title(r'$\rho (\vec{r})$', fontsize=25, y=1.02)
im= ax1.imshow( Rho, vmin=0, vmax=1.6, extent=[0,Length,0,Length], cmap=cmapUpper)

ax1.contour(Backround, extent=[0,Length,0,Length],levels = [0.3], colors = 'mediumslateblue', linewidths = (3))
ax1.contour(PhiCont, extent=[0,Length,0,Length],levels = [0.35], colors = 'black', linewidths = (2.0), hold='on', origin='image')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="8%", pad=0.05)
cbar1 = plt.colorbar(im, cax=cax1, ticks=np.arange(0,2.1,0.4))
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

ax2.set_title(r'Inhibitor $I(\vec{r})$', fontsize=25, y=1.02)
im2 = ax2.imshow( Inh, vmin=-0.05, vmax=0.05, extent=[0,Length,0,Length])
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="8%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2, format="%.3f", ticks=np.arange(-0.05,0.06,0.025))
ax2.contour(Backround, extent=[0,Length,0,Length],levels = [0.3], colors = 'mediumslateblue', linewidths = (3))
ax2.contour(PhiCont, extent=[0,Length,0,Length],levels = [0.35], colors = 'black', linewidths = (2.0), hold='on', origin='image')
ax2.yaxis.set_visible(False)
ax2.set_xlabel("X")


plt.tight_layout()

plt.savefig(os.path.join(args.inputDir,folder, 'Combined_'+CurrentStep+'.png'),bbox_inches='tight', dpi=300)
plt.close(fig)
