# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import interpolate
from random import random
from scipy import ndimage

#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("hit", help="number of hit", type=int)
#args = parser.parse_args()


############# Things to adjust per shot
### Path to data
dirPath ='/Users/Max/Desktop/working 2/doped_13%_800fs_shot_corr_to_1D'+'/'             # NO "/"" at the end
#hitList =np.loadtxt(dirPath+'goodhits_r0108-scratch.txt', dtype="string")
pathToHDF5 = dirPath+'r0174__5973757754469218304.h5'   #hitList[args.hit] #there is a name bug in a previous piece of code.

### Gaps between front top pnCCD and front bottom pnCCD to the middle of the rear pnCCD.
gapTop =231
gapBot =252

### offset between rear and front pnCCDs 
xShift =-1

### Data about the cluster
#clusterSize =3.04*10**-8                        # must be float
#clusterInt = 0.1#390239.0*10**43                     # must be float

#overlapCheck=0
############# 

############## Things to adjust per run/experiment
pixelSizePnccd =75*10**-6 					# Size of a pixel in meter.

distanceOfRearPnccd =0.74					# Distance from IR to front pnCCD in meter.
distanceOfFrontPnccd =0.36			

gainRearPnccd =1./64.						# Detector muiltiplier for gain calibration.
gainFrontPnccd =1.

scatteredWaveLength =1.5*10**-9			# Wavelength of the scattered photons in meter.


### Functions
# Phi/2 - scatter angle
def PhiScatt(pixel,pixelSize, distanceToDetector):
    return np.arctan(pixel*pixelSize/distanceToDetector)

# q vector function
def qVector(pixel, pixelSize, distanceToDetector, waveLength):
	return 4.*np.pi*np.sin(PhiScatt(pixel, pixelSize, distanceToDetector)/2)/waveLength


### Reading detectors
intFrontTop =np.lib.pad(np.asarray(h5py.File(pathToHDF5,'r')['/data/FrontPnCCDLab'],dtype='float32')[0:512,0:1024],((0,0),(2,2)),'constant',constant_values=0)
intRear =np.asarray(h5py.File(pathToHDF5,'r')['/data/RearPnCCDLab'],dtype='float32')
intFrontBottom =np.lib.pad(np.asarray(h5py.File(pathToHDF5,'r')['/data/FrontPnCCDLab'],dtype='float32')[531:1043,0:1024],((0,0),(2,2)),'constant',constant_values=0)


### Intensity normalization and offset of electronic noise.
intRear =intRear*(1/gainRearPnccd)*(distanceOfRearPnccd**2)*(1/distanceOfFrontPnccd**2)

# Offsset rear
intRear[intRear<15*(1/gainRearPnccd)*(distanceOfRearPnccd**2)*(1/distanceOfFrontPnccd**2)] =0.0
# Offset front top
intFrontTop[intFrontTop<350] =0.0                   
# Offset front bottom
intFrontBottom[intFrontBottom<350] =0.0             

### Combining front detector
ztop =np.concatenate((intFrontTop,np.zeros((gapTop+gapBot,1028)),intFrontBottom))

### Combining rear with front detector
# Creating pixel to q-vector correlation vectors
yRear, xRear =np.ogrid[(-len(intRear))/2:(len(intRear))/2,
    -(len(intRear[0]))/2:(len(intRear[0]))/2]

# To transform q values from rear pnCCD to front pnCCD pixel values
qDiv = qVector(1, pixelSize=pixelSizePnccd, distanceToDetector=distanceOfFrontPnccd, waveLength=scatteredWaveLength)

# Actual transform
y_comb = len(ztop)/2
x_comb = len(ztop[0])/2

qyRear = np.round(qVector(yRear, pixelSize=pixelSizePnccd, distanceToDetector=distanceOfRearPnccd, waveLength=scatteredWaveLength)/qDiv).astype(int) + y_comb
qxRear = np.round(qVector(xRear, pixelSize=pixelSizePnccd, distanceToDetector=distanceOfRearPnccd, waveLength=scatteredWaveLength)/qDiv).astype(int) + x_comb + xShift


# Keep track of per pixel additions to create mean
norm = np.zeros(ztop.shape)

# Iterating over the array, is there a faster way than a for loop?
it = np.nditer(intRear, flags=['multi_index'])

while not it.finished:
    # read pixel transform coordinates
    y_idx = qyRear[it.multi_index[0],0]
    x_idx = qxRear[0,it.multi_index[1]] 

    # add intensities and add to norm
    ztop[y_idx,x_idx] += it[0]
    norm[y_idx,x_idx] += 1

    it.iternext()

# Corrections to norm so not 1/0, seems inefficient but is fast enough.
norm -= 1
norm[norm<0]=0
norm += 1

# Actual mean value creation
ztop = ztop/norm

#######################################
################## Plotting ###########
#######################################

z_min, z_max = 0.1, intRear.max()

#plt.subplot(2,1,1)
#plt.pcolormesh(qXRear, qYRear, intRear, norm=LogNorm(vmin=z_min, vmax=z_max), cmap='plasma')
#plt.pcolormesh(qXFrontTop, qYFrontTop, intFrontTop, norm=LogNorm(vmin=z_min, vmax=z_max), cmap='plasma')
#plt.pcolormesh(qXFrontBottom, qYFrontBottom, intFrontBottom, norm=LogNorm(vmin=z_min, vmax=z_max), cmap='plasma')
#plt.colorbar()

#plt.imshow(znew, norm=LogNorm(vmin=z_min, vmax=z_max), cmap='plasma')
#plt.colorbar()


#plt.subplot(2,1,2)

#combinedArray =np.add(intRearDownSmpldReshp,ztop)
w, h = plt.figaspect(1.)

plt.Figure(figsize=(w,h))
plt.imshow(ztop+0.1, norm=LogNorm(vmin=z_min, vmax=z_max), cmap='plasma')
#plt.colorbar()
plt.show()

### saving for further use, uncomment as needed
#np.save('0fs_cluster_weird_shot',combinedArray)
#np.save('0fs_cluster_shot_mask',invertMaskCombined)
#np.save('0fs_cluster_shot_ampSim',ampSimulatedCombinedArray)
#np.save('0fs_cluster_shot_phase',phaseSimulatedCombinedArray)
