# Imported Packages
import numpy as np

### Gaps between front top and bottom pnCCD to middle of rear pnCCD
gapTop =231			# In pixel
gapBot =252			# In pixel

### offset between rear and front pnCCDs 
xShift =-1			# In pixel

############## Things to adjust per run/experiment
pixelSizePnccd=75*10**-6 	# Size of a pixel in meter.

distanceOfRearPnccd=0.74	# Distance IR to front pnCCD in meter
distanceOfFrontPnccd=0.36	# Distance IR to front pnCCD in meter		
gainRearPnccd=1./64.		# Detector gain multiplier
gainFrontPnccd=1.		# Detector gain multiplier

scatteredWaveLength=1.5*10**-9	# Wavelength of photons in meter

pathToHDF5= 			# Adapt to your needs


### Functions
# Phi - scattering angle
def PhiScatt(pixel,pixelSize, distanceToDetector):
	return np.arctan(pixel*pixelSize/distanceToDetector)

# Q-vector function
def qVector(pixel, pixelSize, distanceToDetector, waveLength):
	return 4.*np.pi*np.sin(PhiScatt(pixel, pixelSize,
		distanceToDetector)/2)/waveLength


### Reading intensity files, 
intFrontTop=	# Adapt to your needs
intRear=	# Adapt to your needs
intFrontBottom=	# Adapt to your needs

### Intensity normalization and offset of electronic noise.
# Rear int. normalization
intRear=intRear*(1/gainRearPnccd)*(distanceOfRearPnccd**2)*
	(1/distanceOfFrontPnccd**2)

# Offsset 
intRear[intRear<15*(1/gainRearPnccd)*(distanceOfRearPnccd**2)*
	(1/distanceOfFrontPnccd**2)]=0.0
intFrontTop[intFrontTop<350]=0.0                   
intFrontBottom[intFrontBottom<350]=0.0             

### Combining front detector
ztop=np.concatenate((intFrontTop,np.zeros
	((gapTop+gapBot,1028)),intFrontBottom))

### Combining rear with front detector
# Creating pixel to Q-vector correlation vectors
yRear, xRear =np.ogrid[(-len(intRear))/2:(len(intRear))/2,
	-(len(intRear[0]))/2:(len(intRear[0]))/2]

# To transform Q-values from rear pnCCD to front pnCCD pixel values
qDiv=qVector(1, pixelSize=pixelSizePnccd,
	distanceToDetector=distanceOfFrontPnccd,
	waveLength=scatteredWaveLength)

# Spanning Q-space
y_comb=len(ztop)/2
x_comb=len(ztop[0])/2

qyRear=np.round(qVector(yRear, pixelSize=pixelSizePnccd,
	distanceToDetector=distanceOfRearPnccd,
	waveLength=scatteredWaveLength)/qDiv).astype(int) + y_comb
qxRear=np.round(qVector(xRear, pixelSize=pixelSizePnccd,
	distanceToDetector=distanceOfRearPnccd,
	waveLength=scatteredWaveLength)/qDiv).astype(int) + x_comb + xShift


# Keep track of per pixel additions to create mean intensities
norm=np.zeros(ztop.shape)

# Iterating over the array, slow but works
it=np.nditer(intRear, flags=['multi_index'])

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

### Combined mean intensities of front and rear detector.
ztop = ztop/norm