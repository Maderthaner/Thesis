from psana import *

ds = DataSource('exp=XCS/xcstut13:run=15')
src = Source('DetInfo(XcsBeamline.0:Princeton.0)')

import matplotlib.pyplot as plt

plt.figure('Princeton Camera')
plt.ion()
plt.show()

nevent=0
for evt in ds.events():
    nevent+=1
    frame = evt.get(Princeton.FrameV1, src)

    plt.imshow(frame.data())
    plt.draw()
    if nevent>3:
        break
