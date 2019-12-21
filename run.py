"""
Create and run networks for testing.
"""

from classes import * # necessary to import object classes
from celltypes import *
import numpy as np
import network as net
import matplotlib.pyplot as plt

###############################################################################

# Initialize and run network

structs = {'BipolarCell1':BipolarCell1, 'BipolarCell2':BipolarCell2, 'BipolarCell3a':BipolarCell3a,
           'BipolarCell3b':BipolarCell3b, 'BipolarCell4':BipolarCell4, 'BipolarCell5A':BipolarCell5A,
           'BipolarCell5R':BipolarCell5R, 'BipolarCell5X':BipolarCell5X, 'BipolarCellX':BipolarCellX,
           'BipolarCell6':BipolarCell6, 'BipolarCell7':BipolarCell7, 'BipolarCell8':BipolarCell8,
           'BipolarCell9':BipolarCell9, 'BipolarCellR':BipolarCellR, 'AmacrineCellOff':AmacrineCellOff, 'AmacrineCellOffStar':AmacrineCellOffStar, 
           'AmacrineCellOn':AmacrineCellOn, 'AmacrineCellOnStar':AmacrineCellOnStar, 'AmacrineCellAII':AmacrineCellAII, 'AmacrineCell1': AmacrineCell1,
           'GanglionCellsOFFa':GanglionCellsOFFa, 'GanglionCellFminiOFF':GanglionCellFminiOFF, 'GanglionCellFmidiOFF':GanglionCellFmidiOFF, 
           'GanglionCellPV5':GanglionCellPV5, 'GanglionCellooDS37c':GanglionCellooDS37c, 'GanglionCellooDS37d':GanglionCellooDS37d, 
           'GanglionCellooDS37r':GanglionCellooDS37r, 'GanglionCellooDS37v':GanglionCellooDS37v, 'GanglionCellW3':GanglionCellW3,
           'GanglionCellsONa':GanglionCellsONa, 'GanglionCellFminiON':GanglionCellFminiON, 'GanglionCellFmidiON':GanglionCellFmidiON, 
           'GanglionCelltONa':GanglionCelltONa, 'GanglionCellsOnDS7id':GanglionCellsOnDS7id, 'GanglionCellsOnDS7ir':GanglionCellsOnDS7ir, 
           'GanglionCellsOnDS7iv':GanglionCellsOnDS7iv, 'GanglionCelltOnDS7o':GanglionCelltOnDS7o, 'GanglionCelltONaPre':GanglionCelltONaPre,
           'PresynapticSilencerBip5AAmAII':PresynapticSilencerBip5AAmAII}

# for the sustained on ganglion cell
size = {'BipolarCell6':10,'AmacrineCellAII':10,'GanglionCellsONa':1}
'''

# for the transient on ganglion cell
size = {'BipolarCell5A':5,'BipolarCell6':5,'AmacrineCellAII':5,'GanglionCelltONa':1}

# for the sustained off ganglion cell
size = {'BipolarCell2':5,'BipolarCell6':5,'AmacrineCellAII':5,'GanglionCellsOFFa':1}
'''
classes = {'BipolarCell':BipolarCell,'AmacrineCell':AmacrineCell,'GanglionCell':GanglionCell,'Delay':Delay,'PresynapticSilencer':PresynapticSilencer}

cells = net.network_init(structs,size,classes)

sONa = cells['GanglionCellsONa'][0][0].out()

t = np.arange(sONa.size)*0.002
plt.plot(t,sONa, label='Sustained ONaRGC')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (spikes/s)')
plt.title('Sustained On Alpha Ganglion cell')
plt.legend()
'''

sOFFa = cells['GanglionCellsOFFa'][0][0].out()
t = np.arange(sOFFa.size)*0.002

plt.figure()
plt.subplot(122)
plt.plot(t,sOFFa)
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (spikes/s)')
plt.title('s-OFF aRGC response')
plt.subplot(121)
plt.imshow(image[:,:,0],extent=[0,250,250,0])
plt.title('Stimulus for l = {:.2f} μm'.format(5/(2*k)))
plt.ylabel('y (um)')
plt.xlabel('x (um)')
plt.tight_layout()
plt.show()
'''
###############################################################################
        
# amacrine cells are faster on the rise so no chance to take a transient response
# from them. more likely to get a sustained response because they cancel out initial
# wave but keep the steady state wave