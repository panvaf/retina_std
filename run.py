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

'''
# for transient vs sustained on bipolar cells
size = {'BipolarCell4':10,'BipolarCell5A':10,'BipolarCell6':10,'AmacrineCellAII':10,'PresynapticSilencerBip5AAmAII':10,
        'GanglionCellPV5':1,'GanglionCellsOFFa':1,'GanglionCellsONa':1,'GanglionCelltONa':1,'GanglionCelltONaPre':1}
'''

size = {'BipolarCell5A':5,'BipolarCell6':5,'AmacrineCellAII':5,'GanglionCelltONa':1,'GanglionCellsONa':1}

classes = {'BipolarCell':BipolarCell,'AmacrineCell':AmacrineCell,'GanglionCell':GanglionCell,'Delay':Delay,'PresynapticSilencer':PresynapticSilencer}

cells = net.network_init(structs,size,classes)

tONa = cells['GanglionCelltONa'][0][0].out()
sONa = cells['GanglionCellsONa'][0][0].out()

t = np.arange(tONa.size)*0.002
plt.plot(t,tONa)
plt.plot(t,sONa)
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (spikes/s)')
plt.title('Transient vs Sustained On Alpha Ganglion cells')
plt.legend('Transient ONaRGC','Sustained ONaRGC')

###############################################################################
        
# amacrine cells are faster on the rise so no chance to take a transient response
# from them. more likely to get a sustained response because they cancel out initial
# wave but keep the steady state wave