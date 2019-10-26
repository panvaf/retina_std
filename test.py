"""
Create and run networks for testing.
"""

from classes import * # necessary to import object classes
from celltypes import *
import numpy as np
import network as net

image = np.zeros((round(image_size[0]/pixel),round(image_size[1]/pixel),100))

###############################################################################

# Checking if elements are working fine

BipolarStruct = {"inputs":[image], "weights": np.array([1]), "center": [0, 0], "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': [10, 20], 'on_off_ratio': 3, 'temporal': 'stretched_sin',
     'duration': 10, 'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
# Create multiple bipolar cells of the same type
Bipolars = [ BipolarCell(BipolarStruct["inputs"],BipolarStruct["weights"],BipolarStruct["center"],BipolarStruct["attributes"]) for i in range(10)]

# need to find a more automated way to put together large networks, maybe use 
# classes with parameters that can draw across an entire range of bipolar cells?

AmacrineStruct = {"inputs":[Bipolars[0],Bipolars[2],Bipolars[4]], 
    "weights": np.array([1,1,1]), "attributes": {'temporal': ['stretched_sin', 
     'stretched_sin', 'stretched_sin'], 'duration': [1,1,1], 'coeffs': [[1],[1],[1]],
     'activation': 'relu', 'threshold': 0, 'recurrent': [1, -0.2]}}
    
Amacrine = AmacrineCell(AmacrineStruct["inputs"],AmacrineStruct["weights"],[0, 0],AmacrineStruct["attributes"])

DelayStruct = {"inputs":[Amacrine],"weights": np.array([1]), "attributes": {'t_delay': 2}}

Buffer = Delay(DelayStruct["inputs"],DelayStruct["weights"],[0, 0],DelayStruct["attributes"])

GanglionStruct = {"inputs":[Bipolars[1],Bipolars[3],Bipolars[5],Buffer], 
    "weights": np.array([1,1,1,-3]), "attributes": {'temporal': ['stretched_sin', 'stretched_sin', 'stretched_sin','stretched_sin'],
    'duration': [1,1,1,1], 'coeffs': [[1],[1],[1],[1]] , 'activation': 'relu', 'threshold': 0}}
    
Ganglion = GanglionCell(GanglionStruct["inputs"],GanglionStruct["weights"],[0, 0],GanglionStruct["attributes"])

a = Ganglion.out()

SilencerStruct = {"inputs":[Amacrine,Bipolars[6]],"weights": np.array([-1,1]), "attributes":{}}

Silencer = PresynapticSilencer(SilencerStruct["inputs"],SilencerStruct["weights"],[0, 0],SilencerStruct["attributes"])

b = Silencer.out()

###############################################################################

# Checking if the network is initialized correctly

structs = {'BipolarCellTemplate':BipolarCellTemplate, 'AmacrineCellTemplate':AmacrineCellTemplate}
size = {'BipolarCellTemplate': 100, 'AmacrineCellTemplate':30}
classes = {'BipolarCell':BipolarCell,'AmacrineCell':AmacrineCell,'GanglionCell':GanglionCell,'Delay':Delay,'PresynapticSilencer':PresynapticSilencer}

cells = net.network_init(structs,size,classes)

###############################################################################