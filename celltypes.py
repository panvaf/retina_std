"""
Contains information about all retinal cell types necessary to define them,
stored in dictionaries. 

Information about the cell itself is directly fed into the corresponding class.
Information about the way it connects is manipulated by network_init in network
to produce the appropriate connectivity in overlapping 2-D sheets of cells.
"""

import numpy as np
from classes import *

# note: width should be in um in the structs and then transformed to pixel values
# later in the script. similarly, duration should be in ms and then transformed to
# units of temporal res. All inputs to the network functions and to the classes
# should be normalised in units of pixels and temporal res. Then one only need to
# multiply with those quantities to revert back to physical dimensions

# order of parameters in attributes is order of element position in connectivity,
# and order element types are read in connectivity is determined in inputs

image = np.zeros((img_size[0],img_size[1],100))

###############################################################################

# Bipolar cells

BipolarCell1 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

BipolarCell2 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}   

BipolarCell3a = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

BipolarCell3b = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell4 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell5A = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell5R = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

BipolarCell5X = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
BipolarCellX = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell6 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell7 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

BipolarCell8 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

BipolarCell9 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

BipolarCellR = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
###############################################################################
    
# Amacrine cells    
    
AmacrineCellOffS = {"inputs":['BipolarCell2','BipolarCell3a'], "connectivity": {'BipolarCell2':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0}}
    
AmacrineCellOnS = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell7'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0}}
    
AmacrineCellWidefield = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4',
    'BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCellX'], "connectivity": {'BipolarCell3a':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0}}

AmacrineCellAII = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a','BipolarCell3b',
    'BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCell6','BipolarCell7',
    'BipolarCell8','BipolarCell9','BipolarCellR'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0}}

###############################################################################
    
# Ganglion cells
    
GanglionCellsOFFa = {"inputs":['BipolarCell1','BipolarCell2','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellFminiOFF = {"inputs":['BipolarCell1','BipolarCell2','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellFmidiOFF = {"inputs":['BipolarCell1','BipolarCell2','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellPV5 = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4',
    'AmacrineCellWidefield','AmacrineCellAII'], "connectivity": {'BipolarCell3a':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellooDS37c = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellooDS37d = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellooDS37r = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellooDS37v = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellW3 = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4','BipolarCell5A',
    'BipolarCell5R','BipolarCell5X','BipolarCellX','AmacrineCellWidefield','AmacrineCellAII'],
    "connectivity": {'BipolarCell3a': [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellsONa = {"inputs":['BipolarCell6','BipolarCell7','BipolarCell8','BipolarCell9',
    'BipolarCellR','AmacrineCellAII'], "connectivity": {'BipolarCell6':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellFminiON = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4',
    'BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCellX','AmacrineCellWidefield',
    'AmacrineCellAII'], "connectivity": {'BipolarCell3a':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellFmidiON = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'BipolarCellX','AmacrineCellWidefield','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCelltONa = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'BipolarCellX','AmacrineCellWidefield','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellsOnDS7id = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellsOnDS7ir = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellsOnDS7iv = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCelltOnDS7o = {"inputs":['AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'AmacrineCellOnS':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
# Rescale parameters relative to time and space partitioning    
    
BipolarCell1["attributes"]['width'] = BipolarCell1["attributes"]['width']/pixel
BipolarCell1["attributes"]['duration'] = BipolarCell1["attributes"]['duration']/temporal_res
BipolarCell2["attributes"]['width'] = BipolarCell2["attributes"]['width']/pixel
BipolarCell2["attributes"]['duration'] = BipolarCell2["attributes"]['duration']/temporal_res
BipolarCell3a["attributes"]['width'] = BipolarCell3a["attributes"]['width']/pixel
BipolarCell3a["attributes"]['duration'] = BipolarCell3a["attributes"]['duration']/temporal_res
BipolarCell3b["attributes"]['width'] = BipolarCell3b["attributes"]['width']/pixel
BipolarCell3b["attributes"]['duration'] = BipolarCell3b["attributes"]['duration']/temporal_res
BipolarCell4["attributes"]['width'] = BipolarCell4["attributes"]['width']/pixel
BipolarCell4["attributes"]['duration'] = BipolarCell4["attributes"]['duration']/temporal_res
BipolarCell5A["attributes"]['width'] = BipolarCell5A["attributes"]['width']/pixel
BipolarCell5A["attributes"]['duration'] = BipolarCell5A["attributes"]['duration']/temporal_res
BipolarCell5R["attributes"]['width'] = BipolarCell5R["attributes"]['width']/pixel
BipolarCell5R["attributes"]['duration'] = BipolarCell5R["attributes"]['duration']/temporal_res
BipolarCell5X["attributes"]['width'] = BipolarCell5X["attributes"]['width']/pixel
BipolarCell5X["attributes"]['duration'] = BipolarCell5X["attributes"]['duration']/temporal_res
BipolarCellX["attributes"]['width'] = BipolarCellX["attributes"]['width']/pixel
BipolarCellX["attributes"]['duration'] = BipolarCellX["attributes"]['duration']/temporal_res
BipolarCell6["attributes"]['width'] = BipolarCell6["attributes"]['width']/pixel
BipolarCell6["attributes"]['duration'] = BipolarCell6["attributes"]['duration']/temporal_res
BipolarCell7["attributes"]['width'] = BipolarCell7["attributes"]['width']/pixel
BipolarCell7["attributes"]['duration'] = BipolarCell7["attributes"]['duration']/temporal_res
BipolarCell8["attributes"]['width'] = BipolarCell8["attributes"]['width']/pixel
BipolarCell8["attributes"]['duration'] = BipolarCell8["attributes"]['duration']/temporal_res
BipolarCell9["attributes"]['width'] = BipolarCell9["attributes"]['width']/pixel
BipolarCell9["attributes"]['duration'] = BipolarCell9["attributes"]['duration']/temporal_res
BipolarCellR["attributes"]['width'] = BipolarCellR["attributes"]['width']/pixel
BipolarCellR["attributes"]['duration'] = BipolarCellR["attributes"]['duration']/temporal_res

AmacrineCellOffS["attributes"]['duration'] = [dur/temporal_res for dur in AmacrineCellOffS["attributes"]['duration']]
AmacrineCellOnS["attributes"]['duration'] = [dur/temporal_res for dur in AmacrineCellOnS["attributes"]['duration']]
AmacrineCellWidefield["attributes"]['duration'] = [dur/temporal_res for dur in AmacrineCellWidefield["attributes"]['duration']]
AmacrineCellAII["attributes"]['duration'] = [dur/temporal_res for dur in AmacrineCellAII["attributes"]['duration']]

GanglionCellsOFFa["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellsOFFa["attributes"]['duration']]
GanglionCellFminiOFF["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellFminiOFF["attributes"]['duration']]
GanglionCellFmidiOFF["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellFmidiOFF["attributes"]['duration']]
GanglionCellPV5["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellPV5["attributes"]['duration']]
GanglionCellooDS37c["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellooDS37c["attributes"]['duration']]
GanglionCellooDS37d["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellooDS37d["attributes"]['duration']]
GanglionCellooDS37r["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellooDS37r["attributes"]['duration']]
GanglionCellooDS37v["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellooDS37v["attributes"]['duration']]
GanglionCellW3["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellW3["attributes"]['duration']]
GanglionCellsONa["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellsONa["attributes"]['duration']]
GanglionCellFminiON["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellFminiON["attributes"]['duration']]
GanglionCellFmidiON["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellFmidiON["attributes"]['duration']]
GanglionCelltONa["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCelltONa["attributes"]['duration']]
GanglionCellsOnDS7id["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellsOnDS7id["attributes"]['duration']]
GanglionCellsOnDS7ir["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellsOnDS7ir["attributes"]['duration']]
GanglionCellsOnDS7iv["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCellsOnDS7iv["attributes"]['duration']]
GanglionCelltOnDS7o["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCelltOnDS7o["attributes"]['duration']]