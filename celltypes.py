"""
Contains information about all retinal cell types necessary to define them,
stored in dictionaries. 

Information about the cell itself is directly fed into the corresponding class.
Information about the way it connects is manipulated by network_init in network
to produce the appropriate connectivity in overlapping 2-D sheets of cells.
"""

import numpy as np
from classes import *
from scipy.io import loadmat

# note: width should be in um in the structs and then transformed to pixel values
# later in the script. similarly, duration should be in ms and then transformed to
# units of temporal res. All inputs to the network functions and to the classes
# should be normalised in units of pixels and temporal res. Then one only need to
# multiply with those quantities to revert back to physical dimensions

# order of parameters in attributes is order of element position in connectivity,
# and order element types are read in connectivity is determined in inputs

image = np.zeros((img_size[0],img_size[1],100))
BipolarTemporals = loadmat('C:\\Users\\user\\Documents\\CNS\\1st Rotation\\data\\pantelis_bip_filters.mat')
BipolarTemporals = BipolarTemporals['pantelis_bip_filters']
PV5recurrent = loadmat('C:\\Users\\user\\Documents\\CNS\\1st Rotation\\data\\gc_fdbk_filt.mat')
PV5recurrent = PV5recurrent['feedbackfilt']

###############################################################################

# Bipolar cells

BipolarCell1 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,0], 'activation': 'relu', 'threshold': 0}}

BipolarCell2 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,1], 'activation': 'relu', 'threshold': 0}}   

BipolarCell3a = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,2], 'activation': 'relu', 'threshold': 0}}

BipolarCell3b = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,3], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell4 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,4], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell5A = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,5], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell5R = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,6], 'activation': 'relu', 'threshold': 0}}

BipolarCell5X = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,7], 'activation': 'relu', 'threshold': 0}}
    
BipolarCellX = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,8], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell6 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,9], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell7 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,10], 'activation': 'relu', 'threshold': 0}}

BipolarCell8 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,11], 'activation': 'relu', 'threshold': 0}}

BipolarCell9 = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,12], 'activation': 'relu', 'threshold': 0}}

BipolarCellR = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': np.array([10, 20]),
     'on_off_ratio': 3, 'temporal': BipolarTemporals[:,13], 'activation': 'relu', 'threshold': 0}}
    
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

BipolarCell6TOAmacrineCellAII = [(0,0,5)]
temp =  list(zip(*BipolarCell6TOAmacrineCellAII))
BipolarCell6TOAmacrineCellAIIw = temp[2]; BipolarCell6TOAmacrineCellAIIconn = list(zip(temp[0],temp[1]))
    
AmacrineCellAII = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a','BipolarCell3b',
    'BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCell6','BipolarCell7',
    'BipolarCell8','BipolarCell9','BipolarCellR'], "connectivity": {'BipolarCell6': BipolarCell6TOAmacrineCellAIIconn},
    "weights": np.array(BipolarCell6TOAmacrineCellAIIw), "attributes": {'temporal': ['stretched_sin']*1,'duration': [1]*1,
    'coeffs': [[1]]*1, 'activation': 'relu','threshold': 1.0}}

BipolarCell3bTOAmacrineCell1 = [(-3,0,.5),(-2,0,.5),(-1,0,.5),(0,0,.5),(1,0,.5),(2,0,.5),(3,0,.5),(4,0,.5)]
temp =  list(zip(*BipolarCell3bTOAmacrineCell1))
BipolarCell3bTOAmacrineCell1w = temp[2]; BipolarCell3bTOAmacrineCell1conn = list(zip(temp[0],temp[1]))

BipolarCell5RTOAmacrineCell1 = [(-3,0,.5),(-2,0,.5),(-1,0,.5),(0,0,.5),(1,0,.5),(2,0,.5),(3,0,.5),(4,0,.5)]
temp =  list(zip(*BipolarCell5RTOAmacrineCell1))
BipolarCell5RTOAmacrineCell1w = temp[2]; BipolarCell5RTOAmacrineCell1conn = list(zip(temp[0],temp[1]))
    
AmacrineCell1 = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a','BipolarCell3b',
    'BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCell6','BipolarCell7',
    'BipolarCell8','BipolarCell9','BipolarCellR'], "connectivity": {'BipolarCell3b': BipolarCell3bTOAmacrineCell1conn,'BipolarCell5R':BipolarCell5RTOAmacrineCell1conn},
    "weights": np.array(np.concatenate((BipolarCell3bTOAmacrineCell1w,BipolarCell5RTOAmacrineCell1w))), 
    "attributes": {'temporal': ['stretched_sin']*16,'duration': [1]*16,
    'coeffs': [[1]]*16, 'activation': 'relu','threshold': 0}}

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
    
BipolarCell4TOGanglionCellPV5 = [(-3,0,.5),(-2,0,.5),(-1,0,.5),(0,0,.5),(1,0,.5),(2,0,.5),(3,0,.5),(4,0,.5)]
temp =  list(zip(*BipolarCell4TOGanglionCellPV5))
BipolarCell4TOGanglionCellPV5w = temp[2]; BipolarCell4TOGanglionCellPV5conn = list(zip(temp[0],temp[1]))

AmacrineCellAIITOGanglionCellPV5 = [(-3,0,20),(-2,0,20),(-1,0,20),(0,0,20),(1,0,20),(2,0,20),(3,0,20),(4,0,20)]
temp =  list(zip(*AmacrineCellAIITOGanglionCellPV5))
AmacrineCellAIITOGanglionCellPV5w = temp[2]; AmacrineCellAIITOGanglionCellPV5conn = list(zip(temp[0],temp[1]))
        
GanglionCellPV5 = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4',
    'AmacrineCellWidefield','AmacrineCellAII','AmacrineCell1'], "connectivity":
    {'BipolarCell4': BipolarCell4TOGanglionCellPV5conn, 'AmacrineCellAII': AmacrineCellAIITOGanglionCellPV5conn},
    "weights": np.array(np.concatenate((BipolarCell4TOGanglionCellPV5w,AmacrineCellAIITOGanglionCellPV5w))),
    "attributes": {'temporal': ['stretched_sin']*16,'duration': [1]*16,
    'coeffs': [[1]]*16, 'activation': 'relu','threshold': 1.7, 'recurrent': PV5recurrent}}

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
    
###############################################################################    
    
# Rescale parameters relative to time and space partitioning    
    
BipolarCell1["attributes"]['width'] = BipolarCell1["attributes"]['width']/pixel
BipolarCell2["attributes"]['width'] = BipolarCell2["attributes"]['width']/pixel
BipolarCell3a["attributes"]['width'] = BipolarCell3a["attributes"]['width']/pixel
BipolarCell3b["attributes"]['width'] = BipolarCell3b["attributes"]['width']/pixel
BipolarCell4["attributes"]['width'] = BipolarCell4["attributes"]['width']/pixel
BipolarCell5A["attributes"]['width'] = BipolarCell5A["attributes"]['width']/pixel
BipolarCell5R["attributes"]['width'] = BipolarCell5R["attributes"]['width']/pixel
BipolarCell5X["attributes"]['width'] = BipolarCell5X["attributes"]['width']/pixel
BipolarCellX["attributes"]['width'] = BipolarCellX["attributes"]['width']/pixel
BipolarCell6["attributes"]['width'] = BipolarCell6["attributes"]['width']/pixel
BipolarCell7["attributes"]['width'] = BipolarCell7["attributes"]['width']/pixel
BipolarCell8["attributes"]['width'] = BipolarCell8["attributes"]['width']/pixel
BipolarCell9["attributes"]['width'] = BipolarCell9["attributes"]['width']/pixel
BipolarCellR["attributes"]['width'] = BipolarCellR["attributes"]['width']/pixel

# BipolarCell1["attributes"]['duration'] = BipolarCell1["attributes"]['duration']/temporal_res
# BipolarCell2["attributes"]['duration'] = BipolarCell2["attributes"]['duration']/temporal_res
# BipolarCell3a["attributes"]['duration'] = BipolarCell3a["attributes"]['duration']/temporal_res
# BipolarCell3b["attributes"]['duration'] = BipolarCell3b["attributes"]['duration']/temporal_res
# BipolarCell4["attributes"]['duration'] = BipolarCell4["attributes"]['duration']/temporal_res
# BipolarCell5A["attributes"]['duration'] = BipolarCell5A["attributes"]['duration']/temporal_res
# BipolarCell5R["attributes"]['duration'] = BipolarCell5R["attributes"]['duration']/temporal_res
# BipolarCell5X["attributes"]['duration'] = BipolarCell5X["attributes"]['duration']/temporal_res
# BipolarCellX["attributes"]['duration'] = BipolarCellX["attributes"]['duration']/temporal_res
# BipolarCell6["attributes"]['duration'] = BipolarCell6["attributes"]['duration']/temporal_res
# BipolarCell7["attributes"]['duration'] = BipolarCell7["attributes"]['duration']/temporal_res
# BipolarCell8["attributes"]['duration'] = BipolarCell8["attributes"]['duration']/temporal_res
# BipolarCell9["attributes"]['duration'] = BipolarCell9["attributes"]['duration']/temporal_res
# BipolarCellR["attributes"]['duration'] = BipolarCellR["attributes"]['duration']/temporal_res

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