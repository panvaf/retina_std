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
import stimuli as stim

# note: width should be in um in the structs and then transformed to pixel values
# later in the script. similarly, duration should be in ms and then transformed to
# units of temporal res. All inputs to the network functions and to the classes
# should be normalised in units of pixels and temporal res. Then one only need to
# multiply with those quantities to revert back to physical dimensions

# order of parameters in attributes is order of element position in connectivity,
# and order element types are read in connectivity is determined in inputs

image = stim.expanding_disk([100,100],[0,0],50,0,100,1,200,1000,4000) + \
        stim.expanding_disk([100,100],[0,0],50,0,100,-1,200,2500,4000)

# The receptive fields Dawna gave me were sampled at 2 ms. I keep this time step here
# however I could consider lowering it since the receptive fields of ganglion and
# amacrine cells (differentiators in essense) are quite crude (10 ms duration)
BipolarTemporals = loadmat('C:\\Users\\user\\Documents\\CNS\\1st Rotation\\data\\bip_trf_Franke_spl09999tuk005causNorm.mat')
BipolarTemporals = BipolarTemporals['kernels']
PV5recurrent = loadmat('C:\\Users\\user\\Documents\\CNS\\1st Rotation\\data\\gc_fdbk_filt.mat')
PV5recurrent = PV5recurrent['feedbackfilt']

# make sure any receptive fields that are determined by the user are normalised

###############################################################################

# Bipolar cells

BipolarCell1 = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'Off', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,0], 'activation': 'relu', 'threshold': 5}}

BipolarCell2 = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'Off', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,1], 'activation': 'relu', 'threshold': 5}}

BipolarCell3a = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'Off', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,2], 'activation': 'relu', 'threshold': 20}}

BipolarCell3b = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'Off', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,3], 'activation': 'relu', 'threshold': 20}}
    
BipolarCell4 = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'Off', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,4], 'activation': 'relu', 'threshold': 10}}
    
BipolarCell5A = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,5], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell5R = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,6], 'activation': 'relu', 'threshold': 0}}

BipolarCell5X = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,7], 'activation': 'relu', 'threshold': 0}}
    
BipolarCellX = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,8], 'activation': 'relu', 'threshold': 5}}
    
BipolarCell6 = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,9], 'activation': 'relu', 'threshold': 0}}
    
BipolarCell7 = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,10], 'activation': 'relu', 'threshold': 0}}

BipolarCell8 = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,11], 'activation': 'relu', 'threshold': 0}}

BipolarCell9 = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,12], 'activation': 'relu', 'threshold': 0}}

BipolarCellR = {"inputs":[image], "connectivity": [], "weights": np.array([1]),
    "attributes": {'type': 'On', 'separable': True, 'spatial': 'Gauss', 'width': 30,
    'temporal': BipolarTemporals[:,13], 'activation': 'relu', 'threshold': 0}}
    
###############################################################################
    
# Amacrine cells    
    
AmacrineCellOffS = {"inputs":['BipolarCell2','BipolarCell3a'], "connectivity": {'BipolarCell2':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0}}
    
AmacrineCellOnS = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell7'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0}}
    
AmacrineCellWidefield = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4',
    'BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCellX'], "connectivity": {'BipolarCell3a':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0}}

BipolarCell6TOAmacrineCellAII = [(0,0,1)]
temp =  list(zip(*BipolarCell6TOAmacrineCellAII))
BipolarCell6TOAmacrineCellAIIw = temp[2]; BipolarCell6TOAmacrineCellAIIconn = list(zip(temp[0],temp[1]))

AmacrineCellAII = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a','BipolarCell3b',
    'BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCell6','BipolarCell7',
    'BipolarCell8','BipolarCell9','BipolarCellR'], "connectivity": {'BipolarCell6': BipolarCell6TOAmacrineCellAIIconn},
    "weights": np.array(BipolarCell6TOAmacrineCellAIIw), "attributes": {'temporal': ['stretched_sin']*1,'duration': [10]*1,
    'coeffs': [[2,3]]*1, 'activation': 'relu','threshold': 1.0}}

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
    "attributes": {'temporal': ['stretched_sin']*16,'duration': [10]*16,
    'coeffs': [[2,3]]*16, 'activation': 'relu','threshold': 0}}

###############################################################################

# Presynaptic inhibition

PresynapticSilencerBip5AAmAII = {"inputs":['BipolarCell5A','AmacrineCellAII'], 
    "connectivity": {'BipolarCell5A':(0,0),'AmacrineCellAII':(0,0)},
    "weights": np.array([.5,-1]), "attributes": {}}

###############################################################################
    
# Ganglion cells

phenom_rec_field = np.array([250,250])
n_amII = (phenom_rec_field/BipolarCell6['attributes']['width']).astype(int)  # using bipolar because amacrines do not have spatial rec fields defined
AmacrineCellIITOGanglionCellsOFFa = [(int(x-round(n_amII[0]/2)),int(y-round(n_amII[1]/2)),-1) for x in range(n_amII[0]) for y in range(n_amII[1])]
temp =  list(zip(*AmacrineCellIITOGanglionCellsOFFa))
AmacrineCellIITOGanglionCellsOFFaw = temp[2]; AmacrineCellIITOGanglionCellsOFFaconn = list(zip(temp[0],temp[1]))
total_n = len(AmacrineCellIITOGanglionCellsOFFaconn)

# bipolar 1 and 2 are OFF cells, they could also connect directly to here

GanglionCellsOFFa = {"inputs":['BipolarCell1','BipolarCell2','AmacrineCellAII'],
    "connectivity": {'AmacrineCellAII':AmacrineCellIITOGanglionCellsOFFaconn},
    "weights": np.array(AmacrineCellIITOGanglionCellsOFFaw), "attributes":
    {'temporal': ['stretched_sin']*total_n,'duration': [10]*total_n,
    'coeffs': [[2,3]]*total_n, 'activation': 'relu','threshold': 0}}

GanglionCellFminiOFF = {"inputs":['BipolarCell1','BipolarCell2','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellFmidiOFF = {"inputs":['BipolarCell1','BipolarCell2','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

BipolarCell4TOGanglionCellPV5 = [(-3,0,.5),(-2,0,.5),(-1,0,.5),(0,0,.5),(1,0,.5),(2,0,.5),(3,0,.5),(4,0,.5)]
temp =  list(zip(*BipolarCell4TOGanglionCellPV5))
BipolarCell4TOGanglionCellPV5w = temp[2]; BipolarCell4TOGanglionCellPV5conn = list(zip(temp[0],temp[1]))

AmacrineCellAIITOGanglionCellPV5 = [(-3,0,-20),(-2,0,-20),(-1,0,-20),(0,0,-20),(1,0,-20),(2,0,-20),(3,0,-20),(4,0,-20)]
temp =  list(zip(*AmacrineCellAIITOGanglionCellPV5))
AmacrineCellAIITOGanglionCellPV5w = temp[2]; AmacrineCellAIITOGanglionCellPV5conn = list(zip(temp[0],temp[1]))
        
GanglionCellPV5 = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4',
    'AmacrineCellWidefield','AmacrineCellAII','AmacrineCell1'], "connectivity":
    {'BipolarCell4': BipolarCell4TOGanglionCellPV5conn, 'AmacrineCellAII': AmacrineCellAIITOGanglionCellPV5conn},
    "weights": np.array(np.concatenate((BipolarCell4TOGanglionCellPV5w,AmacrineCellAIITOGanglionCellPV5w))),
    "attributes": {'temporal': ['stretched_sin']*16,'duration': [10]*16,
    'coeffs': [[2,3]]*16, 'activation': 'relu','threshold': 1.7, 'recurrent': PV5recurrent}}

GanglionCellooDS37c = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellooDS37d = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellooDS37r = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellooDS37v = {"inputs":['BipolarCell1','BipolarCell2','BipolarCell3a',
    'BipolarCell3b','BipolarCell4','BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOffS','AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell1':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellW3 = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4','BipolarCell5A',
    'BipolarCell5R','BipolarCell5X','BipolarCellX','AmacrineCellWidefield','AmacrineCellAII'],
    "connectivity": {'BipolarCell3a': [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

# this was modelled in  Schwartz 2012

phenom_rec_field = np.array([250,250])
n_bip6 = (phenom_rec_field/BipolarCell6['attributes']['width']).astype(int)
BipolarCell6TOGanglionCellsONa = [(int(x-round(n_bip6[0]/2)),int(y-round(n_bip6[1]/2)),Gaussian(x-n_bip6[0]/2,y-n_bip6[1]/2,n_bip6[0]/3,n_bip6[1]/3,.5)) for x in range(n_bip6[0]) for y in range(n_bip6[1])]
temp =  list(zip(*BipolarCell6TOGanglionCellsONa))
BipolarCell6TOGanglionCellsONaw = temp[2]; BipolarCell6TOGanglionCellsONaconn = list(zip(temp[0],temp[1]))
total_n = len(BipolarCell6TOGanglionCellsONaconn)

GanglionCellsONa = {"inputs":['BipolarCell6','BipolarCell7','BipolarCell8','BipolarCell9',
    'BipolarCellR','AmacrineCellAII'], "connectivity": {'BipolarCell6':BipolarCell6TOGanglionCellsONaconn},
    "weights": np.array(BipolarCell6TOGanglionCellsONaw), "attributes": {'temporal': ['stretched_sin']*total_n,'duration': [10]*total_n,
    'coeffs': [[2,3]]*total_n, 'activation': 'relu','threshold': 0}}

GanglionCellFminiON = {"inputs":['BipolarCell3a','BipolarCell3b','BipolarCell4',
    'BipolarCell5A','BipolarCell5R','BipolarCell5X','BipolarCellX','AmacrineCellWidefield',
    'AmacrineCellAII'], "connectivity": {'BipolarCell3a':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCellFmidiON = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'BipolarCellX','AmacrineCellWidefield','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

# Bipolar5 was split to 3 categories, and we do not know which connects here. 
# It does not make much of a difference though in terms of receptive fields.

# could also connect bipolar6 to simplify the circuit
    
phenom_rec_field = np.array([200,200])
n_bip5A = (phenom_rec_field/BipolarCell5A['attributes']['width']).astype(int)
BipolarCell5ATOGanglionCelltONa = [(int(x-round(n_bip5A[0]/2)),int(y-round(n_bip5A[1]/2)),Gaussian(x-n_bip5A[0]/2,y-n_bip5A[1]/2,n_bip5A[0]/3,n_bip5A[1]/3,.5)) for x in range(n_bip5A[0]) for y in range(n_bip5A[1])]
temp =  list(zip(*BipolarCell5ATOGanglionCelltONa))
BipolarCell5ATOGanglionCelltONaw = temp[2]; BipolarCell5ATOGanglionCelltONaconn = list(zip(temp[0],temp[1]))

n_amII = (phenom_rec_field/BipolarCell6['attributes']['width']).astype(int) 
AmacrineCellAIITOGanglionCelltONa = [(int(x-round(n_amII[0]/2)),int(y-round(n_amII[1]/2)),-1) for x in range(n_amII[0]) for y in range(n_amII[1])]
temp =  list(zip(*AmacrineCellAIITOGanglionCelltONa))
AmacrineCellAIITOGanglionCelltONaw = temp[2]; AmacrineCellAIITOGanglionCelltONaconn = list(zip(temp[0],temp[1]))

weights = np.array(np.concatenate((BipolarCell5ATOGanglionCelltONaw,AmacrineCellAIITOGanglionCelltONaw)))
total_n = len(weights)
    
GanglionCelltONa = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'BipolarCellX','AmacrineCellWidefield','AmacrineCellAII'], "connectivity": 
    {'BipolarCell5A':BipolarCell5ATOGanglionCelltONaconn,'AmacrineCellAII':AmacrineCellAIITOGanglionCelltONaconn},
    "weights": weights, "attributes": {'temporal': ['stretched_sin']*total_n,'duration': [10]*total_n,
    'coeffs': [[2,3]]*total_n, 'activation': 'relu','threshold': 0}}
    
phenom_rec_field = np.array([200,200])
n_presyn = (phenom_rec_field/BipolarCell5A['attributes']['width']).astype(int)
PresynapticSilencerBip5AAmAIITOGanglionCelltONaPre = [(int(x-round(n_presyn[0]/2)),int(y-round(n_presyn[1]/2)),Gaussian(x-n_presyn[0]/2,y-n_presyn[1]/2,n_presyn[0]/3,n_presyn[1]/3,1)) for x in range(n_presyn[0]) for y in range(n_presyn[1])]
temp =  list(zip(*PresynapticSilencerBip5AAmAIITOGanglionCelltONaPre))
PresynapticSilencerBip5AAmAIITOGanglionCelltONaPrew = temp[2]; PresynapticSilencerBip5AAmAIITOGanglionCelltONaPreconn = list(zip(temp[0],temp[1]))
total_n = len(PresynapticSilencerBip5AAmAIITOGanglionCelltONaPrew)

    
GanglionCelltONaPre = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'BipolarCellX','AmacrineCellWidefield','AmacrineCellAII','PresynInhBip5AAmAII'], "connectivity": 
    {'PresynapticSilencerBip5AAmAII':PresynapticSilencerBip5AAmAIITOGanglionCelltONaPreconn},
    "weights": PresynapticSilencerBip5AAmAIITOGanglionCelltONaPrew, "attributes": {'temporal': ['stretched_sin']*total_n,'duration': [10]*total_n,
    'coeffs': [[2,3]]*total_n, 'activation': 'relu','threshold': 0}}

GanglionCellsOnDS7id = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellsOnDS7ir = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
GanglionCellsOnDS7iv = {"inputs":['BipolarCell5A','BipolarCell5R','BipolarCell5X',
    'AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'BipolarCell5A':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}

GanglionCelltOnDS7o = {"inputs":['AmacrineCellOnS','AmacrineCellAII'], "connectivity": {'AmacrineCellOnS':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [10]*9,
    'coeffs': [[2,3]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}
    
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
GanglionCelltONaPre["attributes"]['duration'] = [dur/temporal_res for dur in GanglionCelltONaPre["attributes"]['duration']]