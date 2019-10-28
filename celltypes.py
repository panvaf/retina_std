"""
Contains information about all retinal cell types necessary to define them,
stored in dictionaries. 

Information about the cell itself is directly fed into the corresponding class.
Information about the way it connects is manipulated by network_init in network
to produce the appropriate connectivity in overlapping 2-D sheets of cells.
"""

import numpy as np
from classes import *

# note: width should be in um and then transformed to pixel values in the code.
# Otherwise there is no generality

img_size = (image_size/pixel).astype(int)
image = np.zeros((img_size[0],img_size[1],100))

BipolarCellTemplate = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': [10, 20],
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

# order of parameters in attributes is order of element position in connectivity,
# and order element types are read in connectivity is determined in inputs
    
AmacrineCellTemplate = {"inputs":['BipolarCellTemplate'], "connectivity": {'BipolarCellTemplate':
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}