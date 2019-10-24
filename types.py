"""
Contains information about all retinal cell types necessary to define them,
stored in dictionaries. 

Information about the cell itself is directly fed into the corresponding class.
Information about the way it connects is manipulated by network_init in network
to produce the appropriate connectivity in overlapping 2-D sheets of cells.
"""

import numpy as np

image = np.zeros((round(image_size[0]/pixel),round(image_size[1]/pixel),100))

BipolarTemplate = {"inputs":[image], "connectivity": [], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': [10, 20],
     'on_off_ratio': 3, 'temporal': 'stretched_sin', 'duration': 10,
     'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}

# order of parameters in attributes is order of element position in connectivity,
# and order element types are read in connectivity is determined in inputs
    
AmacrineTemplate = {"inputs":[BipolarTemplate], "connectivity": {"BipolarTemplate":
    [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]},
    "weights": np.array([1]*9), "attributes": {'temporal': ['stretched_sin']*9,'duration': [1]*9,
    'coeffs': [[1]]*9, 'activation': 'relu','threshold': 0, 'recurrent': [1, -0.2]}}