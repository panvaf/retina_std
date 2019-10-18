"""
Create and run networks for testing.
"""

from classes import * # necessary to import object classes
import numpy as np

image = np.zeros((round(image_size[0]/pixel),round(image_size[1]/pixel)))

BipolarStruct1 = {"inputs":[image], "weights": np.array([1]), "attributes":
    {'type': 'On', 'separable': True, 'spatial': 'DoG', 'width': [10, 20],
     'center': [0, 0], 'on_off_ratio': 3, 'temporal': 'stretched_sin',
     'duration': 10, 'coeffs': [1, -0.3, 0.1], 'activation': 'relu', 'threshold': 0}}
    
# Create multiple bipolar cells of the same type
Bipolars1 = [ BipolarCell(BipolarStruct1["inputs"],BipolarStruct1["weights"],BipolarStruct1["attributes"]) for i in range(10)]

for Bipolar in Bipolars1: Bipolar.output = 0

# need to find a more automated way to put together large networks, maybe use 
# classes with parameters that can draw across an entire range of bipolar cells?

AmacrineStruct1 = {"inputs":[Bipolars1[1],Bipolars1[2],Bipolars1[3]], 
    "weights": np.array([1,1,1]), "attributes": {'temporal': ['stretched_sin', 'stretched_sin', 'stretched_sin'],
    'duration': [1,1,1], 'coeffs': [[1],[1],[1]] , 'activation': 'relu', 'threshold': 0}}
    
Amacrine1 = AmacrineCell(AmacrineStruct1["inputs"],AmacrineStruct1["weights"],AmacrineStruct1["attributes"])