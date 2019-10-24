"""
Set up and manipulate networks of retinal cells.
"""

from classes import *
from types import *
import numpy as np


def network_init(structs,size):
    # Initializes overlapping  2-D cell sheets and sets up their connectivity
    # according to local profiles of connectivity specified in every neuron
    
    # structs contains dictonaries of all possible cell types from types
    # size contains the number of cells in one dimension (Square 2-D sheets are created)
    # precedence should be given to upstream cells when setting up this dictionary
    
    global image_size, pixel
    img_size = [round(image_size[0]/pixel),round(image_size[1]/pixel)]
    neurons = {}
    
    for key, value in size.items():
        
        neuron_list = [[None]*value]*value
        x = np.linspace(1, img_size[0]-1, value)
        y = np.linspace(1, img_size[1]-1, value)
        XX, YY = np.meshgrid(x, y);
        
        struct = structs[key]
        
        for i in range(value):
            for j in range(value):
                
                # counter used to delete connections if at the edge of the image
                counter = 0
                # Build connections of this neuron
                center = np.asarray([XX[i,j],YY[i,j]])
                inputs = []
                
                # Parse through different cell types to find appropriate cells,
                # using the size of their grid
                
                # need to create a different procedure for bipolar cells, which
                # have only the image as an input
                for input_type in struct["inputs"]:
                    
                    connectivity = struct["connectivity"][input_type]
                    temp = center*size[input_type]/value; corr_center = temp.astype(int)
                    
                    for coords in connectivity:
                        
                        input_coords = corr_center + coords
                        
                        if all(0 < input_coords < size[input_type]):
                            
                            inputs.append(neurons[input_type][input_coords[0]][input_coords[1]])
                            
                        else:
                            
                            # delete corresponding weight etc to not mess up everything
                            
                
                
                # If a certain neuron not found, abort the corresponding weight
                
                
        
        
        neurons[key] = neuron_list
    
    return neurons