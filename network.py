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
    celltypes = ['BipolarCell','AmacrineCell','GanglionCell','Delay','PresynapticSilencer']
    
    for key, value in size.items():
        
        for temp in celltypes:
                    if temp in key:
                        celltype = temp
                        
        neuron_list = [[None]*value]*value
        x = np.linspace(1, img_size[0]-1, value)
        y = np.linspace(1, img_size[1]-1, value)
        XX, YY = np.meshgrid(x, y);
        
        struct = structs[key]
        
        for i in range(value):
            for j in range(value):
                
                # counter used to delete connections if at the edge of the image
                counter = 0
                weights = struct["weigths"]
                attributes = struct["attributes"]
                # Build connections of this neuron
                center = np.asarray([XX[i,j],YY[i,j]])
                inputs = []
                
                # Parse through different cell types to find appropriate cells,
                # using the size of their grid
                
                if celltype == 'BipolarCell':
                    inputs = struct['inputs']
                else:
                    
                    for input_type in struct["inputs"]:
                        
                        connectivity = struct["connectivity"][input_type]
                        temp = center*size[input_type]/value; corr_center = temp.astype(int)
                        
                        for coords in connectivity:
                            
                            input_coords = corr_center + coords
                            
                            if all(0 < input_coords < size[input_type]):
                                
                                inputs.append(neurons[input_type][input_coords[0]][input_coords[1]])
                                
                            else:
                                # If coordinates point out of grid:
                                # delete corresponding weight etc to not mess up everything
                                # also delete corresponding coeffs, duration and temporal arguments
                                weights = np.delete(weights,counter)
                                del attributes["temporal"][counter]
                                del attributes["duration"][counter]
                                del attributes["coeffs"][counter]
                                # arrays now are one element shorter
                                counter = counter - 1
                                
                        counter = counter + 1
                        
                neuron_list[i][j] = celltype(inputs,weights,center,attributes)
                    
        neurons[key] = neuron_list
    
    return neurons