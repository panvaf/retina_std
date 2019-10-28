"""
Set up and manipulate networks of retinal cells.
"""

from classes import *
import numpy as np
import copy as cp


def network_init(structs,size,classes):
    # Initializes overlapping  2-D cell sheets and sets up their connectivity
    # according to local profiles of connectivity specified in every neuron
    
    # structs contains dictonaries of all possible cell types from types
    # size contains the number of cells in one dimension (Square 2-D sheets are created)
    # precedence should be given to upstream cells when setting up this dictionary
    
    global img_size
    cells = {}
    
    for key, value in size.items():
        
        # find what kind of cells to create
        celltype = 0
        for temp in classes.keys():
                    if temp in key:
                        celltype = temp
                        
        # initialize 2-D grid of cells
        neuron_list = [[None]*value for i in range(value)]
        x = np.linspace(1, img_size[0]-1, value)
        y = np.linspace(1, img_size[1]-1, value)
        
        struct = structs[key]
        
        # create one cell at a time
        for i in range(value):
            for j in range(value):
                
                # counter used to delete connections if at the edge of the image
                counter = 0
                weights = struct["weights"]
                attributes = cp.deepcopy(struct["attributes"])
                # Build connections of this neuron
                center = np.asarray([x[i],y[j]])
                inputs = []
                
                # Parse through different cell types to find appropriate cells,
                # using the size of their grid
                
                if celltype == 'BipolarCell':
                    inputs = struct['inputs']
                else:
                    
                    for input_type in struct["inputs"]:
                        
                        # find corresponding coordinates of this cell in the 2-D
                        # grid of another cell type
                        connectivity = struct["connectivity"][input_type]
                        temp = np.divide(center,img_size)*size[input_type]; corr_center = temp.astype(int)
                        
                        # find absolute coordinates of each cell connecting to this cell
                        # by using relative coordinates around corr_center
                        for coords in connectivity:
                            
                            input_coords = corr_center + coords
                            
                            if ((0 < input_coords) & (input_coords < size[input_type])).all():
                                
                                inputs.append(cells[input_type][input_coords[0]][input_coords[1]])
                                
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
                        
                neuron_list[i][j] = classes[celltype](inputs,weights,center,attributes)
                    
        cells[key] = neuron_list
    
    return cells