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
    
    neurons = {}
    
    for key, value in size.items():
        
        neuron_list = [[None]*value]*value
        neurons[key] = neuron_list
    
    return neurons