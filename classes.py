# Includes classes for all basic elements of the networks

import numpy as np
from scipy import signal

# Global variables
image = (200, 200)  # in um
pixel = 5           # in um
temporal_res = .1  # in msec


# Generic mother class for anything that is common across the building blocks 
# (elements) of the circuits

class Element:
    
    # Every element contains the name of its inputs and the corresponding weights
    def __init__(self, inputs, weights):
        # Inputs: array of input elements to this element
        # Weights: corresponding weights from these elements
        
        self.n_in = len(inputs)
        self.w = weights
        # Am I allowed to pass an array of objects? I want to compute their
        # output and used it is the output method
        self.inputs = inputs
        # Can be used to see if the cell has computed its output, to avoid
        # uneccesary computations. Can also be replaced by a time marker if the
        # network cannot be computed all at once. We do not focus on simulation for now
        self.output = np.nan
        
    
class BipolarCell(Element):
    
    def __init__(self, inputs, weights, attributes):
        # Attributes: contains a list of attributes needed to define the cell. Can contain:
        #       separable: determines whether spatiotemporal field is separable, boolean
        #       spatiotemporal: contains given spatiotemporal receptive field
        #       spatial: contains spatial receptive field or function name to produce it
        #       width: width of the spatial receptive field
        #       temporal: contains temporal receptive field or function name to produce it
        #       duration: "time constant" of the temporal receptive field
        #       center: the center of the receptive field
        #       activation: the nonlinear activation function of the output
        #       threshold: threshold for the activation function
        #   Can also contain other parameters required to define the receptive
        #   field, look at respective function for what their name should be
        
        super().__init__(inputs, weights)
        
        self.separable = attributes["separable"]
        if self.separable:
            
            if isinstance(attributes["spatial"],np.ndarray):
                self.spatial = attributes["spatial"]
            else:
                self.spatial = Spatial(attributes)
                
            if isinstance(attributes["temporal"],np.ndarray):
                self.temporal = attributes["temporal"]
            else:
                self.temporal = Temporal(attributes)
                
            # Spatiotemporal receptive field is spatial * temporal
            temp = self.temporal[np.newaxis,self.temporal[np.newaxis,:]]
            self.spatiotemporal = self.spatial*temp
        
        else:
            self.spatiotemporal = attributes["spatiotemporal"]
            
        self.activation = attributes["activation"]
        self.threshold = attributes["threshold"]
    
    def output(self,image,t):
        # Since there is no recurrent connectivity involving bipolar cells,
        # we can compute all the output and then sample it from the other cells
        if not np.isnan(self.output):
            pass
        else:
            temp = signal.fftconvolve(image,self.spatiotemporal,axes=2)
            self.output = activation(temp,self.activation,self.threshold)
        
        return self.output[t]
        


def Spatial(attributes):
    # Define spatial receptive fields. Options:
    #    mexican hat: "spatial" should be "mexican hat", other parameters
    #    needed in "attributes": "width", "center"
    
    # Access global variables used throughout
    global image, pixel
    
    if np.array_equal(attributes["spatial"],'mexican hat'):
        spatial = mexican_hat(attributes["width"],attributes["center"],image,pixel)
        
    return spatial


def mexican_hat(sigma,center,image,pixel):
    # The width is the standard deviation of the wavelet
    
    x = np.linspace(0,pixel,np.floor(image[0]/pixel))
    y = np.linspace(0,pixel,np.floor(image[1]/pixel))
    X, Y = np.meshgrid(x,y)
    norm_dist = 1/2*(((X-center[0])**2+(Y-center[1])**2)/sigma**2)
    
    return 1/(np.pi*sigma**2)*(1-norm_dist)*np.exp(-norm_dist)


def Temporal(attributes):
    # Define temporal receptive fields. Options:
    #    Adelson and Bergen 1985: "temporal" should be "Adelson_Bergen", other 
    #       parameters needed in "attributes": "duration"
    
    # Access global variables used throughout
    global temporal_res
    
    if np.array_equal(attributes["temporal"],'Adelson_Bergen'):
        temporal = Adelson_Bergen(1/attributes["duration"],temporal_res)
        
    return temporal


def Adelson_Bergen(alpha,step):
    # Alpha acts as an inverse time constant. Equation (2.29) from Dayan & Abbott
    
    t = np.linspace(0,step,20*alpha)
    norm_t  = alpha*t
    
    return alpha*np.exp(-norm_t)*(norm_t**5/np.math.factorial(5)-norm_t**7/np.math.factorial(7))


def activation(h,function,threshold):
    # Computes output of elements. Options:
    #   "relu", "sigmoid"
    
    if np.array_equal(function,"relu"):
        out = relu(h,threshold)
    elif np.array_equal(function,"sigmoid"):
        out = sigmoid(h,threshold)
        
    return out
    

def relu(h,threshold,gain = 1):
    # could define gain for each cell
    out = h-threshold; out[out<0] = 0
    return gain*out

def sigmoid(h,threshold,k=.5,b=5,s=1):
    # could define k, b and s separately for each cell
    return 1/(1+k*np.exp(-b*(h-threshold)))
    