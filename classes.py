"""
Includes classes for all basic elements of the networks.
"""

# imports
import numpy as np
from scipy import signal
import copy as cp

# Global variables
image_size = [200, 200]      # in um
pixel = 5                    # in um
temporal_res = .1            # in msec
t_time = 100                 # in sec

###############################################################################

# Generic parent class for anything that is common across the building blocks 
# (elements) of the circuits

class Element:

    # Every element contains the name of its inputs and the corresponding weights
    def __init__(self, inputs, weights, center):
        # Inputs: array of input elements to this element
        # Weights: corresponding weights from these elements
        # Center: tuple indicating position in 2-D neural sheet. It is also center
        # of the receptive field for bipolar cells
        
        self.n_in = len(inputs)
        self.w = weights
        # Inputs is a list of objects. I want to compute their
        # output and use it is the output method
        self.inputs = inputs
        # Needed to compute which elements to connect to this element
        self.cemter = center
        
        # preallocate matrix of activity for this cell
        self.output = np.nan
        # Can be used to see if the cell has computed its output, to avoid
        # uneccesary computations. Can also be replaced by a time marker if the
        # network cannot be computed all at once
        
        
###############################################################################   
        
    
class BipolarCell(Element):
    
    def __init__(self, inputs, weights, center, attributes):
        # Attributes: contains a list of attributes needed to define the cell. Can contain:
        #       type: can be "On" or "Off" 
        #       separable: determines whether spatiotemporal field is separable, boolean
        #       spatiotemporal: contains given spatiotemporal receptive field
        #       spatial: contains spatial receptive field or function name to produce it
        #       width: width of the spatial receptive field
        #       temporal: contains temporal receptive field or function name to produce it
        #       duration: "time constant" of the temporal receptive field
        #       activation: the nonlinear activation function of the output
        #       threshold: threshold for the activation function
        #   Can also contain other parameters required to define the receptive
        #   field, look at respective function for what their name and
        #   specification should be
        
        super().__init__(inputs, weights, center)
        
        self.type = attributes["type"]
        self.separable = attributes["separable"]
        if self.separable:
            
            if isinstance(attributes["spatial"],np.ndarray):
                self.spatial = attributes["spatial"]
            else:
                self.spatial = Spatial(self.center,attributes)
                
            if isinstance(attributes["temporal"],np.ndarray):
                self.temporal = attributes["temporal"]
            else:
                self.temporal = Temporal(attributes)
                
            # Spatiotemporal receptive field is spatial * temporal
            temp1 = self.temporal[np.newaxis,:]; temp1 = temp1[np.newaxis,:]
            temp2 = np.expand_dims(self.spatial,2)
            self.spatiotemporal = temp2*temp1
        
        else:
            self.spatiotemporal = attributes["spatiotemporal"]
            
        if np.array_equal(self.type,"Off"):
            self.spatiotemporal = - self.spatiotemporal
        
        self.activation = attributes["activation"]
        self.threshold = attributes["threshold"]
    
    def out(self):
        # Since there is no recurrent connectivity involving bipolar cells,
        # we can compute all the output and then sample it from the other cells
        
        if not np.any(np.isnan(self.output)):
            pass
        else:
            # the first element of the list 'inputs' should contain the image
            temp = signal.fftconvolve(self.inputs[0],self.spatiotemporal,'same',axes = 2)
            self.output = activation(temp,self.activation,self.threshold)
        
        return self.output
    
    
###############################################################################  
    
    
class AmacrineCell(Element):
    
    def __init__(self, inputs, weights, center, attributes):
        # Attributes: contains a list of attributes needed to define the cell. Can contain:
        #       temporal: contains temporal receptive fields as matrix or list
        #       of function names to produce them. Each input has a different
        #       corresponding receptive field
        #       duration: array of "time constants" of temporal receptive fields
        #       activation: the nonlinear activation function of the output
        #       threshold: threshold for the activation function
        #   Can also contain other parameters required to define the receptive
        #   field, look at respective function for what their name should be
        
        super().__init__(inputs, weights, center)
        
        if isinstance(attributes["temporal"],np.ndarray):
                self.temporal = attributes["temporal"]
        else:
                self.temporal = Temporal_multiple(attributes,self.n_in)
        
        self.activation = attributes["activation"]
        self.threshold = attributes["threshold"]
        
    def out(self):
        # Amacrine cells receive recurrent connections

        # assuming that inputs is a list of input objects
        if not np.any(np.isnan(self.output)):
            pass
        else:
            values = np.asarray(list(map(lambda x: x.out(),self.inputs)))
            
            for i in range(self.n_in):
                # Special temporal receptive field for each input
                # Change shape of temporal to convolve
                temp = self.temporal[i][np.newaxis,:]; temp = temp[np.newaxis,:]
                values[i,:] = signal.fftconvolve(values[i,:],temp,'same',axes=2)
                
            # Use transpose to do multiplication with np.dot
            temp = np.dot(values.transpose(),self.w).transpose()
            self.output = activation(temp,self.activation,self.threshold)
        
        return self.output


###############################################################################


class GanglionCell(Element):
    
    def __init__(self, inputs, weights, center, attributes):
        # Attributes: contains a list of attributes needed to define the cell. Can contain:
        #       temporal: contains temporal receptive fields as matrix or list
        #       of function names to produce them. Each input has a different
        #       corresponding receptive field
        #       duration: array of "time constants" of temporal receptive fields
        #       activation: the nonlinear activation function of the output
        #       threshold: threshold for the activation function
        #       recurrent: recurrent filter coefficients
        #   Can also contain other parameters required to define the receptive
        #   field, look at respective function for what their name should be
        
        super().__init__(inputs, weights, center)
        
        if isinstance(attributes["temporal"],np.ndarray):
                self.temporal = attributes["temporal"]
        else:
                self.temporal = Temporal_multiple(attributes,self.n_in)
        
        self.activation = attributes["activation"]
        self.threshold = attributes["threshold"]
        
        if "recurrent" in attributes:
            self.recurrent = attributes["recurrent"]
        else:
            self.recurrent = np.nan
        
    def out(self):
        # Ganglion cells receive recurrent connections

        # assuming that inputs is a list of input objects
        if not np.any(np.isnan(self.output)):
            pass
        else:
            # care must be taken that the length of all arrays is equal. So to it
            # for values, after this it is ok
            values = np.asarray(list(map(lambda x: x.out(),self.inputs)))
            
            for i in range(self.n_in):
                # Special temporal receptive field for each input
                # Change shape of temporal to convolve
                temp = self.temporal[i][np.newaxis,:]; temp = temp[np.newaxis,:]
                values[i,:] = signal.fftconvolve(values[i,:],temp,'same',axes=2)
        
            # Use transpose to do multiplication with np.dot
            temp = np.dot(values.transpose(),self.w).transpose()
            
            if np.isnan(self.recurrent):
                self.output = activation(temp,self.activation,self.threshold)
            else:
                time_p = np.size(temp,2); l = np.size(self.recurrent)
                self.output = np.zeros((time_p,))
                
                for t in range(time_p):
                    if t < l:
                        self.output[t] = activation(temp[t],self.activation,self.threshold)
                    else:
                        temp[t] = temp[t] + np.dot(self.output[t-l:t],self.recurrent)
                        self.output[t] = activation(temp[t],self.activation,self.threshold)
        
        return self.output


###############################################################################


class Delay(Element):
    
    def __init__(self, inputs, weights, center, attributes):
        
        # Attributes can contain:
        # t_delay : time delay the element introduces
        
        super().__init__(inputs, weights, center)
        # convert time to count
        self.delay = int(np.round(attributes["t_delay"]/temporal_res))
    
    def out(self):
        
        if not np.any(np.isnan(self.output)):
            pass
        else:
            self.output = np.roll(self.inputs[0].out(),self.delay)
            self.output[0:self.delay] = 0
        
        return self.output


###############################################################################


class PresynapticSilencer(Element):
    
    # Used so that amacrine cells can silence bipolar cells before reaching
    # cells. Necessary component for OMS cells
    
    def __init__(self, inputs, weights, center, attributes):
        
        super().__init__(inputs, weights, center)
            
    def out(self):
        
        if not np.any(np.isnan(self.output)):
            pass
        else:
            values = np.asarray(list(map(lambda x: x.out(),self.inputs)))
            self.output = np.dot(values.transpose(),self.w).transpose()
        
        return self.output


###############################################################################


def Spatial(center,attributes):
    # Define spatial receptive fields. Options:
    #    difference of gaussians: "spatial" should be "DoG", other parameters
    #    needed in "attributes": "width", "center", "on_off_ratio"
    
    # Access global variables used throughout
    global image_size, pixel
    
    if np.array_equal(attributes["spatial"],'DoG'):
        spatial = DoG(attributes["width"],attributes["on_off_ratio"],center,image_size,pixel)
        
    return spatial


def DoG(sigmas,ratio,center,image_size,pixel):
    # Sigmas contain the standard deviations the positive (center) and negative
    # (surround) part. To invert the parts. use argument "type" in attributes
    # Ratio is the ratio of the peaks of the gaussians (center/surround)
    
    x = np.arange(0,image_size[0],pixel)
    y = np.arange(0,image_size[1],pixel)
    X, Y = np.meshgrid(x,y)
    
    norm_dist1 = 1/2*(((X-center[0])**2+(Y-center[1])**2)/sigmas[0]**2)
    norm_dist2 = 1/2*(((X-center[0])**2+(Y-center[1])**2)/sigmas[1]**2)
    gauss1 = ratio/(2*np.pi*sigmas[0]**2)*np.exp(-norm_dist1)
    gauss2 = 1/(2*np.pi*sigmas[1]**2)*np.exp(-norm_dist2)
    # Normalization? Should it not be zero sum? (no reaction to constant input)
    
    return (gauss1 - gauss2)/(1+ratio) 


###############################################################################


def Temporal_multiple(attributes,n):
    # Unpacks contents of attributes and passes them one at a time to Temporal()
    # Returns list of receptive fields, each one corresponding to one input
    
    temporals = [None]*n
    atts = cp.deepcopy(attributes)      # Changes in atts should not affect attributes
    
    for i in range(n):
        atts["temporal"] = attributes["temporal"][i]
        atts["duration"] = attributes["duration"][i]
        atts["coeffs"] =  attributes["coeffs"][i]
        temporals[i] = Temporal(atts)
        
    return temporals

def Temporal(attributes):
    # Define temporal receptive fields. Options:
    #    Adelson and Bergen 1985: "temporal" should be "Adelson_Bergen", other 
    #       parameters needed in "attributes": "duration"
    #    Stretched sin: "temporal" should be "stretched_sin", other parameters
    #       needed: "duration", "coeffs"
    
    # Access global variables used throughout
    global temporal_res
    
    if np.array_equal(attributes["temporal"],'Adelson_Bergen'):
        temporal = Adelson_Bergen(1/attributes["duration"],temporal_res)
    elif np.array_equal(attributes["temporal"],'stretched_sin'):
        temporal = stretched_sin(attributes["duration"],attributes["coeffs"],temporal_res)
        
    return temporal


def Adelson_Bergen(alpha,step):
    # Alpha acts as an inverse time constant. Equation (2.29) from Dayan & Abbott
    
    t = np.arange(0,20*alpha,step)
    norm_t  = alpha*t
    
    return alpha*np.exp(-norm_t)*(norm_t**5/np.math.factorial(5)-norm_t**7/np.math.factorial(7))

def stretched_sin(tf,coeffs,step):
    # Computes equation (5) from Keat et al 2001
    # tf is the maximal length of the filter
    # coeffs should be a numpy array with the corresponding coefficient of the
    # n-th term of (5) in position n-1
    
    filt = 0
    
    for i in range(len(coeffs)):
        filt = filt + coeffs[i]*stretched_sin_basis(tf,i+1,step)
        
    return filt

def stretched_sin_basis(tf,order,step):
    # Equation (6) from Keat et al 2001
    
    t = np.arange(0,tf,step)
    norm_t  = t/tf
    
    return np.sin(np.pi*order*(2*norm_t-norm_t**2))
    

###############################################################################


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