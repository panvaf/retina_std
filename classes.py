"""
Includes classes for all basic elements of the networks.
"""

# imports
import numpy as np
from scipy import signal
import copy as cp

# Global variables
image_size = np.array([500, 500])      # in um
pixel = 5                    # in um
img_size = (image_size/pixel).astype(int)  # number
temporal_res = 2            # in msec
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
        self.center = center
        
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
        
        self.activation = attributes["activation"]
        self.threshold = attributes["threshold"]
    
    def out(self):
        # Since there is no recurrent connectivity involving bipolar cells,
        # we can compute all the output and then sample it from the other cells
        
        if not np.any(np.isnan(self.output)):
            pass
        else:
            # the first element of the list 'inputs' should contain the image
            temp = signal.fftconvolve(self.inputs[0],self.spatiotemporal,'full',axes = 2)[:,:,0:np.size(self.inputs[0],2)]
            self.output = activation(np.sum(temp,axis = (0,1)),self.activation,self.threshold)
        
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
        
        if "recurrent" in attributes:
            self.recurrent = attributes["recurrent"]
        else:
            self.recurrent = np.nan
        
    def out(self):
        # Amacrine cells receive recurrent connections

        # assuming that inputs is a list of input objects
        if not np.any(np.isnan(self.output)):
            pass
        else:
            values = np.asarray(list(map(lambda x: x.out(),self.inputs)))
            
            for i in range(self.n_in):
                # Different temporal receptive field for each input
                values[i,:] = signal.fftconvolve(values[i,:],self.temporal[i],'full')[0:np.size(values[i,:])]
                
            # Use transpose to do multiplication with np.dot
            temp = np.dot(values.transpose(),self.w).transpose()
            
            try: 
                if np.isnan(self.recurrent):
                    self.output = activation(temp,self.activation,self.threshold)
            except:
                time_p = np.size(temp,0); l = np.size(self.recurrent)
                self.output = np.zeros((time_p,))
                
                for t in range(time_p):
                    if t < l:
                        self.output[t] = activation(temp[t],self.activation,self.threshold)
                    else:
                        temp[t] = temp[t] + np.dot(self.output[t-l:t],self.recurrent)
                        self.output[t] = activation(temp[t],self.activation,self.threshold)
        
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
            # care must be taken that the length of all arrays is equal. So do it
            # for values, after this it is ok
            values = np.asarray(list(map(lambda x: x.out(),self.inputs)))
            
            for i in range(self.n_in):
                # Different temporal receptive field for each input
                values[i,:] = signal.fftconvolve(values[i,:],self.temporal[i],'full')[0:np.size(values[i,:])]
    
            # Use transpose to do multiplication with np.dot
            temp = np.dot(values.transpose(),self.w).transpose()
                        
            try: 
                if np.isnan(self.recurrent):
                    self.output = activation(temp,self.activation,self.threshold)
            except:
                time_p = np.size(temp); l = np.size(self.recurrent)
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
        # t_delay : time delay the element introduces, in time units
        
        super().__init__(inputs, weights, center)
        # convert time to count
        self.delay = int(attributes["t_delay"])
    
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

# Functions for spatial attributes

def Spatial(center,attributes):
    # Define spatial receptive fields. Options:
    #    difference of gaussians: "spatial" should be "DoG", other parameters
    #    needed in "attributes": "width", "center", "on_off_ratio"
    
    # Access global variables used throughout
    global img_size
    
    if np.array_equal(attributes["spatial"],'DoG'):
        spatial = DoG(attributes["width"],attributes["on_off_ratio"],center,img_size)
    if np.array_equal(attributes["spatial"],'Gauss'):
        spatial = Gauss(attributes["width"],center,img_size)
    
    spatial = norm(spatial)
    
    return spatial


def DoG(sigmas,ratio,center,img_size):
    # Sigmas contain the standard deviations the positive (center) and negative
    # (surround) part. Ratio is the ratio of the peaks of the gaussians (center/surround)
    
    x = np.arange(0,img_size[0])
    y = np.arange(0,img_size[1])
    X, Y = np.meshgrid(x,y)
    
    norm_dist1 = 1/2*(((X-center[0])**2+(Y-center[1])**2)/sigmas[0]**2)
    norm_dist2 = 1/2*(((X-center[0])**2+(Y-center[1])**2)/sigmas[1]**2)
    gauss1 = ratio/(2*np.pi*sigmas[0]**2)*np.exp(-norm_dist1)
    gauss2 = 1/(2*np.pi*sigmas[1]**2)*np.exp(-norm_dist2)
    # Normalization? Should it not be zero sum? (no reaction to constant input)
    
    return (gauss1 - gauss2)/(1+ratio)

def Gauss(sigma,center,img_size):
    # Sigma: standard deviation of the receptive field
    
    x = np.arange(0,img_size[0])
    y = np.arange(0,img_size[1])
    X, Y = np.meshgrid(x,y)
    
    norm_dist = 1/2*(((X-center[0])**2+(Y-center[1])**2)/sigma**2)
    gauss =1/(2*np.pi*sigma**2)*np.exp(-norm_dist)
    
    return gauss

def Gaussian(x,y,sigmax,sigmay,peak):
    norm_dist = 1/2*(x**2/sigmax**2+y**2/sigmay**2)
    return peak*np.exp(-norm_dist)

###############################################################################

# Functions for temporal attributes

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
    
    if np.array_equal(attributes["temporal"],'Adelson_Bergen'):
        temporal = Adelson_Bergen(attributes["duration"])
    elif np.array_equal(attributes["temporal"],'stretched_sin'):
        temporal = stretched_sin(attributes["duration"],attributes["coeffs"])
        
    temporal = norm(temporal)
    
    return temporal


def Adelson_Bergen(duration):
    # Alpha acts as an inverse time constant. Equation (2.29) from Dayan & Abbott
    
    alpha = 20/duration
    t = np.arange(duration)
    norm_t  = alpha*t
    
    return alpha*np.exp(-norm_t)*(norm_t**5/np.math.factorial(5)-norm_t**7/np.math.factorial(7))

def stretched_sin(tf,coeffs):
    # Computes equation (5) from Keat et al 2001
    # tf is the maximal length of the filter
    # coeffs should be a numpy array with the corresponding coefficient of the
    # n-th term of (5) in position n-1
    
    filt = 0
    
    for i in range(len(coeffs)):
        filt = filt + coeffs[i]*stretched_sin_basis(tf,i+1)
        
    return filt

def stretched_sin_basis(tf,order):
    # Equation (6) from Keat et al 2001
    
    t = np.arange(tf)
    norm_t  = t/tf
    
    return np.sin(np.pi*order*(2*norm_t-norm_t**2))
    

###############################################################################

# Functions for activations

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
    out = h-threshold; 
    if np.isscalar(out): 
        if out<0: 
            out = 0 
    else: 
        out[out<0] = 0
    return gain*out

def sigmoid(h,threshold,k=.5,b=5,s=1):
    # could define k, b and s separately for each cell
    return 1/(1+k*np.exp(-b*(h-threshold)))

###############################################################################
    
# Utils
    
def norm(array):
    coeff = np.linalg.norm(array)
    if coeff>0:
        array = array/coeff
    return array