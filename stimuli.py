"""
Create stimuli to probe the networks.
"""

import numpy as np

def expanding_disk(pos,speed,width,exp_rate,maxwidth,amplitude,gridsize,appears,duration,order=10):
    # Creates artificial stimuli of expanding disks. Params:
    # pos: 2 dim starting position in grid coordinates
    # speed: 2 dim speed vector, in pixels per time point
    # width: the initial width of the disk
    # exp_rate: the rate with which the disk expands, in pixels per time point
    # maxwidth: the maximum attainable width of the disk
    # amplitude: peak amplitude of the disk
    # gridsize: the size of the grid, in pixels
    # appears: the time point the disk first appears
    # duration: the temporal extent of the stimulus, in units of time
    # order: controls how sharp the transition on the margins of the disk is
    
    disk_dur = duration - appears
    xc = pos[0] + speed[0]*np.arange(disk_dur)
    yc = pos[1] + speed[1]*np.arange(disk_dur)
    w = width + exp_rate*np.arange(disk_dur)
    w[w>maxwidth] = maxwidth
    # correction for negative expansion rates
    if exp_rate<0:
            w[w<1] = 1
        
    # do a meshgrid over 3 coordinates (x,y,w)
    
    x = np.arange(gridsize); y = np.arange(gridsize)
    X, Y, W = np.meshgrid(x,y,w)
    norm_dist = ((X-xc)**2+(Y-yc)**2)/W**2
    stim1 = amplitude*np.exp(-1/2*norm_dist**int(order/2))
    
    stim = np.zeros((gridsize,gridsize,duration))
    stim[:,:,appears:duration] = stim1
    
    return stim

def expanding_annuli(pos,speed,width,init,exp_rate,maxsize,amplitude,gridsize,appears,duration,order=10):
    # Creates artificial stimuli of expanding annuli. Params:
    # pos: 2 dim starting position in grid coordinates
    # speed: 2 dim speed vector, in pixels per time point
    # width: the width of the annulus
    # init: the initial size of the annulus
    # exp_rate: the rate with which the annulus expands, in pixels per time point
    # maxsize: the maximum attainable width of the annulus
    # amplitude: peak amplitude of the annulus
    # gridsize: the size of the grid, in pixels
    # appears: the time point the annulus first appears
    # duration: the temporal extent of the stimulus, in units of time
    # order: controls how sharp the transition on the margins of the annulus is
    
    base = expanding_disk(pos,speed,init,exp_rate,maxsize,amplitude,gridsize,appears,duration,order)
    extract = expanding_disk(pos,speed,init-width,exp_rate,maxsize-width,amplitude,gridsize,appears,duration,order)
    stim = base - extract    
    
    return stim


def moving_bars(k,speed,theta,phase,contrast,gridsize,duration):
    # Creates artificial stimuli of moving bars. Params:
    # k: spatial frequency of the bars, in inverse pixel values
    # speed: amplitude and direction of moving speed
    # theta: orientation of the bars in space in rads, 0 rads being horizontal
    # contrast: amplitude of positive and negative amplitude of negative part
    # gridsize: the size of the grid, in pixels
    # duration: the temporal extent of the stimulus, in units of time
    
    x = np.arange(gridsize); y = np.arange(gridsize); t = np.arange(duration)
    X, Y, T = np.meshgrid(x,y,t)
    
    stim = np.cos(2*np.pi*k*X*np.cos(theta)+2*np.pi*k*Y*np.sin(theta)+phase-2*np.pi*speed*T)
    
    return contrast*np.sign(stim)


def counterphase_grating(k,f,theta,phase,contrast,gridsize,duration):
    # Creates artificial stimuli of moving bars. Equation 2.18 from Dayan & Abbott. Params:
    # k: spatial frequency of the bars, in inverse pixel values
    # f: temporal frequency of the bars, in inverse temporal unit values
    # theta: orientation of the bars in space in rads, 0 rads being horizontal
    # contrast: amplitude of positive and negative amplitude of negative part
    # gridsize: the size of the grid, in pixels
    # duration: the temporal extent of the stimulus, in units of time
    
    x = np.arange(gridsize); y = np.arange(gridsize); t = np.arange(duration)
    X, Y, T = np.meshgrid(x,y,t)
    
    stim = contrast*np.cos(2*np.pi*k*X*np.cos(theta)+2*np.pi*k*Y*np.sin(theta)+phase)*np.cos(2*np.pi*f*T)
    
    return stim


def flashing_disk(pos,width,amplitude,f,gridsize,duration,order=10):
    # Creates artificial stimuli of expanding disks. Params:
    # pos: 2 dim starting position in grid coordinates
    # width: the initial width of the disk
    # amplitude: peak amplitude of the disk
    # f: frequency of flashing
    # gridsize: the size of the grid, in pixels
    # duration: the temporal extent of the stimulus, in units of time
    # order: controls how sharp the transition on the margins of the disk is
        
    x = np.arange(gridsize); y = np.arange(gridsize); t = np.arange(duration)
    X, Y, T = np.meshgrid(x,y,t)
    norm_dist = ((X-pos[0])**2+(Y-pos[1])**2)/width**2
    stim = amplitude*np.exp(-1/2*norm_dist**int(order/2))*np.cos(2*np.pi*f*T)
    
    return stim