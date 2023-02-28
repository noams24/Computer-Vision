import numpy as np
import sys

def log_filt(ksize, sig):
    std2 = float( sig**2 )
    x = np.arange( -(ksize-1)/2, (ksize-1)/2+1, 1)
    y = np.arange( -(ksize-1)/2, (ksize-1)/2+1, 1)
    X, Y = np.meshgrid(x, y)
    arg = -(X*X + Y*Y)/(2*std2);
    
    h = np.exp(arg);
    eps = sys.float_info.epsilon
    h[h < eps*np.max(h)] = 0;
    
    sumh = np.sum(h)
    if sumh != 0:
       h  = h/sumh
       
       # now calculate Laplacian     
    h1 = h*(X*X + Y*Y - 2*std2)/(std2**2);
    h = h1 - np.sum(h1)/(ksize*ksize) # make the filter sum to zero
  
    return h
