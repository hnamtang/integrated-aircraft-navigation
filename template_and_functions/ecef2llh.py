
import numpy as np
from numpy.linalg import inv

"""
    llh = ecef2llh(llh)
    
    Converts Earth-Centered-Earth-Fixed (ECEF) to Lat-Lon-Height (LLH) coordinates (WGS 84 Ellipse)

    Input:      ecef = x,y,z (user units)
    Output:  llh  = lat,long,height location (rad,rad,user units)
    
    See lecture  notes
 
    20.10.2019  MUdH Copyright (c) TU Berlin

"""
def ecef2llh (ecef) :

    # Define constants
    a = np.float64(6378137.0);
    e = np.float64(8.1819190842622e-02);
    esq = e * e;
    
    # Extract the coordinates
    X = ecef[0,0]
    Y = ecef[1,0]
    Z = ecef[2,0]
    
    # R squared
    rsq = (X * X) + (Y * Y)
    
    # Initail estimate of H
    H = esq * Z
    
    cnt = 0
    done = 0
    while (cnt < 10) and (done == 0) :
        
        # Update the loop counter
        cnt = cnt + 1
        
        # Update estimate of Zp
        Zp = Z + H
        
        # Update the estimate of R
        R = np.sqrt(rsq + (Zp * Zp))
        
        # Compute the sin of the latitude
        sp = Zp / R
        
        # Estimate the radius of curvature
        gsq = 1.0 - (esq * sp * sp)
        Rp = a / np.sqrt(gsq)
        
        # New estimate of H (temporarily called P)
        P = Rp * esq * sp
        
        # Check if the change in H is small enough to stop iterating
        if np.abs(H - P) < 5.0e-04 :
            done = 1
        
        H = P
    
    llh = 0.0*ecef
    
    llh[0,0] = np.arctan2( Zp, np.sqrt(rsq) )
    llh[1,0] = np.arctan2( Y, X)
    llh[2,0] = R - Rp
    
    return llh
