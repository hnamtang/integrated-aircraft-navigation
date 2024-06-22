
import numpy as np


"""
    ecef = llh2ecef(llh)
    
    Converts Lat-Lon-Height (LLH) (WGS 84 Ellipse) to Earth-Centered-Earth-Fixed (ECEF) coordinates (WGS84).

    Input:  llh  = lat,long,height location (rad,rad,user units)
    Output: ecef = x,y,z (user units)

    20.10.2019  MUdH Copyright (c) TU Berlin

"""
def llh2ecef( llh ) :

    # Define constants
    a   = np.float64(6378137.0)               # Earth's semi-major axis [m]
    e   = np.float64(8.1819190842622e-02)     # Eccentricity
    esq = e * e
    
    # Short forms for sin and cos of latitude/longitude
    sp  = np.sin(llh[0,0])
    cp  = np.cos(llh[0,0])
    sl  = np.sin(llh[1,0])
    cl  = np.cos(llh[1,0])
    
    # Compute the prime radius of curvature
    gsq = 1.0 - (esq * sp * sp)
    Rp  = a / np.sqrt(gsq)

    # Compute the ECEF coordinates
    ecef = 0.0*llh
    ecef[0,0] = (Rp + llh[2,0]) * cp * cl;
    ecef[1,0] = (Rp + llh[2,0]) * cp * sl;
    ecef[2,0] = (Rp * (1.0 - esq) + llh[2,0]) * sp
    
    return ecef
