
import numpy as np

"""
    enu = ecef2enu(ecef, orgece, orgllh)
    
    Convert ECEF coordinates to East-North-Up with respect to orgece and orgllh (orgece is the same location as orgllh).


    Inputs:     orgece [3x1] - ENU origin in ECEF
                orgllh [3x1] - ENU origin in LLH
                ecef   [3xN] - ECEF coordinates of points ot be converted
    Outputs:    enu    [3xN] - ENU coordinates
"""
def ecef2enu ( ecef, orgece, orgllh) :

    # Compute ECEF vector w.r.t. origin
    difece = ecef - orgece
   
    # Set up terms for sines and cosines
    sla = np.sin(orgllh[0,0]); 
    cla = np.cos(orgllh[0,0]);
    slo = np.sin(orgllh[1,0]); 
    clo = np.cos(orgllh[1,0]);

    # Earth to navigation frame 
    C = np.array([[  -slo,  clo, 0], 
                  [ -sla*clo,  -sla*slo, cla],
                  [ cla*clo, cla*slo, sla] ]);
    
    # Rotate the difece vector from Earth to Navigation frame orientation
    enu = np.matmul(C,difece)
    
    return enu
