"""
    Position Computation
    
    - Calculate the satellite position at a particular time (given)
    - Convert the position to ENU at a specific origin (given)
    
    Copyright (c) - Maarten Uijt de Haag
"""

# ISSUES WITH 6 AND 1  !!!

import time
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as la

from ephcal import *
from llh2ecef import *
from ecef2llh import *
from ecef2enu import *

# GPS value for speed of light
vlight   = 299792458.0    
   
# WGS-84 value for earth rotation rate
eanvel = 7.2921151467e-05   

def compute_svpos_svclock_compensate( gpstime, svid, pr, ephem ) :
    

    
    pr_input = np.zeros((0,1))
    sv_clock = np.zeros((0,0))
    sv_ECEF  = np.zeros((3,0))
    sv_temp  = np.zeros((3,1))
    sv_ident = np.zeros((0,0))
    nosvs    = 0
    
           
    for jj in range(0,len(svid))  :
    
        # Extract the pseudorange
        pseudorange = pr[jj];

        # Find SVID in ephemerides array
        # Extract all time with this time tag
        ind = (ephem[:,0] == svid[jj]).nonzero()[0]
    
        if ind.size > 0 :
            
            # Solve for SV position and clock offset
            
            # Estimate the time-of-transmission corrected for SV clock offset
            tot = gpstime - pseudorange / vlight;
            (svpos, svclock, ecode) = ephcal(tot, ephem, svid[jj]);
            
            # Make sure that the clock offset is OK
            if abs(svclock) < 0.01 : 
                
                # Correct tot also for the satellite clock offset
                tot = gpstime - (pseudorange / vlight) - svclock
                
                # Redo the satellite position and clock offset calculation
                (svpos, svclock, ecode) = ephcal(tot, ephem, svid[jj]);
            
            
            
            # If no errors, prepare the measurements and satellite positions
            if (ecode == 0)  :
                
                # Add the pseudorange measurement
                pr_input = np.append(pr_input,pr[jj])
                
                # Correct satellite position for earth rotation during
                # travel time from satellite to user GPS receiver
                # Note: it is not necessary to correct for the receiver
                # clock offset since this particular GPS receiver
                # synchronizes its measurements to GPS time such that the
                # receiver clock offset is negligible (< 100 ns)
                travel_time = pseudorange / vlight
                if abs(svclock) < 0.01 :
                    travel_time = travel_time + svclock;
                
                # Make sure that the travel_time is not unreasonable
                if (travel_time < 0.06) : 
                    travel_time = 0.06; 
                if (travel_time > 0.10) : 
                    travel_time = 0.10; 
                    
                alfa = travel_time * eanvel;
                
                sv_temp[0,0] = svpos[0,0] + alfa * svpos[1,0];
                sv_temp[1,0] = svpos[1,0] - alfa * svpos[0,0];
                sv_temp[2,0] = svpos[2,0];
                
                sv_ECEF = np.append(sv_ECEF,sv_temp,axis=1)
                sv_clock = np.append(sv_clock,svclock)
                sv_ident = np.append(sv_ident,svid[jj])
                
                # Another good satellite
                nosvs = nosvs + 1;
    
    #            else :
    #                print 'Could not find ephemerides for SV ', ind                
        else:
            print('Crap')                
    
    # Compensate the pseudoranges for the clock
    pr_input = pr_input + sv_clock*vlight
    
    return(sv_ECEF, sv_clock)
  
def compute_pos_ecef( gpstime, pr, sv_ECEF, sv_clock) :
    
    nosvs    = len(pr)
                  
    # Compensate the pseudoranges for the clock
    pr_input = pr + sv_clock*vlight
    
    # Maximum number of iterations        
    itmax = 50             
    
    # Desired accuracy in solution
    epsi = 0.1              

    flag = 0;
    if (nosvs == 4) : flag = 1
    
    # No iterations done, yet
    itflg = 0;

    # Set initial error larger than epsi
    duab = 2.0 * epsi;
    
    # Set initial position and clock (at center of Earth and no clock error)
    #r_usr_est = np.zeros((4,1))
    x = np.zeros((4,1))
    
    # Setup dr and H matrices
    z = np.zeros((nosvs,1))
    H = np.zeros((nosvs,4))
    rho_est = np.zeros((nosvs,1))
    llh_data = np.zeros((4,1))
    
    # Inital error
    pos_error = 10000
    
    if nosvs >= 4 :
    
        while pos_error > epsi and itflg < itmax :
            
            # Increment iteration counter
            itflg = itflg + 1
            
            # Setup the measurement matrix H and the error vector
            for jj in range(0,nosvs) :
                
                # Compute the estimated range
                rho_est[jj,0] = la.norm(x[0:3,[0]] - sv_ECEF[:,[jj]]) + x[3,0]
                #rho_est[jj,0] = la.norm(x[0:3,[0]] - sv_ECEF[:,[jj]]) 
        
        
                # Compute the H-matrix (measurement matrix)
                H[jj,0:3] = (x[0:3,[0]] - sv_ECEF[:,[jj]]).T/rho_est[jj,0]
                H[jj,3] = 1.0
                
                z[jj,0] = pr_input[jj]
                
                #rho_est[jj,0] = la.norm(x[0:3,[0]] - sv_ECEF[:,[jj]]) + x[3,0]
        
            
            # Step 2 - compute the estimated range error
            delta_rho_est = z - rho_est
            
            # Step 4 - Estimate the new user position
            delta_x = la.inv(H.T @ H) @ H.T @ delta_rho_est 
        
            # Step 5 - Update position estimate 
            previous_x = x
            x = x + delta_x
            
            # Compute 
            pos_error = la.norm(x - previous_x)
            
        
        # GPS user position in ECEF coordinates
        user_ECEF = x[0:3,[0]]
        
        user_clock = x[3,[0]]
        
        # Compute GPS user position in Lat-Lon-Height (LLH)
        user_LLH = ecef2llh (user_ECEF);
        
        if user_LLH.size > 0 :
                    
            # Compute
            #user_ECEF = tf.llh2ecef(user_LLH)
            #user_ENU  = tf.ecef2enu(user_ECEF, orgece, orgllh)
            
            llh_data[0,0] = gpstime
            llh_data[1,0] = user_LLH[0,0]
            llh_data[2,0] = user_LLH[1,0]
            llh_data[3,0] = user_LLH[2,0] 
            
            
       
        return(user_ECEF, user_clock)