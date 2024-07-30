"""
    ephcal.py
    
    Module with orbital position functions
    
    ephcal  -   calculate position based on ephemerides, satellite ID and time
    
    
    Copyright (c) - Maarten Uijt de Haag and Frank van Graas
"""

import numpy as np

"""
    [svpos, svclock, ecode] = ephcal(transmit_time, ephem, svid)
    
    Ephemeris calculations

     input:  svid, ephem[svid,parameters (1..28)]
     output: svpos[3][1] and svclock offset
             ecode = 0: No detectable errors
             ecode = 1: Semi-major axis <= 0.0
             ecode = 2: Eccentric anomaly did not converge
             ecode = 3: No ephemeris for this satellite
         
         function [svpos, svclock, ecode] = ephcal(transmit_time, ephem, svid)
"""
def ephcal(transmit_time, ephem, svid) :

    # define constants    
    grav	= np.float64(3.986005e14)
    eanvel	= np.float64(7.2921151467e-05)
    fclk	= np.float64(-4.442807633e-10)
    epsi	= np.float64(1.0e-13)
    itmax	= 20
    pi		= np.float64(3.1415926535898)

    ecode = 0      # Assume success

    # find if the SV is in the list
    ix = (ephem[:,0] == svid).nonzero()[0]
    
    if (ix.size == 0) :
        ecode = 3
    else :
        ix = ix[0]
        tgd    = ephem[ix,5]
        toc    = ephem[ix,6]
        af2    = ephem[ix,7]
        af1    = ephem[ix,8]
        af0    = ephem[ix,9]
        crs    = ephem[ix,11]
        deltan = ephem[ix,12]
        m0     = ephem[ix,13]
        cuc    = ephem[ix,14]
        eccen  = ephem[ix,15]
        cus    = ephem[ix,16]
        sqrsma = ephem[ix,17]
        toe    = ephem[ix,18]
        cic    = ephem[ix,19]
        omega0 = ephem[ix,20]
        cis    = ephem[ix,21]
        i0     = ephem[ix,22]
        crc    = ephem[ix,23]
        omega  = ephem[ix,24]
        omgdot = ephem[ix,25]
        idot   = ephem[ix,27]
    
        tk = transmit_time - toe
        if (tk >  302400.0) : tk = tk - 604800.0
        if (tk < -302400.0) : tk = tk + 604800.0
    
        tmod = transmit_time - toc
        if (tmod >  302400.0) : tmod = tmod - 604800.0
        if (tmod < -302400.0) : tmod = tmod + 604800.0
    
        smx = sqrsma * sqrsma
    	
        if (smx <= 0.0) :
            ecode = 1
        else :
            n0 = np.sqrt( grav / (smx * smx * smx) )
            n = n0 + (deltan * pi)
            mk = (m0 * pi) + (n * tk)
    
            iters = 0
            eatemp = mk
            eadiff = 2 * epsi
            while ( (iters <= itmax) and (abs(eadiff) > epsi) ) :
                if (iters > 0) :
                    eatemp = ea
                ea = mk + (eccen * np.sin(eatemp))
                eadiff = ea - eatemp
                iters = iters + 1
                if (iters > itmax) : 
                    ecode = 2
            
            ek = ea
    
            oneme2 = np.sqrt( 1.0 - (eccen * eccen) )
            coea = np.cos(ek)
            xvk = (coea - eccen) / (1.0 - (eccen * coea) )
            yvk = (oneme2 * np.sin(ek)) /  ( 1.0 - (eccen * coea) )
            vk = np.arctan2 (yvk, xvk)
            phik = vk + (omega * pi)
            phik2 = phik + phik
        
            ds = np.sin(phik2)
            dc = np.cos(phik2)
            deluk = (cus * ds) + (cuc * dc)
            delrk = (crc * dc) + (crs * ds)
            delik = (cic * dc) + (cis * ds)
            uk = phik + deluk
            ik = (i0 * pi) + delik + (idot * tk * pi)
            rk = (smx *  (1.0 - (eccen * coea))) + delrk
            xkprim = rk * np.cos(uk)
            ykprim = rk * np.sin(uk)
            omegak = (omega0 * pi) + (((omgdot * pi) - eanvel) * tk) - (eanvel * toe)
    
            ds = np.sin(omegak)
            dc = np.cos(omegak)
            svpos = np.zeros((3,1))
            svpos[0,0] = (xkprim * dc) - (ykprim * np.cos(ik) * ds)
            svpos[1,0] = (xkprim * ds) + (ykprim * np.cos(ik) * dc)
            svpos[2,0] = ykprim * np.sin(ik)
    
            dtr = fclk * eccen * sqrsma * np.sin(ek)
            svclock = af0 + (af1 * tmod) + (af2 * tmod * tmod) + dtr - tgd
        
        return (svpos, svclock, ecode)

