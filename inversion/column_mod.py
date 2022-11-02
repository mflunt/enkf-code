#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:20:07 2020

Convolve model outputs to satellite columns  

@author: mlunt
"""
import numpy as np
import xarray
import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit


@jit(nopython=True)
def numba_interp(log_p_old_3d, log_p_new_3d, ch4_3d):
    """
    May as well loop through time and varname in this loop as well if I can.
    Think that may speed things up even more.
    
    Performs linear interpolation in the vertical dimension
    
    Requires that inputs are of shape (ntime,nlat,nlon,nlev)
    
    """
    ntime = len(log_p_old_3d[:,0,0,0])
    nlat = len(log_p_old_3d[0,:,0,0])
    nlon = len(log_p_old_3d[0,0,:,0])
    nlev_old = len(log_p_old_3d[0,0,0,:])
    nlev_new = len(log_p_new_3d[0,0,0,:])
    
    
    ch4_new = np.zeros((ntime,nlat,nlon,nlev_new))
    #ch4_out = np.zeros((nlev_new))

        
    for ti in range(ntime):
        for lai in range(nlat):
            for loi in range(nlon):
                
                ch4_i = ch4_3d[ti,lai,loi,:]
                
                if np.isfinite(ch4_i[0]):
                #levi=0
                    for zi in range(nlev_new):
                    
                        logp_i = log_p_new_3d[ti,lai,loi,zi]
                        logp_old_i = log_p_old_3d[ti,lai,loi,:]
                        
                        # If new array goes outside range of old array then fill with max value at bottom and min value at top
                        if logp_i > log_p_old_3d[ti,lai,loi,0]:
                            ch4_new[ti,lai,loi,zi] = ch4_3d[ti,lai,loi,0]
                            
                        elif logp_i <= log_p_old_3d[ti,lai,loi,-1]:
                            ch4_new[ti,lai,loi,zi] = ch4_3d[ti,lai,loi,-1]
                            
                        else:
                            
                            # What if logp_i == logp_old_i[-1] - think sorted in abover elif by doing <=
                            for yi in range(nlev_old-1):
                                if logp_old_i[yi] >= logp_i > logp_old_i[yi+1]:
                                    levi = yi
                                    break
                            
                            ch4_new[ti,lai,loi,zi] = (ch4_i[levi]* (logp_old_i[levi+1] - logp_i) + ch4_i[levi+1] * 
                                   (logp_i - logp_old_i[levi])) / (logp_old_i[levi+1] - logp_old_i[levi])
                        
    return ch4_new

def numba_interp_levs_1d(log_p_old_1d, log_p_new_1d, var_1d):
    """
    May as well loop through time and varname in this loop as well if I can.
    Think that may speed things up even more.
    
    Performs linear interpolation in the vertical dimension
    
    Requires that inputs are of shape (nlev)
    
    """
    nlev_old = len(log_p_old_1d)
    nlev_new = len(log_p_new_1d)
    
    
    var_new = np.zeros((nlev_new))
               
    for zi in range(nlev_new):
    
        logp_i = log_p_new_1d[zi]
        #logp_old_i = log_p_old_1d[:]
        
        # If new array goes outside range of old array then fill with max value at bottom and min value at top
        if logp_i > log_p_old_1d[0]:
            var_new[zi] = var_1d[0]
            
        elif logp_i <= log_p_old_1d[-1]:
            var_new[zi] = var_1d[-1]
            
        else:
            
            # What if logp_i == logp_old_i[-1] - think sorted in abover elif by doing <=
            for yi in range(nlev_old-1):
                if log_p_old_1d[yi] >= logp_i > log_p_old_1d[yi+1]:
                    levi = yi
                    break
            
            var_new[zi] = (var_1d[levi]* (log_p_old_1d[levi+1] - logp_i) + var_1d[levi+1] * 
                   (logp_i - log_p_old_1d[levi])) / (log_p_old_1d[levi+1] - log_p_old_1d[levi])
                        
    return var_new

@jit(nopython=True)
def numba_interp_levs_2d(log_p_old_2d, log_p_new_2d, var_2d):
    """
    May as well loop through time and varname in this loop as well if I can.
    Think that may speed things up even more.
    
    Performs linear interpolation in the vertical dimension
    
    Requires that inputs are of shape (nobs, nlev)
    
    Can I perform this in array space rather than looping through all obs?
    
    """
    ntime = len(log_p_old_2d[:,0])
    nlev_old = len(log_p_old_2d[0,:])
    nlev_new = len(log_p_new_2d[0,:])
    
    
    var_new = np.zeros((ntime,nlev_new))
    
    for ti in range(ntime):
        var_i = var_2d[ti,:]
        
        if np.isfinite(var_i[0]):
        #levi=0
            for zi in range(nlev_new):
            
                logp_i = log_p_new_2d[ti,zi]
                logp_old_i = log_p_old_2d[ti,:]
                
                # If new array goes outside range of old array then fill with max value at bottom and min value at top
                if logp_i > log_p_old_2d[ti,0]:
                    var_new[ti,zi] = var_2d[ti,0]
                    
                elif logp_i <= log_p_old_2d[ti,-1]:
                    var_new[ti,zi] = var_2d[ti,-1]
                    
                else:
                    
                    # What if logp_i == logp_old_i[-1] - think sorted in abover elif by doing <=
                    for yi in range(nlev_old-1):
                        if logp_old_i[yi] >= logp_i > logp_old_i[yi+1]:
                            levi = yi
                            break
                    
                    var_new[ti,zi] = (var_i[levi]* (logp_old_i[levi+1] - logp_i) + var_i[levi+1] * 
                           (logp_i - logp_old_i[levi])) / (logp_old_i[levi+1] - logp_old_i[levi])
                        
    return var_new
    

def combine_plevs(p_mod,p_sat, ground_to_top=True, D4 = False):
    """
    Function to combine the pressure levels from model and satellite into one array
    
    Need to keep track of indices of each combined level
    Inputs:     p_mod = model pressure levels - should be model edges
                p_sat = satellite pressure levels - should be level edges
                ground_to_top = Order of pressure levels i .e high to low pressure
    
    Outputs :   p_comb = ordered combined pressure levels
                wh_sat = indices corresponding to sat levels
                wh_mod = indices corresponding to model levels 
    """
    
    if D4 == False:
        p_comb2 = np.sort(np.concatenate((p_mod,p_sat))) # Order from high to low
        
        # Remove any duplicate  pressure levels and order from high to low:
        if ground_to_top == True:
            p_comb = np.unique(p_comb2)[::-1]
        else:
            p_comb = np.unique(p_comb2)
        
        wh_sat=[]
        wh_mod=[]
        for p in p_sat:
            wh_sat.append(np.where(p_comb == p)[0][0])
        for pi in p_mod:
            wh_mod.append(np.where(p_comb == pi)[0][0])
            
        return p_comb, wh_mod,wh_sat
        
    if D4 == True:
        p_comb2 = np.sort(np.concatenate((p_mod,p_sat),axis=3),axis=3) # Order from high to low
        
        if ground_to_top == True:
            p_comb = p_comb2,axis=2[:,:,::-1]
        else:
            p_comb = p_comb2.copy()
        
        # This won't work for multidimensional p
#        wh_sat=[]
#        wh_mod=[]
#        for p in p_sat:
#            wh_sat.append(np.where(p_comb == p)[0][0])
#        for pi in p_mod:
#            wh_mod.append(np.where(p_comb == pi)[0][0])
    
        return p_comb

def get_pressure_weight(pres, p_surf):
    """
    pres = 1d array of shape (nlev)
    p_surf = scalar value
    Outputs:
        pres_wgt - weights fror each pressure level in forming the columns
    """
    pres_wgt = pres.copy()*0.
    
    #p_surf2 = p_surf[:,:,:,np.newaxis]
    
    pdiff = pres[1:] - pres[:-1] # length n-1
    log_pdiff = np.log(pres[1:]/pres[:-1])
    
    # Lowest level
    pres_wgt[0] = np.abs(-1.*pres[0] + (pdiff[0]/log_pdiff[0]))* 1./p_surf
    
    # Highest level
    pres_wgt[-1] = np.abs(pres[-1] - (pdiff[-1]/log_pdiff[-1]))* 1./p_surf
    
    # Middle levels
    pres_wgt[1:-1] = np.abs( (pdiff[1:]/log_pdiff[1:]) - (pdiff[:-1]/log_pdiff[:-1]))*1./p_surf
    
    return pres_wgt

def calc_air_mass(p_edges, p_min, D4=False, D2=False):
    """
    Calculate the mass of air between successive pressure levels
    """
    g_const = 9.81
    
    if D4 == True:
        nlev = len(p_edges[0,0,0,:])    
        
        mass_air = p_edges.copy()*0.
        
        # Flip levels if they run from low to high 
        if p_edges[0,0,0,2] - p_edges[0,0,0,1] > 0:
            p_edges=p_edges[:,:,:,::-1]
        
        mass_air[:,:,:,:-1] =   (p_edges[:,:,:,:-1] - p_edges[:,:,:,1:])/g_const  # kg/m2
        mass_air[:,:,:,-1] = (p_edges[:,:,:,-1] - p_min)/g_const
        
    elif D2 == True:
        nlev = len(p_edges[0,:])    
        
        mass_air = p_edges.copy()*0.
        
        # Flip levels if they run from low to high 
        if p_edges[0,2] - p_edges[0,1] > 0:
            p_edges=p_edges[:,::-1]
        
        mass_air[:,:-1] =   (p_edges[:,:-1] - p_edges[:,1:])/g_const  # kg/m2
        mass_air[:,-1] = (p_edges[:,-1] - p_min)/g_const
       
    else:
        nlev = len(p_edges)
        
        mass_air = np.zeros((nlev)) 
        
        # Flip levels if they run from low to high 
        if p_edges[2] - p_edges[1] > 0:
            p_edges=p_edges[::-1]
        
        mass_air[:-1] =   (p_edges[:-1] - p_edges[1:])/g_const  # kg/m2
        mass_air[-1] = (p_edges[-1] - p_min)/g_const
        
        
    return mass_air
    

def convolve_tropomi(p_edge_mod, p_edge_sat, ch4_mod, ground_to_top=True):
    """
    Map model fields onto vertical levels of TROPOMI retrieval
    
    Due to layer nature of TROPOMI levels need to interpolate cumulative
    atmospheric mass. 
    
    Do linear interpolation in log10 space.
    
    """
    nlev_mod = len(p_edge_mod)
    nlev_sat = len(p_edge_sat)
    
    p_edge_comb, wh_mod_levs, wh_sat_levs = combine_plevs(p_edge_mod, p_edge_sat, 
                                                          ground_to_top=ground_to_top)
    
    # Calculate total air mass in mass/m2 in each model vertical level
    mass_air_mod = calc_air_mass(p_edge_mod, 0.01)
    mass_air_comb = calc_air_mass(p_edge_comb, 0.01)
    
    
    # Multiply by ch4 mixing ration in each model layer to get CH4 mass
    mass_ch4_mod = mass_air_mod * ch4_mod*1.e-9
    
    # Calculate the cumulative atmosphere mass at each pressure edge of model
    cum_ch4_mass_mod = np.zeros((nlev_mod+1))
    for li in range(nlev_mod):
        cum_ch4_mass_mod[1+li] = np.sum(mass_ch4_mod[:li+1])    
        
    # Interpolate cumulative mass to satellite pressure levels
    
    p_edge_mod2 = np.append(p_edge_mod,[0.01]) # Add additional upper bounding layer to model levels 
    p_edge_comb2 = np.append(p_edge_comb,[0.01])
    fill_cumu = (cum_ch4_mass_mod[-1], cum_ch4_mass_mod[0]) # Lower and upper pressure bounds fill values
    
    # Linear interpolation in log_10 pressure
    interp_func = interpolate.interp1d(np.log10(p_edge_mod2), cum_ch4_mass_mod, bounds_error=False, fill_value=fill_cumu)   
    cum_ch4_mass_comb = interp_func(np.log10(p_edge_comb2))
    
    cum_mass_ch4_out=np.zeros((nlev_sat+1))
    cum_mass_ch4_out[:-1] =  cum_ch4_mass_comb[wh_sat_levs]
    cum_mass_ch4_out[-1] = cum_ch4_mass_mod[-1]
    cum_mass_ch4_out[0]=0.
    
    mass_air_out = np.zeros((nlev_sat))
    for levi in range(nlev_sat):
    # Need to account for when p_edge_comb > plevs_out[0]
    
        if levi == 0:
            wh = np.where(p_edge_comb > p_edge_sat[levi+1])
        elif levi == nlev_sat-1:
            wh = np.where(p_edge_comb <= p_edge_sat[levi])
        else:
            wh = np.where(np.logical_and(p_edge_comb <= p_edge_sat[levi],
                                     p_edge_comb > p_edge_sat[levi+1]))    

        mass_air_out[levi] = np.sum(mass_air_comb[wh]) 
    
    
    mass_ch4_out2 = cum_mass_ch4_out[1:] - cum_mass_ch4_out[:-1]
    ch4_prof_out = mass_ch4_out2/mass_air_out*1.e9
    
    return ch4_prof_out
    

def convolve_gosat(p_edge_mod, p_node_sat, ch4_mod, ground_to_top=True):
    """
    Map model fields onto vertical levels of GOSAT retrieval
    
    Due to node nature of GOSAT levels, just need to interpolate mixing ratio in log10(p) space

    """
    p_cent_mod = p_edge_mod.copy()*0.
    p_cent_mod[:-1] = (p_edge_mod[:-1]+p_edge_mod[1:])/2.
    p_cent_mod[-1] = (p_edge_mod[-1]+ 0.01)/2.
    
    
    p_edge_comb, wh_mod_levs, wh_sat_levs = combine_plevs(p_edge_mod, p_node_sat, 
                                                          ground_to_top=ground_to_top)
    
    if ground_to_top == True:
        fill_ch4 = (ch4_mod[-1], ch4_mod[0])
    interp_func_ratio = interpolate.interp1d(np.log10(p_cent_mod), ch4_mod, bounds_error=False, fill_value=fill_ch4)
    
    ch4_comb_edge = interp_func_ratio(np.log10(p_edge_comb))
    
    ch4_prof_out = ch4_comb_edge[wh_sat_levs]
    
    return ch4_prof_out
    
def setup_plevels_3d(p_edge_mod_3d, p_edge_sat_3d, ch4_mod_3d, ground_to_top=True):
    """
     Map model fields onto vertical levels of TROPOMI retrieval
    
    Due to layer nature of TROPOMI levels need to interpolate cumulative
    atmospheric mass. 
    
    Do linear interpolation in log10 space.
    
    """
    nlev_mod = len(p_edge_mod_3d[0,0,:])
    #nlev_sat = len(p_edge_sat_3d[0,0,:])
    
    nlat_mod = len(p_edge_mod_3d[:,0,0])
    nlon_mod = len(p_edge_mod_3d[0,:,0])
    
    p_edge_comb_3d, wh_mod_levs, wh_sat_levs = combine_plevs(p_edge_mod_3d, p_edge_sat_3d, 
                                                          ground_to_top=ground_to_top, D4=True)
    
    # Calculate total air mass in mass/m2 in each model vertical level
    mass_air_mod = calc_air_mass(p_edge_mod_3d, 0.01)
    mass_air_comb = calc_air_mass(p_edge_comb_3d, 0.01)
    
    
    # Multiply by ch4 mixing ration in each model layer to get CH4 mass
    mass_ch4_mod = mass_air_mod * ch4_mod_3d*1.e-9
    
    # Calculate the cumulative atmosphere mass at each pressure edge of model
    cum_ch4_mass_mod = np.zeros((nlat_mod,nlon_mod,nlev_mod+1))
    for li in range(nlev_mod):
        cum_ch4_mass_mod[:,:,1+li] = np.sum(mass_ch4_mod[:,:,:li+1],axis=2)    
        
    # Interpolate cumulative mass to satellite pressure levels
    
    p_edge_mod2 = np.append(p_edge_mod_3d,[0.01]) # Add additional upper bounding layer to model levels 
    p_edge_comb2 = np.append(p_edge_comb_3d,[0.01])
    #fill_cumu = (cum_ch4_mass_mod[-1], cum_ch4_mass_mod[0]) # Lower and upper pressure bounds fill values
    
    return  p_edge_mod2, p_edge_comb2, cum_ch4_mass_mod, mass_air_comb
    
    
    #%% 
    # Need to perform 1d interpolation over a 3D array
  
    
def convert_mass_to_profile(cum_ch4_mass_new, air_mass_sat):  
    """
    Assumes that all arrays are for the model satellite levels only
    
    Need to check I can do this for total air mass as well.
    
    cum_ch4_mass_top = cumulative mass of ch4 at top of model atmosphere
    """
    
    # Linear interpolation in log_10 pressure
    #interp_func = interpolate.interp1d(np.log10(p_edge_mod2), cum_ch4_mass_mod, bounds_error=False, fill_value=fill_cumu)   
    #cum_ch4_mass_comb = interp_func(np.log10(p_edge_comb2))
    
    #ntime = len(p_edge_sat_3d[:,0,0,0])
    #nlat = len(p_edge_sat_3d[0,:,0,0])
    #nlon = len(p_edge_sat_3d[0,0,:,0])
    
    #nlev_sat = len(p_edge_sat_3d[0,0,0,:])
    
    #cum_mass_ch4_out=np.zeros((ntime,nlat,nlon,nlev_sat+1))
    
    """
    # Might need to do this bit in numba as well
    # But why do I need this combined grid in the first place? 
    # Interpolation will deal with this already won't it. So then can't I just output sat levs from interpolation?
    
    
    What is the advantage of putting things on a combined grid? 
    """
    #cum_mass_ch4_out[:,:,:,:-1] =  cum_ch4_mass_new.copy()
    #cum_mass_ch4_out[:,:,:,-1] = cum_ch4_mass_top
    #cum_mass_ch4_out[:,:,:,0]=0.
    
    
    mass_ch4_out2 = cum_ch4_mass_new[:,:,:,1:] - cum_ch4_mass_new[:,:,:,:-1]
    ch4_prof_out = mass_ch4_out2/air_mass_sat*1.e9
    
    return ch4_prof_out
    
    
    
    