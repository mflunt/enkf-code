#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:57:11 2019

This is script used for inversion for AFrica TROPOMI paper

Base future editing off this e.g. to make a genral version for everyone to use.

Best temporary thing might be to filter out anywhere CH4_REF is greater than 1800.
Or at what magnitude will precision be an issue? Should be 2 dp but never mind. Say 2000 ppb.

@author: mlunt
"""

import numpy as np
#import LEnTKF
import xarray
import matplotlib.pyplot as plt
import scipy.linalg as sp_linalg
import ensemble_mod
from acrg_grid import haversine
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
import scipy.stats as spstats
from dateutil.relativedelta import relativedelta
import time as run_time
import argparse
import pandas as pd
#from enkf_code import GC_setup_mod as setup_mod
import regionmask
from numba import jit
#from joblib import Parallel, delayed

####@jit(nopython=True)
def numba_letkf(nstate, N_ens, nruns, wh_local, R_inv, Y_bg_devs_ens, y_bg_mean,
                n0, cov_inflate,
                x_bg_mean, X_bg_devs_ens, x_bg_ens):
    """
    Numba wrapper for assimilation step
    See if this speeds things up at all?
    
#    """
    w_a_out = np.zeros((nstate,N_ens))
    SQ_rt_out = np.zeros((nstate,N_ens,N_ens))
    y_a_mean3 = y_bg_mean.copy()*0.
    count_y3 = y_bg_mean.copy()*0.
    X_a_ens_out=np.zeros((nstate*nruns, N_ens))
    for xi in range(nstate):
    
        wh_xi = wh_local[xi]   # Indices of observations to be used to constrain state vector index.
        if len(wh_xi) > 0:
        # Doesn't work taking R_inv[wh,wh]  - 
        
        #dum1 = np.dot(R_inv[wh_xi, wh_xi[:,np.newaxis]], Y_bg_devs_ens[wh_xi,:])  # Can't have new axis in numba 
        #wh_xi2 = np.reshape(wh_xi, (len(wh_xi),1))
        #dum0 = R_inv[wh_xi,:] # This indexing of R_inv is a problem
        
        #What if I select R_inv indices outside of the loop?
            #dum1 = np.dot(R_inv[xi], Y_bg_devs_ens[wh_xi,:])
            #dum1 = np.dot(R_inv[wh_xi, np.expand_dims(wh_xi,1)], Y_bg_devs_ens[wh_xi,:])
            
            #dum1 = Rinv_dot_Y[wh_xi,:]
            dum1 = np.dot(R_inv[np.ix_(wh_xi, wh_xi)].T, Y_bg_devs_ens[wh_xi,:])
            #dum1 = np.dot(R_inv_split[xi], Y_bg_devs_ens[wh_xi,:])
            
            dum2 = np.dot(Y_bg_devs_ens[wh_xi,:].T,dum1)
            
            
            P_hat_a_inv = dum2 + np.diag(np.zeros((N_ens))+(N_ens-1)/cov_inflate)
           
            P_hat_a = np.linalg.inv(P_hat_a_inv)
            
            #RHS1 = np.dot(Y_bg_devs_ens[wh_xi,:].T,R_inv[wh_xi, wh_xi[:,None]])
            
            #RHS1 = np.dot(Y_bg_devs_ens[wh_xi,:].T,R_inv[xi])
            
            RHS1 = dum1.T
            
            RHS2 = np.dot(P_hat_a, RHS1)
            
            B,S = np.linalg.eigh((N_ens-1)*P_hat_a)
            diag_B_sq = np.diag(np.sqrt(B))
            S_inv = np.linalg.inv(S)
            SQ_rt=np.dot(np.dot(S,diag_B_sq),S_inv)
            
            
            #SQ_rt = sp_linalg.sqrtm((N_ens-1)*P_hat_a)   # SQ_RT is Wa in Hunt paper
            w_a = np.dot(RHS2,n0[wh_xi])
        
            y_a_mean3[wh_xi] = (y_a_mean3[wh_xi] + 
                         y_bg_mean[wh_xi] + np.dot(Y_bg_devs_ens[wh_xi,:],w_a))
            count_y3[wh_xi] = count_y3[wh_xi]+1.
          
            w_a_out[xi,:] = w_a.copy()
            SQ_rt_out[xi,:,:] = SQ_rt.copy()
            
            for ti in range(nruns):
                
                K_gain = np.dot(X_bg_devs_ens[xi+ti*nstate,:], RHS2)
                x_a_mean = x_bg_mean[xi+ti*nstate] + np.dot(K_gain,n0[wh_xi])
                X_a_devs_ens = np.dot(X_bg_devs_ens[xi+ti*nstate,:], SQ_rt)
                X_a_ens_out[xi+ti*nstate,:] = X_a_devs_ens + x_a_mean
            
        #P_hat_a_out.append(P_hat_a)
        
        else:
            for ti in range(nruns):
                X_a_ens_out[xi+ti*nstate,:] = x_bg_ens[xi+ti*nstate,:]
                
    return w_a_out, SQ_rt_out, y_a_mean3, count_y3, X_a_ens_out

###@jit(nopython=True)
def numba_calc_R(deltaspace, rho, cum_nmeas2, ndays,nobs, y_uncert):
        
        Qinv_temp = np.zeros((nobs,nobs))
        for ti in range(ndays):
            
            
            Q_block = np.exp((-1.)*deltaspace[cum_nmeas2[ti]:cum_nmeas2[ti+1],
                             cum_nmeas2[ti]:cum_nmeas2[ti+1]] / rho)
            Q_block_inv = np.linalg.inv(Q_block)
            Qinv_temp[cum_nmeas2[ti]:cum_nmeas2[ti+1],
                      cum_nmeas2[ti]:cum_nmeas2[ti+1]] = Q_block_inv.copy()
        
        dum1 = np.dot(np.diag(1./y_uncert),Qinv_temp)
        R_inv = np.dot(dum1,np.diag(1./y_uncert))
        
        return R_inv
    

def custom_sqrtm(A):
    
    B,S = np.linalg.eigh(A)
    diag_B_sq = np.diag(np.sqrt(B))
    sqrt_A=np.dot(np.dot(S,diag_B_sq),np.linalg.inv(S))
    
    return sqrt_A

#%%
###
parser = argparse.ArgumentParser(description='This is a demo script by Mark.')              
parser.add_argument("start", help="Start date string yyyymmdd") 
parser.add_argument("end", help="End date string yyyymmdd") 

parser.add_argument("inflation", help="Covariance inflation as a multiple")
args = parser.parse_args()

start_date = args.start
end_date = args.end
cov_inflate = float(args.inflation)


#start_date = "20171201"   # Start date of observation assimilation window 
#end_date = "20171215"     # End date of obervation assimilation window
#cov_inflate=2.

satellite = "TROPOMI"
domain="SSA"
run_str = "SSA_run1"      # Run name
N_ens=140                 # Number of ensembke members  
spc_per_run=70            # Number of ensembele members per GC run
version = "test_sron_priorx2"# + str(cov_inflate)[:1]     # Meaningful moniker to append to output files
land_only=True

uncert_inflation=2.       # Multiplicative factor to apply to observational uncertainty
uncert_offset = 4.        # Additive factor to add to observational uncertainty.

localization_dist=500.    # Localization distance for observations. 
                          # Only obs within this distance of emission gridcell will be assimilated.
baseline_threshold = -5.  # Threshold at which obs are ignored. -2 means XCH4_obs - XCH4_BC cannot be less than -2 ppb
rho = 50.                  # Correlation length of observation errors in km

#cov_inflate = 1.1         # Covariance inflation parameter
SRON=False                # Set whether using SRON processed version of TROPOMI data 
SRON_v2 = True

GOSAT_v2=True

BC_type = "CH4BC"
bc_bias=0.     # For GOSAT subtract 1 ppb

scale_prior =True
prior_scaling=2.
#albedo_bounds=[0.05,0.4] # Lower and upper bound of cut-off for acceptable SWIR albedos
#aod_max = 0.3
#asol_size_min = 2.5
##xco_max = 0.001
##xco_alb_max=0.15
#surf_std_max=80.  #60.

obs_filter = False

albedo_bounds=[0.05,0.3] # Lower and upper bound of cut-off for acceptable SWIR albedos
aod_max = 0.1
asol_size_min = 3.4
surf_std_max=55.

#albedo_bounds=[0.02,0.3] # Lower and upper bound of cut-off for acceptable SWIR albedos
#aod_max = 0.3
#asol_size_min = 3.2
#surf_std_max=80.
  #60.

corr=True              # Perform inversion using correlated observation errors (True/False)
sequential=True      # Perform inversions sequentially, using outputs of previous windows as starting points for new windows.

inv_out_dir = "/home/mlunt/datastore/enkf_output/" + domain + "/" + version + "/"    # Directory where outputs will be stored.


run_mn = int(start_date[4:6])
run_year = int(start_date[:4])

# Cut this down to 3 windows per inversion - the fourth is v. small to make a difference and is a bit dodgy anyway.
if start_date[-2:] == "01":
    run_days = ["01", "16", "01"]
    run_months = [str(run_mn-1).zfill(2), str(run_mn-1).zfill(2), str(run_mn).zfill(2)]
elif start_date[-2:] == "16":
    run_days = ["16", "01", "16"]
    run_months = [str(run_mn-1).zfill(2), str(run_mn).zfill(2), str(run_mn).zfill(2)]

if run_mn==1:
    if start_date[-2:] == "01":
        run_years = [str(run_year-1), str(run_year-1),str(run_year)]
        run_months = ["12", "12","01"]
    elif start_date[-2:] == "16":
        run_years = [str(run_year-1),str(run_year),str(run_year)]
        run_months = [ "12", "01", "01"]
elif run_mn==2:
    if start_date[-2:] == "01":
        run_years = [str(run_year), str(run_year),str(run_year)]
        run_months = ["01", "01","02"]
    elif start_date[-2:] == "16":
        run_years = [str(run_year),str(run_year),str(run_year)]
        
#    else:
#        run_years = [str(run_year-1), str(run_year),str(run_year),str(run_year)]
#        run_months = ["12", "01", "01", "01"]
else:
    run_years = [str(run_year),str(run_year),str(run_year)]
    
    
#if start_date[-2:] == "01":
#    run_days = ["16","01", "16", "01"]
#    run_months = [str(run_mn-2).zfill(2),str(run_mn-1).zfill(2), str(run_mn-1).zfill(2), str(run_mn).zfill(2)]
#elif start_date[-2:] == "16":
#    run_days = ["01", "16", "01", "16"]
#    run_months = [str(run_mn-1).zfill(2),str(run_mn-1).zfill(2), str(run_mn).zfill(2), str(run_mn).zfill(2)]
#
#    
#if run_mn==1:
#    if start_date[-2:] == "01":
#        run_years = [str(run_year-1), str(run_year-1), str(run_year-1),str(run_year)]
#        run_months = ["11", "12", "12","01"]
#    elif start_date[-2:] == "16":
#        run_years = [str(run_year-1),str(run_year-1),str(run_year),str(run_year)]
#        run_months = ["12", "12", "01", "01"]
#elif run_mn==2:
#    if start_date[-2:] == "01":
#        run_years = [str(run_year-1), str(run_year), str(run_year),str(run_year)]
#        run_months = ["12", "01", "01","02"]
#    elif start_date[-2:] == "16":
#        run_years = [str(run_year), str(run_year),str(run_year),str(run_year)]
#        
#else:
#    run_years = [str(run_year), str(run_year),str(run_year),str(run_year)]
    
#run_date0 = pd.to_datetime(start_date) + relat    

if start_date == "20171201":
    run_dates= ["20171201"]
elif start_date == "20171216":
    run_dates= ["20171201", "20171216"]
#elif start_date == "20180101":
#    run_dates= ["20171201", "20171216", "20180101"] #, "20180601"]
    
    
#if start_date == "20190901":
#    run_dates= ["20190901"]
#elif start_date == "20190916":
#    run_dates= ["20190901", "20190916"]
#elif start_date == "20180101":
#    run_dates= ["20171201", "20171216", "20180101"] #, "20180601"]    
    
else:
    run_dates=[]
    for ti in range(3):
        run_dates.append(run_years[ti] + run_months[ti] + run_days[ti])

#%%

# Read in baseline concentrations from separate file in SSA_BC directory 
        
if GOSAT_v2 == True:
    BC_run_str = "SSA_GOSAT_BC_025x03125_CH4" 
    BC_output_dir  = "/geos/u73/mlunt/GC_output/EnKF/" + BC_run_str + "/nc_files/sat_columns/"
else:
    BC_run_str = "SSA_BC_025x03125_CH4" 
    BC_output_dir  = "/geos/d21/mlunt/GC_output/EnKF/" + BC_run_str + "/new_nc_files/sat_columns/"
#fname_BC= BC_output_dir + "XCH4_Model_" + satellite + ".nc" 
if satellite == "TROPOMI":
    if SRON==True:
        BC_file_str = "XCH4_Model_scaled_SRON_" + satellite + "_"
    elif SRON_v2 == True:
        #BC_file_str = "XCH4_Model_scaled_SRON_v2_" + satellite + "_"
        BC_file_str = "XCH4_Model_SRON_v2_" + satellite + "_"
    else:
        BC_file_str = "XCH4_Model_" + satellite + "_"
else:
    if GOSAT_v2 == True:
        BC_file_str = "XCH4_Model_" + satellite + "_"
    else:
        BC_file_str = "XCH4_Model_scaled_" + satellite + "_"

start_date_ed  = start_date[:6] + "01"
# Too much memory for kingie - use holmes
files = ensemble_mod.filenames(BC_output_dir, BC_file_str, start = start_date_ed, end=end_date, freq="MS")

ds_BC_temp = ensemble_mod.read_netcdfs(files)
ds_BC = ds_BC_temp.sel(time=slice(start_date,end_date))

y_BC_3d = ds_BC[BC_type]    # DEfine the BC component of the observations 
y_BC_wnan = np.ravel(y_BC_3d.values)-bc_bias

if satellite == "TROPOMI":
    lat_sat = ds_BC.lat.values
    lon_sat = ds_BC.lon.values
    #if SRON == True:
    mask = regionmask.defined_regions.natural_earth.countries_50.mask(lon_sat, lat_sat, xarray=True)
    albedo_3d = ds_BC.SWIR_albedo
    albedo_wnan = np.ravel(albedo_3d.values)
    
    aod_3d = ds_BC.SWIR_aerosol_thickness
    asol_size_3d = ds_BC.aerosol_size
    if SRON == True or SRON_v2==True:
        surf_std_3d = ds_BC.surf_alt_std
        surf_std_wnan = np.ravel(surf_std_3d.values)
        
    xco_3d = ds_BC.XCO_plus
    #xco_dum_3d = xco_3d.where(np.logical_or(xco_3d < xco_max, albedo_3d < xco_alb_max))
    xco_dum_3d = xco_3d.where(np.isfinite(mask))
    
    aod_wnan = np.ravel(aod_3d.values)
    asize_wnan = np.ravel(asol_size_3d.values)
    xco_wnan = np.ravel(xco_dum_3d.values)
        
    
if satellite == "GOSAT":
    retr_flag_3d = ds_BC.retr_flag
    r_flag_wnan = np.ravel(retr_flag_3d.values)


#%%
# Now read in ensemble files and extract ensemble variables.
# Convert to 1D and filter using the same process as above.   
    
n_ens_runs = int(np.ceil(N_ens/spc_per_run))
ensemb_names=[]
for xi in range(N_ens):    
    ensemb_names.append("CH4_E" + str(xi+1))

nruns = len(run_dates)
    
y_bg_ens_list = []  #np.zeros((nobs,N_ens))
y_REF_3d_all = {}
y_bg_ens_list_all={}

y_ens_3d_all={}


for run_date in run_dates:
    y_bg_ens_list = []  #np.zeros((nobs,N_ens))
    
    for ens_num in range(n_ens_runs):
            
        ens_range = [ens_num*spc_per_run+1, (ens_num+1)*spc_per_run]
        
        ensemb_names_50 = ensemb_names[ens_num*spc_per_run:(ens_num+1)*spc_per_run]
        N_ens_50  = len(ensemb_names_50)
    
        run_name=run_str + "_" + run_date + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) 
        #run_name = run_str + run_date
        
        varnames_short = ensemb_names_50
            
        column_out_dir = "/geos/d21/mlunt/GC_output/EnKF/" + run_name + "/sat_columns/"  
        if SRON == True:
            fname_ens= column_out_dir + "XCH4_Model_SRON_" + satellite + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) + ".nc" 
        elif SRON_v2 == True:
            fname_ens= column_out_dir + "XCH4_Model_SRON_v2_" + satellite + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) + ".nc"

        else:
            fname_ens= column_out_dir + "XCH4_Model_" + satellite + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) + ".nc" 
    
        ds_data_ens2 = ensemble_mod.open_ds(fname_ens)
        
                
        ###################################################
        ds_data_ens = ds_data_ens2.sel(time=slice(start_date,end_date))
        
        
        if ens_num == 0:
                
            if run_date == run_dates[-1]:
                y_obs_3d = ds_data_ens["XCH4_obs"] #- ds_data_ens["IminusA_XCH4_ap"]
                if satellite == "TROPOMI":
                    y_uncert_3d = ds_data_ens.XCH4_precision
                elif satellite == "GOSAT":
                    #y_uncert_3d = ds_data_ens.XCH4_obs*0.004
                    y_uncert_3d = ds_data_ens.XCH4_uncertainty
                y_obs_wnan = np.ravel(y_obs_3d.values)
                y_uncert_wnan = np.ravel(y_uncert_3d.values)
                
                lat = ds_data_ens.lat.values
                lon = ds_data_ens.lon.values
                doy = ds_data_ens["time.dayofyear"].values
                
                nlat = len(ds_data_ens.lat)
                nlon= len(ds_data_ens.lon)
                ntime = len(ds_data_ens.time)
                
                lat_3d = np.transpose(np.tile(lat, (ntime,nlon,1)), (0,2,1))
                lon_3d = np.tile(lon, (ntime,nlat,1))
                doy_3d= np.transpose(np.tile(doy, (nlat,nlon,1)), (2,0,1))
                
                y_lat_wnan = np.ravel(lat_3d)
                y_lon_wnan = np.ravel(lon_3d)
                y_doy_wnan = np.ravel(doy_3d)
            
           
            y_REF_3d_all[run_date] = ds_data_ens["CH4REF"]
            

        for xi, name in enumerate(ensemb_names_50):
            
            y_ensi_3d = ds_data_ens[name] - y_REF_3d_all[run_date]
            if scale_prior == True:
                y_ensi_wnan = np.ravel(y_ensi_3d.values)*prior_scaling   
            else:
                y_ensi_wnan = np.ravel(y_ensi_3d.values) 
            y_bg_ens_list.append(y_ensi_wnan)
            
            
            if xi ==0:
                y_ens_3d_all[run_date] = y_ensi_3d/N_ens
            else:
                y_ens_3d_all[run_date] = y_ens_3d_all[run_date] + y_ensi_3d/N_ens

    y_bg_ens_stack = np.stack(y_bg_ens_list)
    y_bg_ens_list_all[run_date] = np.transpose(y_bg_ens_stack)

    
#####################################################
# Filter out data where model column is severely corrupted
if len(run_dates) > 2:
    y_ref_wnan = np.ravel(y_REF_3d_all[run_dates[1]].values)
elif len(run_dates) == 2:
    y_ref_wnan = np.ravel(y_REF_3d_all[run_dates[0]].values)
else:
    y_ref_wnan = np.ravel(y_REF_3d_all[run_dates[-1]].values)
    
#wh_rub = np.where(y_ref_wnan > 3000.)
wh_rub = np.where(np.abs(y_ref_wnan-np.nanmedian(y_ref_wnan)) > 20.)
if len(wh_rub[0])>0:
    y_obs_wnan[wh_rub] = np.nan
######################################################

y_obs_temp = y_obs_wnan[np.isfinite(y_obs_wnan)]
y_uncert_temp = y_uncert_wnan[np.isfinite(y_obs_wnan)]
y_lat_temp = y_lat_wnan[np.isfinite(y_obs_wnan)]
y_lon_temp = y_lon_wnan[np.isfinite(y_obs_wnan)]
y_doy_temp = y_doy_wnan[np.isfinite(y_obs_wnan)]
y_BC_temp = y_BC_wnan[np.isfinite(y_obs_wnan)]
if satellite == "TROPOMI":
    albedo_temp = albedo_wnan[np.isfinite(y_obs_wnan)]
    aod_temp = aod_wnan[np.isfinite(y_obs_wnan)]
    asize_temp = asize_wnan[np.isfinite(y_obs_wnan)]
    xco_plus_temp = xco_wnan[np.isfinite(y_obs_wnan)]
    if SRON == True or SRON_v2 ==True:
        surf_std_temp = surf_std_wnan[np.isfinite(y_obs_wnan)]

if satellite == "GOSAT":
    r_flag_temp = r_flag_wnan[np.isfinite(y_obs_wnan)]


##########
#xch4_reg_mn = xch4_reg_mn.where(np.logical_or(lats_reg<11,lons_reg<45))
#xch4_reg_mn = xch4_reg_mn.where(np.logical_or(lats_reg<12,lons_reg<42))
# Filter out Middle East and Horn of AFrica - too close to boundary and dodgy retrievals
if satellite == "TROPOMI":
    wh_me = np.where(np.logical_and(y_lat_temp >= 12.75, y_lon_temp>=42.))
    #wh_horn = np.where(np.logical_and(y_lat_temp >= 11., y_lon_temp>=45.))
    #wh_me = np.where(np.logical_and(lats_reg<12.75,lons_reg<42))
    
#    wh_me = np.where(np.logical_and(y_lat_temp >= 12., y_lon_temp>=43.))
#    wh_horn = np.where(np.logical_and(y_lat_temp >= 8., y_lon_temp>=45.))
    
    wh_sh_alb = np.where(np.logical_and(y_lat_temp <= -6., albedo_temp>=albedo_bounds[1]-0.08))
    wh_nh_alb = np.where(np.logical_and(y_lat_temp >= 15., albedo_temp>=albedo_bounds[1]-0.04))
    
    xco_plus_temp[wh_me] = np.nan
    #xco_plus_temp[wh_horn] = np.nan
    
    xco_plus_temp[wh_sh_alb] = np.nan
    xco_plus_temp[wh_nh_alb] = np.nan
    
    if SRON==True or SRON_v2==True:
        wh_jag = np.where(surf_std_temp >= surf_std_max)
        xco_plus_temp[wh_jag] = np.nan

#%%

if satellite == "TROPOMI":    
    
#    wh_gt_bg = np.where(np.logical_and(np.isfinite(xco_plus_temp),
#            np.logical_and(asize_temp > asol_size_min, np.logical_and(aod_temp < aod_max,
#            np.logical_and(y_obs_temp-y_BC_temp > baseline_threshold,
#                    np.logical_and(albedo_temp >albedo_bounds[0],
#                                   albedo_temp < albedo_bounds[1]))))))
    
    wh_gt_bg = np.where(np.logical_and(np.isfinite(xco_plus_temp),
            np.logical_and(asize_temp > asol_size_min, np.logical_and(aod_temp < aod_max,
            np.logical_and(y_obs_temp-y_BC_temp > baseline_threshold,
                    np.logical_and(albedo_temp >albedo_bounds[0],
                                   albedo_temp < albedo_bounds[1]))))))
elif satellite == "GOSAT":
    #y_BC_temp5 = y_BC_temp*1.0065/0.997
    #y_BC_temp  = y_BC_temp*1.0065#/0.997
    if land_only == True:
        wh_gt_bg = np.where(np.logical_and(y_obs_temp-y_BC_temp > baseline_threshold,
                                       r_flag_temp == 0))
    else:
        wh_gt_bg = np.where(y_obs_temp-y_BC_temp > baseline_threshold)

#wh_gt_bg = np.where(np.logical_and(y_obs_temp-y_BG_temp > baseline_threshold,
#                                   y_lon_temp < 42))

#%%


y_obs = y_obs_temp[wh_gt_bg]
y_BC = y_BC_temp[wh_gt_bg]
y_lat = y_lat_temp[wh_gt_bg]
y_lon = y_lon_temp[wh_gt_bg]
y_doy = y_doy_temp[wh_gt_bg]
y_uncert = np.sqrt((y_uncert_temp[wh_gt_bg]*uncert_inflation)**2 + uncert_offset**2)
if satellite == "TROPOMI":
    albedo = albedo_temp[wh_gt_bg]
    aod = aod_temp[wh_gt_bg]
    asize = asize_temp[wh_gt_bg]
############################################################
##########################################################

nobs=len(y_obs)

if obs_filter == True:
    
    wh_obs_ind = np.unique(np.random.randint(0, high=nobs-1, size=nobs//8*7))
    
    y_obs = y_obs[wh_obs_ind]
    y_BC = y_BC_temp[wh_obs_ind]
    y_lat = y_lat[wh_obs_ind]
    y_lon = y_lon[wh_obs_ind]
    y_doy = y_doy[wh_obs_ind]
    y_uncert = y_uncert[wh_obs_ind]
    
    if satellite == "TROPOMI":
        albedo = albedo[wh_obs_ind]
        aod = aod[wh_obs_ind]
        asize = asize[wh_obs_ind]
    
    nobs = len(y_obs)
y_emis=y_obs*0.
y_bg_ens_all_temp={}


y_bg_ens_temp = np.zeros((nobs,N_ens))
y_bg_ens_rdate_temp={}
y_bg_mean_rdate_temp={}
Y_bg_devs_ens_rdate_temp={}

for run_date in run_dates:
    y_bg_ens_wnan = y_bg_ens_list_all[run_date]
    y_bg_ens_temp2 = y_bg_ens_wnan[np.isfinite(y_obs_wnan),:]
    y_bg_ens_xi = y_bg_ens_temp2[wh_gt_bg[0],:]
    
    if obs_filter == True:
        y_bg_ens_xi = y_bg_ens_xi[wh_obs_ind,:]
        
    y_emis = y_emis + y_bg_ens_xi.mean(axis=1)
    
    y_bg_ens_rdate_temp[run_date] = y_bg_ens_xi.copy()
    y_bg_mean_rdate_temp[run_date] = y_bg_ens_xi.mean(axis=1)
    Y_bg_devs_ens_rdate_temp[run_date] = y_bg_ens_rdate_temp[run_date] - y_bg_mean_rdate_temp[run_date][:,None] 
    
    y_bg_ens_all_temp[run_date] = y_bg_ens_xi.copy()
    y_bg_ens_temp = y_bg_ens_temp + y_bg_ens_xi

y_obs_emis = y_obs - y_BC


#%%
# Need to read in the initial state ensemble values as well: x_bg_ens        
# Going to need to read in more than one ensemble file at a time
# nstate_total  = nstate*n_run_dates (probably nstate*4)

state_root  = "/geos/u73/mlunt/ensemb_files/"
state_dir  = state_root + run_str + "/"

x_bg_ens_all={}

if sequential == True:
    # If using outputs of previous assim window as inputs for next assim window
    
    if nruns > 1:
        post_fname = inv_out_dir + "window_" + run_dates[-2] + "_" + version + ".nc"
        post_ds = ensemble_mod.open_ds(post_fname)
        
        for run_date in run_dates[:-1]:
            x_bg_ens_all[run_date] = post_ds["X_a_ens_"+ run_date][:,:N_ens] # ML Added indexing 23/7/20
#            w_a_all  = post_ds.w_a
#            SQ_rt

    fname_x_state = state_dir + "ensemble_values_" + run_dates[-1] + ".nc"
    ds_state = ensemble_mod.open_ds(fname_x_state)
    if scale_prior == True:
        x_bg_ens_all[run_dates[-1]] = np.squeeze(ds_state.x_ens.values)[:,:N_ens]*prior_scaling
    else:
        x_bg_ens_all[run_dates[-1]] = np.squeeze(ds_state.x_ens.values)[:,:N_ens] 

else:
    # If performing all windows independently start from base prior each time
    for xi, run_date in enumerate(run_dates):

        fname_x_state = state_dir + "ensemble_values_" + run_date + ".nc"
        ds_state = ensemble_mod.open_ds(fname_x_state)
        x_bg_ens_all[run_date] = np.squeeze(ds_state.x_ens.values)[:,:N_ens]

# Take this part out of the loop    
nstate = len(ds_state.state)

x_lat = ds_state.x_lat.values
x_lon = ds_state.x_lon.values

lonmin = ds_state.lonmin
lonmax = ds_state.lonmax
latmin = ds_state.latmin
latmax = ds_state.latmax
dlon_state = ds_state.dlon
dlat_state = ds_state.dlat

lon_state = np.arange(lonmin, lonmax+dlon_state, dlon_state)
lat_state = np.arange(latmin, latmax+dlat_state, dlat_state)    
nlon_state = len(lon_state)
nlat_state = len(lat_state)
## Get all land-based grid cells
land_index, nland = ensemble_mod.create_land_mask(lat_state,lon_state)

#####################################

#%%
# Create local indices
wh_local=[]
count=[]
for xi in range(nstate):

    distance_xi = haversine.multiple_dist_arrays([x_lat[xi],x_lon[xi]], y_lat,y_lon)
    wh_local_xi = np.where(distance_xi < localization_dist)
    
    wh_local.append(wh_local_xi[0])
    count.append(len(wh_local_xi[0]))

count_np = np.asarray(count)
nobs_xi_max = np.max(count_np)

wh_local_np = np.zeros((nobs_xi_max, nstate))

for xi in range(nstate):
    count_xi = count_np[xi]
    wh_local_np[:count_xi,xi] = wh_local[xi]
    
nobs_xi  = count_np.copy()
#%%

y_bg_mean_rdate={}
Y_bg_devs_ens_rdate={}
y_bg_ens_rdate={}
count_y_ap={}
y_bg_ens = np.zeros((nobs,N_ens))
if sequential == True and nruns > 1:
    for run_date in run_dates:
        y_bg_mean_rdate[run_date] = np.zeros((nobs))
        count_y_ap[run_date] = np.zeros((nobs))
        Y_bg_devs_ens_rdate[run_date] = np.zeros((nobs,N_ens))
         
    for xi in range(nstate):
        wh_xi = wh_local[xi] 
        if len(wh_xi > 0):
            SQ_rt_xi = post_ds.SQ_rt[xi,:N_ens,:N_ens].values
            w_a_xi = post_ds.w_a[xi,:N_ens].values
            
            for ti in range(nruns-1):
            
                y_bg_mean_rdate[run_dates[ti]][wh_xi] = (y_bg_mean_rdate[run_dates[ti]][wh_xi] + 
                         y_bg_mean_rdate_temp[run_dates[ti]][wh_xi] + np.dot(Y_bg_devs_ens_rdate_temp[run_dates[ti]][wh_xi,:],w_a_xi))
                count_y_ap[run_dates[ti]][wh_xi] = count_y_ap[run_dates[ti]][wh_xi]+1.
                
                Y_bg_devs_ens_rdate[run_dates[ti]][wh_xi,:] = (Y_bg_devs_ens_rdate[run_dates[ti]][wh_xi,:] +
                np.dot(Y_bg_devs_ens_rdate_temp[run_dates[ti]][wh_xi,:], SQ_rt_xi))
            
    y_bg_ens_rdate[run_dates[-1]] = y_bg_ens_all_temp[run_dates[-1]]
    y_bg_mean_rdate[run_dates[-1]] = y_bg_mean_rdate_temp[run_dates[-1]]
    
    for run_date in run_dates:
        if run_date != run_dates[-1]:   
            
            y_bg_mean_rdate[run_date] = y_bg_mean_rdate[run_date]/count_y_ap[run_date]
            Y_bg_devs_ens_rdate[run_date] = Y_bg_devs_ens_rdate[run_date]/count_y_ap[run_date][:,None]

            y_bg_ens_rdate[run_date] = y_bg_mean_rdate[run_date][:,None] + Y_bg_devs_ens_rdate[run_date]
        y_bg_ens = y_bg_ens + y_bg_ens_rdate[run_date]
        
                
else:
    for run_date in run_dates:
        y_bg_ens_rdate[run_date] = y_bg_ens_all_temp[run_date].copy()
        y_bg_mean_rdate[run_date] = y_bg_mean_rdate_temp[run_date]
        y_bg_ens = y_bg_ens + y_bg_ens_rdate[run_date]
    
y_bg_mean0=y_obs_emis*0.
for run_date in run_dates:
    y_bg_mean0  = y_bg_mean0 + y_bg_mean_rdate_temp[run_date] 



#y_obs = y_true.copy()
#y_uncert = y_obs*0. + 5.
#R_inv = np.diag(1./y_uncert**2)

nmeasure = len(y_obs)

#%%
#########################################################
# Only proceed if there are observations in this asismilation window
if len(ds_BC.time) > 0:  
    
    # Define a measurement erro correlation matrix
    # Correlate measurements by distance

    
    deltatime = np.zeros((nmeasure,nmeasure))
    deltaspace = np.zeros((nmeasure,nmeasure))
    count_day=[]
    for ti in range(nmeasure):
        tdelta = np.absolute(y_doy - y_doy[ti])
        
        wh_day = np.where(tdelta == 0)[0]
        
        
        deltatime[ti,:] = tdelta
    #    deltaspace[ti,:] = fn_multipledistances([sat_lat[ti],sat_lon[ti]], 
    #              sat_lat[:nmeasure],sat_lon[:nmeasure])
        deltaspace[ti,wh_day] = haversine.multiple_dist_arrays([y_lat[ti],y_lon[ti]], 
                  y_lat[wh_day],y_lon[wh_day])
    
    
    
    #%%
    
    days_unique, nobs_day = np.unique(y_doy, return_counts=True)
    nobs_day_max = np.max(nobs_day)
    ndays=len(days_unique)
    cum_nmeas=np.zeros((ndays),dtype=np.int16)
    for ti in range(ndays):
        cum_nmeas[ti] = np.sum(nobs_day[:ti])
    
    tau = 0.01  # 0.01 days
    
    ###########################
    # Also treat different lat bands independently 3 separate bands.
#    lat_band_edges = [-36,-12,4,22]
#    nobs_day_lat=[]
#    for ti in range(ndays):
#        for lati in range(3):
#            
#            wh_day_lat = np.where(np.logical_and(y_doy == days_unique[ti],
#                                                    np.logical_and(y_lat >=lat_band_edges[lati],
#                                                                  y_lat < lat_band_edges[lati+1])))
#            if len(wh_day_lat[0])>0:
#                nobs_day_lat.append(len(wh_day_lat[0]))
#                
#    ndays_lat = len(nobs_day_lat)
#    cum_nmeas_dayslat=np.zeros((ndays_lat),dtype=np.int16)
#    for ti in range(ndays_lat):
#        cum_nmeas_dayslat[ti] = np.sum(nobs_day_lat[:ti])
    ###############################

    #Q_temp = np.exp((-1.)*deltaspace/rho) * np.exp((-1.)*deltatime/tau) 
    
    #Q_inv = np.linalg.inv(Q_temp)  # Very expensive - just approximate as symmetric toeplitz
    
    # Try doing as block diagonal matrix, so no correlations between days. Might be quicker.
    
#    if corr == True:
#        
#        cum_nmeas2 = np.append(cum_nmeas, nobs)
#        
#        midt1 = run_time.time()
#        #R_inv = numba_calc_R(deltaspace, rho, cum_nmeas2, ndays,nobs, y_uncert)
#        Rinv_dot_Y = solve_Rinv_dot_Y_bg(Y_bg_devs_ens, 
#                                         deltaspace, rho, cum_nmeas2, ndays,nobs, y_uncert)
#        midt2 = run_time.time()
#        
#        print("Time taken for R_inv_dot_Y is ", midt2-midt1, " seconds")
#
#    
#    else:
#        R_inv = np.diag(1./y_uncert**2)
#        rho=0.

    #%%
    
    ###########################
    # 1. Define X_bg_ens 
    
    x_bg_ens = np.zeros((nstate*nruns, N_ens))
    
    for xi,run_date in enumerate(run_dates):
        x_bg_ens[xi*nstate:(xi+1)*nstate,:] = x_bg_ens_all[run_date]
        
    
    #statesize = len(x_ens_dict[ensemb_names[0]])
    
    # Define x as scaling factors applied to underlying distribution
    
    x_bg_mean = np.mean(x_bg_ens,axis=1)
    X_bg_devs_ens = x_bg_ens - x_bg_mean[:,None]
    
    ############################
    # 2. Calculate Y_bg terms (by running geos-chem)
    
    
    y_bg_mean = np.mean(y_bg_ens,axis=1)
    Y_bg_devs_ens = y_bg_ens - y_bg_mean[:,None] 
    
    Y_bg_devs_ens_rdate = {}
    for run_date in run_dates:
        Y_bg_devs_ens_rdate[run_date] = y_bg_ens_rdate[run_date] - y_bg_mean_rdate[run_date][:,None] 
    
    n0 = y_obs_emis - y_bg_mean


    if corr == True:
        
        cum_nmeas3 = np.append(cum_nmeas, nobs)
        #cum_nmeas2 = np.append(cum_nmeas_dayslat, nobs)
        
        #midt1 = run_time.time()
        
        #R_inv = numba_calc_R(deltaspace, rho, cum_nmeas3, ndays,nobs, y_uncert)
        
        #R_inv = numba_calc_R(deltaspace, rho, cum_nmeas2, ndays_lat, nobs, y_uncert)
        
        #Rinv_dot_Y = solve_Rinv_dot_Y_bg(Y_bg_devs_ens, 
        #                                 deltaspace, rho, cum_nmeas2, ndays,nobs, y_uncert)
        #midt2 = run_time.time()
        
        #print("Time taken for R_inv is ", midt2-midt1, " seconds")
        
        midt1 = run_time.time()
        R_inv = numba_calc_R(deltaspace, rho, cum_nmeas3, ndays,nobs, y_uncert)
        midt2 = run_time.time()
        print("Time taken for R_inv is ", midt2-midt1, " seconds")
        

    
    else:
        rho=0.
        midt1 = run_time.time()
        R_inv = np.diag(1./y_uncert**2)
        
        R_inv_split=[]
        for xi in range(nstate):
            wh_xi  = wh_local[xi]
            R_inv_split.append(np.diag(1/y_uncert[wh_xi]**2))
        midt2 = run_time.time()
        print("Time taken for R_inv_split is ", midt2-midt1, " seconds")
        #Rinv_dot_Y = np.dot(R_inv,Y_bg_devs_ens)
    
    # Calculate R_inv_in for fortran module
    # Too expensive computationlly - don't do this!!!!s
    #R_inv_in = np.zeros((nobs_xi_max,nobs_xi_max,nstate))
    #for xi in range(nstate):
    #    wh_xi = wh_local[xi]
    #    if len(wh_xi > 0):
    #        
    #        nobs_temp = nobs_xi[xi]
    #        R_inv_in[:nobs_temp,:nobs_temp,xi] = R_inv[wh_xi, wh_xi[:,None]]
    
    startt1 = run_time.time()
    ############################
    # 3. Update bg ensemble t analysis ensemble
    #wh_local=None
    
    X_a_ens_out = np.zeros((nstate*nruns, N_ens))
    y_a_mean2={}
    count_y={}
    Y_a_devs_ens_rdate2 ={}
    
    y_a_mean3 = np.zeros((nobs))
    count_y3 = np.zeros((nobs))
    for run_date in run_dates:
        y_a_mean2[run_date] = np.zeros((nobs))
        count_y[run_date] = np.zeros((nobs))
        Y_a_devs_ens_rdate2[run_date] = np.zeros((nobs,N_ens))
    #Y_a_devs_ens2 = np.zeros((nobs,N_ens))
    #count_y = np.zeros((nobs))
    #for xi in range(nstate):
    
    
    #%%
    # Call numba inversion function
    #startt = run_time.time()
#    if corr == True:
#        # THIS IS VERY SLOW!!!
#        R_inv_split = []
#        midt1 = run_time.time()
#        for xi in range(nstate):
#            wh_xi  = wh_local[xi]
#            #R_inv_split.append(R_inv[wh_xi,wh_xi[:,None]])
#            
#            R_inv_split.append(R_inv[np.ix_(wh_xi,wh_xi)])
#        midt2 = run_time.time()
#        print("Time taken for R_inv_split is ", midt2-midt1, " seconds")
#        
#        R_inv_split2 = []
#        midt1 = run_time.time()
#        for xi in range(nstate):
#            wh_xi  = wh_local[xi]
#            R_inv_split2.append(R_inv[wh_xi,wh_xi[:,None]])
#            
#           
#        midt2 = run_time.time()
#        print("Time taken for R_inv_split old is ", midt2-midt1, " seconds")
#    
    #midt1 = run_time.time()
    #dum_w, dum_SQ, dum_y3, dum_c3, dum_X = numba_letkf(2, N_ens, 2, wh_local,
    #                                                        R_inv_split, Y_bg_devs_ens, y_bg_mean,
    #                                                        n0, cov_inflate)
    midt2 = run_time.time()

    
    w_a_out, SQ_rt_out, y_a_mean3, count_y3, X_a_ens_out = numba_letkf(nstate, N_ens, nruns, wh_local,
                                                            R_inv, Y_bg_devs_ens, y_bg_mean,
                                                            n0, cov_inflate,
                                                            x_bg_mean, X_bg_devs_ens, x_bg_ens)
    
    endt1 = run_time.time()
    #print("Finished inversion in ",  endt1-startt1, "seconds")

    #print("Dummy run time is  ",  midt2-midt1, "seconds")
    #print("Proper run time is  ",  endt1-midt2, "seconds")
    
    

    for xi in range(nstate):
        wh_xi = wh_local[xi]   # Indices of observations to be used to constrain state vector index.
        if len(wh_xi > 0):
            for ti in range(nruns):
                
#                K_gain = np.dot(X_bg_devs_ens[xi+ti*nstate,:], RHS2)
#                x_a_mean = x_bg_mean[xi+ti*nstate] + np.dot(K_gain,n0[wh_xi])
#                X_a_devs_ens = np.dot(X_bg_devs_ens[xi+ti*nstate,:], SQ_rt)
#                X_a_ens_out[xi+ti*nstate,:] = X_a_devs_ens + x_a_mean
                
                y_a_mean2[run_dates[ti]][wh_xi] = (y_a_mean2[run_dates[ti]][wh_xi] + 
                         y_bg_mean_rdate[run_dates[ti]][wh_xi] + np.dot(Y_bg_devs_ens_rdate[run_dates[ti]][wh_xi,:],w_a_out[xi,:]))
                count_y[run_dates[ti]][wh_xi] = count_y[run_dates[ti]][wh_xi]+1.
                
                Y_a_devs_ens_rdate2[run_dates[ti]][wh_xi,:] = (Y_a_devs_ens_rdate2[run_dates[ti]][wh_xi,:] +
                np.dot(Y_bg_devs_ens_rdate[run_dates[ti]][wh_xi,:], SQ_rt_out[xi,:,:]))
                
        #else:
        #    for ti in range(nruns):
        #        X_a_ens_out[xi+ti*nstate,:] = x_bg_ens[xi+ti*nstate,:]
    
    #%%
    x_post = np.mean(X_a_ens_out,axis=1)
    y_a_mean_comb = np.zeros((nobs))
    y_a_mean={}
    Y_a_devs_ens_rdate={}
    Y_a_ens_out={}
    for run_date in run_dates:
        y_a_mean[run_date] = y_a_mean2[run_date]/count_y[run_date]
        y_a_mean_comb = y_a_mean_comb + y_a_mean[run_date]
        
        Y_a_devs_ens_rdate[run_date] = Y_a_devs_ens_rdate2[run_date]/count_y[run_date][:,None]
        
        Y_a_ens_out[run_date]  = Y_a_devs_ens_rdate[run_date] + y_a_mean[run_date][:,None]
        
    y_post = y_a_mean3/count_y3
    
    
    endt2 = run_time.time()
    print("Finished " + start_date + " inversion in ",  endt2-startt1, "seconds")
    #y_a_mean = y_a_mean2/count_y
    
    
    
    #%%
    
#    x_post_map_v = np.zeros((nlat_state*nlon_state))
#    x_post_map_v[land_index] = x_post[nstate*(nruns-1):]*1.
#    x_post_map = np.reshape(x_post_map_v, (nlat_state,nlon_state))
#    
#    proj = ccrs.PlateCarree()
#    fig2,ax2=plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
#    h2 = ax2.pcolormesh(lon_state, lat_state, x_post_map, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-2., vmax=4.)
#    cax2,kw2 = cbar.make_axes(ax2,location='bottom',pad=0.05,shrink=0.7)
#    out=fig2.colorbar(h2,cax=cax2,extend='both',**kw2, label="Posterior x")
#    ax2.add_feature(cfeature.BORDERS)
#    
#%%
else:
    
    print("No data in assimilation window. Skipping updates, posterior=prior for " + start_date )
    #X_a_ens_out = x_bg_ens.copy()
    X_a_ens_out = np.zeros((nstate*nruns, N_ens))
    for xi,run_date in enumerate(run_dates):
        X_a_ens_out[xi*nstate:(xi+1)*nstate,:] = x_bg_ens_all[run_date]
    Y_a_ens_out = y_bg_ens_rdate.copy()
#    w_a_out  = w_a  # These are defined within the if statement - Can I just copy form before or do I need to recalculate?
#    SQ_rt_out = SQ_rt   # Can I copy or do I need to calculate?
    
    SQ_rt_out = post_ds.SQ_rt.values # If no data in window, use previous 
    w_a_out = post_ds.w_a.values
    
    y_bg_mean = y_bg_ens.mean(axis=1) 
    y_post = y_bg_ens.mean(axis=1) 
    

ds_out= xarray.Dataset()

for ti,run_date in enumerate(run_dates):
    ds_out["X_a_ens_"+run_date] = (('nstate', 'Nens'), X_a_ens_out[ti*nstate:(ti+1)*nstate,:])
    ds_out["Y_a_ens_"+run_date] = (('nobs', 'Nens'), Y_a_ens_out[run_date])
    
ds_out["w_a"] = (('nstate', 'Nens'), w_a_out)
ds_out["SQ_rt"] = (('nstate', 'Nens', 'Nens'), SQ_rt_out)

ds_out["localization_dist"] = localization_dist
ds_out["bl_threshold"] = baseline_threshold
ds_out["uncert_offset"] = uncert_offset
ds_out["uncert_inflation"] = uncert_inflation
ds_out["obs_correlation_length"] = rho
ds_out["y_mod_BC"] = (('nobs'),y_BC)
ds_out["y_obs"] = (('nobs'),y_obs)
ds_out["y_lon"] = (('nobs'),y_lon)
ds_out["y_lat"] = (('nobs'),y_lat)
ds_out["y_doy"] = (('nobs'),y_doy)
ds_out["y_prior"] = (('nobs'),y_bg_mean)
ds_out["y_ap0"] = (('nobs'),y_bg_mean0)
ds_out["sigma_y"]  = (('nobs'),y_uncert)
ds_out["y_post"] = (('nobs'),y_post)
ds_out["covariance_inflation"] = cov_inflate

fname_out = inv_out_dir + "window_" + start_date + "_" + version + ".nc"

for key in list(ds_out.keys()):
    ds_out[key].encoding['zlib'] = True 
    
compress_keys = list(ds_out.keys())
#compress_keys.remove("w_a")
#compress_keys.remove("SQ_rt")

for key in compress_keys:
    ds_out[key].encoding['least_significant_digit'] = 6 
    
ds_out.to_netcdf(path=fname_out)
