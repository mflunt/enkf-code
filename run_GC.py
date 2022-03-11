#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:38:27 2019

Module to speciy inputs and call other scripts 
to write input files for GEOS-Chem ensemble runs.

Inpus:
    From inputs.ini file. 
    
    This script will create ensemble run directories and files for 1 assimilation window. 

    Parse in start date from command line to loop through multiple assim windows

@author: mlunt
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import GC_setup_mod as ensemble_mod
import subprocess
import glob
import ast
import os
import configparser
###########################################################################
# Read these quantities from inputs.ini file
def open_config(config_file):
    '''
    The open_config function is used to open configuration files in the ini format.
    
    Args:
        config_file (str): 
            Filename for input configuration file (str).
    
    Returns:
        configparser.ConfigParser object
    '''
    config = configparser.ConfigParser(inline_comment_prefixes=(';','#'))
    config.optionxform=str # Keeps case when inputting option names
    
    with open(config_file) as fp:
        config.read_file(fp)
    
    return config

# 1. REad in inputs file to get quantities of interest
config_file = 'inputs.ini'
config = open_config(config_file)

species = config["SETUP"]["species"]
emis_start = config["SETUP"]["run_start"]
#run_end = config["SETUP"]["run_end"]
run_name_prefix = config["SETUP"]["run_name"]
#region_short = config["SETUP"]["domain"]
nested_tf = ast.literal_eval(config["SETUP"]["nested"])

N_ens = int(config["ENKF"]["N_ens"])
#localization = float(config["ENKF"]["d_local"])
lag_period = int(config["ENKF"]["lag_period"])
lag_unit = config["ENKF"]["lag_unit"]
assim_window = int(config["ENKF"]["assim_window"])
window_unit = config["ENKF"]["assim_unit"]

bc_split = config["STATE"]["bc_split"]
sigma_x_ap = float(config["STATE"]["ap_uncertainty"])
state_res  = config["STATE"]["state_resolution"]
x_ap_cov_length = float(config["STATE"]["ap_cov_length"])
land_only  = ast.literal_eval(config["STATE"]["land_only"])

model_res = config["MODEL"]["resolution"]
met_version = config["MODEL"]["met_version"]
ref_conc = float(config["MODEL"]["ref_conc"])
spc_per_run = int((config["MODEL"]["tracers_per_run"]))

surface_bool = ast.literal_eval(config["MODEL_OUTPUTS"]["surface_sites"])
satellite_bool = ast.literal_eval(config["MODEL_OUTPUTS"]["satellites"])

surf_out_freq = config["MODEL_OUTPUTS"]["out_freq"]

out_bounds={}
if surface_bool:
    surface_sites = ast.literal_eval(config["MODEL_OUTPUTS"]["site_list"])
    site_json_file = config["MODEL_OUTPUTS"]["site_json"]
    
    out_bounds["lat"] = ast.literal_eval( config["MODEL_OUTPUTS"]["lat_bounds"])
    out_bounds["lon"] = ast.literal_eval(config["MODEL_OUTPUTS"]["lon_bounds"])
    out_bounds["lev"] = ast.literal_eval(config["MODEL_OUTPUTS"]["lev_bounds"])
else:
    surface_sites=[]
    out_bounds["lat"] = []
    out_bounds["lon"] = []
    out_bounds["lev"] = []


data_root = config["DIRECTORIES"]["data_root"]
gc_code_dir = config["DIRECTORIES"]["gc_code_dir"]+ "Code." + config["MODEL"]["gc_version"]
#obs_dir = config["DIRECTORIES"]["obs_dir"]
run_root = config["DIRECTORIES"]["run_root"]
output_root = config["DIRECTORIES"]["output_root"]
bc_input_root = config["DIRECTORIES"]["bc_input_root"]
ens_file_root = config["DIRECTORIES"]["ensemble_file_dir"]
restart_template_root = config["DIRECTORIES"]["restart_template_root"]

restart_fstr = config["DIRECTORIES"]["restart_file_str"]

bc_input_dir = bc_input_root + species + "/"
restart_template_dir = restart_template_root + species + "/"
restart_template_file = restart_template_dir + restart_fstr

emission_keys={}
emis_key_list = list(config[species + "_EMISSIONS"])
for key in emis_key_list:
    emission_keys[key] = ast.literal_eval(config[species + "_EMISSIONS"][key])
    
    
if nested_tf:
    region_short = config["NESTED_MODEL"]["region_short"]
    lonmin = float(config["NESTED_MODEL"]["lonmin"])
    lonmax = float(config["NESTED_MODEL"]["lonmax"])
    latmin = float(config["NESTED_MODEL"]["latmin"])
    latmax = float(config["NESTED_MODEL"]["latmax"])
    check_BC = True
else:
    region_short = None
    check_BC = False
    
inv_only = ast.literal_eval(config["SUBMODULES"]["inv_only"])
write_bc_files = ast.literal_eval(config["SUBMODULES"]["write_bc_files"])
write_restart_file = ast.literal_eval(config["SUBMODULES"]["write_restart_file"])
overwrite_ensemble = ast.literal_eval(config["SUBMODULES"]["overwrite_ensemble"])
    

#read_gc_output = ast.literal_eval(config["SUBMODULES"]["read_gc_output"])
#localize = ast.literal_eval(config["SUBMODULES"]["localize"])

#inv_only=False
#write_bc_files= True
#write_restart_file = True
#overwrite_ensemble = True

#read_gc_output=False
#localize=False

####################################################################################
#%%

template_dir = "./templates/"
gc_make_dir = "./gc_files/"

sigma_x_bc_ap = sigma_x_ap*1.
#spc_IC = ["CH4IC", "CH4BC", "CH4REF"] 
#spc_IC = ["CH4REF"] 

spc_IC = [species+ "IC"]   #,species + "REF"]

#region_short = None
# Work out met dir from combination of data_root, met_name, res and region_short

if region_short:
    met_dir = data_root + "GEOS_" + model_res + "_" + region_short.upper() + "/" + met_version.upper() + "/"
else:
    met_dir = data_root + "GEOS_" + model_res + "/" +  met_version.upper() + "/"
    
# Check met_dir exists    
if os.path.isdir(met_dir) == False:
    raise ValueError("Met directory :" + met_dir + "does not exist!")
    
hemco_dir = data_root + "HEMCO/"
if os.path.isdir(hemco_dir) == False:
    raise ValueError("HEMCO directory :" + hemco_dir + "does not exist!")

#%%

pd_run_start = pd.to_datetime(emis_start)

# Do each forward model run independently. 
# Parse start date from command line
n_assim_windows = 1

n_ens_runs = int(np.ceil(N_ens/spc_per_run))



if bc_split == "North-South":    
    nBC = 2
elif bc_split == "NESW":
    nBC=4
    for xi in range(nBC):
        spc_IC.append(species + "_BC" + str(xi+1))
elif bc_split == "None":
    nBC=0
else:
    raise ValueError("Incorrect bc_split definition. Must be North-South or NESW")

# Add reference species concentration
    
spc_IC.append(species + "REF")

# For CO add chemical production terms
if species == "CO":
    
    if emission_keys["GCCH4"] == True:
        
        spc_IC.append("COCH4")
        
    if emission_keys["GCNMVOC"] == True:
        
        spc_IC.append("CONMVOC")


if model_res == "0.5x0.625":
    dlat_native = 0.5
    dlon_native = 0.625
elif model_res == "0.25x0.3125":
    dlat_native = 0.25
    dlon_native = 0.3125
elif model_res == "4x5":
    dlat_native = 4.
    dlon_native = 5.
    
if state_res == "0.5x0.625":
    dlat_state = 0.5
    dlon_state = 0.625
elif state_res == "0.25x0.3125":
    dlat_state = 0.25
    dlon_state = 0.3125
elif state_res == "1x1.25":
    dlat_state = 1.
    dlon_state = 1.25
elif state_res == "2x2.5":
    dlat_state = 2.
    dlon_state = 2.5
elif state_res == "4x5":
    dlat_state = 4.
    dlon_state = 5.
    
if nested_tf == False:
    lonmin  = -180.
    lonmax = 180.-dlon_native
    latmin = -90.
    latmax=90.

lon_state = np.arange(lonmin, lonmax+dlon_state, dlon_state)
lat_state = np.arange(latmin, latmax+dlat_state, dlat_state)    
nlon_state = len(lon_state)
nlat_state = len(lat_state)

lon_native = np.arange(lonmin, lonmax+dlon_native, dlon_native)
lat_native = np.arange(latmin, latmax+dlat_native, dlat_native)    
nlon_native = len(lon_native)
nlat_native = len(lat_native)

# Get all land-based grid cells - not appropriate for CO2 and CO 
if land_only == True:
    land_index, nland = ensemble_mod.create_land_mask(lat_state,lon_state)

ensemb_names=[]
for xi in range(N_ens):    
    ensemb_names.append(species + "_E" + str(xi+1))

#%%

# Don't need to loop if only one date per run.
# But do need to loop through different ensemble groups (1-50, 51-100 etc.)

# Define times for first run
    
pd_emis_start = pd.to_datetime(emis_start)

if window_unit =="d":
    pd_assim_window_end = pd_emis_start + pd.Timedelta(assim_window, unit='d') 
    emis_end = (pd_assim_window_end - pd.Timedelta(1, unit='d')).strftime('%Y%m%d')
    n_emis_windows = int(np.round(((pd_assim_window_end - pd_emis_start).days)/assim_window))
elif window_unit == "MS":
    pd_assim_window_end = pd_emis_start + relativedelta(months=assim_window)
    emis_end = (pd_assim_window_end - pd.Timedelta(1, unit='d')).strftime('%Y%m%d')
    n_emis_windows = int(np.round(((pd_assim_window_end - pd_emis_start).days/30.)/assim_window))


# The below lines work to split windows into dekads (approximate 10 day periods evenly splitting months)
# Seems to work but not sure if it's totally foolproof. - it's not update to be like above...
window_start_dates_pd=[]
pd_scale_factor_dates=[]
for ti in range(n_emis_windows+1):  
    if window_unit =="d":
        new_date = pd_emis_start + pd.Timedelta(assim_window*ti, unit='d')
    elif window_unit =="MS":
        new_date = pd_emis_start + relativedelta(months=assim_window*ti)
        
    if assim_window == 10:    
        
        if new_date.day == 31:
            new_date2 = new_date + pd.Timedelta(1, unit='d')
            window_start_dates_pd.append(new_date2)
            pd_scale_factor_dates.append(new_date2)
        else:
            d = np.mod(new_date.day,10)
            if d == 1:
                window_start_dates_pd.append(new_date)
                pd_scale_factor_dates.append(new_date)
            else:
                #if new_date.month == pd_emis_start.month:
                #    new_date2 = new_date + pd.offsets.MonthBegin(0)
                #    window_start_dates.append(new_date2)
                #else:
                days_past = np.mod(new_date.day, 10)
                new_date2 = new_date - pd.Timedelta(days_past-1, unit='d')
                window_start_dates_pd.append(new_date2)
                pd_scale_factor_dates.append(new_date2)
                
    elif assim_window == 15:    
        
        if new_date.day == 31:
            new_date2 = new_date + pd.Timedelta(1, unit='d')
            window_start_dates_pd.append(new_date2)
            pd_scale_factor_dates.append(new_date2)
            
        elif new_date.day == 16:
            window_start_dates_pd.append(new_date)
            pd_scale_factor_dates.append(new_date)
        else:
            d = np.mod(new_date.day,10)
            if d == 1:
                window_start_dates_pd.append(new_date)
                pd_scale_factor_dates.append(new_date)
            else:
                #if new_date.month == pd_emis_start.month:
                #    new_date2 = new_date + pd.offsets.MonthBegin(0)
                #    window_start_dates.append(new_date2)
                #else:
                days_past = np.mod(new_date.day, 10)
                new_date2 = new_date - pd.Timedelta(days_past-1, unit='d')
                window_start_dates_pd.append(new_date2)
                pd_scale_factor_dates.append(new_date2)

    else:
        window_start_dates_pd.append(new_date)
        pd_scale_factor_dates.append(new_date)

pd_lag_start = window_start_dates_pd[-1]
lag_start_date = pd_lag_start.strftime('%Y%m%d')
if lag_unit =="d":
    pd_lag_end = pd_lag_start + pd.Timedelta(lag_period, unit="d")
elif lag_unit =="MS":    
    pd_lag_end = pd_lag_start + relativedelta(months=lag_period)
lag_end_date = pd_lag_end.strftime('%Y%m%d')

#%%
# Set up GC run directories and output directories for this run...
# Use subprocess to make directories and copy/move files


pd_scale_factor_dates.append(pd_lag_end)

#x0 = np.arange(nland)/nland+0.5
    
if land_only == True:
    ngrid  = nland*1
else:
    ngrid = nlon_state*nlat_state

nBC_all = nBC * n_emis_windows
ngrid_all = ngrid * n_emis_windows
nstate = nBC_all + ngrid_all # nstate = number of grid cells x number of time steps to be optimized.
# i,e. if assim_window = 10 days
# lag period = 50 days
# n_assim_windows = 5. so nstate will be around 36500 big.   
lons,lats = np.meshgrid(lon_state,lat_state)
lats_v = np.ravel(lats)
lons_v = np.ravel(lons)

#wh_nh = np.where(lats_land >= 0)[0]
#wh_sh = np.where(lats_land < 0)[0]

if land_only  == True:
    lats_land = lats_v[land_index]
    lons_land = lons_v[land_index]
    lats_grid = lats_land.copy()
    lons_grid = lons_land.copy()
else:
    lats_grid = lats_v.copy()
    lons_grid = lons_v.copy()

if inv_only == False: 
# Need to also include boundary condition variables as well in statesize
    ens_file_dir = ens_file_root + run_name_prefix 
    subprocess.call(["mkdir", '-p', ens_file_dir])
    fname_ens = ens_file_dir +"/" + species + "_ensemble_values_" + emis_start  + ".nc"
    
    if overwrite_ensemble == False and len(glob.glob(fname_ens)) > 0:
        # load x_ens file
        
        ds_ensemble = ensemble_mod.open_ds(fname_ens)
    
        x_ens_3d= ds_ensemble.x_ens.values
        
        x_region_2d = ds_ensemble.x_region
        #x_time_2d
       
            
            # Load pre-created x_ens file
    else:
        # Need to create the initial ensemble file
        # Crete based on covariance structure of P
        x_ap = np.ones((ngrid))
        
        
        dist_to_site = ensemble_mod.calc_dist_to_site(surface_sites, site_json_file, lats_v,lons_v)
        
        P,deltaspace = ensemble_mod.create_prior_covariance(x_ap, sigma_x_ap, lats_grid, lons_grid, 
                                                 correlated=True, length_scale = x_ap_cov_length,
                                                 corr_type="exp", site_dist = [])
        
        X_devs_ens = ensemble_mod.create_ensemble_from_P(P, N_ens, ngrid)
        
        x_ens = x_ap[:,None] + X_devs_ens
        
        # This function could be changed to something more intelligent.
        # At the moment, just randomly assigning scale factor values of ensemble members.
        
        x_ens_3d,x_region_2d,x_time_2d = ensemble_mod.create_ensemble_file(x_ens, N_ens, ngrid,nBC, n_emis_windows,
                                                                             window_start_dates_pd[:-1],
                                                                             lats_grid, lons_grid,
                                                                             sigma_x_ap, sigma_x_bc_ap,
                                                                             lonmin, lonmax, latmin,latmax,
                                                                             dlon_state, dlat_state, land_index=None,
                                                                             fname_ens = fname_ens,
                                                                             pseudo=False)

# Old cdoe        
#        x_ens_3d,x_region_2d,x_time_2d = ensemble_mod.create_initial_ensemble(N_ens, nland+nBC,nBC, n_emis_windows,
#                                                                             window_start_dates_pd[:-1],
#                                                                             lats_land, lons_land,
#                                                                             sigma_x_ap, sigma_x_bc_ap,
#                                                                             fname_ens = fname_ens,
#                                                                             pseudo=True)
#       
    x_region = np.ravel(x_region_2d)
    #x_time = np.ravel(x_time_2d)
    
#%%
    
    # Output the region (BC or land) and time window of eahc state vector
    
    # Reshape 3d (ntime, nBC+nland,N_ens) to 2D (nstate, N_ens)
    # Need to keep track of x_time and x_region here. 
    
    #x_ens = np.zeros((nstate,N_ens))
    
    scale_map_ensemble={}
   
    for xi,label in enumerate(ensemb_names):
        scale_map_ensemble[label] = ensemble_mod.map_state_vector(x_ens_3d[:,:,xi], 
                          lat_state,lon_state,n_emis_windows,land_index=None, land_only = land_only)
    
        #x_ens[:,xi] = np.ravel(x_ens_3d[:,:,xi])

    #if aggregate_afar == True:
        
    #dist_2d = np.reshape(dist_to_site, (nlat_state,nlon_state))

#    wh = np.where(dist_2d > 800)
#    
#    keys = list(scale_map_ensemble.keys())
#    for key in keys:
#        
#        idx_lat = wh[0][::4]
#        idx_lon = wh[1][::4]
#        
#        scale_dum = scale_map_ensemble[key]  #[0,:,:]
#        
#        scale_dum_temp = scale_dum.copy()
#        
#        for ti in range(len(idx_lat)):
#            
#            #if idx_lat[ti] < np.max(idx_lat):
#            scale_dum[:,idx_lat[ti]:idx_lat[ti]+4, idx_lon[ti]:idx_lon[ti]+4] = scale_dum_temp[:,idx_lat[ti],idx_lon[ti]]
#        
#        scale_map_ensemble[key] = scale_dum

        #scale_map_ensemble[label] = ensemble_mod.aggregate_afar(scale_map_ensemble, dist_to_site,
        #                                                          nlat_state,nlon_state)

#%%

for ens_num in range(n_ens_runs):

# Set up GC run directories and output directories for this run...
# Use subprocess to make directories and copy/move files

    ens_range = [ens_num*spc_per_run+1, (ens_num+1)*spc_per_run]
    
    ensemb_names_50 = ensemb_names[ens_num*spc_per_run:(ens_num+1)*spc_per_run]
    N_ens_50  = len(ensemb_names_50)
    
    
    run_name=run_name_prefix + "_" + species + "_" + emis_start + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) 
    run_dir = run_root + run_name + "/"
    output_dir = output_root + run_name + "/"
    
    #bc_output_dir = output_root + run_name + "/BC/"
    #restart_out_dir = output_dir + "restarts/"
    
    bc_output_dir = run_dir + "BC/"
    restart_out_dir = run_dir + "restarts/"
    
    subprocess.call(["mkdir", '-p', run_dir])
    subprocess.call(["mkdir", '-p', output_dir])
    subprocess.call(["mkdir", '-p', run_dir + 'scale_factors'])
    
    subprocess.call(["mkdir", '-p', run_dir + 'input_files'])
    
    subprocess.call(["mkdir", '-p', run_dir + 'OutputDir'])
    
    subprocess.call(["ln", '-s', gc_code_dir, run_dir + "CodeDir"])
    
    make_files = glob.glob(gc_make_dir + "/*")
    
    for mfile in make_files:
        subprocess.call(["cp", mfile, run_dir]) # Can't use * wildcard without shell interpretation. 
    
    #subprocess.call(["mkdir", '-p', bc_run_dir])
    subprocess.call(["mkdir", '-p', bc_output_dir])
    subprocess.call(["mkdir", '-p', restart_out_dir])
    
    fname_scale_factors = run_dir + "/scale_factors/ensemb_scalings.nc"
    fname_restart_out = restart_out_dir + "GEOSChem.Restart." + emis_start + "_0000z.nc4"
    
    # Copy template directory files for pre-compiled GC executable
    
    #%%
    # Create Scale Factor files for GC
    
#    x0 = np.arange(nland)/nland+0.5
#    
#    nBC_all = nBC * n_emis_windows
#    nland_all = nland * n_emis_windows
#    nstate = nBC_all + nland_all # nstate = number of grid cells x number of time steps to be optimized.
#    # i,e. if assim_window = 10 days
#    # lag period = 50 days
#    # n_assim_windows = 5. so nstate will be around 36500 big.   
#    lons,lats = np.meshgrid(lon,lat)
#    lats_v = np.ravel(lats)
#    lons_v = np.ravel(lons)
#    lats_land = lats_v[land_index]
#    lons_land = lons_v[land_index]
#    wh_nh = np.where(lats_land >= 0)[0]
#    wh_sh = np.where(lats_land < 0)[0]
#    
#    if inv_only == False: 
#    # Need to also include boundary condition variables as well in statesize
#        fname_ens = run_dir + "ensemble_values.nc"
#        
#        if overwrite_ensemble == False and len(glob.glob(fname_ens)) > 0:
#            # load x_ens file
#            
#            ds_ensemble = ensemble_mod.open_ds(fname_ens)
#        
#            x_ens_3d= ds_ensemble.x_ens.values
#            
#            x_region_2d = ds_ensemble.x_region
#            #x_time_2d
#           
#                
#                # Load pre-created x_ens file
#        else:
#            # Need to create the initial ensemble file
#            #if emis_start == assim_start_dates[0]:
#            x_ens_3d,x_region_2d,x_time_2d = ensemble_mod.create_initial_ensemble(N_ens, nland+nBC,nBC, n_emis_windows,
#                                                                                 window_start_dates_pd[:-1],
#                                                                                 lats_land, lons_land,
#                                                                                 sigma_x_ap, sigma_x_bc_ap,
#                                                                                 fname_ens = fname_ens,
#                                                                                 pseudo=True)
#            #else:
#            #    x_ens_3d,x_region_2d,x_time_2d = ensemble_mod.forecast_new_ensemble()
#                
#                # Don't need this if short-cutting and keeping model and ensembe calculations totally separate. 
#    
#        x_region = np.ravel(x_region_2d)
#        #x_time = np.ravel(x_time_2d)
        
    #%%
    if inv_only == False: 
        # Output the region (BC or land) and time window of eahc state vector
        
        # Reshape 3d (ntime, nBC+nland,N_ens) to 2D (nstate, N_ens)
        # Need to keep track of x_time and x_region here. 
        
#        x_ens = np.zeros((nstate,N_ens_50))
#        
#        scale_map_ensemble={}
#       
#        for xi,label in enumerate(ensemb_names):
#            scale_map_ensemble[label] = ensemble_mod.map_state_vector(x_ens_3d[:,nBC:,xi], lat,lon,land_index, n_emis_windows)
#        
#            x_ens[:,xi] = np.ravel(x_ens_3d[:,:,xi])
#            
    
        
        
        if overwrite_ensemble == False and len(glob.glob(fname_scale_factors)) > 0:
            print("Scale factor file already exists and overwrite is False")
        else:
            ensemble_mod.write_scale_factor_file(pd_scale_factor_dates,lat_state,lon_state, 
                                           scale_map_ensemble, ensemb_names_50, fname_scale_factors,
                                           assim_window, emis_start, lag_end_date)
        
        fname_mask =  run_dir + "/scale_factors/NAF_mask.nc"
        ensemble_mod.write_mask_file(pd_scale_factor_dates, lat_native,lon_native, fname_mask,
                            assim_window, emis_start, lag_end_date)
    #%%
    # Set up GC run directories and output directories for this run...
        
        # Use subprocess to make directories and copy/move files
        
        # Write the input file for Geos-chem
       
        ensemble_mod.write_input_file_v12(emis_start, lag_start_date, ensemb_names_50, spc_IC,
                                      run_dir, output_dir, template_dir,  
                                      lonmin, lonmax, latmin,latmax,
                                      model_res, species, met_version, 
                                      region_short=region_short)
        # surface_sites, satellite_bool,
                                      
        
        # Write the input file for the lag period
        ensemble_mod.write_input_file_v12(lag_start_date,lag_end_date, ensemb_names_50, spc_IC,
                                      run_dir, output_dir, template_dir,  
                                      lonmin, lonmax, latmin,latmax,
                                      model_res, species, met_version, 
                                      region_short=region_short, lag=True)
        # surface_sites, satellite_bool,
        
        
        
        
       #%%     
        
        # Write the HEMCO_Config file for GEOS-Chem
        ensemble_mod.write_hemco_config_v12(emis_start, lag_end_date, ensemb_names_50, fname_scale_factors,
                                        run_dir, template_dir, species, emission_keys,
                                        met_dir, hemco_dir, restart_out_dir, bc_output_dir, fname_mask, region_short=region_short)
        
        ensemble_mod.write_hemco_lag_config_v12(run_dir, template_dir, species, 
                                        met_dir, hemco_dir, restart_out_dir, bc_output_dir, region_short=region_short)
        
        
        # Write the HISTORY.rc file for GEOS-Chem
        ensemble_mod.write_history_rc_v12(run_dir, template_dir, save_bc = False, 
                                          save_conc=surface_bool, conc_freq = surf_out_freq,
                                          lon_bounds = out_bounds["lon"], lat_bounds = out_bounds["lat"],
                                          lev_bounds = out_bounds["lev"] )
        
        # Write HEMCO diagnostics file
        
        spc_emit_diagn = ensemb_names_50.copy()
        if species == "CO":
            spc_emit_diagn.append("COCH4")
            spc_emit_diagn.append("CONMVOC")
        
        ensemble_mod.write_hemco_diagn_file(run_dir, spc_emit_diagn, species)
                             
       
        #%%
 
    
        # Write new restart file for each run. Base on output of optimized ensemble mean run 
        # But for first run, base on a spun-up global/regional run
        
        # For now define this as something specific, but will be the restart file from previous mean run in future.
        
        # Restart file only needs to contain CH4 and CH4IC for first run. 
        # Assuming I'm tracking emissions from each month/assim window individually. 
        
        
        #restart_input_file = "/geos/u73/mlunt/GC_code/restarts/GEOSChem.Restart.20160701_0000z_05x0625_CA.nc4"
        if write_restart_file == True:
            
            ds_restart = ensemble_mod.write_restart_file_v12(restart_template_file, 
                                                               fname_restart_out, species, ensemb_names_50,spc_IC, 
                                                               emis_start, ref_conc = ref_conc, 
                                                               write_IC = True,
                                                               write_ens=True)
    #            ds_restart = ensemble_mod.write_restart_file(restart_input_file, ensemb_names,  
    #                                                         fname_restart_out, emis_start, spc_copy="CH4")
    
        #%%
           
#        if nBC > 0:
#            bc_scalings_3d = x_ens_3d[:,:nBC,:]
#            bc_scalings =  {}
#            for xi, spc in enumerate(ensemb_names):
#                bc_scalings[spc]  = bc_scalings_3d[:,:,xi]
#        else:
#            bc_scalings =  {}
#            for xi, spc in enumerate(ensemb_names):
#                bc_scalings[spc]  = x_ens_3d[:,:1,xi]*0.+1.
            
        # Need to include these terms in initial statesize. 23/7/19
        bc_str = "GEOSChem.BoundaryConditions."
        
        if check_BC:
            # Check whether BC files from global run exist or not. Raise an error if not.
            bc_check = ensemble_mod.filenames(bc_input_dir, bc_str, start=emis_start, end=lag_end_date, freq="D")
            if len(bc_check) == 0:
                raise ValueError("No BC files available. Make sure you run global model first before attempting nested ensemble run. Or turn check_BC to False") 
       
        if write_bc_files == True:
            
            # Only write BCs for CH4 species
            # All ensemble fields have BC=0. Only interested in emissions component.
            ensemble_mod.write_bc_ensemble_v12(emis_start, lag_start_date, lag_end_date, ensemb_names_50, bc_input_dir, bc_str,
                                bc_output_dir, species, spc_IC,
                                latmin,latmax,lonmin,lonmax,
                                bc_split = bc_split, BC_ensemble=False)
            
#            ensemble_mod.write_bc_ensemble_v12(emis_start, lag_end_date, ensemb_names_50, bc_input_dir, bc_str,
#                                bc_output_dir, species = "CH4", ic_spc_names=["CH4IC"], BC_ensemble=False)
   
    else:
        print("Inversion only is True")
#%%
# Now run model

#subprocess.call(["make" "mprun"])  # Might be best to do this from shell script.

#%%
# Write final input files with no emissions

# Would easier thing to do just be to make scaling files time dependent?
# After end of assim window all scalings should be 0.
# Same for BC - that way only need one run.

#ensemble_mod.write_input_file_v12(emis_start, window_end_date, ensemb_names, spc_IC,
#                                  run_dir, output_dir, template_dir,  
#                                  lonmin, lonmax, latmin,latmax,
#                                  model_res, species, met_version, region_short="CA")

	#python ${py_dir}/write_final_05x0625_input.py ${run_name} ${start_run} ${tperiod} ${period_units} 1
	#python ${py_dir}/write_final_hemco_05x0625.py ${run_name} ${start_run} ${tperiod} ${period_units} 1

	#cp ${run_dir}/input.geos ${run_dir}/input_files/input.final
	#cp ${run_dir}/HEMCO_Config.rc ${run_dir}/input_files/HEMCO_Config.final
	#cp ${run_dir}/MyDiagnFile.rc ${run_dir}/input_files/MyDiagnFile.final

	#cd ${run_dir}
	#make mprun
