#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:46:53 2019

Modeule containg functions for setting up Ensemble GC runs

@author: mlunt
"""

import pandas
import xarray
import regionmask
import numpy as np
import xbpch
import glob
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import re

def open_ds(fname):
    """
    Open netcdf file as xarray dataset. 
    Loads to memor then closes the file.
    """
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

def open_bpch_ds(fname, tracer_file, diag_file):
    with xbpch.open_bpchdataset(fname, tracerinfo_file=tracer_file, diaginfo_file=diag_file) as ds:
        ds.load()
    return ds

def filenames(file_dir, file_str, start=None, end=None, freq='D'):
    """
    Output a list of available file names,for given directory and date range.
    """
    files = []
    # Convert into time format
    if (start is not None) and (end is not None):
        #days = pandas.DatetimeIndex(start = start, end = end, freq = "D").to_pydatetime()
        dates = pandas.date_range(start = start, end = end, freq = freq)
        
        if freq =="D":
            yearmonthday = [str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2) for d in dates]
        elif freq =="MS":
            yearmonthday = [str(d.year) + str(d.month).zfill(2) for d in dates]
        elif freq =="YS":
            yearmonthday = [str(d.year) for d in dates]
    
        for ymd in yearmonthday:
            f=glob.glob(file_dir + "/" + file_str + "*" + ymd + "*")
            if len(f) > 0:
                files += f
        files.sort()
        
    else:
        f=glob.glob(file_dir + "/" + file_str + "*.")
        if len(f) > 0:
            files += f     # Not entirely sure if this will work - might be worth checking! - Yes it works
        files.sort()

    if len(files) == 0:
        raise ValueError("Can't find file: " + file_dir + "/" + file_str + ymd[:4] + "*")
                        
    return files

def days_in_month(month):
    """
    Define the (maximum) number of days in a month
    Inputs: Month (Integer)
    """
    if month in ([1,3,5,7,8,10,12]):
        ndays = 31
    elif month in ([4,6,9,11]):
        ndays = 30
    else:
        ndays = 29
    return ndays

def multiple_dist_arrays(origin, lats,lons):
    
    """
    Calculate distance between a lat/lon point (origin) and an array of other coordinates
    """
    radius = 6371 #km
    
    lat0 = origin[0]
    lon0 = origin[1]
    
    dlats = np.radians(lats-lat0)
    dlons = np.radians(lons-lon0)
    
    a = np.sin(dlats/2) * np.sin(dlats/2) + np.cos(np.radians(lat0)) \
            * np.cos(np.radians(lat0)) * np.sin(dlons/2) * np.sin(dlons/2)
            
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
    distances = radius * c        
    
    return distances

def create_land_mask(lat,lon):
    """
    Find the indices of grid cells in the domain which are over land
    Args:
        lat - grid latitudes
        lon - grid longitudes
    """
    lon = lon-0.0001 # Need to shfit the lons slightly as otherwise an issue between Egypt and Libya.
    mask = regionmask.defined_regions.natural_earth.countries_50.mask(lon, lat, xarray=True)
    mask_np_v = np.ravel(mask.values)
    land_index = np.where(np.isfinite(mask_np_v)==True)[0]
    nland = len(land_index)
    return land_index, nland

def map_state_vector(x_state, lat, lon, land_index, ntime):
    """
    Map state vector values onto land grid cells
    
    Call this function in a loop? i.e Call for each ensemble member - maybe. 
    
    """

    nlat=len(lat)
    nlon=len(lon)

    scaling_v = np.zeros((nlat*nlon))
    scaling_map = np.zeros((ntime,nlat,nlon))
    for ti in range(ntime):
        scaling_v[land_index] = x_state[ti,:]
        scaling_map[ti,:,:] = np.reshape(scaling_v, (nlat,nlon))
    
    return scaling_map
    

def create_prior_covariance(x_ap, sigma_x, lat_state, lon_state, correlated=False, corr_type="exp", length_scale = 250):
    
    nstate = len(x_ap)
    
    #P = np.zeros((nstate,nstate))
    
    P_diag = np.diag(x_ap*sigma_x)
    
    if correlated == False:
        
        P = P_diag**2
        
    else:
        
        deltaspace = np.zeros((nstate,nstate))
        for ti in range(nstate):
           
            deltaspace[ti,:] = multiple_dist_arrays([lat_state[ti],lon_state[ti]], 
                      lat_state,lon_state)
        
        if corr_type == "exp":
            SIGMA = np.exp((-1.)*deltaspace/length_scale)   # * np.exp((-1.)*deltatime/tau) 
        elif corr_type == "balgovind":
            SIGMA = (1.+ deltaspace/length_scale) *np.exp((-1.)*deltaspace/length_scale)
            
        else:
            raise ValueError("Need to define a valid covariance structure")
        
        P = np.dot(P_diag,np.dot(SIGMA,P_diag))
        
    return P


def create_ensemble_from_P(P, N_ens, nstate):
    
    """
    Create the ensemble deviations from the mean based on covariance structure of P
    P  = X.X^T
    
    X = (nstate x N_ens)
    """
    # Cholesky decomposition
    L = np.linalg.cholesky(P)
    
    
    # How do I get this in right size: nstate x N_ens?
    mu = np.zeros((nstate,N_ens))
    
    for xi in range(nstate):
        mu[xi,:] = np.random.normal(loc=0., scale=1., size=(N_ens))
        
    mu_mean = np.mean(mu, axis=1)
    mu2 = mu - mu_mean[:,None]   # Ensure mean of deviations of each state element is exactly 0.

    X_bg_devs_ens = np.dot(L,mu2)
    
    return X_bg_devs_ens

    

def create_ensemble_file(x_ens_temp,N_ens, nstate0, nBC, ntime, dates, lats_land,lons_land,
                            sigma_x_land, sigma_x_bc, lonmin, lonmax, latmin,latmax,
                            dlon_state,dlat_state,land_index,
                            fname_ens=None, pseudo=False):
    """
    Create an random ensemble array
    
    nstate0 = nland+nBC
    """
        
    x_ens = np.zeros((ntime,nstate0,N_ens))
    for ti in range(ntime):
        x_ens[ti,:,:] = x_ens_temp.copy()*1.
    
    #x_ens_temp = np.random.normal(loc=1., scale=sigma_x_land, size=(ntime, nstate0, N_ens))
    
    # Make sure ensemble mean of each column is 1
    
#    mean_xi  = np.mean(x_ens_temp,axis=2)
#    x_ens = x_ens_temp -mean_xi[:,:,None] + 1.
    
    ###############################################################
    # Ver ydodgy comment out I think
    # Make sure all values are +ve - Not sure I can do this without affecting the stats
#    wh_neg = np.where(x_ens < 0.)
#    if np.sum(wh_neg)>0:
#        x_ens[wh_neg] = np.random.uniform(low=0.01,high=0.5)
    #################################################################
    
    if pseudo == True:
        # Perturb NH and SH values for pseudo case
        
        wh1 = np.where(lons_land >=24.)[0]
        wh2 = np.where(lons_land < 24.)[0]
        x_ens[:,wh1,:] = x_ens[:,wh1,:] +0.2
        x_ens[:,wh2,:] = x_ens[:,wh2,:] -0.2
    
    #if BC are defined then overwrite first nBC entries with different ensemble
    if nBC > 0:
        x_ens_bc = np.random.normal(loc=1., scale=sigma_x_bc, size=(ntime, nBC, N_ens))
        x_ens[:,:nBC,:] = x_ens_bc.copy()*1.

    x_region=[]
    x_time=[]
    
    for ti in range(ntime):
        x_region0=[]
        x_time0=[]
        for xi in range(nstate0):
            
            if xi < nBC:
                x_region0.append("BC" + str(xi+1))
            else:
                x_region0.append("Q" + str(xi-nBC+1))
            x_time0.append(dates[ti].strftime('%Y%m%d'))

        x_time.append(x_time0)
        x_region.append(x_region0)

    x_region_out = np.vstack(x_region)
    x_time_out = np.vstack(x_time)
    
    # Write the ensemble values to file so a record is kept and they can be reloaded
    if fname_ens != None:
        ds_ensemble = xarray.Dataset()
        ds_ensemble["x_ens"] = (('time', 'state', 'member'), x_ens)
        ds_ensemble["x_region"] = (('time','state'), x_region_out)
        ds_ensemble["x_lon"] = (("state"), lons_land)
        ds_ensemble["x_lat"] = (("state"), lats_land)
        ds_ensemble["land_indices"] = (("state"), land_index)
        ds_ensemble['time'] = dates
        ds_ensemble["nBC"] = nBC
        ds_ensemble["nland"] = nstate0-nBC
        ds_ensemble["sigma_x_land"] = sigma_x_land
        ds_ensemble["sigma_x_bc"] = sigma_x_bc
        ds_ensemble["lonmin"] = lonmin  
        ds_ensemble["lonmax"] = lonmax 
        ds_ensemble["latmin"] = latmin  
        ds_ensemble["latmax"] = latmax 
        ds_ensemble["dlat"] = dlat_state
        ds_ensemble["dlon"] = dlon_state
        
        ds_ensemble.to_netcdf(path=fname_ens, mode="w")
    
    
    return x_ens, x_region_out, x_time_out


def write_scale_factor_file(dates_out, lat,lon,scale_map_ensemble, spc_prefix, fname_out,
                            assim_window, start_date, end_date):
    """
    Write scale factor netcdf file for GC run. 
    
    2 timesteps - 1st is emissions period.
                - 2nd is lag period where scale factors should be 0.
                 -3rd is end of lag period 
    
    """
    ds_out = xarray.Dataset()
    
    if len(dates_out) != 3:
        raise ValueError("Must have 3 scale factor dates. 1 for emis start and other for lag start")
    
    nlat = len(lat)
    nlon=len(lon)
    
    dlat = lat[1]-lat[0]
    dlon = lon[1]-lon[0]
    lat_out = np.arange(lat[0]-dlat*3, lat[-1]+dlat*4, dlat)
    lon_out = np.arange(lon[0]-dlon*3, lon[-1]+dlon*4, dlon)
    
    #pd_days = pandas.date_range(start_date,end_date)
    
    #ndays = len(pd_days)
    ndays = len(dates_out)
    #scale_factor_field = np.zeros((ndays, nlat+6, nlon+6))
    
    for label in spc_prefix:
        scale_factor_field = np.zeros((ndays, nlat+6, nlon+6))
#        scale_factor_field[:assim_window,2:-2,2:-2] = scale_map_ensemble[label]
#        scale_factor_field[assim_window:,:,:] = 0.
        
        scale_factor_field[0,3:-3,3:-3] = scale_map_ensemble[label]*1.
        #scale_factor_field[1:,:,:] = 0.
        
        ds_out[label] = (("time", "lat","lon"), scale_factor_field)
    
    
    
#    for label in spc_prefix:
#    
#        scale_fac_1step = scale_map_ensemble[label]
#        scale_fac_2step = np.append(scale_fac_1step, np.zeros((2,nlat,nlon)),axis=0)
#        
#        ds_out[label] = (("time", "lat","lon"), scale_fac_2step)
        #ds_out[label] = (("time", "lat","lon"), scale_map_ensemble[label])
        
#    ds_out.coords['lat'] = lat
#    ds_out.coords['lon'] = lon
    
    ds_out.coords['lat'] = lat_out
    ds_out.coords['lon'] = lon_out
    
    #ds_out.coords['time'] = [pandas.to_datetime("2010-01-01")]
    ds_out.coords['time'] = dates_out
    #ds_out.coords['time'] = pd_days
    
    ds_out.time.attrs["standard_name"] =  'time'
    ds_out.time.attrs["long_name"] =  'time'
    #ds_out.time.attrs["units"] =  'hours since 2000-01-01'
    
    ds_out["lat"].attrs["standard_name"] =  'latitude'
    ds_out["lat"].attrs["long_name"] =  'latitude'
    ds_out["lat"].attrs["units"] =  'degrees_north'
    ds_out["lat"].attrs["axis"] =  'Y'
    
    ds_out["lon"].attrs["standard_name"] =  'longitude'
    ds_out["lon"].attrs["long_name"] =  'longitude'
    ds_out["lon"].attrs["units"] =  'degrees_east'
    ds_out["lon"].attrs["axis"] =  'X'
    
    for label in spc_prefix:
        ds_out[label].attrs["units"] = "1"
        ds_out[label].attrs["long_name"] = "Scaling factor for " +  label + " variable"
    
    ds_out.to_netcdf(path=fname_out, mode='w')
    
    print("Successfully written file: " + fname_out)
    return
    
def write_mask_file(dates_out, lat,lon, fname_out,
                            assim_window, start_date, end_date):
    """
    Write scale factor netcdf file for GC run. 
    
    2 timesteps - 1st is emissions period.
                - 2nd is lag period where scale factors should be 0.
                 -3rd is end of lag period 
    
    """
    ds_out = xarray.Dataset()
    
    if len(dates_out) != 3:
        raise ValueError("Must have 3 scale factor dates. 1 for emis start and other for lag start")
    
    nlat = len(lat)
    nlon=len(lon)
    
    dlat = lat[1]-lat[0]
    dlon = lon[1]-lon[0]
    lat_out = np.arange(lat[0]-dlat*3, lat[-1]+dlat*4, dlat)
    lon_out = np.arange(lon[0]-dlon*3, lon[-1]+dlon*4, dlon)
    
    pd_days = pandas.date_range(start_date,end_date)
    
    ndays = len(pd_days)
    mask_field = np.zeros((ndays, nlat+6, nlon+6))
    
    if assim_window == 10:
        if start_date[-2:]=="21":
            if start_date[4:6] in ["04", "06", "09", "11"]:
                mask_field[:10,3:-3,3:-3] = 1.
            elif start_date[4:6] == "02":
                mask_field[:8,3:-3,3:-3] = 1.
            else:
                mask_field[:11,3:-3,3:-3] = 1.
        else:        
            mask_field[:assim_window,3:-3,3:-3] = 1.
            
    elif assim_window == 15:
        if start_date[-2:]=="16":
            if start_date[4:6] in ["04", "06", "09", "11"]:
                mask_field[:15,3:-3,3:-3] = 1.
            elif start_date[4:6] == "02":
                mask_field[:13,3:-3,3:-3] = 1.
            else:
                mask_field[:16,3:-3,3:-3] = 1.
        else:        
            mask_field[:assim_window,3:-3,3:-3] = 1.
            
    #mask_field[assim_window:,:,:] = 0.
        
    ds_out["MASK"] = (("time", "lat","lon"), mask_field)
    
    
    ds_out.coords['lat'] = lat_out
    ds_out.coords['lon'] = lon_out

    #ds_out.coords['time'] = dates_out
    ds_out.coords['time'] = pd_days
    
    ds_out.time.attrs["standard_name"] =  'time'
    ds_out.time.attrs["long_name"] =  'time'
    #ds_out.time.attrs["units"] =  'hours since 2000-01-01'
    
    ds_out["lat"].attrs["standard_name"] =  'latitude'
    ds_out["lat"].attrs["long_name"] =  'latitude'
    ds_out["lat"].attrs["units"] =  'degrees_north'
    ds_out["lat"].attrs["axis"] =  'Y'
    
    ds_out["lon"].attrs["standard_name"] =  'longitude'
    ds_out["lon"].attrs["long_name"] =  'longitude'
    ds_out["lon"].attrs["units"] =  'degrees_east'
    ds_out["lon"].attrs["axis"] =  'X'
    
    
    ds_out["MASK"].attrs["units"] = "unitless"
    ds_out["MASK"].attrs["long_name"] = "NAF mask"
    ds_out["MASK"].attrs["scale_factor"] = 1.
    
    ds_out.to_netcdf(path=fname_out, mode='w')
    
    print("Successfully written file: " + fname_out)

    return

def replace_inplace(file_path, pattern, substitute):
    """
    Replace a file line inplace (i.e. don't create a new file)
    """
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, substitute))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)
    
    return


def write_input_file_v12(start_date, end_date, spc_names, spc_BC,
                     run_dir, output_dir, template_dir, 
                     lonmin,lonmax,latmin,latmax,
                     grid_res, species, met_type_in,
                     data_dir = "/geos/d21/GC_DATA/",
                     sat_TF = "T",
                     region_short=None, lag=False):
    """
    Write the input.geos file for each assimilation run. 
    Length of run needs to be for whole of lag period.
    
    Inputs:
        start_date (string): "YYYYMMDD" start of emissions window
        lag_start (string): "YYYYMMDD" Start of lag window/end of emissions window
        end_date (string): "YYYYMMDD" End of lag window
        N_ens (int): NUmber of ensemble members
        bc_dir (string): Directory where BC files are kept.
        spc_BC (list): list of boundary variables to include. - Not neededI don't think
        
    Requires:
        Template file - input_ensemble.template - to be setup correctly. 
    """

    res_str={}
    res_str["0.25x0.3125"] = "025x03125"
    res_str["0.5x0.625"] = "05x0625"
    res_str["2x2.5"] = "2x25"
    res_str["4x5"] = "4x5"
    
    sim_number={}
    sim_number["CH4"] = "9"
    sim_number["CO2"] = "12"
    
    dlon_dict = {}
    dlon_dict["0.25x0.3125"] = 0.3125
    dlon_dict["0.5x0.625"] = 0.625
    dlon_dict["2x2.5"] = 2.5
    dlon_dict["4x5"] = 5.
     
    dlat_dict = {}
    dlat_dict["0.25x0.3125"] = 0.25
    dlat_dict["0.5x0.625"] = 0.5
    dlat_dict["2x2.5"] = 2.
    dlat_dict["4x5"] = 4.
    
    if met_type_in == "GEOS_FP":
        met_type = "geos-fp"
    else:
        met_type = met_type_in

    template_file = template_dir + "input_file_v12.template"
    
    if region_short:       # If a regional simulation is defined
        gc_run_name = met_type.lower() + "_" + res_str[grid_res] + "_" + species + "_" + region_short.lower()
        nested = "T"
        Buffer = "3"
        tstep_trans_conv = "600"     # Transport timestep in seconds # was 300 edited 12/11/19
        
        if lag ==True:
            tstep_emis_chem = "3600"   # was 600
        else:
            tstep_emis_chem = "1800"
    else:
        # GLOBAL simulation
        gc_run_name = met_type.lower() + "_" + res_str[grid_res] + "_" + species 
        nested = "F"
        Buffer = "0"
        tstep_trans_conv = "600"     # Transport timestep in seconds
        tstep_emis_chem = "1200"
    
    
    dlon = dlon_dict[grid_res]
    dlat = dlat_dict[grid_res]
    
    latmin3 = latmin - dlat*3   # Add buffer to grid size
    latmax3 = latmax + dlat*3   # Add buffer to grid size
    lonmin3 = lonmin - dlon*3   # Add buffer to grid size
    lonmax3 = lonmax + dlon*3   # Add buffer to grid size
    
    nlon_grid = (lonmax3-lonmin3)/dlon + 1
    nlat_grid = (latmax3-latmin3)/dlat + 1
    
    Imin_region = str(1 + int(Buffer)*2)
    Jmin_region = str(1 + int(Buffer)*2)
    
    Imax_region = str(int(nlon_grid) - int(Buffer)*2)
    Jmax_region = str(int(nlat_grid) - int(Buffer)*2)
    
    
    #pd_start = pandas.to_datetime(start_date)
    #pd_lag_start = pandas.to_datetime(lag_start)
    pd_end = pandas.to_datetime(end_date)
    
    if type(spc_BC) == list:
        spc_IC = ["CH4"] + spc_BC
    else:
        spc_IC = ["CH4"]
        
    
    month_index = {1:"JAN",
                   2:"FEB",
                   3:"MAR",
                   4:"APR",
                   5:"MAY",
                   6:"JUN",
                   7:"JUL",
                   8:"AUG",
                   9:"SEP",
                   10:"OCT",
                   11:"NOV",
                   12:"DEC"}
    
    
    
    #if pd_end.month == pd_lag_start.month:
        
    days_in_end_month = days_in_month(pd_end.month)
    string_list=[]
    for ti in range(days_in_end_month):
        string_list.append('0')
        
    string_list[pd_end.day-1] = '3' 
    #string_list[pd_lag_start.day-1] = '3'
    string_30s = "".join(string_list)
    
    # define diagnostic outfile name
    diagn_outfile_name = output_dir + "trac_avg." + gc_run_name + ".YYYYMMDDhhmm"
    
    # define satellite outfile name
    sat_outfile_name = output_dir + "ts_satellite.YYYYMMDD.bpch"  # Fine to leave thi as YYMMDD as GC works this out.
    
    #####################################################################################
    # 1. Need to define advected species i .e include species from any previous months
    if type(spc_names) == list:
        spc_advect = spc_IC + spc_names
    else:
        spc_advect = spc_IC 
    nAdvect = len(spc_advect)  # Have to remeber to add 2 for CH4 and CH4REF
    
    
    # 3. Define tracer number list to write 
    
    tracer_list = "512 "    # 512 = pressure fields
    for si in range(nAdvect):
                    tracer_list = tracer_list + str(si+1)+ " "    
    

    #%%
    # Read in template input file
    # Read in the file
    #with open(template_dir + "input.template", 'r') as file :
    #  filedata = file.read()
    ## Replace the target string
    
    str_len=len("Species Entries ------->")      
    
    if lag == True:
        input_file_name = run_dir + "input_files/lag_input.geos"
    else:
        input_file_name = run_dir + "input_files/window_input.geos"
     
    #with open(template_dir + "input_ensemble_CH4.template", "r") as in_file:
    with open(template_file, "r") as in_file:
        filedata = in_file.readlines()
    
    with open(input_file_name, "w") as out_file:
        for line in filedata:
            
            
            if "gc_run_name" in line: 
                line = line.replace('{gc_run_name}', gc_run_name)
            if "StartDate" in line: 
                line = line.replace('{StartDate}', start_date)
            if "EndDate" in line:
                line = line.replace('{EndDate}', end_date)          
            if "DataRoot" in line:
                line = line.replace('{DataRoot}', data_dir)
                
            if "met_field" in line:
                line = line.replace('{met_field}', met_type.lower())    
            if "grid_res" in line:
                line = line.replace('{grid_res}', grid_res)    
                
                
            if "lonmin" in line:
                line = line.replace('{lonmin}', str(lonmin3))
                line = line.replace('{lonmax}', str(lonmax3))
            if "latmin" in line:
                line = line.replace('{latmin}', str(latmin3))
                line = line.replace('{latmax}', str(latmax3))
            if "Nested?" in line:
                line = line.replace('{Nested?}', nested)
            if "Buffer" in line:
                line = line.replace('{Buffer}', Buffer)
            if "Tstep_tran/conv" in line:
                line = line.replace('{Tstep_tran/conv}', tstep_trans_conv)
            if "Tstep_chem/emis" in line:
                line = line.replace('{Tstep_chem/emis}', tstep_emis_chem)
            if "sim_number" in line:
                line = line.replace('{sim_number}', sim_number[species])
            if "DiagnosticOutputFile" in line:
                line = line.replace('{DiagnosticOutputFile}', diagn_outfile_name)     
                
                
            if "SatOutput?" in line:
                line = line.replace("{SatOutput?}", sat_TF )
            if "SatelliteOutputFile" in line:
                line = line.replace("{SatelliteOutputFile}", sat_outfile_name )
            if "SatelliteTracers" in line:
                line = line.replace("{SatelliteTracers}", tracer_list )
            if "Imin_region" in line:
                line = line.replace("{Imin_region}", Imin_region )
                line = line.replace("{Imax_region}", Imax_region )
            if "Jmin_region" in line:
                line = line.replace("{Jmin_region}", Jmin_region )
                line = line.replace("{Jmax_region}", Jmax_region )
                
            
            if "Species Entries ------->: Name" in line:    
                for si, species in enumerate(spc_advect):
                    dummy_str = "Species #"+str(si+1)
                    line = line + dummy_str.ljust(str_len) + ": " + species + "\n"  
                    #line = line + "Species #"+str(si+1)+"              : " + species + "\n"    
            
            if "OUTPUT_DATES" in line:
                if "Schedule output for " + month_index[pd_end.month] in line:
                    line = line.replace("OUTPUT_DATES", string_30s)
#                elif "Schedule output for " + month_index[pd_lag_start.month] in line:
#                    if pd_end.month !=pd_lag_start.month:
#                        line = line.replace("OUTPUT_DATES", string_30s2)
                else:
                    if any(s in line for s in ["JAN", "MAR", "MAY", "JUL", "AUG", "OCT", "DEC"]):      
                        string_0s = "0"*31
                    elif any(s in line for s in ["APR", "JUN", "SEP", "NOV"]):
                        string_0s = "0"*30
                    else:
                        string_0s = "0"*29                    
                    line=line.replace("OUTPUT_DATES", string_0s)
                
            out_file.write(line)

    print("Successfully written file " + run_dir + "input.geos")
    
    return

def write_hemco_config_v12(start_date, end_date, spc_emit, fname_scale_factors,
                       run_dir, template_dir, species, emission_keys, 
                       met_dir, hemco_dir, restart_dir, BC_dir,fname_masks,
                       region_short=None):
    """
    Write the HEMCO_Config.rc file for each assimilation run. 
    
    Inputs:
        start_date (string): "YYYYMMDD" start of emissions lag period
        end_date (string): "YYYYMMDD" End of obs assimilation window
        spc_emit (N_ens): Name of ensemble members
        fname_Scale_factors (string): Full path of scale factor file
        run_dir (string): Directory where run code is located.
        template_dir (string): Directory where template file is located
        
        emission_keys: Dictionary of True/False values for different emission types
        
    Requires:
        Template file - input_ensemble.template - to be setup correctly. 
    """
    
    template_file = template_dir + "HEMCO_Config_" + species + "_v12.template"

    
    spc_fixed = [species]
    
    pd_start = pandas.to_datetime(start_date)
    pd_end = pandas.to_datetime(end_date)
    
    start_year = pd_start.year
    end_year = pd_end.year
    
    # define diagnostic outfile name
    #diagn_infile_path = run_dir + "MyDiagnFile.rc" 
    diagn_infile_path = run_dir + "HEMCO_Diagn.rc"
    
    diagn_freq = "End"  # Will probabl want to use "Monthly"
    diagn_prefix = "HEMCO_diagnostics" #+ start_date
    
    #%%
    #####################################################################################
    # 3. Define tracer list to write 
    spc_list1 = "/".join(spc_fixed) 
    if type(spc_emit) == list:
        spc_list2 = "/".join(spc_emit)
        spc_list = spc_list1 + "/" + spc_list2
    else:
        spc_list = spc_list1
    #%%
    mask_no = {}
    if type(spc_emit) == list:
        for si,spc in enumerate(spc_emit):
            mask_no[spc] = str(1101+ si)
    #%%
    with open(template_file, "r") as in_file:
        filedata = in_file.readlines()
    
    with open(run_dir + "input_files/window_HEMCO_Config.rc", "w") as out_file:
        for line in filedata:
            
            
            if "HEMCO_DIR" in line: 
                line = line.replace("{HEMCO_DIR}", hemco_dir)
            if "MET_DIR" in line: 
                line = line.replace("{MET_DIR}", met_dir)
                
            if "RESTART_DIR" in line: 
                line = line.replace("{RESTART_DIR}", restart_dir)
            if "BC_DIR" in line: 
                line = line.replace("{BC_DIR}", BC_dir)
            
            if "HemcoDiagnosticFileName" in line: 
                line = line.replace("{HemcoDiagnosticFileName}", diagn_infile_path)
            if  "HemcoDiagnosticPrefix" in line:
                line = line.replace("{HemcoDiagnosticPrefix}", diagn_prefix)          
            if "HemcoDiagnosticFreq" in line:
                line = line.replace("{HemcoDiagnosticFreq}", diagn_freq)   
                        
#            if "SpeciesList" in line:
#                line = line.replace("SpeciesList", spc_list )
               
                
            if "{REG_SHORT}" in line:
                if region_short:
                    line = line.replace("{REG_SHORT}", region_short + ".")
                else:
                    line = line.replace("{REG_SHORT}", "")
                    
                
            if type(spc_emit) == list:
                
                
                if emission_keys["GFED"] == True:
                    
                    if "111     GFED" in line:
                        line = line.replace("off", "on")
                        
                    if "SpeciesList" in line:
                        line = line.replace("SpeciesList", spc_list )
                    
                # Only need to do this if I have tagged species
                    if "--> fraction POG1" in line:    
                            for si, species in enumerate(spc_emit):
                                spc_no = species[5:]
                                line = line + "    --> ScaleField_" + species +"     :    GFED_SCALEFIELD_"+spc_no+"\n"
                                
                    if "## Insert scalefields here ##" in line:
                        for si, species in enumerate(spc_emit):
                            spc_no = species[5:]
                            line = line + "\n" + "111 GFED_SCALEFIELD_" + spc_no + \
                            "   1  -  2010/1/1/0 C xy unitless * " + mask_no[species] + "/1012" + " 1 1"
                
                if emission_keys["EDGAR_v432"] == True:
                    
                    if "--> EDGARv432" in line:
                        line = line.replace("false", "true")
                
                    if "0 CH4_OILGAS" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_OILGAS_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 1 1\n"    
                        
                    if "0 CH4_COAL" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_COAL_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 3 1\n"    
                                  
                    if "0 CH4_LIVESTOCK__4A" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_LIVESTOCK__4a_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 4 1\n"    
                    
                    if "0 CH4_LIVESTOCK__4B" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_LIVESTOCK__4b_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + "10/" + mask_no[species] + "/1012" + " 4 1\n"    
                           
                    if "0 CH4_LANDFILLS__6A_6D" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_LANDFILLS__6a6d_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 5 1\n"    
                            
                    if "0 CH4_WASTEWATER__6B" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_WASTEWATER__6b_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 6 1\n"    
                   
                    if "0 CH4_RICE__4C_4D" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_RICE__4c4d_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + "11/" + mask_no[species] + "/1012" + " 7 1\n"   
                            
                 
                   
                    cat_list = ["0 CH4_OTHER__1A1_1B1_1B2", 
                                "0 CH4_OTHER__1A1a", 
                                  "0 CH4_OTHER__1A2", 
                                  "0 CH4_OTHER__1A3a_CDS",                             
                                  "0 CH4_OTHER__1A3a_CRS",                               
                                  "0 CH4_OTHER__1A3a_LTO",                              
                                  "0 CH4_OTHER__1A3a_SPS",                             
                                  "0 CH4_OTHER__1A3b",                                   
                                  "0 CH4_OTHER__1A3c_1A3e",                                
                                  "0 CH4_OTHER__1A3d_1C2",                       
                                  "0 CH4_OTHER__1A4",                                   
                                  "0 CH4_OTHER__6B",                                  
                                  "0 CH4_OTHER__2C",             
                                  "0 CH4_OTHER__4F",                                        
                                  "0 CH4_OTHER__6C"  ]
                    
                    for cat in cat_list:
                        
                        if cat in line:
                            for si, species in enumerate(spc_emit):
                                line = line + cat + "_"+ str(si)+"   -   -    -   -  -   -  " + \
                                species + " " + mask_no[species] + "/1012" + " 8 1\n"   
                    
                if emission_keys["QFED"] == True:      
                    
                    if "--> QFED" in line:
                        line = line.replace("false", "true")
                        
                    if "0 QFED_CH4" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 QFED_CH4_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 9 2\n"   
                            
                if emission_keys["Wetcharts"] == True:        
                    
                    if "--> JPL_WETCHARTS" in line:
                        line = line.replace("false", "true")
                    
                    if "0 JPLW_CH4" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 JPLW_CH4_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 10 1\n"   
                        
                if emission_keys["Fung"] == True:
                    
                    if "--> FUNG" in line:
                        line = line.replace("false", "true")
                    
                    if "0 CH4_SOILABSORB" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_SOILABSORBTION_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + "/1" + " 14 1\n"   
                            
                    if "0 CH4_TERMITES" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_TERMITES_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species]  + "/1012" + " 13 1\n"   
                
                if emission_keys["Seeps"] == True:        
                    
                    if "--> SEEPS" in line:
                        line = line.replace("false", "true")
                        
                    if "0 CH4_SEEPS" in line:
                        for si, species in enumerate(spc_emit):
                            line = line + "0 CH4_SEEPS_"+ str(si)+"   -   -    -   -  -   -  " + \
                            species + " " + mask_no[species] + "/1012" + " 11 1\n"   
                        
      
                        
        
        # Need to write in the mask definitions and ensure that numbers correspond with definitions above
        
                if "## Insert scale factors here ##" in line:
                    for si, species in enumerate(spc_emit):
                        spc_no = species[5:]
                      
                        if start_year == end_year:
                            line = line + mask_no[species] + "  ENSEMB_" + spc_no + "  " \
                            + fname_scale_factors + "   " + species + "  " + \
                            str(start_year) + "/1-12/1-31/0 RF xy 1 1 " + "\n"
                        else:
                            line = line + mask_no[species] + "  ENSEMB_" + spc_no + "  " \
                            + fname_scale_factors + "   " + species + "  " + \
                            str(start_year) + "-" + str(end_year) + "/1-12/1-31/0 RF xy 1 1 " + "\n"  # Changed from RF
                        
                    # RF  = Range forced. HEMCO will raise an error if simulation dates outside of the stated range.

                if "## Insert masks here ##" in line:
                    if start_year == end_year:
                        line = line + "1012" + " NAF_MASK" + "  " \
                                + fname_masks + "   " + "MASK" + "  " + \
                                str(start_year) + "/1-12/1-31/0 EF xy 1 1 " + "-25/-40/60/40" + "\n"
                    else:
                        line = line + "1012" + " NAF_MASK" + "  " \
                                + fname_masks + "   " + "MASK" + "  " + \
                                str(start_year) + "-" + str(end_year) + "/1-12/1-31/0 EF xy 1 1 " + "-25/-40/60/40" + "\n"

            out_file.write(line)
    
    print("Successfully written file " + run_dir + "HEMCO_Config.rc")
    return
    
def write_hemco_lag_config_v12(run_dir, template_dir, species, 
                       met_dir, hemco_dir, restart_dir, BC_dir,
                       region_short=None):
    """
    Write the HEMCO_Config.rc file for each assimilation run lag period. 
    
    Inputs:
        start_date (string): "YYYYMMDD" start of emissions lag period
        end_date (string): "YYYYMMDD" End of obs assimilation window
        spc_emit (N_ens): Name of ensemble members
        fname_Scale_factors (string): Full path of scale factor file
        run_dir (string): Directory where run code is located.
        template_dir (string): Directory where template file is located
        
        emission_keys: Dictionary of True/False values for different emission types
        
    Requires:
        Template file - input_ensemble.template - to be setup correctly. 
    """
    
    template_file = template_dir + "HEMCO_Config_" + species + "_v12.template"

    # define diagnostic outfile name
    #diagn_infile_path = run_dir + "MyDiagnFile.rc" 
    diagn_infile_path = run_dir + "HEMCO_Diagn.rc"
    
    diagn_freq = "End"  # Will probabl want to use "Monthly"
    diagn_prefix = "HEMCO_diagnostics" #+ start_date
    
    
    #%%
    with open(template_file, "r") as in_file:
        filedata = in_file.readlines()
    
    with open(run_dir + "input_files/lag_HEMCO_Config.rc", "w") as out_file:
        for line in filedata:
            
            
            if "HEMCO_DIR" in line: 
                line = line.replace("{HEMCO_DIR}", hemco_dir)
            if "MET_DIR" in line: 
                line = line.replace("{MET_DIR}", met_dir)
                
            if "RESTART_DIR" in line: 
                line = line.replace("{RESTART_DIR}", restart_dir)
            if "BC_DIR" in line: 
                line = line.replace("{BC_DIR}", BC_dir)
            
            if "HemcoDiagnosticFileName" in line: 
                line = line.replace("{HemcoDiagnosticFileName}", diagn_infile_path)
            if  "HemcoDiagnosticPrefix" in line:
                line = line.replace("{HemcoDiagnosticPrefix}", diagn_prefix)          
            if "HemcoDiagnosticFreq" in line:
                line = line.replace("{HemcoDiagnosticFreq}", diagn_freq)   
                            
                
            if "{REG_SHORT}" in line:
                if region_short:
                    line = line.replace("{REG_SHORT}", region_short + ".")
                else:
                    line = line.replace("{REG_SHORT}", "")
                    

            out_file.write(line)
    
    print("Successfully written HEMCO lag file " + run_dir + "HEMCO_Config.rc")
    return
    
    
def write_hemco_diagn_file(run_dir, spc_emit):
    # Write diagnostic HEMCO file to specify emissions totals    
    
    # Probably want to write sectors to output as well. Particularly GFED and wetlands. 
    # Other sectors don't matter as much as can work them out.
    
    diagn_infile_path = run_dir + "HEMCO_Diagn.rc"
    
    with open(diagn_infile_path, "w") as out_file2:
        
        if type(spc_emit) == list:
            nlines  = len(spc_emit)+2
        else:
            nlines = 2
        
        for ti in range(nlines):
            
            if ti ==0:
                line = "# Name        Spec    ExtNr Cat Hier Dim OutUnit \n"
    
            elif ti ==1: 
                line = "CH4" + "     " +  "CH4" + "  -1   -1  -1   2   kg/m2/s \n"  
            else: 
                line = spc_emit[ti-2] + "     " +  spc_emit[ti-2] + "  -1   -1  -1   2   kg/m2/s \n"  
    
            out_file2.write(line)
    
    print ("Written file: " + "HEMCO_Diagn.rc")
    return
    
    
def write_history_rc_v12(run_dir,template_dir,save_conc=False, save_bc = False,
                           conc_freq = '6H'):
    """
    Write the HEMCO_Config.rc file for each assimilation run. 
    
    Inputs:
        run_dir (string): Directory where run code is located.
        template_dir (string): Directory where template file is located
        
        save_conc: Save species concentration fields
        save_bc: Save boundary conditions (from global runs for use in nested runs )
        
    """
    template_file = template_dir + "HISTORY_rc_v12.template"
    with open(template_file, "r") as in_file:
        filedata = in_file.readlines()
    
    with open(run_dir + "HISTORY.rc", "w") as out_file:
        for line in filedata:
                
            if save_conc == True:
                if "#'SpeciesConc'," in line:
                    line = line.replace("#" "")
                                        
                if "SpeciesConc.frequency:" in line:
                    
                    if conc_freq[-1] == "H":
                        rep_str = "00000000 " + conc_freq[:-1].zfill(2) + "0000"
                    elif conc_freq[-1] == "D":
                        rep_str = "000000" + conc_freq[:-1].zfill(2) + " 000000"
                    elif conc_freq[-1] == "D":
                        rep_str = "000000" + conc_freq[:-1].zfill(2) + " 000000"
                    
                    line = line.replace("00000100 000000", rep_str)
                                            
            if save_bc == True:
                if "#'BoundaryConditions'," in line:
                    line = line.replace("#" "")

            out_file.write(line)
    
    print("Successfully written file " + run_dir + "HISTORY.rc")    
    return
    
def write_restart_file_v12(fname_in, fname_out, species, ensemb_names, spc_IC, emis_start, 
                           ref_ch4 = 1700, write_IC=False, write_ens=False):
    """
    Write restart file for ensemble runs
    
    Need to copy an input CH4 field.
    For first run this will be spinup field.
    
    Need restart file to be in right format for v12.4 and v12.5
    Assume it is, but might need to add some more checks to make sure this works in future...
    """
    
    species_main = "SpeciesRst_"+ species
    
    ds = open_ds(fname_in)
    
    if species_main in list(ds.keys()):
        ds_out = ds.copy()
    else:
        rename_dict={}
        rename_dict["SPC_"+species] = "SpeciesRst_"+species
        ds_out = ds.rename(name_dict=rename_dict)
    
    ds_out.coords["time"]  = [pandas.to_datetime(emis_start)]
    
    ref_field = ds_out["SpeciesRst_"+species].copy()*0.+ref_ch4*1.e-9  # Convert to mol/mol
    ref_field.attrs = ds_out["SpeciesRst_"+species].attrs
    
    if write_IC ==True:
        for spc in spc_IC:
            if spc == "CH4IC":
                ds_out["SpeciesRst_"+spc] = ds_out["SpeciesRst_"+species].copy()
            elif spc == "CH4REF":
                ds_out["SpeciesRst_"+spc] = ref_field

    if write_ens == True:
        for spc in ensemb_names:
            ds_out["SpeciesRst_"+spc] = ref_field
            
    ds_out.to_netcdf(path = fname_out)
    print ("Successfully written " + fname_out + " to disk")
    return ds_out
    
def write_bc_ensemble_v12(start_date, start_lag, end_lag, ensemb_names, bc_dir,bc_str,
                            out_dir, species = "CH4", ic_spc_names=["CH4IC"],
                            BC_ensemble = False):
    """
    Write BC input files for ensemble GEOS-Chem runs.
    
    Easiest way then is to read the CH4 field from global run. Then:
        1) Apply scaling factors of 1 for new fields
        2) apply oter scaling factos for older fields.
    
    bc scalings =  dictionary of arrays of size (nwindows, nBC) e.g. if north-south = (nwindows, 2)
    
    v12.4/v12.5
    Read in netcdf files from global run with CH4 field
    Copy this field for each ensemble member and apply scale factors from x_ensemble
    
    Only apply scale factors for length of assim window.
    But - these may require I use a longer lag window. Is there a way of avoiding this?
    
    """
    
    # Find list of file names that span date range of interest
    files = filenames(bc_dir, bc_str, start=start_date, end=end_lag, freq="D")
    #fname = bc_dir + "BC.20140102"
    # Now loop through all files and create new daily files in turn:

    # New version for 12.4 - Need to edit this.

    for bc_file in files:
        
        date_str = re.search("([0-9]{8})", bc_file)[0]
        ds = open_ds(bc_file)
        
        ch4_field = ds.SpeciesBC_CH4.copy()
        
#        if pandas.to_datetime(date_str) == pandas.to_datetime(start_lag):
#            ch4_field[1:,:,:,:]=0.
#        elif int(date_str) > int(start_lag):
#            ch4_field[:,:,:,:]=0.         # Make CH4BC field 0 once the assimilation window has passed.
             
        if int(date_str) >= int(start_lag):   
            ch4_field[:,:,:,:]=0.            # Set BC to 0 from the start hour of lag period.
            
        ch4_field_out0 = ch4_field*0.
        ch4_field_out0.attrs = ch4_field.attrs
        
        # Attributes not carried over after opearting on data array. 
        # HEMCO reads in units, so need to include attrs in copied fields.
        
        # No need to write BC for emitted ensemble species anymore - saves space. 
        # Unless I include these in the ensemble, in which case I will have to...
        if BC_ensemble == True:
            for name in ensemb_names:
                ds["SpeciesBC_" + name] = ch4_field_out0 
                ds["SpeciesBC_" + name].attrs["long_name"] = "Dry mixing ratio of species " + name
   
        # But I do still want to include IC species
        if len(ic_spc_names) > 0:
            for name2 in ic_spc_names:
                if name2 == "CH4BC":
                    ds["SpeciesBC_" + name2] = ch4_field 
                    ds["SpeciesBC_" + name2].attrs["long_name"] = "Dry mixing ratio of species " + name2
                elif name2 == "CH4IC":
                    ds["SpeciesBC_" + name2] = ch4_field_out0
                    ds["SpeciesBC_" + name2].attrs["long_name"] = "Dry mixing ratio of species " + name2
                    #ds["SpeciesBC_" + name2].attrs = ch4_field.attrs 
                #ds["SpeciesBC_" + name2].attrs["long_name"] = "Dry mixing ratio of species " + name2
            
        
        fname_out = out_dir + bc_str + date_str + "_0000z.nc4"
        ds.to_netcdf(path = fname_out)
     
    print("Written BC files between " + start_date + " and " + end_lag + " to disk." )
    
    return