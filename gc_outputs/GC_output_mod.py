#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:24:03 2019

Module for GC output functions for ensemble run code

@author: mlunt
"""

import numpy as np
import xarray
import xbpch
import glob
import pandas as pd
from bisect import bisect_left
#import flib as flb
import test_flib as flb
import re

mh2o = 18.04
mg=28.96 # Assuming here that mg is molar mass of air

def nearest(items, pivot):
    """
    Deprecated.
    This function is very slow
    better to use find_nearest using numpy
    """
    return min(items, key=lambda x: abs(x - pivot))

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()    
    return idx, array[idx]

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

def open_bpch_ds(fname, tracer_file, diag_file, fields=None):
    if fields == None:
        with xbpch.open_bpchdataset(fname, tracerinfo_file=tracer_file, diaginfo_file=diag_file) as ds:
            ds.load()
    else:
        with xbpch.open_bpchdataset(fname, tracerinfo_file=tracer_file, diaginfo_file=diag_file, fields=fields) as ds:
            ds.load()
    return ds

def filenames(file_dir, file_str, file_type="bpch", freq="D", start=None, end=None):
    """
    Output a list of available file names,
    for given directory and date range.
    Assumes monthly files
    """
    files = []
    # Convert into time format
    if (start is not None) and (end is not None):
        if freq == "MS":
            days = pd.date_range(start=start[:6] + "01", end=end, freq=freq)
        else:
            days = pd.date_range(start=start, end=end, freq=freq)
        
        #days = pd.DatetimeIndex(start = start, end = end, freq = freq).to_pydatetime()
        
        if freq == "MS":
            yearmonthday = [str(d.year) + str(d.month).zfill(2) for d in days]
        else:
            yearmonthday = [str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2) for d in days]
    
        for ymd in yearmonthday:
            f=glob.glob(file_dir + "/" + file_str + ymd + "*." +file_type)
            if len(f) > 0:
                files += f
        files.sort()
        
    else:
        f=glob.glob(file_dir + "/" + file_str + "*." +file_type)
        if len(f) > 0:
            files += f     # Not entirely sure if this will work - might be worth checking! - Yes it works
        files.sort()
        ymd=""

    if len(files) == 0:
        print("Can't find file: " + file_dir + "/" + file_str + ymd[:4] + "*." + file_type)
                        
    return files

def read_irregular_bpch(files, tracer_file, diag_file, fields=None):
    """
    Subroutine to merge bpch datasets where not every variable exists
    in each file.
    Required where read_netcdfs won't work
    """
    datasets = [open_bpch_ds(p, tracer_file, diag_file, fields=fields) for p in sorted(files)]
    date_str=[]
    for f in files:     
        #date_str.append(re.search("([0-9]{8})", f).group(0)+"-13:00")
        date_str.append(re.findall("([0-9]{8})", f)[-1]+"-13:00")
    days=pd.to_datetime(date_str)   
    dum_xrda = xarray.DataArray(np.arange(len(files)), coords=[days], dims=['time'])
    days_xr = dum_xrda.time
    
    if "time" not in (datasets[0].keys()):
    
        for ti, dataset in enumerate(datasets):
            
            dum = dataset.expand_dims('time')
            dum.coords["time"] = [days[ti]]    
            datasets[ti] = dum
        
    #combined=xarray.merge(datasets)
    
    #return combined
    return datasets,days_xr


def read_netcdfs(files, dim = "time", file_type = "nc", 
                 tracer_file=None, diag_file=None,
                 list_output=False):
    '''
    Use xray to open sequential netCDF or bpch files. 
    Makes sure that file is closed after open_dataset call.
    '''
    if file_type == "nc":
        datasets = [open_ds(p) for p in sorted(files)]
    elif file_type == "bpch":
        
        if tracer_file is None:
            raise ValueError("Need to define tracer_file to read bpch output files")
        elif tracer_file is None:
            raise ValueError("Need to define diag_file to read bpch output files")
        
        datasets = [open_bpch_ds(p, tracer_file, diag_file) for p in sorted(files)]
    
    if list_output == True:
        time_list = []
        for dataset in datasets:
            time_list.append(dataset.time)
    
        return datasets, time_list
    else:
        combined = xarray.concat(datasets, dim)
        return combined   

def read_satellite(sat_dir, sat_str,start_date=None, end_date=None, region=None):
    #sat_files = glob.glob(sat_dir + sat_str + "*.nc")
    
    sat_files = filenames(sat_dir, sat_str, file_type="nc", freq="MS", start=start_date, end=end_date)
    # Only want to read files between given dates though. So use filenames ratehr than glob.
    
    ds_sat = read_netcdfs(sat_files)
    
    if start_date is not None:
        if end_date is not None:
            ds_sat = ds_sat.sel(time = slice(start_date,end_date))
        else:
            raise ValueError("Need to specify end date if start date is specified") 
            
    if region is not None:
        ds_sat = ds_sat.sel(lat=slice(region[0],region[1]), lon=slice(region[2], region[3]))
        
    return ds_sat



def read_GC_out_files(output_dir, file_str, start_date=None, end_date=None, 
                      tracer_file=None, diag_file=None, irregular_vars=False, fields=None,
                      file_type='bpch', list_output=False):
    
    """
    Read Geos_chem output fiels into a xarray dataset
    Add time dimension to dataset as this doesn't get done automatically
    """
    
    files = filenames(output_dir, file_str, file_type=file_type, start=start_date, end=end_date)
    
    if irregular_vars:
        ds_gc = read_irregular_bpch(files, tracer_file, diag_file, fields=fields)
    elif list_output == True:
        ds_gc_list, gc_time = read_netcdfs(files, file_type=file_type, 
                                           tracer_file=tracer_file, diag_file=diag_file,
                                           list_output=True)
        return ds_gc_list, gc_time
    else:
        ds_gc = read_netcdfs(files, file_type=file_type, tracer_file=tracer_file, diag_file=diag_file)
    
        # Need to asign times to GC time dimension
        # need to do this from filenames rather than input and ouput start-dates
        date_str=[]
        for f in files:     
            #date_str.append(re.search("([0-9]{8})", f).group(0)+"-13:00")
            date_str.append(re.findall("([0-9]{8})", f)[-1]+"-13:00")

        days=pd.to_datetime(date_str)
        #days = pd.date_range(start=start_date + "-13:00" , end=end_date+"-13:00", freq='D' )
        ds_gc["time"]=days
        
        return ds_gc


def setup_vert_intpl(grd_pres, mod_pres, obs_pres):

    """
    Vertical interpolation parameter. It calculates
    
    1) vpl, vpr, vwgt

    location (vpl/vpr) and  coefficicents (vwgt) for vertical interpolations from model grid (mod_pres)
    to observation vertical grid (obs_pres)

    2) col_wgt  
    coefficent for calculating column integration (i.e., mass weight)

    """
    
    do_vert_reverse=False
    
    if (mod_pres[0,2]>mod_pres[0,6]):
        do_vert_reverse=True
        lg_mod_pres=np.log10(mod_pres[:, ::-1])
    else:
        lg_mod_pres=np.log10(mod_pres)
        
    usd_idx=np.where(grd_pres>-990.0)
    lg_grd_pres=np.array(grd_pres)
    lg_grd_pres[usd_idx]=np.log10(grd_pres[usd_idx])
    # lg_opres=obs_pres
    usd_idx=np.where(obs_pres>-990.0)
    lg_obs_pres=np.array(obs_pres)
    lg_obs_pres[usd_idx]=np.log10(obs_pres[usd_idx])
    
    
    
    vpl, vpr, vwgt=flb.get_vertical_wgt_1d(lg_mod_pres, lg_grd_pres)

    obs_vpl, obs_vpr, obs_vwgt=flb.get_vertical_wgt_1d(lg_obs_pres, lg_grd_pres)
    
    colwgt=flb.get_col_wgt_1d(lg_grd_pres)
    colwgt=np.squeeze(colwgt)
    
    
    # print 'do_vert_reverse', do_vert_reverse
    return vpl, vpr, vwgt, obs_vpl, obs_vpr, obs_vwgt, colwgt, do_vert_reverse

def get_col_dry_air(grd_colwgt, wv_at_grd):
    """ calculate dry-air column (ratio)
    from water-vapor profile prof_wv """
    #rmass=gc.mh2o/gc.mg
    prof_dry_air=np.ones(np.shape(wv_at_grd), float)
    used_idx=np.where(wv_at_grd<-990.0)
    prof_dry_air[used_idx]=-999.0
    
    
    
    prof_dry_air=flb.array_divide_array_2d(prof_dry_air, wv_at_grd, 1.0)
    col_dry_air=flb.col_int_1d(prof_dry_air, grd_colwgt)
    
    return col_dry_air

def get_xgp0(obs_apr, \
             obs_ak,\
             obs_vpl,\
             obs_vpr,\
             obs_vwgt,\
             wv_at_grd,\
             grd_colwgt):
    
    """
    integration of   (1-ak)*apriori

    """
    no_use_idx=np.where(obs_apr != obs_apr)
    
    obs_ak[no_use_idx]=-999.0
    obs_apr[no_use_idx]=-999.0
    obs_apr=np.where(obs_apr<-990.0, -999.0, obs_apr)  # wheere obs_apr < -999 return -999 else return obs_apr
    
    
    new_ak = obs_ak.copy()*1.
    new_apr = obs_apr.copy()*1.
    
    # I've commented these 2 lines out as function doesn't exist in 
    
#    new_ak=flb.refill_bad_points_1d(obs_ak)   
#    new_apr=flb.refill_bad_points_1d(obs_apr)

    # project apr to combined grid
    
    apr_at_grd=flb.prof_vertical_intpl_wv(obs_vpl, obs_vpr, obs_vwgt, wv_at_grd, new_apr)
    ak_at_grd=flb.prof_vertical_intpl_1d(obs_vpl, obs_vpr, obs_vwgt, new_ak)
    res_ak_at_grd=1.0-ak_at_grd
    # convert wet mixing ratio to dry-air ratio 
    # but there is no need to do that, as GEOS-Chem messed these two things. 

    rmass=mh2o/mg
    #rmass=gc.mh2o/gc.mg
    wv_org=np.array(wv_at_grd)
    usd_idx=np.where(wv_org>-990.0)
    wv_org[usd_idx]=1.0-wv_org[usd_idx]/rmass
    
    apr_at_grd=flb.array_divide_array_2d(apr_at_grd, wv_org, 0.0)
    
    
    xgp0=flb.ak_col_int_1d(grd_colwgt, res_ak_at_grd, apr_at_grd)
    return  ak_at_grd, xgp0


def get_xgp(grd_pres,
            prof,\
            wv_at_grd,\
            ak_at_grd, \
            vpl, \
            vpr,\
            vwgt,\
            grd_colwgt,\
            mod_offset=0.0,\
            do_vert_reverse=False, debug=False):
    
    """ calculate xgp (sum of ak*model_profile)

    
    Notes:
    1) this version use column water vapor instead of the profile, which could be problematic
 
    2) the resulting xgp does not include contribution of a-priori
    
    """
    
    # calculate xgp at observation locations
        
    # ch4 profile after removing model_offset
    
    prof=prof-mod_offset
    
    if (do_vert_reverse):
        prof=prof[:,::-1]
        
    # print prof[42,:]
    # wv_at_grd[:,:]=0.0
    
    prof_at_grd=flb.prof_vertical_intpl_wv(vpl, vpr, vwgt, wv_at_grd, prof)
    # print prof_at_grd[42,:]
    
    xgp=flb.ak_col_int_1d(grd_colwgt, ak_at_grd, prof_at_grd)
    
    return xgp

def select_gosat_region(ds_gosat, lon_limits,lat_limits, keep_bad = False):
    '''
    Sub-select GOSAT measurements only within region of interest.  
    lon_limits  = array(2) [0]= min, [1] = max
    
    keep_bad: False = Only keep good measurements (with flag =0)
    '''
# 3. Sub-select GOSAT measuements within GC lat-lon domain
   
    
    # Is there an easier way than using lots of logical ands and wheres?
    wh_domain = np.where(np.logical_and(np.logical_and(ds_gosat.lon >lon_limits[0], 
                                                       ds_gosat.lon < lon_limits[1] ),
                                        np.logical_and(ds_gosat.lat >lat_limits[0], 
                                                       ds_gosat.lat < lat_limits[1])))
        
    ds_gosat_domain = ds_gosat.isel(time = wh_domain[0])
    if keep_bad is False:
        wh_good = np.where(ds_gosat_domain.xch4_quality_flag == 0)
        ds_gosat_domain = ds_gosat_domain.isel(time = wh_good[0])
        
    return ds_gosat_domain


def calc_horizontal_weights(lon_gosat, lat_gosat, lon_model, lat_model):
    """
    Calculate the weights needed to interpolate model fields horizontally 
    onto measurement lons and lats
    
    If doing this lots of times may be quicker just to pass in lons and lats 
    rather than whole dataset.
    
    Function assumes that lons and lats are in regular ascending order
    """
    # Now start to follow Liang's procedure for converting onto same grid. 
    
    #1. Work out horizontal weights and nearest lons and lats for each GOSAT obs point 
    # For each Gosat obs find 2 nearest lons and two nearest lats
    # Establish weighting based on normalised difference:
   
    
    dlon_model = lon_model[3]-lon_model[2]
    dlat_model = lat_model[3]-lat_model[2]
    
    
    #nobs=20
    nobs = len(lat_gosat)
    
    plon1 = np.zeros((nobs), dtype=np.int16)
    plon2 = np.zeros((nobs), dtype=np.int16)
    plat1 = np.zeros((nobs), dtype=np.int16)
    plat2 = np.zeros((nobs), dtype=np.int16)
    
    for ti in range(nobs):
        loni = lon_gosat[ti]
        lati = lat_gosat[ti]
        
        pos_lon  = bisect_left(lon_model,loni)
        pos_lat  = bisect_left(lat_model,lati)
        plon1[ti] = pos_lon-1
        plon2[ti] = pos_lon
        plat1[ti] = pos_lat-1
        plat2[ti] = pos_lat
    
    
    latwgt = (lat_model[plat2]-lat_gosat)/dlat_model
    lonwgt = (lon_model[plon2]-lon_gosat)/dlon_model
    
    w1=latwgt*lonwgt
    w2=(1.0-latwgt)*lonwgt
    w3=latwgt*(1.-lonwgt)
    w4=(1-latwgt)*(1.0-lonwgt)
    
    return w1,w2,w3,w4, plon1,plon2, plat1,plat2

#def get_tropomi_p_levels(psurf_sat, dp_sat, nlev):
#    
#    p_sat_low = np.zeros((nlat,nlon,nlev))
#    
#    for ilev in range(nlev_sat):
#        
#        p_sat_low[:,:,ilev] = psurf_sat  - dp_sat*(nlev_sat-ilev)
#        p_sat_high = psat_low - dp_sat
#    
#    
#def interpolate_vertical(psurf_sat, dp_sat, nlev_sat, pedge_mod, ch4_mod, nlev_mod):
#    
#    
#    pcent_mod = pedge_mod.copy()*0.
#    
#    pcent_mod[:,:,:,:-1] = (pedge_mod[:,:,:,:-1] + pedge_mod[:,:,:,1:])/2.
#    pcent_mod[:,:,:,-1] = pedge_mod[:,:,:,-1]/2.
#    
#    for ilev in range(nlev_sat):
#        
#        p_sat = psurf_sat  - dp_sat*(nlev_sat-ilev)  #(time,lat,lon)
#        
#        p_diff = p_sat - pcent_mod
        
        
        
    
    
