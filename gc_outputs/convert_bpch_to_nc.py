#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:46:41 2019

Convert Geos-Chem bpch output files into 1 dataset and save to netcdf

@author: mlunt
"""

import numpy as np
#import process_GC_mod as process
#import flib as flb
#import test_flib as flb
import time as run_time
import datetime
import xarray
import glob
import matplotlib.pyplot as plt
import pandas as pd
import re
import GC_output_mod as process
from scipy import interpolate

############################################
# Inputs

    
#run_range=["20171216",
#           "20180101", "20180116",
#           "20180201", "20180216",
#           "20180301", "20180316",
#           "20180401", "20180416",
#           "20180501", "20180516",
#           "20180601", "20180616",
#           "20180701", "20180716"]

#run_range=["20180801", "20180816",
#           "20180901", "20180916",
#           "20181001", "20181016",
#           "20181101", "20181116"]

#run_range=["20181201", "20181216", "20190101", "20190116"]
           
#run_range = ["20190301","20190316",
#             "20190401", "20190416",
#             "20190501", "20190516",
#             "20190601", "20190616",
#             "20190701", "20190716"]   

run_range = ["20191101", "20191116",
             "20191201", "20191216"]
            
                   

#run_str = "SA_true_025x03125" 
#run_str = "SA_pseudo_cov" 
#run_str = "SA_true2" 
#run_str = "NAF_mk3" 
#run_str = "test_NAF2" 
run_str = "SSA_run2"
#run_str = "NAF_run1"

N_ens = 70
#N_ens = 140
spc_per_run = 70

n_ens_runs = int(np.ceil(N_ens/spc_per_run))

ensemb_names=[]
for xi in range(N_ens):    
    ensemb_names.append("CH4_E" + str(xi+1))

#satellite = "TROPOMI"
#satellite = None

#varnames_IC = ["CH4IC", "CH4BC", "CH4REF"]
varnames_IC = ["CH4REF"]

#column_out_dir = "/geos/d21/mlunt/TROPOMI/"           
#fname_out= column_out_dir + "GC_TROPOMI_XCH4_025x03125_daily_201901-201902.nc"

for run_date in run_range:
    
    
    for ens_num in range(n_ens_runs):
        
        ch4_mod_new={}
        ch4_mod={}
        xch4_column={}


        ens_range = [ens_num*spc_per_run+1, (ens_num+1)*spc_per_run]
        
        ensemb_names_50 = ensemb_names[ens_num*spc_per_run:(ens_num+1)*spc_per_run]
        N_ens_50  = len(ensemb_names_50)
    
        run_name=run_str + "_" + run_date + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) 
        #run_name = run_str + run_date
        
        if ens_num == 0:
            varnames_short=varnames_IC + ensemb_names_50
        else:
            varnames_short = ensemb_names_50

        varnames_long=[]
        for varname in varnames_short:
                varnames_long.append("IJ_AVG_S_" + varname)
        
        ############################################################################
        #%%
        
        # Define directories for each run
        
        run_dir  = "/geos/u73/mlunt/gc_rundirs/ensemb_runs/" + run_name + "/" 
        output_dir  = "/geos/d21/mlunt/GC_output/EnKF/" + run_name + "/" 
        #output_dir  = "/geos/u73/mlunt/GC_output/EnKF/" + run_name + "/"
        #column_out_dir = "/geos/d21/mlunt/GC_output/EnKF/" + run_name + "/sat_columns/"  
        #fname_out= column_out_dir + "XCH4_Model_" + satellite + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) + ".nc" 
        
#        if satellite == "GOSAT":
#            obs_dir = "/home/mlunt/datastore/GOSAT/"
#            obs_str = "GOSAT-L2-ACOS_GCPR-CH4-avg-CA-merra2-"
#        elif satellite == "TROPOMI":
#            obs_dir = "/geos/d21/sat_data/TROPOMI/"
#            obs_dir = "/geos/d21/mlunt/TROPOMI/processed/daily/"
#            obs_str = "TROPOMI_XCH4_025x03125_daily_"
#            
#        else:
#            obs_dir = None
        
        tracer_file = run_dir + "tracerinfo.dat"
        diag_file = run_dir + "diaginfo.dat"

        gc_file_str = "ts_satellite."
        #%%
        startt = run_time.time()
        files = process.filenames(output_dir, gc_file_str, file_type="bpch")
    #    start_date = re.search("([0-9]{8})", files[0]).group(0)
    #    end_date = re.search("([0-9]{8})", files[-1]).group(0)
    
        if len(files) < 1:
            raise ValueError("No files found in " + output_dir )
        
        m = re.search(gc_file_str +'(.+?)' + '.bpch', files[0])
        if m:
            start_date = m.group(1)
        else:
            raise ValueError("START_DATE: GC output file string does not fit expected format")
            
        m2 = re.search(gc_file_str +'(.+?)' + '.bpch', files[-1])
        if m:
            end_date = m2.group(1)
        else:
            raise ValueError("END_DATE: GC output file string does not fit expected format")
            
       
#        ds_gc_list, gc_time = process.read_GC_out_files(output_dir, gc_file_str, start_date=start_date, 
#                                          end_date=end_date, tracer_file=tracer_file,
#                                          diag_file=diag_file, irregular_vars=False)
        
#        ds_gc = process.read_GC_out_files(output_dir, gc_file_str, start_date=start_date, 
#                                          end_date=end_date, tracer_file=tracer_file,
#                                          diag_file=diag_file, irregular_vars=False)
       
#%%
         ######################################
         # Commented this seciton out for now - try doing one at a time instead
            
#        ds_gc = process.read_GC_out_files(output_dir, gc_file_str, start_date=start_date, 
#                                          end_date="20190410", tracer_file=tracer_file,
#                                          diag_file=diag_file, irregular_vars=False)
#        
#        # In this instance ds_gc is a list of datasets
#        # Each one has a time dimension applied
#        
#        
#        
#        gc_time = ds_gc.time
#
#        time_mod_int = []
#        for ti_day in gc_time:
#            time_int = int(pd.to_datetime(ti_day.values).strftime('%Y%m%d'))
#            time_mod_int.append(time_int)
#        
#        lat_mod = ds_gc.lat.values
#        lon_mod = ds_gc.lon.values
#        
#            
##%%        
#        # Calculate pressure weight matrix
#        # Can probably do as array operation in Python rather than using f2py.
#        
#        ds_gc  = ds_gc.transpose("time", "lat", "lon", "lev")
#        
#        ds_out = ds_gc.copy()
#        for key in list(ds_gc.keys()):
#            del(ds_out[key].attrs["scale_factor"])
#            del(ds_out[key].attrs["hydrocarbon"])
#            del(ds_out[key].attrs["chemical"])
#            del(ds_out[key].encoding["scale_factor"])
#            if key != "PEDGE_S_PSURF":
#                ds_out[key].attrs["units"] = "ppbv"
#        
#        for key in list(ds_gc.keys()):
#            ds_out[key].encoding['zlib'] = True  
#            ds_out[key].encoding['least_significant_digit'] = 2 
#            #ds_out[key].encoding['dtype'] = 
#    
#        fname_gc_out = output_dir + "satellite_output_all.nc"
#        ds_out.to_netcdf(path=fname_gc_out, mode='w')
        
        
#ds_out2 = ds_out["IJ_AVG_S_CH4"]
#ds_out2.encoding['least_significant_digit'] = 3
#ds_out2.attrs['units'] = "ppbv"
#del(ds_out2.encoding["scale_factor"])
#
#ds_out2.to_netcdf(path=output_dir+"test_file.nc", mode='w')
 
#%%       
        
        # Loope through all files and save output from each day individually
        
        for file in files:
            
            #ds_gc  = process.open_ds(file)
            ds_gc = process.open_bpch_ds(file, tracer_file, diag_file, fields=None)
        
            date_str=[]
            date_str.append(re.findall("([0-9]{8})", file)[-1]+"-13:00")
            days=pd.to_datetime(date_str)
            #days = pd.date_range(start=start_date + "-13:00" , end=end_date+"-13:00", freq='D' )
            #ds_gc["time"]=days
            
            #ds_gc  = ds_gc.transpose("time", "lat", "lon", "lev")
            ds_gc  = ds_gc.transpose("lat", "lon", "lev")
            ds_gc2 = ds_gc.expand_dims(["time"])
            ds_gc2["time"]=days
        
            ds_out = ds_gc2.copy()
            for key in list(ds_gc.keys()):
                del(ds_out[key].attrs["scale_factor"])
                del(ds_out[key].attrs["hydrocarbon"])
                del(ds_out[key].attrs["chemical"])
                del(ds_out[key].encoding["scale_factor"])
                if key != "PEDGE_S_PSURF":
                    ds_out[key].attrs["units"] = "ppbv"
            
            for key in list(ds_gc.keys()):
                ds_out[key].encoding['zlib'] = True  
                ds_out[key].encoding['least_significant_digit'] = 2 
                #ds_out[key].encoding['dtype'] = 
        
            fdate = re.findall("([0-9]{8})", file)[-1]
        
            fname_gc_out = output_dir + "sat_output_" + fdate + ".nc"
            ds_out.to_netcdf(path=fname_gc_out, mode='w')
        
            keyi = list(ds_gc.keys())[10]
            max_ch4 = ds_gc[keyi].max().values
            print( "Max CH4 on " + date_str[0][:8] + " is: ", max_ch4)
    #%% 
    
#            xch4_obs = ds_obs.XCH4
#            xch4_precision = ds_obs.XCH4_precision
#            dates_out = ds_obs.time
#        
#            ds_out = xarray.Dataset({"XCH4_obs": (["time", "lat", "lon"],xch4_obs),
#                                     "IminusA_XCH4_ap": (["time", "lat", "lon"],I_minus_A_xap),
#                                     "XCH4_ap": (["time", "lat", "lon"],xch4_ap),
#                                     "XCH4_precision": (["time", "lat", "lon"],xch4_precision),
#                                     "surface_pressure":(["time", "lat", "lon"],psurf_sat),
#                                     "pressure_interval":(["time", "lat", "lon"],dp_sat),
#                                     "ch4_profile_apriori":(["time", "lat", "lon", "layer"],ch4_ap),
#                                     "ch4_averaging_kernel":(["time", "lat", "lon","layer"],averaging_kernel),
#                                     "dry_air_subcolumn":(["time", "lat", "lon","layer"],air_subcolumn)
#                                     },
#                                     coords={"lon":lon_mod, "lat": lat_mod, "time":dates_out})
#            ds_out.attrs["Comments"]='XCH4 mean at 0.25x0.3125'
#            
#            
#            for varname in varnames_short:
#                ds_out[varname]  = (["time", "lat", "lon"],xch4_column[varname])
#            
#            ds_out.attrs["qa_threshold"] = 0.5
#            #ds_out.attrs["Comments"]='Monthly mean XCH4 at 0.1x0.1'
#            
#            ds_out["XCH4_obs"].attrs = xch4_obs.attrs
#            ds_out["XCH4_precision"].attrs = xch4_precision.attrs
#            ds_out["surface_pressure"].attrs = psurf_sat.attrs
#            ds_out["pressure_interval"].attrs = dp_sat.attrs
#            ds_out["ch4_profile_apriori"].attrs = ch4_ap.attrs
#            ds_out["ch4_averaging_kernel"].attrs = averaging_kernel.attrs
#            ds_out["dry_air_subcolumn"].attrs = air_subcolumn.attrs
#            
#            ds_out["surface_pressure"].attrs["units"] = "hPa"
#            ds_out["pressure_interval"].attrs["units"] = "hPa"
#            
#            for key in list(ds_out.keys()):
#                    ds_out[key].encoding['zlib'] = True                    
#            ds_out.to_netcdf(path=fname_out, mode='w')
                       
           
            
            


          
                       
