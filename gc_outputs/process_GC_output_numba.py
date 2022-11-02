#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:29:08 2020

This script is really memory intensive

Suspect because of use of large 4d arrays - but is it just from reading GC outputs? 

Yes - just reading in files takes up 13% of memory on holmes.

Maybe do day by day and then stitch together at the end. 

@author: mlunt
"""

import numpy as np
#import process_GC_mod as process
#import flib as flb
import test_flib as flb
import time as run_time
import datetime
import xarray
import glob
import matplotlib.pyplot as plt
import pandas as pd
import re
import GC_output_mod as process
from scipy import interpolate
from enkf_code import column_mod

############################################
# Inputs
    
file_type='nc'
#run_range=["20180901", "20180911", "20180921"]

#run_range=["20171201", "20171216",
#           "20180101", "20180116",
#           "20180201", "20180216",
#           "20180301", "20180316",
#           "20180401", "20180416",
#           "20180501", "20180516",
#           "20180601", "20180616",
#           "20180701", "20180716",
#           "20180801", "20180816",
#           "20180901", "20180916",
#           "20181001", "20181016",
#           "20181101", "20181116",
#           "20181201", "20181216",
#           "20190101", "20190116", 
#           "20190201", "20190216",
#           "20190301", "20190316",
#           "20190401", "20190416",
#           "20190501", "20190516",
#           "20190601", "20190616",
#           "20190701", "20190716",
#           "20190801", "20190816",
#           "20190901", "20190916",
#           "20191001", "20191016",
#           "20191101", "20191116"]

run_range = ["20191001", "20191016",
             "20191101", "20191116",
             "20191201", "20191216"]
#            
#run_range=["20200201", "20200216",
#           "20200301", "20200316"]
         
#run_range= ["20191201", "20191216"]
            #"20200101", "20200116"]

#run_range = ["20181116",
#             "20181201", "20181216",
#             "20190101", "20190116"]

#run_range = ["20190601", "20190616", 
#             "20190701", "20190716",
#             "20190801", "20190816"]

 
run_str = "SSA_run1" 
N_ens = 70   #140
#N_ens = 140
spc_per_run = 70

n_ens_runs = int(np.ceil(N_ens/spc_per_run))


ensemb_names=[]
for xi in range(N_ens):    
    ensemb_names.append("CH4_E" + str(xi+1))

satellite = "TROPOMI"
#satellite = "GOSAT"
#satellite = None

sron=True

#varnames_IC = ["CH4IC", "CH4BC", "CH4REF"]
varnames_IC = ["CH4REF"]

#column_out_dir = "/geos/d21/mlunt/TROPOMI/"           
#fname_out= column_out_dir + "GC_TROPOMI_XCH4_025x03125_daily_201901-201902.nc"

for run_date in run_range:
    
    
    for ens_num in range(n_ens_runs):
        
        ch4_mod_new={}
        ch4_mod={}
        xch4_column={}
        xch4_column_time={}

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
        #run_dir = "/geos/u73/mlunt/GC_output/EnKF/" + run_name + "/" 
        output_dir  = "/geos/d21/mlunt/GC_output/EnKF/" + run_name + "/" 
        column_out_dir = "/geos/d21/mlunt/GC_output/EnKF/" + run_name + "/sat_columns/"  
        if sron==True:
            fname_out= column_out_dir + "XCH4_Model_SRON_v2_" + satellite + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) + ".nc" 
        else:
            fname_out= column_out_dir + "XCH4_Model_" + satellite + "_ENS_" + str(ens_range[0]).zfill(2) + "-" +str(ens_range[1]) + ".nc" 
        
        if satellite == "GOSAT":
            obs_dir = "/geos/d21/mlunt/GOSAT/processed/SSA/"
            obs_str = "GOSAT_XCH4_025x03125_daily_SSA_"
        elif satellite == "TROPOMI":
            if sron == True:
                obs_dir = "/geos/d21/mlunt/TROPOMI/SRON/processed_v202008/SSA/"
                obs_str = "TROPOMI_XCH4_025x03125_daily_SSA_"
            else:
                obs_dir = "/geos/d21/mlunt/TROPOMI/processed/daily/"
                obs_str = "TROPOMI_XCH4_025x03125_daily_SSA_"
            
        else:
            obs_dir = None
        
        tracer_file = output_dir + "tracerinfo.dat"
        diag_file = output_dir + "diaginfo.dat"

        if file_type == "bpch":
            gc_file_str = "ts_satellite."
        elif file_type == "nc":
            gc_file_str = "sat_output_"
        #%%
        #startt = run_time.time()
        files = process.filenames(output_dir, gc_file_str, file_type=file_type)
    #    start_date = re.search("([0-9]{8})", files[0]).group(0)
    #    end_date = re.search("([0-9]{8})", files[-1]).group(0)
    
        if len(files) < 1:
            raise ValueError("No files found in " + output_dir )
        
        
        # Loop through file list to limit memory usage
        startt=run_time.time()
        
        for varname in varnames_short:
            xch4_column_time[varname]=[]
        
        
        # Find lat and lon bounds for reading in satellite data
        
        #%%
        # Loop through each file individually to save memory
        #for file in files[:1]:
        
            #m = re.search(gc_file_str +'(.+?)' + '.' + file_type, file)
            #if m:
            #    start_date = m.group(1)
            #    end_date = m.group(1)
            #else:
            #    raise ValueError("START_DATE: GC output file string does not fit expected format")
                
        m = re.search(gc_file_str +'(.+?)' + '.' + file_type, files[0])
        if m:
            start_date = m.group(1)
        else:
            raise ValueError("START_DATE: GC output file string does not fit expected format")
            
        m2 = re.search(gc_file_str +'(.+?)' + '.' + file_type, files[-1])  # Should be -1
        if m:
            end_date = m2.group(1)
        else:
            raise ValueError("END_DATE: GC output file string does not fit expected format")
            
       
#        ds_gc_list, gc_time = process.read_GC_out_files(output_dir, gc_file_str, start_date=start_date, 
#                                          end_date=end_date, tracer_file=tracer_file,
#                                          diag_file=diag_file, irregular_vars=False)
        
        ds_gc_list, gc_time = process.read_GC_out_files(output_dir, gc_file_str, start_date=start_date, 
                                          end_date=end_date, tracer_file=tracer_file,
                                          diag_file=diag_file, irregular_vars=False,
                                          file_type=file_type, list_output=True)            
        # In this instance ds_gc is a list of datasets
        # Each one has a time dimension applied
        #gc_time = ds_gc.time

        mod_time_int = []
        for ti_day in gc_time:
            time_int = int(pd.to_datetime(ti_day[0].values).strftime('%Y%m%d'))
            mod_time_int.append(time_int)
        
        mod_time_int_arr = np.asarray(mod_time_int)
        
        lat_mod = ds_gc_list[0].lat.values
        lon_mod = ds_gc_list[0].lon.values
        #lat_mod = ds_gc.lat.values
        #lon_mod = ds_gc.lon.values
        
        
        
        #%% 
        # Read satellite data
        #if satellite in(["GOSAT", "TROPOMI"]):
            
        latmin = lat_mod.min()
        latmax = lat_mod.max()
        lonmin = lon_mod.min()
        lonmax = lon_mod.max()
        
        ds_obs = process.read_satellite(obs_dir, obs_str, start_date=start_date, 
                                        end_date=end_date, region=[latmin,latmax,lonmin,lonmax])

        obs_time  = ds_obs.time
        obs_time_int = []
        for ti_day2 in obs_time:
            time_int2 = int(pd.to_datetime(ti_day2.values).strftime('%Y%m%d'))
            obs_time_int.append(time_int2)   
            
        obs_time_int_arr=np.asarray(obs_time_int)
        #ds_temp = ds_gc.copy()
        #ds_gc = ds_temp.reindex_like(ds_obs, method="nearest")
            
#%%        
        # Loop through each day in ds_obs
        # Find entry in ds_gc_list 
        ntime  = len(ds_obs.time)
        nlat= len(ds_obs.lat)
        nlon=len(ds_obs.lon)
        
        if satellite == "TROPOMI":
            
            nlev_sat=12
            dp_sat = ds_obs.pressure_interval
            psurf_sat = ds_obs.surface_pressure
            
            for ti in range(ntime):
                if np.max(psurf_sat[ti,:,:]) > 5000:
                    dp_sat[ti,:,:] = dp_sat[ti,:,:]/100.
                    psurf_sat[ti,:,:] = psurf_sat[ti,:,:]/100.
            
            # Satellite files go from top of atmsphere to bottom
            # GC goes from bottom to top.
            # Need to be consistent!!!!!!!
            air_subcolumn = ds_obs.dry_air_subcolumn[:,:,:,::-1]
            ch4_ap = ds_obs.ch4_profile_apriori[:,:,:,::-1]   # mol/m2
            averaging_kernel = ds_obs.ch4_averaging_kernel[:,:,:,::-1]
            
            p_cent_sat = np.zeros((ntime,nlat,nlon,nlev_sat))
            p_edge_sat = np.zeros((ntime,nlat,nlon,nlev_sat))
                
            # Need to create 4d arraay pf p_sat_cent and p_sat_edge before looping
            for levi in range(nlev_sat):
                p_edge_sat[:,:,:,levi] = (psurf_sat.values  - dp_sat.values*levi)
            
        elif satellite == "GOSAT":
            
            nlev_sat=20
            psurf_sat = ds_obs.surface_pressure
            plevels_sat = ds_obs.pressure_levels
            
            
            # GOSAT files go from bottom of atmosphere to top like model
            # GC goes from bottom to top.
            # Need to be consistent!!!!!!!
            #air_subcolumn = ds_obs.dry_air_subcolumn[:,:,:,::-1]
            ch4_ap = ds_obs.ch4_profile_apriori#[:,:,:,::-1]   # mol/m2
            averaging_kernel = ds_obs.ch4_averaging_kernel#[:,:,:,::-1]
            
    
        ntime1=1
        I_minus_A_time = []
        xch4_ap_time=[]
        for di, day in enumerate(obs_time_int_arr):
            
            midt1=run_time.time()
            
            wh_day_int = np.where(mod_time_int_arr == day)[0]
            ds_gc = ds_gc_list[wh_day_int[0]]
            #ds_gc  = ds_gc_temp.transpose("time", "lat", "lon", "lev") # Not needed - already correct shape
            pres_mod = ds_gc["PEDGE_S_PSURF"].values
            nlev_mod = len(ds_gc.lev)
            
            pedge_mod = pres_mod.copy()
            pcent_mod = pres_mod.copy()*0.
    
            # Calculate pressure at centre of model grid cells
            pcent_mod[:,:,:,:-1] = (pedge_mod[:,:,:,:-1] + pedge_mod[:,:,:,1:])/2.
            pcent_mod[:,:,:,-1] = pedge_mod[:,:,:,-1]/2.
            
            if satellite == None:
                p_surf = pres_mod[:,:,:,0]
                pres_wgt = column_mod.get_pressure_weight(pres_mod, p_surf)
                # Pressure weighting function from Connor et al 2008 https://doi.org/10.1029/2006JD008336
                for varname in varnames_short:
                    xch4_column[varname] = (pres_wgt * ds_gc["IJ_AVG_S_" + varname]).sum(dim="lev")
                    
            elif satellite == "TROPOMI":
                
                
                # Create a merged array where surfae pressure is max of model or sat surface pressures
                p_edge_sat_mod = p_edge_sat[di:di+1,:,:,:].copy()
                psurf_mod = pedge_mod[:,:,:,0]
                max_p_surf = np.maximum(psurf_sat[di:di+1,:,:].values, psurf_mod)
                p_edge_sat_mod[:,:,:,0] = max_p_surf
                pedge_mod[:,:,:,0]=max_p_surf
                
                # Need to loop through all species
                # Separate interpolation function for each one.
                         
                for varname in varnames_short:
                
                    ch4_mod_new[varname] = np.zeros((ntime1,nlat,nlon,nlev_sat))
                    ch4_mod[varname]  = ds_gc["IJ_AVG_S_" + varname]
                
                #startt = run_time.time()
                
                # Calculate model cumulative mass
                # Calculate total air mass in mass/m2 in each model vertical level
                mass_air_mod = column_mod.calc_air_mass(pedge_mod, 0.01, D4=True)
                mass_air_sat = column_mod.calc_air_mass(p_edge_sat_mod, 0.01, D4=True)
                
                # Calculate model CH4 mass
                cum_ch4_mass_mod={}
                mass_ch4_mod={}
                for varname in varnames_short:
                    mass_ch4_mod[varname] = mass_air_mod * ch4_mod[varname]*1.e-9
            
                    # Calculate the cumulative atmosphere mass at each pressure edge of model
                    #cum_ch4_mass_mod[varname] = mass_air_mod.copy()*0.
                    cum_ch4_mass_mod[varname] = np.zeros((ntime1,nlat,nlon,nlev_mod+1))
                    for li in range(nlev_mod):
                        cum_ch4_mass_mod[varname][:,:,:,1+li] = np.sum(mass_ch4_mod[varname][:,:,:,:li+1],axis=3)
                
                # Interpolate cumulative mass to satellite pressure edges
                
                arr_extra = np.zeros((ntime1,nlat,nlon,1))+0.01
                
                p_edge_mod2 = np.append(pedge_mod,arr_extra,axis=3) # Add additional upper bounding layer to model levels 
                p_edge_sat_mod2 = np.append(p_edge_sat_mod,arr_extra,axis=3)
                
                logp_edge_mod2 = np.log10(p_edge_mod2)
                logp_edge_sat_mod2=np.log10(p_edge_sat_mod2)
                
                ch4_cum_mass_new={}
                # Numba doesn't support dictionary inputs/outputs so have to loop here instead.
                for varname in varnames_short:
                    ch4_cum_mass_new[varname] = column_mod.numba_interp(logp_edge_mod2, 
                                    logp_edge_sat_mod2, cum_ch4_mass_mod[varname])
                
                
                # Convert cumulative mass profile to mole fraction profile
                
                for varname in varnames_short:
                    ch4_mod_new[varname] = column_mod.convert_mass_to_profile(ch4_cum_mass_new[varname],
                               mass_air_sat)
                    
    #                ch4_mod_new[varname] = column_mod.covert_mass_to_profile(ch4_cum_mass_new[varname],
    #                           mass_air_sat, cum_ch4_mass_mod[varname][:,:,:,-1], p_edge_sat_mod)
                    
                
                # Apply averaging kernel to retrieve final XCH4 column data
                total_air_column = np.sum(air_subcolumn[di:di+1,:,:,:].values,axis=3)
                p_wgt_sat = air_subcolumn[di:di+1,:,:,:].values/total_air_column[:,:,:,None]
                
            
                # Convert ch4_ap subcolumn to mol/mol dry air
                ch4_ap_subcolumn = ch4_ap[di:di+1,:,:,:].values/air_subcolumn[di:di+1,:,:,:].values
                xch4_ap = np.sum((p_wgt_sat * ch4_ap_subcolumn), axis=3)*1.e9
                
                
                #xch4_obs_no_ap = xch4_obs - 
                I_minus_A_xap = xch4_ap - np.sum((p_wgt_sat*averaging_kernel[di:di+1,:,:,:].values *ch4_ap_subcolumn*1.e9),axis=3)
                
                for varname in varnames_short:
                    if varname in (["CH4"]):
                        xch4_column[varname] = xch4_ap + np.sum(p_wgt_sat*averaging_kernel[di:di+1,:,:,:].values * 
                                               (ch4_mod_new[varname] - ch4_ap_subcolumn*1.e9), axis=3)
                        
                    else:
                        xch4_column[varname] = np.sum(p_wgt_sat*averaging_kernel[di:di+1,:,:,:].values*ch4_mod_new[varname], axis=3)
                
                
                #endt=run_time.time()
                #%%
                
            elif satellite == "GOSAT":
                
                p_node_sat = plevels_sat[di:di+1,:,:,:].values
                for varname in varnames_short:
                
                    ch4_mod_new[varname] = np.zeros((ntime1,nlat,nlon,nlev_sat))
                    ch4_mod[varname]  = ds_gc["IJ_AVG_S_" + varname]
                
                
                logp_cent_mod = np.log10(pcent_mod)
                logp_node_sat = np.log10(p_node_sat)
                
                #startt = run_time.time()
                for varname in varnames_short:
                    ch4_mod_new[varname] = column_mod.numba_interp(logp_cent_mod, 
                                    logp_node_sat, ch4_mod[varname].values)
                
                
    
                p_wgt_sat = ds_obs.pressure_weight[di:di+1,:,:,:] 
                xch4_ap = (p_wgt_sat * ch4_ap[di:di+1,:,:,:]).sum(dim="layer", min_count=1)#*1.e9
               
                I_minus_A_xap = xch4_ap - np.sum((p_wgt_sat*averaging_kernel[di:di+1,:,:,:].values *ch4_ap[di:di+1,:,:,:]),axis=3)
                
                for varname in varnames_short:
                    if varname in (["CH4"]):
                        xch4_column[varname] = xch4_ap + np.sum(p_wgt_sat.values*averaging_kernel[di:di+1,:,:,:].values * 
                                               (ch4_mod_new[varname] - ch4_ap[di:di+1,:,:,:]), axis=3)
                        
                    else:
                        xch4_column[varname] = np.sum(p_wgt_sat.values*averaging_kernel[di:di+1,:,:,:].values
                                   *ch4_mod_new[varname], axis=3)
                
            for varname in varnames_short:    
                xch4_column_time[varname].append(xch4_column[varname])
                
            I_minus_A_time.append(I_minus_A_xap)
            xch4_ap_time.append(xch4_ap)
            
            
        midt2 = run_time.time()
        print("Time taken for loop is: ", midt2-midt1, " seconds")
                
        endt = run_time.time()
            
            
        print("Time taken to convolve columns is ", endt-startt, " seconds")
    #%% 
    
        xch4_obs = ds_obs.XCH4
        dates_out = ds_obs.time
        
        I_minus_A_xap2 = np.vstack(I_minus_A_time)
        xch4_ap2 = np.vstack(xch4_ap_time)
    
        ds_out = xarray.Dataset({"XCH4_obs": (["time", "lat", "lon"],xch4_obs),
                                 "IminusA_XCH4_ap": (["time", "lat", "lon"],I_minus_A_xap2),
                                 "XCH4_ap": (["time", "lat", "lon"],xch4_ap2),
                                 "surface_pressure":(["time", "lat", "lon"],psurf_sat),
                                 "ch4_profile_apriori":(["time", "lat", "lon", "layer"],ch4_ap),
                                 "ch4_averaging_kernel":(["time", "lat", "lon","layer"],averaging_kernel) 
                                 },
                                 coords={"lon":lon_mod, "lat": lat_mod, "time":dates_out})
        ds_out.attrs["Comments"]='XCH4 mean at 0.25x0.3125'
        
        if satellite == "TROPOMI":
            xch4_precision = ds_obs.XCH4_precision
            albedo = ds_obs.SWIR_albedo
            
            ds_out["XCH4_precision"] = (["time", "lat", "lon"],xch4_precision)
            ds_out["pressure_interval"] = (["time", "lat", "lon"],dp_sat)
            ds_out["SWIR_albedo"] = (["time", "lat", "lon"],albedo)
            ds_out["dry_air_subcolumn"] = (["time", "lat", "lon", "layer"],air_subcolumn)
            
            ds_out["XCH4_precision"].attrs = xch4_precision.attrs
            ds_out["pressure_interval"].attrs = dp_sat.attrs
            ds_out["SWIR_albedo"].attrs = albedo.attrs
            ds_out["dry_air_subcolumn"].attrs = air_subcolumn.attrs
            
            ds_out["pressure_interval"].attrs["units"] = "hPa"
            
            ds_out.attrs["qa_threshold"] = 0.5
        
        elif satellite == "GOSAT":
            xch4_uncert = ds_obs.XCH4_uncertainty
            retr_flag = ds_obs.retr_flag
            ds_out["XCH4_uncertainty"] = xch4_uncert
            ds_out["pressure_weight"] = p_wgt_sat
            ds_out["pressure_levels"] = plevels_sat
            ds_out["retr_flag"] = retr_flag
            
            ds_out["XCH4_uncertainty"].attrs = xch4_uncert.attrs
            ds_out["pressure_weight"].attrs = p_wgt_sat.attrs
            ds_out["pressure_levels"].attrs = plevels_sat.attrs
            ds_out["retr_flag"].attrs = retr_flag.attrs
        
        #for varname in varnames_short:
        #    ds_out[varname]  = (["time", "lat", "lon"],xch4_column[varname])
            
        for varname in varnames_short:
            xch4_array = np.vstack(xch4_column_time[varname])
            ds_out[varname]  = (["time", "lat", "lon"],xch4_array) 
        
        
        
        ds_out["XCH4_obs"].attrs = xch4_obs.attrs
        ds_out["surface_pressure"].attrs = psurf_sat.attrs
        ds_out["ch4_profile_apriori"].attrs = ch4_ap.attrs
        ds_out["ch4_averaging_kernel"].attrs = averaging_kernel.attrs
        
        ds_out["surface_pressure"].attrs["units"] = "hPa"
        
        
        for key in list(ds_out.keys()):
                ds_out[key].encoding['zlib'] = True                    
        ds_out.to_netcdf(path=fname_out, mode='w')
