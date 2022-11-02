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
import gchem
import xbpch
import glob
from datetime import datetime

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

def read_netcdfs(files, dim = "time"):
    '''
    Use xray to open sequential netCDF or bpch files. 
    Makes sure that file is closed after open_dataset call.
    '''
    
    datasets = [open_ds(p) for p in sorted(files)]
    combined = xarray.concat(datasets, dim)
    
    return combined 

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
    
def create_initial_ensemble(N_ens, nstate0, nBC, ntime, dates, lats_land,lons_land,
                            sigma_x_land, sigma_x_bc, 
                            fname_ens=None, pseudo=False):
    """
    Create an random ensemble array
    
    nstate0 = nland+nBC
    """
        
    x_ens_temp = np.random.normal(loc=1., scale=sigma_x_land, size=(ntime, nstate0, N_ens))
    
    # Make sure ensemble mean of each column is 1
    
    mean_xi  = np.mean(x_ens_temp,axis=2)
    x_ens = x_ens_temp -mean_xi[:,:,None] + 1.
    
    # ake sure all values are +ve
    wh_neg = np.where(x_ens < 0.)
    if np.sum(wh_neg)>0:
        x_ens[wh_neg] = np.random.uniform(low=0.01,high=0.5)
    
    if pseudo == True:
        # Perturb NH and SH values for pseudo case
        
        wh_nh = np.where(lats_land >=0)[0]
        wh_sh = np.where(lats_land < 0)[0]
        x_ens[:,wh_nh,:] = x_ens[:,wh_nh,:] +0.2
        x_ens[:,wh_sh,:] = x_ens[:,wh_sh,:] -0.2
    
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
        ds_ensemble['time'] = dates
        ds_ensemble["nBC"] = nBC
        ds_ensemble["nland"] = nstate0-nBC
        ds_ensemble["sigma_x_land"] = sigma_x_land
        ds_ensemble["sigma_x_bc"] = sigma_x_bc
        ds_ensemble.to_netcdf(path=fname_ens, mode="w")
    
    
    return x_ens, x_region_out, x_time_out


def write_scale_factor_file(dates_out, lat,lon,scale_map_ensemble, spc_prefix, fname_out):
    """
    Write scale factor netcdf file for GC run. 
    
    """
    ds_out = xarray.Dataset()
    for label in spc_prefix:
        
        ds_out[label] = (("time", "lat","lon"), scale_map_ensemble[label])
        
    ds_out.coords['lat'] = lat
    ds_out.coords['lon'] = lon
    #ds_out.coords['time'] = [pandas.to_datetime("2010-01-01")]
    ds_out.coords['time'] = dates_out
    
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


def write_input_file(start_date, end_date, spc_names, bc_dir, spc_BC,
                     run_dir, output_dir, template_dir, restart_fname, 
                     global_res, post_run=False):
    """
    Write the input.geos file for each assimilation run. 
    Length of run needs to be for whole of lag period.
    
    Inputs:
        start_date (string): "YYYYMMDD" start of emissions lag period
        end_date (string): "YYYYMMDD" End of obs assimilation window
        N_ens (int): NUmber of ensemble members
        bc_dir (string): Directory where BC files are kept.
        spc_BC (list): list of boundary variables to include. - Not neededI don't think
        
    Requires:
        Template file - input_ensemble.template - to be setup correctly. 
    """

    if global_res == "2x2.5":
        T_F_res = "T"
    else:
        T_F_res = "F"
    #pd_start = pandas.to_datetime(start_date)
    pd_end = pandas.to_datetime(end_date)
    
    
#    run_dir = "/geos/d21/mlunt/gc11_rundirs/" + run_name + "/"
#    template_dir = "/home/mlunt/programs/GC/rundirs/templates/EnKF/"
#    output_dir = "/geos/d21/mlunt/GC_output/" + run_name + "/"
    
    #bc_dir = "/geos/d21/mlunt/GC_output/liang_BC_4x5/perturbed/"
    # Perturbed BC shoudl include BC_N E S W
    
    if type(spc_BC) == list:
        spc_IC = ["CH4"] + spc_BC
    else:
        spc_IC = ["CH4"]
        
    #spc_IC = ["CH4", "CH4IC", "CH4BC_N", "CH4BC_S", "CH4BC_NE",  "CH4BC_SE","CH4BC_SW", "CH4BC_NW"] 
    

#    spc_prefix=[]
#    for xi in range(N_ens):    
#        spc_prefix.append("CH4_E" + str(xi+1))
    
    
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
    
    
   # if period_unit == "d":
   #     pd_end = pd_start + pandas.Timedelta(lag_period, unit='d') 
   # elif period_unit == "m":
   #     pd_end = pd_start + relativedelta(months=lag_period)
   # end_date = pd_end.strftime('%Y%m%d')
    
    
    if pd_end.month in ([1,3,5,7,8,10,12]):
        days_in_out_month = 31
    elif pd_end.month in ([4,6,9,11]):
        days_in_out_month = 30
    else:
        days_in_out_month = 29
    
    string_list=[]
    for ti in range(days_in_out_month):
        string_list.append('0')
        
    string_list[pd_end.day-1] = '3'    
    
    string_30s = "".join(string_list)
    
    
    # Define restart file name
    #restart_filename = "./restart_file." + start_date + "0000.nc"
    #restart_filename = output_dir+ "restarts/GEOSChem_restart." + start_date + "0000.nc"
    
    # define diagnostic outfile name
    diagn_outfile_name = output_dir + "trac_avg.merra2_05x0625_CH4.YYYYMMDDhhmm"
    
    # define satellite outfile name
    sat_outfile_name = output_dir + "ts_satellite.YYYYMMDD.bpch"  # Fine to leave thi as YYMMDD as GC works this out.
    
    #%%
    #####################################################################################
    # 1. Need to define advected species i .e include species from any previous months
    if type(spc_names) == list:
        spc_advect = spc_IC + spc_names
    else:
        spc_advect = spc_IC 
    nAdvect = len(spc_advect)  # Have to remeber to add 2 for CH4 and CH4REF
    
    # When writing to file need to loop through nAdvect and write:
    # Species #si              : spc_name[si]
    
    # 3. Define tracer number list to write 
    
    tracer_list = "162 "    # 162 = pressure fields
    for si in range(nAdvect):
                    tracer_list = tracer_list + str(si+1)+ " "    
    

    #%%
    # Read in template input file
    # Read in the file
    #with open(template_dir + "input.template", 'r') as file :
    #  filedata = file.read()
    ## Replace the target string
    
    str_len=len("Species Entries ------->")      
           
     
    with open(template_dir + "input_ensemble.template", "r") as in_file:
        filedata = in_file.readlines()
    
    with open(run_dir + "input.geos", "w") as out_file:
        for line in filedata:
            
            if "StartDate" in line: 
                line = line.replace('StartDate', start_date)
            if "EndDate" in line:
                line = line.replace('EndDate', end_date)          
            if "RestartFileName" in line:
                line = line.replace('RestartFileName', restart_fname)        
            if "NumberAdvectedSpecies" in line:
                line = line.replace('NumberAdvectedSpecies', str(nAdvect))
            if "DiagnosticOutputFile" in line:
                line = line.replace('DiagnosticOutputFile', diagn_outfile_name)     
            if "SatelliteOutputFile" in line:
                line = line.replace("SatelliteOutputFile", sat_outfile_name )
            if "Input BCs at 2x2.5?" in line:
                line = line.replace("BC_RES", T_F_res)
            if "TPCORE AS BC directory" in line:
                line = line.replace("BC_DIRECTORY", bc_dir )
            
            if "SatelliteTracers" in line:
                line = line.replace("SatelliteTracers", tracer_list )
            
            if "Species Entries ------->: Name" in line:    
                for si, species in enumerate(spc_advect):
                    dummy_str = "Species #"+str(si+1)
                    line = line + dummy_str.ljust(str_len) + ": " + species + "\n"  
                    #line = line + "Species #"+str(si+1)+"              : " + species + "\n"    
            
            if "OUTPUT_DATES" in line:
                if "Schedule output for " + month_index[pd_end.month] in line:
                    line = line.replace("OUTPUT_DATES", string_30s)
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

#run_name='test_masks_05x0625_CA'
#assim_window = 10
#lag_period = 50
#period_unit="d"
#start_date = "20140101"
#N_ens = 50

def write_hemco_config(start_date, end_date, spc_emit, fname_scale_factors,
                       run_dir, template_dir, post_run=False):
    """
    Write the HEMCO_Config.rc file for each assimilation run. 
    
    Inputs:
        start_date (string): "YYYYMMDD" start of emissions lag period
        end_date (string): "YYYYMMDD" End of obs assimilation window
        spc_emit (N_ens): Name of ensemble members
        fname_Scale_factors (string): Full path of scale factor file
        run_dir (string): Directory where run code is located.
        template_dir (string): Directory where template file is located
        
    Requires:
        Template file - input_ensemble.template - to be setup correctly. 
    """
    #run_dir = "/geos/d21/mlunt/gc11_rundirs/" + run_name + "/"
    #template_dir = "/home/mlunt/programs/GC/rundirs/templates/EnKF/"
    
    #scale_factor_file = "/geos/d21/mlunt/gc11_rundirs/" + run_name + "/scale_factors/"
    
    spc_fixed = ["CH4"]
    
#    spc_emit=[]
#    for xi in range(N_ens):    
#        spc_emit.append("CH4_E" + str(xi+1))
    
    
    pd_start = pandas.to_datetime(start_date)
    pd_end = pandas.to_datetime(end_date)
    
    start_year = pd_start.year
    end_year = pd_end.year
    
    # define diagnostic outfile name
    diagn_infile_path = run_dir + "MyDiagnFile.rc" 
    
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
    with open(template_dir + "HEMCO_Config_ensemble.template", "r") as in_file:
        filedata = in_file.readlines()
    
    with open(run_dir + "HEMCO_Config.rc", "w") as out_file:
        for line in filedata:
            
            if "HemcoDiagnosticFileName" in line: 
                line = line.replace("HemcoDiagnosticFileName", diagn_infile_path)
            if  "HemcoDiagnosticPrefix" in line:
                line = line.replace("HemcoDiagnosticPrefix", diagn_prefix)          
            if "HemcoDiagnosticFreq" in line:
                line = line.replace("HemcoDiagnosticFreq", diagn_freq)   
                        
            if "SpeciesList" in line:
                line = line.replace("SpeciesList", spc_list )
            if type(spc_emit) == list:
                # Only need to do this if I have tagged species
                if "--> fraction POG1" in line:    
                        for si, species in enumerate(spc_emit):
                            if post_run == True:
                                spc_no = str(si+1)
                            else:
                                spc_no = species[5:]
                            line = line + "    --> ScaleField_" + species +"     :    GFED_SCALEFIELD_"+spc_no+"\n"
                
                if "0 CH4_GAS__1B2a" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_GAS__1b2a_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 1 1\n"    
                    
                if "0 CH4_GAS__1B2b" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_GAS__1b2b_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 1 1\n"    
                    
                #if "0 CH4_COAL__1B1" in line:
                #    for si, species in enumerate(spc_emit):
                #        line = line + "0 CH4_COAL__1b1_"+ str(si)+"   -   -    -   -  -   -  " + \
                #        species + " " + mask_no[species] + " 2 1\n"    
                              
                if "0 CH4_LIVESTOCK__4A" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_LIVESTOCK__4a_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 2 1\n"    
                
                if "0 CH4_LIVESTOCK__4B" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_LIVESTOCK__4b_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 2 1\n"    
                       
                if "0 CH4_WASTE__6A_6C" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_WASTE__6a6c_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 3 1\n"    
                        
                if "0 CH4_WASTE__6B" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_WASTE__6b_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 3 1\n"    
                        
                if "0 CH4_BIOFUEL__1A4" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_BIOFUEL__1a4_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 4 1\n"   
                
                if "0 CH4_OTHER__1A1_1A2" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_OTHER__1a1a2_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 5 1\n"   
                        
                if "0 CH4_OTHER__1A3a_c_d_e" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_OTHER__1a3_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 5 1\n"   
                        
                if "0 CH4_OTHER__1A3b" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_OTHER__1a3b_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 5 1\n"   
                        
                if "0 CH4_OTHER__2" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_OTHER__2_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 5 1\n"   
                        
                if "0 CH4_OTHER__7A" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_OTHER__7a_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 5 1\n"   
                        
                if "0 CH4_RICE__4c_4d" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_RICE__4c4d_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 6 1\n"   
                        
                if "0 CH4_WETLANDS" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_WETLANDS_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + " 7 1\n"   
                        
                if "0 CH4_SOILABSORBTION" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_SOILABSORBTION_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species] + "/1" + " 8 1\n"   
                        
                if "0 CH4_TERMITES" in line:
                    for si, species in enumerate(spc_emit):
                        line = line + "0 CH4_TERMITES_"+ str(si)+"   -   -    -   -  -   -  " + \
                        species + " " + mask_no[species]  + " 9 1\n"   
                        
        # Need to write this line to file for each species - omit number from species, just write letters
                        
                if "## Insert scalefields here ##" in line:
                    for si, species in enumerate(spc_emit):
                        if post_run == True:
                            spc_no = str(si+1)
                        else:
                            spc_no = species[5:]
                        line = line + "\n" + "111 GFED_SCALEFIELD_" + spc_no + \
                        "   1  -  2010/1/1/0 C xy unitless * " + mask_no[species] + " 1 1"
                    
        
        # Need to write in the mask definitions and ensure that numbers correspond with definitions above
        
                if "## Insert scale factors here ##" in line:
                    for si, species in enumerate(spc_emit):
                        if post_run == True:
                            spc_no = str(si+1)
                        else:
                            spc_no = species[5:]
                      
                        if start_year == end_year:
                            line = line + mask_no[species] + "  ENSEMB_" + spc_no + "  " \
                            + fname_scale_factors + "   " + species + "  " + \
                            str(start_year) + "/1-12/1-31/0 RF xy 1 1 " + "\n"
                        else:
                            line = line + mask_no[species] + "  ENSEMB_" + spc_no + "  " \
                            + fname_scale_factors + "   " + species + "  " + \
                            str(start_year) + "-" + str(end_year) + "/1-12/1-31/0 RF xy 1 1 " + "\n"
                        
                    # RF  = Range forced. HEMCO will raise an error if simulation dates outside of the sttated range.

            out_file.write(line)
    
    print("Successfully written file " + run_dir + "HEMCO_Config.rc")
            
    #%%
    # Write diagnostic HEMCO file to specify emissions totals    
    
    # Probably want to write sectors to output as well. Particularly GFED and wetlands. 
    # Other sectors don't matter as much as can work them out.
    
    with open(diagn_infile_path, "w") as out_file2:
        
        if type(spc_emit) == list:
            nlines  = len(spc_emit)+2
        else:
            nlines = 2
        
        for ti in range(nlines):
            
            if ti ==0:
                line = "# Name        Spec    ExtNr Cat Hier Dim OutUnit \n"
    
            elif ti ==1: 
                line = "CH4" + "     " +  "CH4" + "  -1   -1  -1   2   kg \n"  
            else: 
                line = spc_emit[ti-2] + "     " +  spc_emit[ti-2] + "  -1   -1  -1   2   kg \n"  
    
            out_file2.write(line)
            
            
def write_bc_ensemble_files(start_date, end_date,window_start_dates, bc_scalings, ensemble_names, 
                            bc_dir, out_dir, bc_split = "North-South", global_res = "2x2.5", region="AFRICA",
                            species = "CH4", ic_spc_names=None):
    """
    Write BC input files for ensemble GEOS-Chem runs.
    
    Easiest way then is to read the CH4 field from global run. Then:
        1) Apply scaling factors of 1 for new fields
        2) apply oter scaling factos for older fields.
    
    bc scalings =  dictionary of arrays of size (nwindows, nBC) e.g. if north-south = (nwindows, 2)
    
    """
    tracer_file = bc_dir + "tracerinfo.dat"
    diag_file = bc_dir + "diaginfo.dat"
    
    #fname = "/home/mlunt/datastore/GC_output/trac_avg.geosfp_4x5_CH4.201301010000"
    bc_str = "BC."
    
    # Find list of file names that span date range of interest
    files = filenames(bc_dir, bc_str, start=start_date, end=end_date, freq="D")
    #fname = bc_dir + "BC.20140102"
    # No loop through all files and create new daily files in turn:
    if global_res=="2x2.5":
        resolution = (2.5,2.0)   
        if region == "AFRICA":
            origin = (65,33,1)  # origin for 2x2.5
               
    elif global_res == "4x5":
            resolution = (5.0,4.0)
            if region == "AFRICA":
                origin = (33,17,1)  
                
    
    for bc_file in files:
    
        ds = open_bpch_ds(bc_file, tracer_file, diag_file)
        
        date_str = bc_file[-8:] # Assumes BC files end in YYYYMMDD
        
        #date_str2 = bc_file[-8:-4] + "04" + bc_file[-2:]
        # Only want to perturb the CH4 field I think. Want to leave CH4IC as is. -  Migth need to think about this though.
        
        # Check resolution of BC files matches definition.
        if ds.res != resolution:
            raise ValueError("BC input resolution does not match specified global resolution")
        
        # Only apply this correction if needed - need to check somehow...
        if species == "CH4":
            if np.max(ds.IJ_AVG_S_CH4) > 1800:
                ch4_field = ds.IJ_AVG_S_CH4/28.97*16.04
            else:
                ch4_field = ds.IJ_AVG_S_CH4
                
        
        #ch4_field = ds.IJ_AVG_S_CH4/28.97*16.04    # (time, lon,lat,lev) Need to apply correction to make mass mixing ratio
        nlat= len(ch4_field[0,0,:,0])
        ch4_bc_dict={"CH4": ch4_field.values}
        
        if ic_spc_names != None:
            for spc in ic_spc_names:
                ch4_bc_dict[spc] = ch4_field.values
        
        # Split into diferent ensemble time periods:
        # Need to find between which window start dates the current date is. 
        time_diff_min = 1000
        time_diff_min_id = -999
        for ti, wind_start in enumerate(window_start_dates):
            if type(wind_start) == str:
                time_diff = (pandas.to_datetime(date_str) - pandas.to_datetime(wind_start)).days
            else:
                time_diff = (pandas.to_datetime(date_str) - wind_start).days
        
            if time_diff < time_diff_min and time_diff >= 0:
                time_diff_min = time_diff
                time_diff_min_id = ti
        
        if bc_split == "North-South":
            
            for spc in ensemble_names:
                ch4_ens_field = ch4_field.values.copy()*0.
            
                # North
                ch4_ens_field[:,:, nlat//2:,:] = (ch4_field[:,:,nlat//2:,:].values
                             *bc_scalings[spc][time_diff_min_id,0])
                # South             
                ch4_ens_field[:,:, :nlat//2,:] = (ch4_field[:,:,:nlat//2,:].values
                             *bc_scalings[spc][time_diff_min_id,1])
                             
                ch4_bc_dict[spc] = ch4_ens_field
                
        elif bc_split == "None":
            for spc in ensemble_names:
                ch4_bc_dict[spc] = ch4_field.values.copy()

        
#        # For MERRA2 - because 3x0.5 > 1 degree. For GEOSFP 3x0.25 < 1 degree so not a problem.
#        # For 2x2.5:
#            if global_res == "2x2.5":
#                ch4_bc_n[:,:,-2:,:] = ch4_field[:,:,-2:,:]*1.
#                ch4_bc_s[:,:,:2,:] = ch4_field[:,:,:2,:]*1.
##                ch4_bc_ne[:,-2:,nlat//2:-2,:] = ch4_field[:,-2:,nlat//2:-2,:]*1.
##                ch4_bc_se[:,-2:,2:nlat//2,:] = ch4_field[:,-2:,2:nlat//2,:]*1.
##                ch4_bc_sw[:,:2,2:nlat//2,:] = ch4_field[:,:2,2:nlat//2,:]*1
##                ch4_bc_nw[:,:2,nlat//2:-2,:] = ch4_field[:,:2,nlat//2:-2,:]*1.
#            else:
#        # For 4x5:
#                ch4_bc_n[:,1:-1,-1,:] = ch4_field[:,1:-1,-1,:]*1.
#                ch4_bc_s[:,1:-1,0,:] = ch4_field[:,1:-1,0,:]*1.
##                ch4_bc_ne[:,-1, nlat//2:,:] = ch4_field[:,-1,nlat//2:,:]*1.
##                ch4_bc_se[:,-1,:nlat//2,:] = ch4_field[:,-1,:nlat//2,:]*1.
##                ch4_bc_sw[:,0,:nlat//2,:] = ch4_field[:,0,:nlat//2,:]*1
##                ch4_bc_nw[:,0,nlat//2:,:] = ch4_field[:,0,nlat//2:,:]*1.
#        
        
        # Now write each tracer to file. CH4, CH4IC, ch4_bc_{nesw} - 6 in total.
        
        # create bpch file
        if ic_spc_names != None:
            names = ["CH4"] + ic_spc_names + ensemble_names
        elif ensemble_names !=None:
            names = ["CH4"] + ensemble_names
        else:
            names=["CH4"]
            
        fullnames=[]
        for name in names:
            fullnames.append(name+"_tracer")
#        fullnames = ["CH4 tracer", "CH4IC tracer", "CH4BC_N tracer", "CH4BC_S tracer", "CH4BC_NE tracer", 
#                     "CH4BC_SE tracer", "CH4BC_SW tracer", "CH4BC_NW tracer"]
        n_tracers=len(names)
        tracer_numbers = range(1,n_tracers+1)
        
    #    n_tracers=1
    #    tracer_numbers = range(1,n_tracers+1)
        
        
#        ch4_bc_dict={"CH4": ch4_field.values, 
#                     "CH4IC": ch4_field.values}
#        if bc_split == "North-South":
#                ch4_bc_dict["CH4BC_N"] = ch4_bc_n
#                ch4_bc_dict["CH4BC_S"] = ch4_bc_s
#                     "CH4BC_N": ch4_bc_n,
#                     "CH4BC_S": ch4_bc_s,
#                     "CH4BC_NE": ch4_bc_ne,
#                     "CH4BC_SE": ch4_bc_se,
#                     "CH4BC_SW": ch4_bc_sw,
#                     "CH4BC_NW": ch4_bc_nw,     
#                     
        
       
        
        #dates_temp = ds.time_bnds.values
        ntime = len(ds.time)
        dates=[]
        
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:])
        
        for ti in range(ntime):
            #dates.append(pd.Timestamp(dates_temp[ti,0]) - relativedelta(months=1))
        
            dates.append(datetime(year,month,day, 3*ti))
        # The date is the problem. Could it be that pandas timestamp is the issue?
        # What if I use datetime.datetime instead?
        
        shape = ds.IJ_AVG_S_CH4[0,:,:,:].shape   # Ignore time dimension
        #values = np.ones((21,17,47,8))
        
        datablocks = []
        for xi,number in enumerate(tracer_numbers):
            
            values = ch4_bc_dict[names[xi]]
            for ti in range(8):
                datablocks.append(
                    gchem.bpch.DataBlock(
                        category = 'IJ-AVG-$',
                        center180 = True,
                        halfpolar = True,
                        #modelname = 'GEOSFP_47L',
                        modelname = 'MERRA2_47L',
                        number = number,
                        resolution = resolution,
                        origin = origin,
                        shape = shape,  # Need to change this to shape of BC files
                        times = (dates[ti],dates[ti]), # Will need to be times in the BC file I think, suspect this needs to be of size (ntime,2)
                        unit = "1",
                        values = values[ti,:,:,:]/1.e9, # Presumably the values of the array I want to write.
                        name = names[xi],
                        fullname = fullnames[xi],
                        molecular_weight = 1.604e-2   # in kg/mole
                    )
                )
        
        
        f = gchem.bpch.File(datablocks=datablocks)
        
        
        #perturbed_dir = out_dir + "perturbed/"
      
        f.save_as(out_dir + bc_str + date_str)  # save BC file to disk
        
    print("Written BC files between " + start_date + " and " + end_date + " to disk." )
    
    return


def write_restart_file(input_file, ensemb_names, spc_post_names, fname_restart_out, emis_start, spc_copy="CH4P"):
    """
    Write restart file for ensemble runs
    
    Need to copy an input CH4 field.
    For first run this will be spinup field.
    
    After that will be field generated by ensemble mean - needs to be CH4P
    So need an option to select what species to copy.
    """
    
    ds = open_ds(input_file)
    ch4 = ds["SPC_"+spc_copy]
    
    ds_out = xarray.Dataset()
    
    if ensemb_names !=None:
        for name in ensemb_names:
            ds_out["SPC_"+name] = ch4
    if spc_post_names !=None:
        for name in spc_post_names:
            ds_out["SPC_"+name] = ch4
            
        
    ds_out["SPC_CH4"] = ch4
    ds_out.coords["time"]  = [pandas.to_datetime(emis_start)]
    ds_out.to_netcdf(path = fname_restart_out, mode='w')
    print ("Successfully written " + fname_restart_out + " to disk")
    return ds_out