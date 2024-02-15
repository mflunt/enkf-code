# enkf-code
Code to set up ensemble runs for GC model

Contains:
  ./
  
  inputs.ini - Config file to specify all relevant inputs
  run_GC.py - Python script that reads in variables from input.ini and sets up input files for GEOS-Chem runs
  GC_setup_mod.py - Module file containing various functions that is called by run_GC.py
  

  
  ./templates/
  
  input_file_v12.template - Template file for input.geos. Used to set up run inputs
  HEMCO_Config_CH4_v12.template - Template file for HEMCO_Config.rc
  HISTORY_rc_v12.template - Template file for History.rc
  HEMCO_Diagn_CH4_v12.template - Template file for HEMCO_Diagn.rc
  
  To Run:
  1. Edit variables in inputs.ini file - particularly path directories
  2. In python run "run_GC.py" (no input arguments needed)
  3. This will create all run directories and files and outpout directories. You will need to manually set the runs going, but      I am still working on that part.
