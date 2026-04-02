#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday April 2 16:46:42 2024

@author: bellemva
"""

name_experiment = 'config_QGSW_notime'

path_data = "../.." # Change path according to where data are saved
path_save = "../.." # Change path where to save files (observation, observational operator, control vectors)

#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime,timedelta
from math import pi

#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = name_experiment, # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = name_experiment, # name of output files

    path_save = f"{path_data}/data/mapping_outputs/{name_experiment}", # path of output files

    tmp_DA_path = f"{path_save}/tmp/DA/{name_experiment}", # temporary data assimilation directory path

    flag_plot = 0, # between 0 and 4. 0 for none plot, 4 for full plot

    init_date = datetime(2012,5,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2012,7,31,23),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=1),  # assimilation time step (corresponding to observation update timestep)

    saveoutput_time_step = timedelta(hours=1),  # time step at which the states are saved 

    plot_time_step = timedelta(days=1),  #  time step at which the states are plotted (for debugging),

    time_obs_min = None, 

    time_obs_max =  None,

    write_obs = True, # save observation dictionary in *path_obs*

    compute_obs = False, # force computing observations 

    path_obs = f"{path_save}/tmp/obs/", # if set to None, observations are saved in *tmp_DA_path*

    coriolis_force = True, # if set to False, coriolis force is set to 0 (for idealized case for instance)

    n_workers = 10 # number of workers to parallelize experiment preparation (like Obsop, ...)

)

#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_GEO',

    lon_min = 190.,                                         # domain min longitude

    lon_max = 200,                                         # domain max longitude

    lat_min = 20.,                                          # domain min latitude

    lat_max = 30.,                                          # domain max latitude

    dlon = 1/16,                                           # zonal grid spatial step (in degree)

    dlat = 1/16,

    name_init_mask = None, 

    name_var_mask = {'lon':'longitude','lat':'latitude','var':'ssh_it1'}

)

#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = ['myMOD_QG','myMOD_SW']

myMOD_QG = dict(

    super = "MOD_QG1L_JAX",

    name_var = {'SSH':'ssh_bm'},

    dtmodel = 300, # model timestep

    init_from_bc = True,

    time_scheme = 'Euler',

    c0 = 2.7,

)

myMOD_SW = dict(

    super = 'MOD_SW1L_JAX',

    name_var = {'U':'u_it', 'V':'v_it', 'SSH':'ssh_it'},

    name_params = {'HE':'He', 'HE_OFFSET':'He_offset', 'HBCX':'hbcx','HBCY':'hbcy','ITG':'itg'},

    var_to_save = ['SSH'],

    dtmodel = 300, # model timestep

    time_scheme = 'rk4', # Time scheme of the model (e.g. Euler,rk4)

    bc_island = "dirichlet", # Either "dirichlet" (orthogonal velocity forced to zero) or "radiative" (dissipative boundaries)

    bc_kind = '1d', # Either 1d or 2d

    w_waves = [2*pi/(12.42060121*3600),2*pi/(12.*3600),2*pi/(12.65834751*3600)], # igw frequencies (in seconds)

    w_names = ["m2","s2","n2"], 

    He_init = 0.95, # Mean height (in m)

    Ntheta = 2, # Number of angles (computed from the normal of the border) of incoming waves,

    g = 9.81,

    path_bathymetry = f"../aux/Bathymetry_hawaii.nc", # path to read bathymetry netcdf file.   

    name_var_bathy = {'lon':'lon','lat':'lat','var':'elevation'},

    smooth_wavelength = 36000, # wavelength for the smoothing of bathymetry (in meters), if None no smoothing is applied 

    path_tidal_velocity = "../aux/FES_tide",

)

#################################################################################################################################
# BOUNDARY CONDITIONS
#################################################################################################################################
NAME_BC = 'myBC'

myBC = dict(

    super = 'BC_EXT',

    file = f'{path_data}/data/OSSE/lowpass_ref_bm/*.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'longitude',

    name_lat = 'latitude',

    name_var = {'SSH':'ssh'}, # name of the boundary conditions variable

)

#################################################################################################################################
# OBSERVATIONAL OPERATORS
#################################################################################################################################
NAME_OBSOP = ["myOBSOP_Nadirs", "myOBSOP_SWOT"]

myOBSOP_Nadirs = dict(

    super = 'OBSOP_INTERP_L3_JAX',

    path_save = f"{path_save}/tmp/obsop/obsop_nadirs", # Directory where to save observational operator

    name_obs = ['ALG','C2','J3','S3A','S3B','SWOT_NADIR'],

    write_op = True, # Write operator data to *path_save*

    compute_op = False, # Force computing H 

    Npix = 4, # Number of pixels to perform projection y=Hx

    mask_coast = False,

    mask_borders = False,

    normalize_misfit = False # normalizing misfit by the number of observations 

)

myOBSOP_SWOT = dict(

    super = 'OBSOP_INTERP_L4',

    path_save = f"{path_save}/tmp/obsop/obsop_swot", # Directory where to save observational operator

    name_obs = ['SWOT'],

    write_op = True, # Write operator data to *path_save*

    compute_op = False, # Force computing H 

    mask_borders = False,

    interp_method = 'nearest' # either 'nearest', 'linear', 'cubic' (use only 'cubic' when data is full of non-NaN)

)


#################################################################################################################################
# REDUCED BASIS 
#################################################################################################################################
NAME_BASIS = ['myBASIS_BM','myBASIS_He','myBASIS_He_offset','myBASIS_HBC','myBASIS_ITG']

myBASIS_BM = dict(

    super = 'BASIS_BMaux_JAX',

    name_mod_var = 'ssh_bm', # Name of the related model variable 
    
    flux = False, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    facns = 1., #factor for wavelet spacing in space

    facnlt = 2., #factor for wavelet spacing in time

    npsp = 3.5, # Defines the wavelet shape

    facpsp = 1.5, # factor to fix df between wavelets

    file_aux = f"../aux/aux_reduced_basis_BM.nc", # Name of auxilliary file in which are stored the std and tdec for each locations at different wavelengths.

    lmin = 80, # minimal wavelength (in km)

    lmax = 900., # maximal wavelength (in km)

    factdec = 7.5, # factor to be multiplied to the computed time of decorrelation 

    tdecmin = 2., # minimum time of decorrelation 

    tdecmax = 20., # maximum time of decorrelation 

    facQ = 1, # factor to be multiplied to the estimated Q

    norm_time = True

)

myBASIS_He = dict(

    super = "BASIS_GAUSS3D_JAX",

    name_mod_var = 'He', # Name of the related model variable 

    flux = True, # Whether making a component signature in space appear/disappear in time. For dynamical mapping, use flux=False

    time_dependant = False,
    
    facns = 3., #factor for wavelet spacing in space
    
    facnlt = 3., #factor for wavelet spacing in time
    
    sigma_D = 970, # Spatial scale (km)
    
    sigma_T = 25, # Time scale (days)

    fcor = 1, 
    
    sigma_Q = 3*0.005, # Standard deviation for matrix Q
    
    normalize_fact = False,

)

myBASIS_He_offset = dict(

    super = "BASIS_OFFSET",

    name_mod_var = "He_offset",

    sigma_B = 0.05,

)

myBASIS_HBC = dict(

    super = "BASIS_HBC",

    name_params = ['hbcx', 'hbcy'], # list of parameters to control (among 'He', 'hbcx', 'hbcy', 'itg')

    ### COMMON PARAMETER ### 

    facns = 3.5, # factor for gaussian spacing in space

    facnlt = 2.5, # factor for gaussian spacing in time 

    time_dependant = False,

    ### - HBC PARAMETER ### 

    sigma_B_bc = 2e-2, # Background variance for bc

    D_bc = 300, # Space scale of gaussian decomposition for boundary conditions (in km)

    T_bc = 20, # Time scale of gaussian decomposition for boundary conditions (in days)

    Nwaves = 3, # igw frequencies (in seconds)

    Ntheta = 2, # Number of angles (computed from the normal of the border) of incoming waves,

)

myBASIS_ITG = dict(

    super = 'BASIS_GAUSS_ITG',

    name_mod_var = 'itg', # Name of the related model variable 

    facns = 3.5, # Factor for gaussian spacing in space

    facnlt = 1., # Factor for gaussian spacing in time

    D_itg = 300, # Spatial scale (km)

    sigma_Q = 3.5*5e-5, # Standard deviation for matrix Q 

    Nwaves = 3 # number of tidal components 

)

#################################################################################################################################
# Analysis parameters
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(

    super = 'INV_4DVAR',

    compute_test = False, # TLM, ADJ & GRAD tests

    path_init_4Dvar = None,

    restart_4Dvar = False, 

    ftol = 1e-3, # Cost function value norm must be less than ftol before successful termination.

    n_consecutive = 100, # Number of consecutive iterations over which ftol should be fulfilled

    maxiter = 1000, # Maximal number of iterations for the minimization process

    opt_method = 'L-BFGS-B', # method for scipy.optimize.minimize

    save_minimization = True, # save cost function and its gradient at each iteration 

    timestep_checkpoint = timedelta(hours=6), #  timesteps separating two consecutive analysis 

    sigma_R = 1E-1, # Observational standard deviation

    prec = True, # preconditioning,

)


#################################################################################################################################
# OBSERVATIONS
#################################################################################################################################
NAME_OBS = ['SWOT','ALG','C2','J3','S3A','S3B','SWOT_NADIR']

SWOT = dict(

    super = 'OBS_SSH_SWATH',

    path = f'{path_data}/data/OSSE/obs/dc_obs_swot/SSH_SWOT_2012-0*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',

    name_var = {'SSH':'ssh'},

    concat_dim = "num_lines",
    
    varmax = None,

    sigma_noise = 0.7869967393890203

)

ALG = dict(

    super = 'OBS_SSH_NADIR',

    path = f'{path_data}/data/OSSE/obs/dc_obs_nadirs/alg/SSH_NADIR_2012-0*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    varmax = None,

    sigma_noise = 0.2130032606109797

)

C2 = dict(

    super = 'OBS_SSH_NADIR',

    path = f'{path_data}/data/OSSE/obs/dc_obs_nadirs/c2/SSH_NADIR_2012-0*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    varmax = None,

    sigma_noise = 0.2130032606109797

)

J3 = dict(

    super = 'OBS_SSH_NADIR',

    path = f'{path_data}/data/OSSE/obs/dc_obs_nadirs/j3/SSH_NADIR_2012-0*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    varmax = None,

    sigma_noise = 0.2130032606109797

)

S3A = dict(

    super = 'OBS_SSH_NADIR',

    path = f'{path_data}/data/OSSE/obs/dc_obs_nadirs/s3a/SSH_NADIR_2012-0*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    varmax = None,

    sigma_noise = 0.2130032606109797

)

S3B = dict(

    super = 'OBS_SSH_NADIR',

    path = f'{path_data}/data/OSSE/obs/dc_obs_nadirs/s3b/SSH_NADIR_2012-0*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    varmax = None,

    sigma_noise = 0.2130032606109797

)

SWOT_NADIR = dict(

    super = 'OBS_SSH_NADIR',

    path = f'{path_data}/data/OSSE/obs/dc_obs_nadirs/swot/SSH_NADIR_2012-0*.nc',

    name_time = 'time',
    
    name_lon = 'longitude',

    name_lat = 'latitude',
    
    name_var = {'SSH':'ssh'},

    varmax = None,

    sigma_noise = 0.2130032606109797

)

#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = None
