import os
import pickle
import numpy as np
import netCDF4 as nc
from itertools import compress
import datetime as dt
from joblib import Parallel
import xarray as xr

#-------------------------------------------------------------

years=[2015]
results_folder = 'Stephan-2022-transport/'

# coordinates of boundaries for which to calculate fluxes:----
NSi = np.arange(1580,1630); NSj = 630; # Nares Strait
LSi = 1584; LSj = np.arange(496,534);  # Lancaster Sound
JSi = 1609; JSj = np.arange(554,583);  # Jones Sound

#-------------------------------------------------------------
# Load files:
mask  = nc.Dataset('/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mask.nc')  
umask = np.array(mask.variables['umask'])[0,:,:,:]
vmask = np.array(mask.variables['vmask'])[0,:,:,:]

mesh = nc.Dataset('/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mesh1.nc')
e1v_base  = np.array(mesh.variables['e1v'])[0,:,:]
e2u_base  = np.array(mesh.variables['e2u'])[0,:,:]
e3t       = np.array(mesh.variables['e3t_0'])[0,:,:,:]
e3v       = np.array(mesh.variables['e3v_0'])[0,:,:,:]
e3u       = np.array(mesh.variables['e3u_0'])[0,:,:,:]
depth     = np.array(mesh.variables['nav_lev'])

e1v = np.empty_like(e3t[:,:,:]); e1v[:] = e1v_base
e2u = np.empty_like(e3t[:,:,:]); e2u[:] = e2u_base

#-------------------------------------------------------------
def files_time_series(year, start_date, end_date):
    #start_date and end_date are datetime objects
    
    # Create list of filenames that fall within the start and end date time range:
    dyn_file_list = np.sort(os.listdir('/data/brogalla/ANHA12/'))
    
    Vlist = [i[26:31]=='gridV' for i in dyn_file_list]
    Ulist = [i[26:31]=='gridU' for i in dyn_file_list]
    
    gridV_list = list(compress(dyn_file_list, Vlist))
    gridU_list = list(compress(dyn_file_list, Ulist))
    
    dateV_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridV_list]
    dateU_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridU_list]
    
    gridV_file_list = list(compress(gridV_list, [V > start_date and V < end_date for V in dateV_list]))
    gridU_file_list = list(compress(gridU_list, [U > start_date and U < end_date for U in dateU_list]))
       
    return gridV_file_list, gridU_file_list

def main_calc(year, filenameU, filenameV): 
    
    # Load 5-day velocity file
    dyn_folder = '/data/brogalla/ANHA12/'
    file_u  = nc.Dataset(dyn_folder + filenameU)
    file_v  = nc.Dataset(dyn_folder + filenameV)
    u_vel   = np.array(file_u.variables['vozocrtx'])[0,:,:,:] 
    v_vel   = np.array(file_v.variables['vomecrty'])[0,:,:,:] 
    
    # For each of the boundaries, call function to calculate the flux:
    flx_LS = calc_flux(LSi, LSj, u_vel, v_vel)
    flx_NS = calc_flux(NSi, NSj, u_vel, v_vel)
    flx_JS = calc_flux(JSi, JSj, u_vel, v_vel)
    
    return  flx_LS, flx_NS, flx_JS

def calc_flux(i, j, u_vel, v_vel): 
    i = np.array(i)
    j = np.array(j)

    # horizontal boundary
    if i.size > j.size: 
        bdy_vel   = u_vel[:, i[0]:i[-1], j]
        area      =   e2u[:, i[0]:i[-1], j] * e3u[:,i[0]:i[-1],j]
        cond_mask = (umask[:,i[0]:i[-1],j] < 0.1)
        
    # vertical boundary
    else: 
        bdy_vel   = v_vel[:, i  , j[0]:j[-1]]
        area      =   e1v[:, i  , j[0]:j[-1]] * e3v[:,i,j[0]:j[-1]]
        cond_mask = (vmask[:,i,j[0]:j[-1]] < 0.1)
        
    # Point-wise multiplication with areas of each of the grid boxes:
    flx_V  = np.multiply(bdy_vel, area)
    
    # Mask the depth levels that correspond to points on land:
    flx_mask_V  = np.ma.masked_where(cond_mask, flx_V)
    
    return flx_mask_V

#-------------------------------------------------------------

for year in years:
    # Create list of five-day files:
    print(year)
    start_date = dt.datetime(year,1,1)
    end_date   = dt.datetime(year,12,31)
    gridV_files, gridU_files = files_time_series(year, start_date, end_date)
    print(len(gridV_files), len(gridU_files))
    
    # call the function for each file that is within range of start date, end date
    a = len(gridV_files)
    time_series_LS = np.empty((a, 50, LSj.size-1));  
    time_series_NS = np.empty((a, 50, NSi.size-1));  
    time_series_JS = np.empty((a, 50, JSj.size-1));  
    
    for f, V_file in enumerate(gridV_files):
         print(f)
         time_LS, time_NS, time_JS = main_calc(year, gridU_files[f], V_file)
         time_series_LS[f,:,:] = time_LS
         time_series_NS[f,:,:] = time_NS
         time_series_JS[f,:,:] = time_JS
    
    #Save time series to files:
    file_write = xr.Dataset(
        {'transport_LS': (("t","z","yLS"), time_series_LS),
         'transport_NS': (("t","z","xNS"), time_series_NS),
         'transport_JS': (("t","z","yJS"), time_series_JS)},
        coords = {
            "t": np.zeros(a),
            "z": depth, 
            "yLS": np.zeros(LSj.size-1),
            "xNS": np.zeros(NSi.size-1),
            "yJS": np.zeros(JSj.size-1)
        },
        attrs = {
            'long_name':'Volume transport time series',
            'units':'m3/s',
        }
    )
    file_write.to_netcdf(f'{results_folder}time-series-{year}.nc')
