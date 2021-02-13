import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap, cm
import cmocean
import netCDF4 as nc
from itertools import compress
import datetime as dt
from joblib import Parallel

#-------------------------------------------------------------

years=[2017,2018]

#-------------------------------------------------------------
# Set region:
imin, imax = 1480, 2180
jmin, jmax = 160, 800

# Load files:
mesh  = nc.Dataset('/ocean/brogalla/GEOTRACES/data/ANHA12-EXH006_5d_gridT_y2002m01d05.nc')
lons  = np.array(mesh.variables['nav_lon'])
lats  = np.array(mesh.variables['nav_lat'])

mask  = nc.Dataset('/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mask.nc') #ariane_runs/ANHA12_Ariane_mesh.nc')
tmask = np.array(mask.variables['tmask'])
umask = np.array(mask.variables['umask'])
vmask = np.array(mask.variables['vmask'])

land_mask = np.ma.masked_where((tmask[0,:,:,:] > 0.1), tmask[0,:,:,:])
tmask = tmask[0,:,imin:imax,jmin:jmax]
umask = umask[0,:,imin:imax,jmin:jmax]
vmask = vmask[0,:,imin:imax,jmin:jmax]

mesh  = nc.Dataset('/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mesh1.nc')

e1t_base = np.array(mesh.variables['e1t'])[0,imin:imax,jmin:jmax]
e1v_base = np.array(mesh.variables['e1v'])[0,imin:imax,jmin:jmax]
e2t_base = np.array(mesh.variables['e2t'])[0,imin:imax,jmin:jmax]
e2u_base = np.array(mesh.variables['e2u'])[0,imin:imax,jmin:jmax]
e3t = np.array(mesh.variables['e3t_0'])[0,:,imin:imax,jmin:jmax]

e1t = np.empty_like(e3t); e1t[:] = e1t_base
e1v = np.empty_like(e3t); e1v[:] = e1v_base
e2t = np.empty_like(e3t); e2t[:] = e2t_base
e2u = np.empty_like(e3t); e2u[:] = e2u_base

e3t = np.array(mesh.variables['e3t_0'])[0,:,:,:]

e3v = np.zeros_like(e3t)
e3u = np.zeros_like(e3t)
for layer in range(50):
    for i in range(imin-5, imax+5):
        for j in range(jmin-5, jmax+5):
            e3v[layer, i, j] = min(e3t[layer, i, j], e3t[layer, i+1, j])
            e3u[layer, i, j] = min(e3t[layer, i, j], e3t[layer, i, j+1])

# boundary coordinates: --------------------------------------
# Central Sills:
CSi = 1680-imin; CSj = np.arange(457-jmin,483-jmin)
# Jones Sound:
JSi = 1607-imin; JSj = np.arange(554-jmin,583-jmin)
# Nares Strait:
N1i = np.arange(1570-imin,1630-imin); N1j = 635-jmin 
# Lancaster Sound: 
P1i = 1584-imin; P1j = np.arange(485-jmin,538-jmin) 

#-------------------------------------------------------------
def files_time_series(year, start_date, end_date):
    #start_date and end_date are datetime objects
    
    # Create list of filenames that fall within the start and end date time range:
    file_list2 = np.sort(os.listdir('/data/brogalla/ANHA12/'))
    
    Vlist = [i[26:31]=='gridV' for i in file_list2]
    Ulist = [i[26:31]=='gridU' for i in file_list2]
    
    gridV_list = list(compress(file_list2, Vlist))
    gridU_list = list(compress(file_list2, Ulist))
   
    dateV_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridV_list]
    dateU_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridU_list]
    
    gridV_file_list = list(compress(gridV_list, [V > start_date and V < end_date for V in dateV_list]))
    gridU_file_list = list(compress(gridU_list, [U > start_date and U < end_date for U in dateU_list]))
    return gridV_file_list, gridU_file_list

def main_calc(year, filenameU, filenameV):
    
    # Load 5-day velocity file
    folder2  = '/data/brogalla/ANHA12/'
    file_u  = nc.Dataset(folder2 + filenameU)
    file_v  = nc.Dataset(folder2 + filenameV)
    u_vel   = np.array(file_u.variables['vozocrtx'])[0,:,imin:imax,jmin:jmax] 
    v_vel   = np.array(file_v.variables['vomecrty'])[0,:,imin:imax,jmin:jmax] 
    
    # For each of the boundaries, call function to calculate the flux:    
    flx_CS = calc_flux(CSi, CSj, u_vel, v_vel) # Central Sills
    flx_JS = calc_flux(JSi, JSj, u_vel, v_vel) # Jones Sound
    flx_NS = calc_flux(N1i, N1j, u_vel, v_vel) # Nares Strait
    flx_LS = calc_flux(P1i, P1j, u_vel, v_vel) # Lancaster Sound
    
    return flx_CS, flx_JS, flx_NS, flx_LS

def calc_flux(i, j, u_vel, v_vel, area=e3t): 
    i = np.array(i)
    j = np.array(j)

    # horizontal boundary
    if i.size > j.size: 
        bdy_vel   = u_vel[:, i[0]:i[-1], j]
        area      = e2u[:, i[0]:i[-1], j]*e3u[:, (i[0]+imin):(i[-1]+imin), j]
        cond_mask = (umask[:,i[0]:i[-1],j] < 0.1)     
    # vertical boundary
    else: 
        bdy_vel   = v_vel[:, i, j[0]:j[-1]]
        area      = e1v[:, i, j[0]:j[-1]]*e3v[:, (i+imin), (j[0]+jmin):(j[-1]+jmin)]
        cond_mask = (vmask[:,i,j[0]:j[-1]] < 0.1)
        
    # Point-wise multiplication with areas of each of the grid boxes:
    flx_V  = np.multiply(bdy_vel, area)
    
    # Mask the depth levels that correspond to points on land:
    flx_mask_V  = np.ma.masked_where(cond_mask, flx_V)
    
    return flx_mask_V

def joblib_solver(main_calc, year, gridU_file, gridV_file):
    
    flx_CS, flx_JS, flx_NS, flx_LS = main_calc(year, gridU_file, gridV_file) 
    
    return flx_CS, flx_JS, flx_NS, flx_LS

#-------------------------------------------------------------

for year in years:
    # Create list of five-day files:
    print(year)
    start_date = dt.datetime(year,1,1)
    end_date   = dt.datetime(year,12,31)
    gridV_files, gridU_files = files_time_series(year, start_date, end_date)
    print(len(gridV_files), len(gridU_files))
    
    # call the function for each file that is within range of start date, end date
    time_series_CS = np.empty((len(gridV_files), 50, CSj.size-1)); time_series_JS = np.empty((len(gridV_files), 50, JSj.size-1)); 
    time_series_NS = np.empty((len(gridV_files), 50, N1i.size-1)); time_series_LS = np.empty((len(gridV_files), 50, P1j.size-1)); 
    
    files=range(0,len(gridV_files))
    joblist=[]
    for file in files:
        positional_args=[main_calc, year, gridU_files[file], gridV_files[file]]
        keyword_args={}
        joblist.append((joblib_solver,positional_args,keyword_args))

    ncores=8
    with Parallel(n_jobs=ncores,backend='threading') as parallel:
        results = parallel(joblist)
        
    time_CS,time_JS,time_NS,time_LS = zip(*results)

    time_series_CS[:,:,:]=time_CS[:][:][:]
    time_series_JS[:,:,:]=time_JS[:][:][:]
    time_series_NS[:,:,:]=time_NS[:][:][:]
    time_series_LS[:,:,:]=time_LS[:][:][:]
    
    #Save time series to pickle (since it takes a long time to calculate):
    pickle.dump((time_series_CS, time_series_JS, time_series_NS, time_series_LS),\
            open('/ocean/brogalla/GEOTRACES/time-series/Manuel-2021-transport/time-series-'+str(year)+'.pickle','wb'))
