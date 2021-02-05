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

years=[2019]
base = 'ref'
results_folder = 'Mn-set5-202008/'
#-------------------------------------------------------------
# Set region:
imin, imax = 1480, 2180
jmin, jmax = 160, 800

# Load files:
ref   = nc.Dataset('/data/brogalla/run_storage/Mn-set4-202004/ref-2002/ANHA12_EXH006_2002_monthly.nc',  'r')
tlons = np.array(ref.variables['nav_lon'])
tlats = np.array(ref.variables['nav_lat'])
depth = np.array(ref.variables['deptht'])

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
x1=imin
y1=jmin

l1i = 2013-x1; l1j = np.arange(300-y1,392-y1)
l2i = 1935-x1; l2j = np.arange(450-y1,530-y1)
l3i = np.arange(1850-x1,1885-x1); l3j = 555-y1
l4i = np.arange(1753-x1,1837-x1); l4j = 568-y1
l5i = np.arange(1720-x1,1790-x1); l5j = 605-y1
l6i = 1730-x1; l6j = np.arange(660-y1,690-y1)

t1i = np.arange(1635-x1,1653-x1); t1j = 760-y1

r1i = 1520-x1; r1j = np.arange(505-y1,673-y1)
r2i = 1520-x1; r2j = np.arange(385-y1,405-y1)

N1i = np.arange(1570-x1,1630-x1); N1j = 635-y1 #Nares
P1i = 1585-x1; P1j = np.arange(485-y1,538-y1)  #Parry channel

#-------------------------------------------------------------
def files_time_series(base, year, start_date, end_date):
    #start_date and end_date are datetime objects
    
    # Create list of filenames that fall within the start and end date time range:
    file_list1 = np.sort(os.listdir('/data/brogalla/run_storage/'+results_folder+base+'-'+str(year)+'/'))
    file_list2 = np.sort(os.listdir('/data/brogalla/ANHA12/'))
    
    Vlist = [i[26:31]=='gridV' for i in file_list2]
    Ulist = [i[26:31]=='gridU' for i in file_list2]
    Plist = [i[35:39]=='ptrc' for i in file_list1]
    
    gridV_list = list(compress(file_list2, Vlist))
    gridU_list = list(compress(file_list2, Ulist))
    gridP_list = list(compress(file_list1, Plist))
    
    dateV_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridV_list]
    dateU_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridU_list]
    dateP_list = [dt.datetime.strptime(i[42:50], "%Y%m%d")    for i in gridP_list]
    
    gridV_file_list = list(compress(gridV_list, [V > start_date and V < end_date for V in dateV_list]))
    gridU_file_list = list(compress(gridU_list, [U > start_date and U < end_date for U in dateU_list]))
    gridP_file_list = list(compress(gridP_list, [P > start_date and P < end_date for P in dateP_list]))
    
    if len(gridP_file_list ) > len(gridU_file_list):
        gridP_file_list = gridP_file_list[0:-1]
    elif len(gridU_file_list) > len(gridP_file_list):
        diff = len(gridP_file_list) - len(gridU_file_list)
        gridU_file_list = gridU_file_list[0:diff]
        gridV_file_list = gridV_file_list[0:diff]
       
    return gridV_file_list, gridU_file_list, gridP_file_list

def main_calc(base, year, filenameU, filenameV, filenameP): 
    # Load 5-day ptrc file
    folder1 = '/data/brogalla/run_storage/'+results_folder+base+'-'+str(year)+'/'
    file1   = nc.Dataset(folder1+filenameP)
    dmn     = np.array(file1.variables['dissolmn'])[0,:,:,:] 
    
    # Load 5-day velocity file
    folder2  = '/data/brogalla/ANHA12/'
    file_u  = nc.Dataset(folder2 + filenameU)
    file_v  = nc.Dataset(folder2 + filenameV)
    u_vel   = np.array(file_u.variables['vozocrtx'])[0,:,imin:imax,jmin:jmax] 
    v_vel   = np.array(file_v.variables['vomecrty'])[0,:,imin:imax,jmin:jmax] 
    
    # For each of the boundaries, call function to calculate the flux:
    flx_mnl1, flx_Vl1 = calc_flux(l1i, l1j, dmn, u_vel, v_vel)
    flx_mnl2, flx_Vl2 = calc_flux(l2i, l2j, dmn, u_vel, v_vel)
    flx_mnl3, flx_Vl3 = calc_flux(l3i, l3j, dmn, u_vel, v_vel)
    flx_mnl4, flx_Vl4 = calc_flux(l4i, l4j, dmn, u_vel, v_vel)
    flx_mnl5, flx_Vl5 = calc_flux(l5i, l5j, dmn, u_vel, v_vel)
    flx_mnl6, flx_Vl6 = calc_flux(l6i, l6j, dmn, u_vel, v_vel)

    flx_mnt1, flx_Vt1 = calc_flux(t1i, t1j, dmn, u_vel, v_vel)
    
    flx_mnr1, flx_Vr1 = calc_flux(r1i, r1j, dmn, u_vel, v_vel)
    flx_mnr2, flx_Vr2 = calc_flux(r2i, r2j, dmn, u_vel, v_vel)
    
    flx_mnN1, flx_VN1 = calc_flux(N1i, N1j, dmn, u_vel, v_vel)
    flx_mnP1, flx_VP1 = calc_flux(P1i, P1j, dmn, u_vel, v_vel)
    
    return flx_mnl1, flx_mnl2, flx_mnl3, flx_mnl4, flx_mnl5, \
            flx_mnl6, flx_mnt1, flx_mnr1, flx_mnr2, flx_mnN1, flx_mnP1, flx_Vl1, \
            flx_Vl2, flx_Vl3, flx_Vl4, flx_Vl5, \
            flx_Vl6, flx_Vt1, flx_Vr1, flx_Vr2, flx_VN1, flx_VP1

def calc_flux(i, j, dmn, u_vel, v_vel, area=e3t): 
    i = np.array(i)
    j = np.array(j)

    # horizontal boundary
    if i.size > j.size: 
        bdy_vel   = u_vel[:, i[0]:i[-1], j]
        dmn_bdyl  =   dmn[:, i[0]:i[-1], j]
        dmn_bdyr  =   dmn[:, i[0]:i[-1], j+1]
        area      = e2u[:, i[0]:i[-1], j]*e3u[:, (i[0]+imin):(i[-1]+imin), j]
        cond_mask = (umask[:,i[0]:i[-1],j] < 0.1)
        
    # vertical boundary
    else: 
        bdy_vel   = v_vel[:, i, j[0]:j[-1]]
        dmn_bdyl  =   dmn[:, i, j[0]:j[-1]]
        dmn_bdyr  =   dmn[:, i+1, j[0]:j[-1]]
        area      = e1v[:, i, j[0]:j[-1]]*e3v[:, (i+imin), (j[0]+jmin):(j[-1]+jmin)]
        cond_mask = (vmask[:,i,j[0]:j[-1]] < 0.1)
        
    # Point-wise multiplication with areas of each of the grid boxes:
    flx_V  = np.multiply(bdy_vel, area)
    
    # Mn flux for each grid cell on the U or V point on the boundary:
    dmn_bdy = 0.5*(dmn_bdyl + dmn_bdyr) 
    flx_mn  = np.multiply(dmn_bdy, flx_V)
    
    # Mask the depth levels that correspond to points on land:
    flx_mask_mn = np.ma.masked_where(cond_mask, flx_mn)
    flx_mask_V  = np.ma.masked_where(cond_mask, flx_V)
    
    return flx_mask_mn, flx_mask_V
def joblib_solver(main_calc, base, year, gridU_file, gridV_file, gridP_file):
#     interp = main_calc(gridU_file, gridV_file, gridP_file) 
    flx_mnl1, flx_mnl2, flx_mnl3, flx_mnl4, flx_mnl5, \
    flx_mnl6, flx_mnt1, flx_mnr1, flx_mnr2, flx_mnN1, flx_mnP1, flx_Vl1, \
    flx_Vl2, flx_Vl3, flx_Vl4, flx_Vl5, flx_Vl6, \
    flx_Vt1, flx_Vr1, flx_Vr2, flx_VN1, flx_VP1 = main_calc(base, year, gridU_file, gridV_file, gridP_file) 
    
    return flx_mnl1, flx_mnl2, flx_mnl3, flx_mnl4, flx_mnl5, flx_mnl6, flx_mnt1, flx_mnr1, flx_mnr2, \
            flx_mnN1, flx_mnP1, flx_Vl1, flx_Vl2, flx_Vl3, flx_Vl4, flx_Vl5, flx_Vl6, flx_Vt1, flx_Vr1, \
            flx_Vr2, flx_VN1, flx_VP1

#-------------------------------------------------------------

for year in years:
    # Create list of five-day files:
    print(year)
    start_date = dt.datetime(year,1,1)
    end_date   = dt.datetime(year,12,31)
    gridV_files, gridU_files, gridP_files = files_time_series(base, year, start_date, end_date)
    print(len(gridV_files), len(gridU_files), len(gridP_files))
    
    # call the function for each file that is within range of start date, end date
    a = len(gridV_files)
    time_series_mn1 = np.empty((a, 50, l1j.size-1)); time_series_mn2 = np.empty((a, 50, l2j.size-1)); 
    time_series_mn3 = np.empty((a, 50, l3i.size-1)); time_series_mn4 = np.empty((a, 50, l4i.size-1)); 
    time_series_mn5 = np.empty((a, 50, l5i.size-1)); time_series_mn6 = np.empty((a, 50, l6j.size-1)); 
    time_series_mn7 = np.empty((a, 50, t1i.size-1)); time_series_mn8 = np.empty((a, 50, r1j.size-1)); 
    time_series_mn9 = np.empty((a, 50, r2j.size-1)); time_series_mn10 = np.empty((a, 50, N1i.size-1));
    time_series_mn11 = np.empty((a, 50, P1j.size-1)); 

    time_series_V1 = np.empty_like(time_series_mn1); time_series_V2 = np.empty_like(time_series_mn2); 
    time_series_V3 = np.empty_like(time_series_mn3); time_series_V4 = np.empty_like(time_series_mn4); 
    time_series_V5 = np.empty_like(time_series_mn5); time_series_V6 = np.empty_like(time_series_mn6); 
    time_series_V7 = np.empty_like(time_series_mn7); time_series_V8 = np.empty_like(time_series_mn8); 
    time_series_V9 = np.empty_like(time_series_mn9); time_series_V10 = np.empty_like(time_series_mn10);
    time_series_V11 = np.empty_like(time_series_mn11);
    
    files=range(0,len(gridV_files))
    joblist=[]
    for file in files:
        positional_args=[main_calc, base, year, gridU_files[file], gridV_files[file], gridP_files[file]]
        keyword_args={}
        joblist.append((joblib_solver,positional_args,keyword_args))

    ncores=8
    with Parallel(n_jobs=ncores,backend='threading') as parallel:
        results = parallel(joblist)
        
    time_mn1,time_mn2,time_mn3,time_mn4,time_mn5, \
    time_mn6,time_mn7,time_mn8,time_mn9,time_mn10,time_mn11,time_V1,time_V2,time_V3,time_V4,time_V5, \
    time_V6,time_V7,time_V8,time_V9,time_V10,time_V11 = zip(*results)

    time_series_mn1[:,:,:]=time_mn1[:][:][:]
    time_series_mn2[:,:,:]=time_mn2[:][:][:]
    time_series_mn3[:,:,:]=time_mn3[:][:][:]
    time_series_mn4[:,:,:]=time_mn4[:][:][:]
    time_series_mn5[:,:,:]=time_mn5[:][:][:]
    time_series_mn6[:,:,:]=time_mn6[:][:][:]
    time_series_mn7[:,:,:]=time_mn7[:][:][:]
    time_series_mn8[:,:,:]=time_mn8[:][:][:]
    time_series_mn9[:,:,:]=time_mn9[:][:][:]
    time_series_mn10[:,:,:]=time_mn10[:][:][:]
    time_series_mn11[:,:,:]=time_mn11[:][:][:]

    time_series_V1[:,:,:]=time_V1[:][:][:]
    time_series_V2[:,:,:]=time_V2[:][:][:]
    time_series_V3[:,:,:]=time_V3[:][:][:]
    time_series_V4[:,:,:]=time_V4[:][:][:]
    time_series_V5[:,:,:]=time_V5[:][:][:]
    time_series_V6[:,:,:]=time_V6[:][:][:]
    time_series_V7[:,:,:]=time_V7[:][:][:]
    time_series_V8[:,:,:]=time_V8[:][:][:]
    time_series_V9[:,:,:]=time_V9[:][:][:]
    time_series_V10[:,:,:]=time_V10[:][:][:]
    time_series_V11[:,:,:]=time_V11[:][:][:]
    
    #Save time series to pickle (since it takes a long time to calculate):
    pickle.dump((time_series_V1, time_series_V2, time_series_V3, time_series_V4, time_series_V5, \
            time_series_V6, time_series_V7, time_series_V8, time_series_V9, time_series_V10, \
            time_series_V11, time_series_mn1, time_series_mn2, time_series_mn3, time_series_mn4, \
            time_series_mn5, time_series_mn6, time_series_mn7, time_series_mn8, time_series_mn9, \
            time_series_mn10,  time_series_mn11),\
            open(results_folder+'time-series-'+str(year)+'.pickle','wb'))

