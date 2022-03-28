import numpy as np
import xarray as xr
import netCDF4 as nc
import time
import sys

write_folder = '/data/brogalla/ANHA4/remapped_files/'
read_folder  = '/data/brogalla/ANHA4/original_files/'

imin, imax = 1480, 2180
jmin, jmax = 160, 800

## ANHA4 coordinate files to load:
nc_ANHA4_gridT  = xr.open_dataset(f'{read_folder}ANHA4-EXH005_y2002m01d05_gridT.nc')
nc_ANHA4_gridU  = xr.open_dataset(f'{read_folder}ANHA4-EXH005_y2002m01d05_gridU.nc')
nc_ANHA4_gridV  = xr.open_dataset(f'{read_folder}ANHA4-EXH005_y2002m01d05_gridV.nc')
nc_ANHA4_gridW  = xr.open_dataset(f'{read_folder}ANHA4-EXH005_y2002m01d05_gridW.nc')
nc_ANHA4_icemod = xr.open_dataset(f'{read_folder}ANHA4-EXH005_y2002m01d05_icemod.nc')
lon_ANHA4_gridT = nc_ANHA4_gridT['nav_lon'].values; lat_ANHA4_gridT = nc_ANHA4_gridT['nav_lat'].values;
lon_ANHA4_gridU = nc_ANHA4_gridU['nav_lon'].values; lat_ANHA4_gridU = nc_ANHA4_gridU['nav_lat'].values;
lon_ANHA4_gridV = nc_ANHA4_gridV['nav_lon'].values; lat_ANHA4_gridV = nc_ANHA4_gridV['nav_lat'].values;
lon_ANHA4_gridW = nc_ANHA4_gridW['nav_lon'].values; lat_ANHA4_gridW = nc_ANHA4_gridW['nav_lat'].values;

## ANHA12 coordinate files to load:
nc_ANHA12_gridT  = xr.open_dataset('/data/brogalla/ANHA12/ANHA12-EXH006_5d_gridT_y2002m01d05.nc')
nc_ANHA12_gridU  = xr.open_dataset('/data/brogalla/ANHA12/ANHA12-EXH006_5d_gridU_y2002m01d05.nc')
nc_ANHA12_gridV  = xr.open_dataset('/data/brogalla/ANHA12/ANHA12-EXH006_5d_gridV_y2002m01d05.nc')
nc_ANHA12_gridW  = xr.open_dataset('/data/brogalla/ANHA12/ANHA12-EXH006_5d_gridW_y2002m01d05.nc')
nc_ANHA12_icemod = xr.open_dataset('/data/brogalla/ANHA12/ANHA12-EXH006_5d_icemod_y2002m01d05.nc')
lon_ANHA12_gridT = nc_ANHA12_gridT['nav_lon'].values; lat_ANHA12_gridT = nc_ANHA12_gridT['nav_lat'].values;
lon_ANHA12_gridU = nc_ANHA12_gridU['nav_lon'].values; lat_ANHA12_gridU = nc_ANHA12_gridU['nav_lat'].values;
lon_ANHA12_gridV = nc_ANHA12_gridV['nav_lon'].values; lat_ANHA12_gridV = nc_ANHA12_gridV['nav_lat'].values;
lon_ANHA12_gridW = nc_ANHA12_gridW['nav_lon'].values; lat_ANHA12_gridW = nc_ANHA12_gridW['nav_lat'].values;

## ANHA12 meshmask:
mm = xr.open_dataset('/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mesh1.nc')
landmask = mm['tmask'][0,:,:,:].values
navlon = mm['nav_lon'].values
navlat = mm['nav_lat'].values

## Functions:
def interp_np(nav_lon, nav_lat, var_in, lon_ANHA12, lat_ANHA12, land_mask):
    ''' Interpolate some field to ANHA12 grid.
        The function is based on the bilinear interpolation in scipy, griddata 
        =======================================================================
        nav_lon, nav_lat        : input field lon/lat
        lon_ANHA12, lat_ANHA12  : ANHA12 defined lons/lats
        var_in                  : 2-D model variable
    '''
    
    from scipy.interpolate import griddata

    LatLonPair = (nav_lon, nav_lat)
    var_out = griddata(LatLonPair, var_in, (lon_ANHA12, lat_ANHA12), method='linear')
    #print('Variable out: ',np.sum(np.isnan(var_out)))
    var_fill = griddata(LatLonPair, var_in, (lon_ANHA12, lat_ANHA12), method='nearest')
    #print('Variable fill: ',np.sum(np.isnan(var_fill)))
    var_out[np.isnan(var_out)] = var_fill[np.isnan(var_out)] # first try replacing it with the nearest value
    #print('Variable out replaced: ', np.sum(np.isnan(var_out)))

    #var_out[np.isnan(var_out)] = 0.0
    # Ocean values that are still NaN, set to the nanmean of the array:
    var_out[np.isnan(var_out)] = np.nanmean(var_out[imin:imax,jmin:jmax])

    # Place mean values anywhere on land
    #var_out[land_mask == 0.0] = np.nanmean(var_out[imin:imax,jmin:jmax])
    
    # Little island instability in Lancaster Sound, fill with local values.
    var_out[imin+70:imin+76,jmin+362:jmin+366] = np.nanmean(var_out[imin+74:imin+77,jmin+366:jmin+368])
    var_out[imin+80:imin+86,jmin+353:jmin+358] = np.nanmean(var_out[imin+80:imin+86,jmin+358:jmin+361])
    #print('Variable out end: ', np.sum(np.isnan(var_out)))

    var_out[land_mask == 0.0] = np.nanmean(var_out[imin:imax,jmin:jmax])  

    return var_out

def interp_gridT(filename, lon_ANHA4=lon_ANHA4_gridT, lat_ANHA4=lat_ANHA4_gridT, lon_ANHA12=lon_ANHA12_gridT, lat_ANHA12=lat_ANHA12_gridT, landmask=landmask):
    filenameT = filename[30:63]
    print('Interpolating file: ', filenameT)
    # Load file
    file    = xr.open_dataset(f'{read_folder}{filenameT}')
    varT    = file['votemper'].values
    varS    = file['vosaline'].values
    varx    = file['somxl010'].values
    
    # Interpolate ANHA4 variables onto ANHA12 grid:
    ANHA12_votemper = np.empty((1,50,2400,1632))
    ANHA12_vosaline = np.empty((1,50,2400,1632))
    for depth in range(0,50):
        ANHA12_votemper[0,depth,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varT[0,depth,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[depth,:,:])
        ANHA12_vosaline[0,depth,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varS[0,depth,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[depth,:,:])
        
    ANHA12_somxl010 = np.empty((1,2400,1632))    
    ANHA12_somxl010[0,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varx[0,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[0,:,:])
    
    # Write interpolated values to file:
    file_write = xr.Dataset(
            {'votemper': (("time_counter","deptht","y","x"), ANHA12_votemper),
                'vosaline': (("time_counter","deptht","y","x"), ANHA12_vosaline),
                'somxl010': (("time_counter","y","x"), ANHA12_somxl010)
                }, 
            coords = {
                "time_counter": np.zeros(1),
                "deptht": np.zeros(50),
                "y": np.zeros(2400),
                "x": np.zeros(1632),
                },
            )
    
    file_write.to_netcdf(f'{write_folder}ANHA4-EXH005_5d_gridT_{filenameT[13:24]}.nc', unlimited_dims='time_counter')
    return

def interp_icemod(filename, lon_ANHA4=lon_ANHA4_gridT, lat_ANHA4=lat_ANHA4_gridT, lon_ANHA12=lon_ANHA12_gridT, lat_ANHA12=lat_ANHA12_gridT, landmask=landmask):
    filenameI = filename[30:64]
    print('Interpolating file: ', filenameI)

    # Load file
    file    = xr.open_dataset(f'{read_folder}{filenameI}')
    varp    = file['iiceprod'].values
    varf    = file['iicesflx'].values
    varl    = file['ileadfra'].values
    varw    = file['iwinstru'].values
                                   
    # Interpolate ANHA4 variables onto ANHA12 grid:
    ANHA12_iiceprod = np.empty((1,2400,1632)); ANHA12_iicesflx = np.empty((1,2400,1632));
    ANHA12_ileadfra = np.empty((1,2400,1632)); ANHA12_iwinstru = np.empty((1,2400,1632));
    ANHA12_iiceprod[0,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varp[0,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[0,:,:])
    ANHA12_iicesflx[0,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varf[0,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[0,:,:])
    ANHA12_ileadfra[0,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varl[0,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[0,:,:])
    ANHA12_iwinstru[0,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varw[0,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[0,:,:])
    
    # Write interpolated values to file:
    file_write = xr.Dataset(
            {'iiceprod': (("time_counter","y","x"), ANHA12_iiceprod),
             'iicesflx': (("time_counter","y","x"), ANHA12_iicesflx),
             'ileadfra': (("time_counter","y","x"), ANHA12_ileadfra),
             'iwinstru': (("time_counter","y","x"), ANHA12_iwinstru)
             }, 
            coords = {
                "time_counter": np.zeros(1),
                "y": np.zeros(2400),
                "x": np.zeros(1632),
                },
            )
    file_write.to_netcdf(f'{write_folder}ANHA4-EXH005_5d_icemod_{filenameI[13:24]}.nc', unlimited_dims='time_counter')
    return 

def interp_gridU(filename, lon_ANHA4=lon_ANHA4_gridU, lat_ANHA4=lat_ANHA4_gridU, lon_ANHA12=lon_ANHA12_gridU, lat_ANHA12=lat_ANHA12_gridU, landmask=landmask):
    filenameU = filename[30:63]
    print('Interpolating file: ', filenameU)
    
    # Load file
    file    = xr.open_dataset(f'{read_folder}{filenameU}')
    varx    = file['vozocrtx'].values
    
    # Interpolate ANHA4 variables onto ANHA12 grid:
    ANHA12_vozocrtx = np.empty((1,50,2400,1632))
    for depth in range(0,50):
        ANHA12_vozocrtx[0,depth,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varx[0,depth,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[depth,:,:])
        
    # Write interpolated values to file:
    file_write = xr.Dataset(
            {'vozocrtx': (("time_counter","deptht","y","x"), ANHA12_vozocrtx)}, 
            coords = {
                "time_counter": np.zeros(1),
                "deptht": np.zeros(50),
                "y": np.zeros(2400),
                "x": np.zeros(1632),
                },
            )
        
    file_write.to_netcdf(f'{write_folder}ANHA4-EXH005_5d_gridU_{filenameU[13:24]}.nc', unlimited_dims='time_counter')
    return 

def interp_gridV(filename, lon_ANHA4=lon_ANHA4_gridV, lat_ANHA4=lat_ANHA4_gridV, lon_ANHA12=lon_ANHA12_gridV, lat_ANHA12=lat_ANHA12_gridV, landmask=landmask):
    filenameV = filename[30:63]
    print('Interpolating file: ', filenameV)
    
    # Load file
    file    = xr.open_dataset(f'{read_folder}{filenameV}')
    vary    = file['vomecrty'].values
    
    # Interpolate ANHA4 variables onto ANHA12 grid:
    ANHA12_vomecrty = np.empty((1,50,2400,1632))
    for depth in range(0,50):
        ANHA12_vomecrty[0,depth,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), vary[0,depth,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[depth,:,:])
        
    # Write interpolated values to file:
    file_write = xr.Dataset(
            {'vomecrty': (("time_counter","deptht","y","x"), ANHA12_vomecrty)}, 
            coords = {
                "time_counter": np.zeros(1),
                "deptht": np.zeros(50),
                "y": np.zeros(2400),
                "x": np.zeros(1632),
                },
            )
    
    file_write.to_netcdf(f'{write_folder}ANHA4-EXH005_5d_gridV_{filenameV[13:24]}.nc', unlimited_dims='time_counter')
    return

def interp_gridW(filename, lon_ANHA4=lon_ANHA4_gridW, lat_ANHA4=lat_ANHA4_gridW, lon_ANHA12=lon_ANHA12_gridW, lat_ANHA12=lat_ANHA12_gridW, landmask=landmask):
    filenameW = filename[30:63]
    print('Interpolating file: ', filenameW)
    
    # Load file
    file    = xr.open_dataset(f'{read_folder}{filenameW}')
    varv    = file['vovecrtz'].values
    vart    = file['votkeavt'].values
    
    # Interpolate ANHA4 variables onto ANHA12 grid:
    ANHA12_vovecrtz = np.empty((1,50,2400,1632))
    ANHA12_votkeavt = np.empty((1,50,2400,1632))
    for depth in range(0,50):
        ANHA12_vovecrtz[0,depth,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), varv[0,depth,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[depth,:,:])
        ANHA12_votkeavt[0,depth,:,:] = interp_np(lon_ANHA4.flatten(), lat_ANHA4.flatten(), vart[0,depth,:,:].flatten(), lon_ANHA12, lat_ANHA12, landmask[depth,:,:])

    # Write interpolated values to file:
    file_write = xr.Dataset(
            {'vovecrtz': (("time_counter","deptht","y","x"), ANHA12_vovecrtz),
                'votkeavt': (("time_counter","deptht","y","x"), ANHA12_votkeavt),
                }, 
            coords = {
                "time_counter": np.zeros(1),
                "deptht": np.zeros(50),
                "y": np.zeros(2400),
                "x": np.zeros(1632),
                },
            )
    
    file_write.to_netcdf(f'{write_folder}ANHA4-EXH005_5d_gridW_{filenameW[13:24]}.nc', unlimited_dims='time_counter') 
    return

if __name__=='__main__':
    if sys.argv[1] == 'interp_icemod':
        interp_icemod(*sys.argv[2:])
    elif sys.argv[1] == 'interp_gridT':
        interp_gridT(*sys.argv[2:])
    elif sys.argv[1] == 'interp_gridU':
        interp_gridU(*sys.argv[2:])
    elif sys.argv[1] == 'interp_gridV':
        interp_gridV(*sys.argv[2:])
    elif sys.argv[1] == 'interp_gridW':
        interp_gridW(*sys.argv[2:])
    else:
        print('Called a function which does not exist')
