import numpy as np
from datetime import timedelta as delta, datetime
import xarray as xr
from parcels import FieldSet, Variable, ParticleSet, JITParticle, AdvectionRK4_3D, ErrorCode
import glob
import matplotlib.pyplot as plt
import sys
import pandas as pd

# Sub-domain dimensions:
jmin, jmax = 159, 799
imin, imax = 1139, 2179

# Open particle initialization location file:
ini_loc_PC = xr.open_dataset('/ocean/brogalla/GEOTRACES/parcels/Pb-chapter/western-PC-transect-parcels.nc')
PC_lons    = ini_loc_PC['longitude'].values
PC_lats    = ini_loc_PC['latitude'].values
PC_deps    = ini_loc_PC['depth'].values

year      = int(sys.argv[1])
month     = 1
bdy_cond  = pd.datetime(year,month,1)
print(year, month)
# Mesh mask
mesh_mask = '/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mesh1.nc'

# Lists of ANHA12 NEMO U,V,W files (on full ANHA12 domain)
folder_ANHA12      = '/data/brogalla/ANHA12/'
gridU_files = sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year}m??d??_gridU.nc'))   + \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year+1}m??d??_gridU.nc')) + \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year+2}m??d??_gridU.nc')) + \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year+3}m??d??_gridU.nc'))
gridV_files = sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year}m??d??_gridV.nc'))   + \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year+1}m??d??_gridV.nc')) + \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year+2}m??d??_gridV.nc')) + \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_y{year+3}m??d??_gridV.nc'))
gridW_files = sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_5d_gridW_y{year}m??d??.nc'))  + \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_5d_gridW_y{year+1}m??d??.nc'))+ \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_5d_gridW_y{year+2}m??d??.nc'))+ \
              sorted(glob.glob(f'{folder_ANHA12}ANHA12-EXH006_5d_gridW_y{year+3}m??d??.nc'))

# Lists of ANHA12 NEMO Pb tracer files (on sub-domain of full ANHA12 domain)
folder_Pb1 = f'/data/brogalla/run_storage/Pb-reference-202303/Pb-reference-{year}_20230308/'
folder_Pb2 = f'/data/brogalla/run_storage/Pb-reference-202303/Pb-reference-{year+1}_20230308/'
folder_Pb3 = f'/data/brogalla/run_storage/Pb-reference-202303/Pb-reference-{year+2}_20230308/'
folder_Pb4 = f'/data/brogalla/run_storage/Pb-reference-202303/Pb-reference-{year+3}_20230308/'
files_Pb1  = sorted(glob.glob(f'{folder_Pb1}ANHA12_EXH006_5d_{year}0101_{year}1231_ptrc_T_{year}*')) 
files_Pb2  = sorted(glob.glob(f'{folder_Pb2}ANHA12_EXH006_5d_{year+1}0101_{year+1}1231_ptrc_T_{year+1}*'))
files_Pb3  = sorted(glob.glob(f'{folder_Pb3}ANHA12_EXH006_5d_{year+2}0101_{year+2}1231_ptrc_T_{year+2}*')) 
files_Pb4  = sorted(glob.glob(f'{folder_Pb4}ANHA12_EXH006_5d_{year+3}0101_{year+3}1231_ptrc_T_{year+3}*'))

# remove months from list that are before the intended start date:
datesU   = pd.to_datetime(gridU_files, format=f"{folder_ANHA12}ANHA12-EXH006_y%Ym%md%d_gridU.nc")
datesV   = pd.to_datetime(gridV_files, format=f"{folder_ANHA12}ANHA12-EXH006_y%Ym%md%d_gridV.nc")
datesW   = pd.to_datetime(gridW_files, format=f"{folder_ANHA12}ANHA12-EXH006_5d_gridW_y%Ym%md%d.nc")
datesPb1 = pd.to_datetime([file.split('-')[5][:-3] for file in files_Pb1], format=f"%Y%m%d")
datesPb2 = pd.to_datetime([file.split('-')[5][:-3] for file in files_Pb2], format=f"%Y%m%d")
datesPb3 = pd.to_datetime([file.split('-')[5][:-3] for file in files_Pb3], format=f"%Y%m%d")
datesPb4 = pd.to_datetime([file.split('-')[5][:-3] for file in files_Pb4], format=f"%Y%m%d")
ANHA12_gridU_files = list(np.array(gridU_files)[datesU >= bdy_cond])
ANHA12_gridV_files = list(np.array(gridV_files)[datesV >= bdy_cond])
ANHA12_gridW_files = list(np.array(gridW_files)[datesW >= bdy_cond])
files_Pb           = list(np.array(files_Pb1)[datesPb1 >= bdy_cond]) + list(np.array(files_Pb2)[datesPb2 >= bdy_cond]) + \
                     list(np.array(files_Pb3)[datesPb3 >= bdy_cond]) + list(np.array(files_Pb4)[datesPb4 >= bdy_cond])

# Create timestamps from file dates: (because of time_origin issue in ANHA12 netcdf dynamics files)
time_stamps_files = []
for n, file in enumerate(ANHA12_gridU_files):
    dateU  = datetime.strptime(ANHA12_gridU_files[n][36:47],'y%Ym%md%d')
    dateV  = datetime.strptime(ANHA12_gridV_files[n][36:47],'y%Ym%md%d')
    dateW  = datetime.strptime(ANHA12_gridW_files[n][45:56],'y%Ym%md%d')
    datePb = datetime.strptime(files_Pb[n][125:133],'%Y%m%d')

    # double-check that the date is the same for all files
    if (dateU != dateV) | (dateU != dateW) | (dateU != datePb):
        print('Dates not equal!')
    else:   
        time_stamps_files.append(np.datetime64(f'{dateU.year:02}-{dateU.month:02}-{dateU.day:02}'))

# array of file timestamps
timestamps_files = np.expand_dims(np.array(time_stamps_files), axis=1)

filenames  = {'U': {'lon': mesh_mask     , 'lat': mesh_mask  , 'depth': ANHA12_gridW_files[0], 'data': ANHA12_gridU_files},
              'V': {'lon': mesh_mask     , 'lat': mesh_mask  , 'depth': ANHA12_gridW_files[0], 'data': ANHA12_gridV_files},
              'W': {'lon': mesh_mask     , 'lat': mesh_mask  , 'depth': ANHA12_gridW_files[0], 'data': ANHA12_gridW_files},
              'Pb': {'lon': files_Pb[0]  , 'lat': files_Pb[0], 'depth': files_Pb[0], 'data': files_Pb}}

dimensions = {'U':  {'lon': 'nav_lon', 'lat': 'nav_lat', 'depth': 'depthw'},
              'V':  {'lon': 'nav_lon', 'lat': 'nav_lat', 'depth': 'depthw'},
              'W':  {'lon': 'nav_lon', 'lat': 'nav_lat', 'depth': 'depthw'},
              'Pb': {'lon': 'nav_lon', 'lat': 'nav_lat', 'depth': 'deptht'}}

variables  = {'U': 'vozocrtx',
              'V': 'vomecrty',
              'W': 'vovecrtz',
              'Pb': 'dissolpb'}

# needed to provide time as timestamps in order to avoid time origin type issue
fieldset   = FieldSet.from_nemo(filenames, variables, dimensions, \
                                timestamps=timestamps_files, \
                                allow_time_extrapolation=False)

def DeleteParticle(particle, fieldset, time):
    print("Particle [%d] lost !! (%g %g %g %g)" % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()
    
# define new particle class for Pb
class PbParticle(JITParticle): 
    dissolvedPb = Variable('dissolpb', initial=0)  # Variable 'dissolpb' initialised by sampling the dissolved Pb field
    
def SamplePb(particle, fieldset, time):
    particle.dissolpb = fieldset.Pb[particle.time, particle.depth, particle.lat, particle.lon]
    
def pushback(particle, fieldset, time):
    particle.depth = 1.5 
    
def run_parcels(fieldset, plon, plat, pdepth, runtime=0, sampling=False, filename='test.zarr'):

    # Define particle locations on a grid in my domain:
    pset = ParticleSet(fieldset=fieldset, pclass=PbParticle, lon=plon, lat=plat, depth=pdepth, 
                       repeatdt=delta(hours=24*7*2).total_seconds()) # release every 2 weeks

    output_file = pset.ParticleFile(name=f"/ocean/brogalla/GEOTRACES/parcels/Pb-chapter/{filename}", 
                                    outputdt=delta(hours=12))

    # Nudge the particle set locations closer to the actual grid locations:               
    for i in range(len(pset)):
        for gi, g in enumerate(fieldset.gridset.grids):
            d = np.hypot((pset.lon[i] - g.lon), (pset.lat[i] - g.lat))
            ind = np.unravel_index(np.argmin(d, axis=None), d.shape)
            pset.xi[i, gi] = ind[1]
            pset.yi[i, gi] = ind[0]
    
    # Option to run with tracer sampling kernel (SamplePb) or only with 3D advection kernel
    if sampling:
        pset.execute(AdvectionRK4_3D + pset.Kernel(SamplePb), 
                     runtime = delta(hours=24*365), dt = delta(hours=6), 
                     output_file = output_file,
                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, 
                               ErrorCode.ErrorThroughSurface: pushback})        

        pset.repeatdt = None
        pset.execute(AdvectionRK4_3D + pset.Kernel(SamplePb),
                     runtime = runtime, dt = delta(hours=6),
                     output_file = output_file,
                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                               ErrorCode.ErrorThroughSurface: pushback})
    else:
        pset.execute(AdvectionRK4_3D, runtime=runtime, dt=delta(hours=6),  
                     output_file = output_file,
                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                               ErrorCode.ErrorThroughSurface: pushback})
    
    return pset


pset = run_parcels(fieldset, PC_lons, PC_lats, PC_deps, runtime=delta(hours=24*30*24), 
        sampling=True, filename=f'run-{year}-repeatdt1yr-24months_20230326.zarr') 
