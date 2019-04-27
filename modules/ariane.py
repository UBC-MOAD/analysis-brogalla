import netCDF4 as nc
import numpy as np

def find_closest_model_point(
    lon, lat, model_lons, model_lats, grid='NEMO', land_mask=None,
    tols={
        'NEMO': {'tol_lon': 0.104, 'tol_lat': 0.0388},
        'GEM2.5': {'tol_lon': 0.016, 'tol_lat': 0.012},
        }
):
    """Returns the grid coordinates of the closest model point
    to a specified lon/lat. If land_mask is provided, returns the closest
    water point.

    Example:

    .. code-block:: python

        j, i = find_closest_model_point(
                   -125.5,49.2,model_lons,model_lats,land_mask=bathy.mask)

    where bathy, model_lons and model_lats are returned from
    :py:func:`salishsea_tools.tidetools.get_bathy_data`.

    j is the y-index(latitude), i is the x-index(longitude)

    :arg float lon: longitude to find closest grid point to

    :arg float lat: latitude to find closest grid point to

    :arg model_lons: specified model longitude grid
    :type model_lons: :py:obj:`numpy.ndarray`

    :arg model_lats: specified model latitude grid
    :type model_lats: :py:obj:`numpy.ndarray`

    :arg grid: specify which default lon/lat tolerances
    :type grid: string

    :arg land_mask: describes which grid coordinates are land
    :type land_mask: numpy array

    :arg tols: stored default tols for different grid types
    :type tols: dict

    :returns: yind, xind
    """

    if grid not in tols:
        raise KeyError(
            'The provided grid type is not in tols. '
            'Use another grid type or add your grid type to tols.')

    # Search for a grid point with longitude and latitude within
    # tolerance of measured location
    j_list, i_list = np.where(
        np.logical_and(
            (np.logical_and(model_lons > lon - tols[grid]['tol_lon'],
                            model_lons < lon + tols[grid]['tol_lon'])),
            (np.logical_and(model_lats > lat - tols[grid]['tol_lat'],
                            model_lats < lat + tols[grid]['tol_lat']))
        )
    )

    if len(j_list) == 0:
        # Added by BMM March 2017
        # If including points outside of domain:
        return np.nan, np.nan
        # raise ValueError(
        #    'No model point found. tol_lon/tol_lat too small or '
        #    'lon/lat outside of domain.')
    try:
        j, i = map(np.asscalar, (j_list, i_list))
    except ValueError:
        # Several points within tolerance
        # Calculate distances for all and choose the closest

        # Avoiding array indexing because some functions
        # pass in model_lons and model_lats as netcdf4 objects
        # (which treat 'model_lons[j_list, i_list]' differently)
        lons = [model_lons[j_list[n], i_list[n]] for n in range(len(j_list))]
        lats = [model_lats[j_list[n], i_list[n]] for n in range(len(j_list))]
        dists = haversine(
            np.array([lon] * i_list.size), np.array([lat] * j_list.size),
            lons, lats)
        n = dists.argmin()
        j, i = map(np.asscalar, (j_list[n], i_list[n]))

    # If point is on land and land mask is provided
    # try to find closest water point
    if land_mask is None or not land_mask[j, i]:
        return j, i
    try:
        return _spiral_search_for_closest_water_point(
            j, i, land_mask, lon, lat, model_lons, model_lats)
    except ValueError:
        raise ValueError(
            'lat/lon on land and no nearby water point found')


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance in kilometers between two points
    on a sphere from their longitudes and latitudes.

    Reference: http://www.movable-type.co.uk/scripts/latlong.html

    :arg lon1: Longitude of point 1.
    :type lon1: float or :py:class:`numpy.ndarray`

    :arg lat1: Latitude of point 1.
    :type lat1: float or :py:class:`numpy.ndarray`

    :arg lon2: Longitude of point 2.
    :type lon2: float or :py:class:`numpy.ndarray`

    :arg lat2: Latitude of point 2.
    :type lat2: float or :py:class:`numpy.ndarray`

    :returns: Great-circle distance between two points in km
    :rtype: float or :py:class:`numpy.ndarray`
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def find_release_coord(lon_vector, lat_vector, model_lons, model_lats, tmask):
    vector_i = np.array([])
    vector_j = np.array([])
    for k in range(0,len(lon_vector)):
        i, j = find_closest_model_point(lon_vector[k], lat_vector[k], model_lons, model_lats)
        vector_i = np.append(vector_i, i)
        vector_j = np.append(vector_j, j)
        
    vector_k = np.zeros(vector_i.shape)
    for n in range(0,len(vector_i)): #make array of depth of bottom
        k_level = np.sum(tmask[0,:,vector_i[n],vector_j[n]])
        vector_k[n] = k_level
        
    print('i:', [int(vector_i[n]+1) for n in range(0,len(vector_i))])
    print('j:', [int(vector_j[n]+1) for n in range(0,len(vector_j))])
    print('k:', [int(vector_k[n]) for n in range(0,len(vector_k))])
    
    return vector_i, vector_j, vector_k
