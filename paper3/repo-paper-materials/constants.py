# Constants used throughout the notebooks

# model dimensions:
jmin, jmax = 159, 799
imin, imax = 1139, 2179
isize = imax-imin
jsize = jmax-jmin

# Boundary rimwidths:
rimwidthN  = 10
rimwidthS  = 10
rimwidthW  = 10
rimwidthE  = 20

# Figure folder path:
path_figure = '/ocean/brogalla/GEOTRACES/figures/Pb_202406/'

# Boundary coordinates: (i1,i2,j1,j2)
bdy_NCB = (1598,2177,778,798) # Northern Canada Basin 
bdy_WCB = (2168,2178,390,797) # Western Canada Basin 
bdy_LS  = (1140,1150,446,672) # Baffin Bay
bdy_HB  = (1190,1472,293,303) # Hudson Bay

# colors
land_color   = '#a9a7a2'
light_land   = '#d0d0cf'
land_edge    = '#373736' 
light_gray   = '#b1b1b1'
masked_color = '#eaeae9'