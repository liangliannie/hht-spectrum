from netCDF4 import Dataset

f = Dataset('./171002_parmmods_monthly_obs.nc')

fsh = f.variables['FSH']
time = f.variables['time']

one_site = np.ma.masked_invalid(fsh[0,:])
time = time[~one_site.mask]

data = one_site.compressed() 
