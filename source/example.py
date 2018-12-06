from netCDF4 import Dataset
from hht import hht
import matplotlib.pyplot as plt
import numpy as np
'''import all the needed packages '''
f = Dataset('./source/obs.nc')
# read one example data
fsh = f.variables['FSH']
time = f.variables['time']
one_site = np.ma.masked_invalid(fsh[0,:])
time = time[~one_site.mask]
data = one_site.compressed()
hht(data, time)
plt.show()

'''from netCDF4 import Dataset
from hht import hht
import matplotlib.pyplot as plt
import numpy as np
import all the needed packages
f = Dataset('./source/obs.nc')
# read one example data
fsh = f.variables['FSH']
time = f.variables['time']
one_site = np.ma.masked_invalid(fsh[0,:])
time = time[~one_site.mask]
data = one_site.compressed()
hht(data, time)
plt.show()

'''
