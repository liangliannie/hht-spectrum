
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pyhht
import math
from matplotlib import colors, ticker, cm

def plot_imfs(signal, imfs, time_samples = None, fig=None):
    ''' Author jaidevd https://github.com/jaidevd/pyhht/blob/dev/pyhht/visualization.py '''
    '''Original function from pyhht, but without plt.show()'''
    n_imfs = imfs.shape[0]
    # print(np.abs(imfs[:-1, :]))
    # axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
    # Plot original signal
    ax = plt.subplot(n_imfs + 1, 1, 1)
    ax.plot(time_samples, signal)
    ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal')
    ax.set_title('Empirical Mode Decomposition')

    # Plot the IMFs
    for i in range(n_imfs - 1):
        # print(i + 2)
        ax = plt.subplot(n_imfs + 1, 1, i + 2)
        ax.plot(time_samples, imfs[i, :])
        # ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('imf' + str(i + 1))

    # Plot the residue
    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    ax.plot(time_samples, imfs[-1, :], 'r')
    ax.axis('tight')
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res.')
    return ax


def plot_frequency(signal, imfs, time_samples = None, fig=None):
    ''' Author jaidevd https://github.com/jaidevd/pyhht/blob/dev/pyhht/visualization.py '''
    '''Original function from pyhht, but without plt.show()'''
    n_imfs = imfs.shape[0]
    # print(np.abs(imfs[:-1, :]))
    # axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
    # Plot original signal
    ax = plt.subplot(n_imfs + 1, 1, 1)
    ax.plot(time_samples, signal)
    ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal')
    ax.set_title('Instantaneous frequency of IMFs')

    # Plot the IMFs
    for i in range(n_imfs - 1):
        # print(i + 2)
        ax = plt.subplot(n_imfs + 1, 1, i + 2)
        ax.plot(time_samples, imfs[i, :])
        # ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.yaxis.tick_right()
        # ax.yaxis.set_ticks(np.logspace(1, 5, 5))
        plt.tick_params(axis='right', which='minor', labelsize=6)
        ax.grid(False)
        ax.set_ylim((0, np.max(imfs[i, :])))
        ax.set_ylabel('imf' + str(i + 1))

    # Plot the residue
    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    ax.plot(time_samples, imfs[-1, :], 'r')
    ax.axis('tight')
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res.')
    return ax


def hilb(s, unwrap=False):
    """
    Performs Hilbert transformation on signal s.
    Returns amplitude and phase of signal.
    Depending on unwrap value phase can be either
    in range [-pi, pi) (unwrap=False) or
    continuous (unwrap=True).
    """
    from scipy.signal import hilbert
    H = hilbert(s)
    amp = np.abs(H)
    phase = np.arctan2(H.imag, H.real)
    if unwrap: phase = np.unwrap(phase)

    return amp, phase

def FAhilbert(imfs, dt):
    n_imfs = imfs.shape[0]
    f = []
    a = []
    for i in range(n_imfs - 1):
        # upper, lower = pyhht.utils.get_envelops(imfs[i, :])
        inst_imf = imfs[i, :] #/upper
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        inst_freq = (2*math.pi)/np.diff(phase)#

        inst_freq = np.insert(inst_freq, len(inst_freq), inst_freq[-1])
        inst_amp = np.insert(inst_amp, len(inst_amp), inst_amp[-1])

        f.append(inst_freq)
        a.append(inst_amp)
    return np.asarray(f).T, np.asarray(a).T


# f = Dataset('/Users/lli51/Documents/ornl_project/171002_parmmods_monthly_obs.nc')
f = Dataset('/Users/lli51/Documents/ornl_project/171002_parmmods_daily_obs.nc')

fsh = f.variables['FSH']
time = f.variables['time']
one_site = np.ma.masked_invalid(fsh[0,:])
time = time[~one_site.mask]

data = one_site.compressed()
# print(data)
decomposer2 = pyhht.emd.EmpiricalModeDecomposition(data)
# imfs = decomposer2.decompose()
from PyEMD import EEMD
eemd = EEMD()
imfs = eemd.eemd(data)
# imfs = data2
# data2 = np.fromfile('data.text')

# print(imfs)
fig1 = plt.figure(figsize=(5, 5))

plot_imfs(data, imfs, time_samples=time, fig=fig1)

#   give frequency - axis resolution for hilbert - spectrum
freqsol = 33
#   give time - axis resolution for hilbert - spectrum
timesol = 50

t0=time[0]
t1=time[-1]

dt = (t1-t0)/(len(time)-1)
freq, amp = FAhilbert(imfs, dt)
# freq, amp = ff, aa
print(freq.shape, imfs.shape)
fig2 = plt.figure(figsize=(5, 5))
plot_frequency(data, freq.T, time_samples=time, fig=fig2)

fw0 = np.min(np.min(freq))
fw1 = np.max(np.max(freq))

if fw0 <= 0:
    fw0 = np.min(np.min(freq[freq > 0]))

fw = fw1-fw0
tw = t1-t0
bins = np.linspace(0, 12, freqsol) #np.logspace(0, 10, freqsol, base=2.0)
p = np.digitize(freq, 2**bins)
# print(p)
t = np.ceil((timesol-1)*(time-t0)/tw)
t = t.astype(int)

hilbert_spectrum = np.zeros([timesol, freqsol])
for i in range(len(time)):
    for j in range(imfs.shape[0]-1):
        if p[i, j] >= 0 and p[i, j]<freqsol:
            hilbert_spectrum[t[i], p[i, j]] += amp[i, j]
# for i in range(timesol):
#     for j in range(freqsol):
#         if hilbert_spectrum[i, j]== 0.:
#             hilbert_spectrum[i, j] = -999.
#         else:
#             hilbert_spectrum[i, j] = math.log(hilbert_spectrum[i, j], 2)
# print(hilbert_spectrum[hilbert_spectrum>0].shape)
hilbert_spectrum = abs(hilbert_spectrum)
fig0 = plt.figure(figsize=(5, 5))
# ax = plt.subplot(1, 1, 1)
ax = plt.gca()
# c = ax.contourf(np.linspace(t0,t1,timesol),np.linspace(fw0,fw1,freqsol), hilbert_spectrum.T) #, colors=('whites','lategray','navy','darkgreen','gold','red')
c = ax.contourf(np.linspace(t0,t1,timesol), bins, hilbert_spectrum.T) #, colors=('whites','lategray','navy','darkgreen','gold','red')
ax.invert_yaxis()
ax.set_yticks(np.linspace(1, 11, 11))
Yticks = [float(math.pow(2, p)) for p in np.linspace(1, 11, 11)]  # make 2^periods
ax.set_yticklabels(Yticks)
ax.set_xlabel('Time', fontsize=8)
ax.set_ylabel('Period', fontsize=8)
position = fig0.add_axes([0.2, 0.05, 0.6, 0.01])
cbar = plt.colorbar(c, cax=position, orientation='horizontal')
cbar.set_label('Power')

plt.show()
