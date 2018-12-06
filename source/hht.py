from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import math
from PyEMD import EEMD


def plot_imfs(signal, imfs, time_samples=None, fig=None):
    """
    plot_imfs function for the Hilbert Huang Transform is adopted from pyhht.
    Author: jaidevd https://github.com/jaidevd/pyhht/blob/dev/pyhht/visualization.py
    Plot the signal, IMFs.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
       The input signal.
    imfs : array-like, shape (n_imfs, n_samples)
       Matrix of IMFs as generated with the `EMD.decompose` method.
    time_samples : array-like, shape (n_samples), optional
       Time instants of the signal samples.
       (defaults to `np.arange(1, len(signal))`)
    -------
    `matplotlib.figure.Figure`
       The figure (new or existing) in which the decomposition is plotted.
        """
    n_imfs = imfs.shape[0]
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


def plot_frequency(signal, imfs, time_samples=None, fig=None):
    """
    plot_imfs function for the Hilbert Huang Transform is adopted from pyhht.
    Author: jaidevd https://github.com/jaidevd/pyhht/blob/dev/pyhht/visualization.py
    Plot the signal, IMFs.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
       The input signal.
    imfs : array-like, shape (n_imfs, n_samples)
       Matrix of IMFs as generated with the `EMD.decompose` method.
    time_samples : array-like, shape (n_samples), optional
       Time instants of the signal samples.
       (defaults to `np.arange(1, len(signal))`)
    -------
    `matplotlib.figure.Figure`
       The figure (new or existing) in which the instance frequency is plotted.
        """
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
    """
    Performs Hilbert transformation on imfs.
    Returns frequency and amplitude of signal.
    """
    n_imfs = imfs.shape[0]
    f = []
    a = []
    for i in range(n_imfs - 1):
        # upper, lower = pyhht.utils.get_envelops(imfs[i, :])
        inst_imf = imfs[i, :]  # /upper
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        inst_freq = (2 * math.pi) / np.diff(phase)  #

        inst_freq = np.insert(inst_freq, len(inst_freq), inst_freq[-1])
        inst_amp = np.insert(inst_amp, len(inst_amp), inst_amp[-1])

        f.append(inst_freq)
        a.append(inst_amp)
    return np.asarray(f).T, np.asarray(a).T


def hht(data, time, freqsol=33, timesol=50):
    """
    hht function for the Hilbert Huang Transform spectrum

    Parameters
    ----------
    data : array-like, shape (n_samples,)
       The input signal.
    time : array-like, shape (n_samples), optional
       Time instants of the signal samples.
       (defaults to `np.arange(1, len(signal))`)
    -------
    `matplotlib.figure.Figure`
       The figure (new or existing) in which the hht spectrum is plotted.

    example:
    --------------------

    .. sourcecode:: ipython
        f = Dataset('./source/obs.nc')
        # read one example data
        fsh = f.variables['FSH']
        time = f.variables['time']
        one_site = np.ma.masked_invalid(fsh[0,:])
        time = time[~one_site.mask]
        data = one_site.compressed()
        hht(data, time)


    ----------------
        """
    #   freqsol give frequency - axis resolution for hilbert - spectrum
    #   timesol give time - axis resolution for hilbert - spectrum
    t0 = time[0]
    t1 = time[-1]
    dt = (t1 - t0) / (len(time) - 1)

    eemd = EEMD()
    imfs = eemd.eemd(data)
    freq, amp = FAhilbert(imfs, dt)

    #     fw0 = np.min(np.min(freq)) # maximum frequency
    #     fw1 = np.max(np.max(freq)) # maximum frequency

    #     if fw0 <= 0:
    #         fw0 = np.min(np.min(freq[freq > 0])) # only consider positive frequency

    #     fw = fw1-fw0
    tw = t1 - t0

    bins = np.linspace(0, 12, freqsol)  # np.logspace(0, 10, freqsol, base=2.0)
    p = np.digitize(freq, 2 ** bins)
    t = np.ceil((timesol - 1) * (time - t0) / tw)
    t = t.astype(int)

    hilbert_spectrum = np.zeros([timesol, freqsol])
    for i in range(len(time)):
        for j in range(imfs.shape[0] - 1):
            if p[i, j] >= 0 and p[i, j] < freqsol:
                hilbert_spectrum[t[i], p[i, j]] += amp[i, j]

    hilbert_spectrum = abs(hilbert_spectrum)
    fig1 = plt.figure(figsize=(5, 5))
    plot_imfs(data, imfs, time_samples=time, fig=fig1)

    fig2 = plt.figure(figsize=(5, 5))
    plot_frequency(data, freq.T, time_samples=time, fig=fig2)

    fig0 = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    c = ax.contourf(np.linspace(t0, t1, timesol), bins,
                    hilbert_spectrum.T)  # , colors=('whites','lategray','navy','darkgreen','gold','red')
    ax.invert_yaxis()
    ax.set_yticks(np.linspace(1, 11, 11))
    Yticks = [float(math.pow(2, p)) for p in np.linspace(1, 11, 11)]  # make 2^periods
    ax.set_yticklabels(Yticks)
    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('Period', fontsize=8)
    position = fig3.add_axes([0.2, -0., 0.6, 0.01])
    cbar = plt.colorbar(c, cax=position, orientation='horizontal')
    cbar.set_label('Power')
    plt.show()
