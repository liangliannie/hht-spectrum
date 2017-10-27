
def compute_hht(imfs, time):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.tri as mtri
    # time /= 365
    n_imfs = imfs.shape[0]
    all_frequency = []
    ax = pl.subplot(1, 1, 1)
    for i in range(n_imfs - 1):
        inst_amp, phase = hilb(imfs[i, :], unwrap=True)
        inst_freq = np.diff(phase)
        inst_freq = np.insert(inst_freq, 1, len(inst_freq))
        # inst_freq
        all_frequency.extend(inst_freq)
    all_frequency = map(abs, all_frequency)
    max_frequency = max(all_frequency)
    min_frequency = min(all_frequency)
    bins = np.linspace(math.log(min_frequency, 2), math.log(max_frequency, 2), 20)
    hilbert_spectrum = np.zeros((len(bins),len(time)))

    for i in range(n_imfs - 1):
        inst_amp, phase = hilb(imfs[i, :], unwrap=True)
        inst_freq = np.diff(phase)
        inst_freq = np.insert(inst_freq, 1, len(inst_freq))
        inst_freq = map(abs, inst_freq)
        # print([math.log(x, 2) for x in inst_freq])

        # inst_freq
        index = np.digitize([math.log(x, 2) for x in inst_freq], bins)
        for j, amp in enumerate(inst_amp):
            hilbert_spectrum[index[j]-1, j] += abs(amp*np.exp(1j*inst_freq[j]))
    # print(hilbert_spectrum[hilbert_spectrum!=0])
    c = ax.contourf(time, bins, hilbert_spectrum, 10) #, colors=('whites','lategray','navy','darkgreen','gold','red')
    pl.colorbar(c, ax=ax)
