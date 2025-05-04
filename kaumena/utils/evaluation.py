import numpy as np


def sdr(est, ref):
    ratio = np.sum(ref**2) / np.sum((ref-est)**2)
    return 10*np.log10(ratio + 1e-10)