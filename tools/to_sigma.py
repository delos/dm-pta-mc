# converts dm-pta-mc output to sigma based on Appendix C of 2012.09857

import numpy as np
from sys import argv
from glob import glob
from os.path import basename
from scipy.special import gammainc, erf, erfinv

def sigma_from_p(p):
    return np.sqrt(2) * erfinv(1 - 2*p)

def p_from_snr_earth(snr, NP=np.inf):
    return 1. - gammainc(0.5, np.sqrt(2*(1 - 1./NP)) * snr)

def p_from_snr_pulsar(snr, NP):
    return 1. - erf(snr/np.sqrt(2))**NP

if len(argv) < 2:
    raise ValueError("python script.py <directory>")

with open(argv[1] + "/sigma.txt", "wt") as fp:
    pass

filenames = glob(argv[1]+'/*')

for filename in filenames:
    try:
        universe, pulsar, snr = np.loadtxt(filename, delimiter=',').T

        if len(np.unique(pulsar)) == 1 and pulsar[0] == -1:
            calc = 'earth'
            NU = len(np.unique(universe))
            with open(filename) as fp:
                while True:
                    line = fp.readline()
                    if not line or len(line.lstrip()) == 0 or line.lstrip()[0] != '#':
                        NP = np.inf
                        break
                    if line.lstrip().split(' ')[1] == 'NUM_PULSAR':
                        NP = int(line.lstrip().split(' ')[3])
                        break
            snr = np.percentile(snr,10)
            sigma = sigma_from_p(p_from_snr_earth(snr, NP))
        else:
            calc = 'pulsar'
            NU = len(np.unique(universe))
            NP = len(np.unique(pulsar))
            snr.shape = (NU,NP)
            snr = snr.max(axis=1) # max over pulsars
            snr = np.percentile(snr,10)
            sigma = sigma_from_p(p_from_snr_pulsar(snr, NP))
        
        with open(argv[1] + "/sigma.txt", "at") as fp:
            fp.write("%s , %f\n"%(basename(filename),sigma))
    except Exception as e:
        continue

    print("%s : %s, %d universes, %s pulsars; SNR=%f; sigma=%f"%(filename, calc, NU, str(NP),snr,sigma))

print("wrote to sigma.txt")
