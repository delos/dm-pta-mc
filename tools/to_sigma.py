# converts dm-pta-mc output to sigma based on Appendix C of 2012.09857

import numpy as np
from sys import argv
from glob import glob
from os.path import basename
from scipy.special import gammainc, erf, erfcinv

# sigma from p
def sigma_from_p(p):
    return np.sqrt(2) * erfcinv(2*p)

# p for earth calculation

def p_from_snr_earth(snr, NP=np.inf):
    return 1. - gammainc(0.5, np.sqrt(2*(1 - 1./NP)) * snr)

# p for pulsar calculation


def _p_from_snr_pulsar_s(snr, NP):
  return 1. - erf(snr/np.sqrt(2))**NP
def _p_from_snr_pulsar_l(snr, NP):
  x = snr/np.sqrt(2)
  return NP * np.exp(-x**2) / (x * np.sqrt(np.pi))
def p_from_snr_pulsar(snr, N):
  return np.select([snr<7.5,snr>=7.5],[
                      _p_from_snr_pulsar_s(snr,N),
                      _p_from_snr_pulsar_l(snr,N),
                      ])

# sigma for pulsar calculation  

def _sigma_p_s(snr,N):
  return sigma_from_p(p_from_snr_pulsar(snr,N))
  
def _sigma_p_l(snr,N):
  L1 = snr**2 + 2*np.log(snr/N)
  L2 = np.log(L1)
  W = L1 - L2 + L2/L1 + L2*(-2+L2)/(2*L1**2) + L2*(6-9*L2+2*L2**2)/(6*L1**3) + L2*(-12+36*L2-22*L2**2+3*L2**3)/(12*L1**4)
  return np.sqrt(W)

def sigma_p(snr,N):
  return np.piecewise(snr,[snr<=36,snr>36],[
                      lambda s: _sigma_p_s(s,N),
                      lambda s: _sigma_p_l(s,N),
                      ])

# sigma for earth calculation

def sigma_e(snr,N):
  return sigma_from_p(p_from_snr_earth(snr, N))

# ---

if len(argv) < 2:
    raise ValueError("python script.py <directory>")

with open(argv[1] + "/sigma.txt", "wt") as fp:
    fp.write("# outputfile , NP , SNR , sigma\n")

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
            sigma = sigma_e(snr, NP)
        else:
            calc = 'pulsar'
            NU = len(np.unique(universe))
            NP = len(np.unique(pulsar))
            snr.shape = (NU,NP)
            snr = snr.max(axis=1) # max over pulsars
            snr = np.percentile(snr,10)
            sigma = sigma_p(snr, NP)
        
        with open(argv[1] + "/sigma.txt", "at") as fp:
            fp.write("%s , %d , %f , %f\n"%(basename(filename),NP,snr,sigma))
    except Exception as e:
        continue

    print("%s : %s, %d universes, %s pulsars; SNR=%f; sigma=%f"%(filename, calc, NU, str(NP),snr,sigma))

print("wrote to sigma.txt")
