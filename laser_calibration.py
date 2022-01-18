import Inst_control as inst
import time
import numpy as np
import matplotlib.pyplot as plt
import Data_analysis as data
import sys
from scipy.optimize import curve_fit as curve_fit
import peeemtee as pt
filename = "/home/pmttest/Desktop/Frederik/Testdaten/BA0455.h5"
reader = pt.WavesetReader(filename)
waveset = reader["1088"]
waveforms = waveset.waveforms
def gauss(x,mu,sigma,a):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
try:
    SN = int(sys.argv[1])
    HV = int(sys.argv[2])
except IndexError:
    print('usage: python {} <serial number> <high voltage>'.format(sys.argv[0]))
    HV = 1400
    SN = 1
except TypeError:
    print('usage: python {} <serial number> <high voltage>'.format(sys.argv[0]))
    sys.exit()

h5_filename = 'KM3Net_{}_{}'.format(HV,SN)
y_values, Measurement_time, YOFF, YMU, samplerate, HV = data.values(filename = '{}.h5'.format(h5_filename))
x,y = data.x_y_values(y_values, Measurement_time, YOFF, YMU, samplerate)
fig,ax = plt.subplots(figsize =(10,5))
j  = data.transit_time_speed(y)
hist, bin = np.histogram(x[j], bins = 200)
p, cov = curve_fit(data.gauss,bin[:-1],hist,p0=[2e-8,1e-8,35])
ax.plot(bin,data.gauss(bin,*p),label = 'sigma = {} s'.format(p[1]))
ax.set_ylabel('# of Events')
ax.set_xlabel('time')
ax.plot(bin[:-1],hist)
ax.legend()
fig.savefig('Laser_new_transit_time_speed.pdf')
plt.show()
