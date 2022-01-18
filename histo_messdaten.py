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

h5_filename = 'Testdaten/KM3Net_{}_{}'.format(HV,SN)
pdf_name = 'KM3Net_{}_{}'.format(HV,SN)
y_values, Measurement_time, YOFF, YMU, samplerate, HV = data.values(filename = '{}.h5'.format(h5_filename))
x,y = data.x_y_values(y_values, Measurement_time, YOFF, YMU, samplerate)
#data.mean_plot(waveforms)
hist, bin = data.hist(waveforms)#,sig_min=750,sig_max=1500, plot = False, name = '{}.pdf'.format(pdf_name))
p_opt, cov = curve_fit(data.doppel_gauss,bin[:-1],hist, p0 = [1.4,3e-3,1.8,1.2,0.01,200])
fig,ax = plt.subplots(figsize = (10,5))
ax.plot(bin[:-1],hist)
ax.plot(bin,data.doppel_gauss(bin,*p_opt))
plt.semilogy()
plt.ylim(.1, 1e5)
plt.show()
#[1.39810324e+00 3.32207994e-03 1.76845189e+00 1.40254820e+00 1.86465628e-02 2.15435983e+02]
hist, bin = data.hist(y,sig_min=750,sig_max=1500, plot = False, name = '{}.pdf'.format(pdf_name))
p_opt, cov = curve_fit(data.doppel_gauss,bin[:-1],hist, p0 = [0,3e-3,1.8,0.1,0.01,200])
fig,ax = plt.subplots(figsize = (10,5))
ax.plot(bin[:-1],hist)
ax.plot(bin,data.doppel_gauss(bin,*p_opt))
#plt.semilogy()
#plt.ylim(.1, 1e5)
plt.show()
