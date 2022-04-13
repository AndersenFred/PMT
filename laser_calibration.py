import numpy as np
import matplotlib.pyplot as plt
import Data_analysis as data
import sys
from scipy.optimize import curve_fit as curve_fit
def gauss(x,mu,sigma,a):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
try:
    HV = int(sys.argv[2])
except IndexError:
    print('usage: python {} <serial number> <high voltage>'.format(sys.argv[0]))
    sys.exit()
except TypeError:
    print('usage: python {} <serial number> <high voltage>'.format(sys.argv[0]))
    sys.exit()

h5_filename = 'Laser_Transit_Spread/Laser_kallibrierung_{}'.format(HV)
y_values, Measurement_time, YOFF, YMU, samplerate, HV = data.values(filename = '{}.h5'.format(h5_filename))
x,y = data.x_y_values(y_values, Measurement_time, YOFF, YMU, samplerate)
#print(y.shape)
#x = np.linspace(1,len(waveforms), len(waveforms))
fig,ax = plt.subplots(figsize =(10,5))
j  = data.transit_time_spread(y)
bins = len(x)
x = x*1e9
hist, bin, patches = ax.hist(x[j], bins = bins)
start_fit = 0
p, cov = curve_fit(data.gauss,bin[start_fit:-1],hist[start_fit:],p0 =[10,1,100] )
start_fit = int((p[0]-2*p[1])/x[-1]*len(x))
end_fit =int((p[0]+2*p[1])/x[-1]*len(x))
print(p)
p, cov = curve_fit(data.gauss,bin[start_fit:end_fit],hist[start_fit:end_fit],p0 =[10,1,100] )
ax.plot(bin,data.gauss(bin,*p),label = 'sigma = {} ns'.format(p[1]))
ax.set_ylabel('# of Events')
ax.set_xlabel('time in ns')

ax.legend()
print(1-len(j)/len(y[0,:]))
fig.savefig('Laser_Transit_Spread/Laser_new_transit_time_spread_{}.pdf'.format(HV))
data.log_transit_spread(n=len(j),N=len(y[0,:]),bins=bins, binwidth = '{}ns'.format((x[1]-x[0])),p0=p,cov=cov)
plt.show()
