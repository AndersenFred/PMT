import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.optimize as opt
from datetime import datetime
import scipy.constants
from peeemtee.pmt_resp_func import ChargeHistFitter
e = scipy.constants.e
from scipy.optimize import curve_fit as curve_fit

class WavesetReader:
    def __init__(self, filename):
        self.filename = filename
        self._wavesets = None

    @property
    def wavesets(self):
        if self._wavesets is None:
            with h5py.File(self.filename, "r") as f:
                self._wavesets = list(f.keys())
        return self._wavesets

    def __getitem__(self, key):
        with h5py.File(self.filename, "r") as f:
            raw_waveforms = f[f"{key}/waveforms"][:]
            v_gain = f[f"{key}/waveform_info/v_gain"][()]
            h_int = f[f"{key}/waveform_info/h_int"][()]
            Measurement_time = f[f"{key}/waveform_info/Measurement_time"][()]
            y_off = f[f"{key}/waveform_info/y_off"][()]
        return Waveset(raw_waveforms, v_gain, h_int,Measurement_time,y_off)

class Waveset:
    def __init__(self, raw_waveforms, v_gain, h_int,Measurement_time, y_off):
        self.raw_waveforms = raw_waveforms
        self.v_gain = v_gain
        self.h_int = h_int
        self.Measurement_time = Measurement_time
        self.samplerate = 1/h_int
        self.y_off = y_off
        self._waveforms = None

    @property
    def waveforms(self):
        if self._waveforms is None:
            self._waveforms = self.raw_waveforms * self.v_gain+self.y_off
        return self._waveforms

    def zeroed_waveforms(self, baseline_min, baseline_max):
        return (self.waveforms.T- np.mean(self.waveforms[:, baseline_min:baseline_max], axis=1)).T

def save_rawdata_to_file(h5_filename, data, Measurement_time, y_off, YMULT, samplerate, HV):
    f = h5py.File(h5_filename, 'a')
    i=0
    while(True):
        try:
            f.create_dataset(f"{HV}_{i}/waveforms", data=data, dtype=np.int8)
            wf_info = f.create_group(f"{HV}_{i}/waveform_info")
            break
        except ValueError:
            i+=1
            print(i)
    wf_info["h_int"] = 1/samplerate
    wf_info["v_gain"] =  YMULT
    wf_info["Measurement_time"] = Measurement_time
    wf_info["y_off"] = y_off
    f.close()


def doppel_gauss(x,mu_1,sig_1,ampl_1,mu_2,sig_2,ampl_2):
    '''A function with two normal distributions'''
    return ampl_1/(np.sqrt(2*np.pi)*sig_1)*np.exp(-(x-mu_1)**2/(2*sig_1**2))\
    +ampl_2/(np.sqrt(2*np.pi)*sig_2)*np.exp(-(x-mu_2)**2/(2*sig_2**2))

def gauss(x,mu,sigma,a):
    ''' A function with a normal distribution'''
    return a*np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)



def x_y_values(data,  y_off, YMULT, Measurement_time = None, samplerate = None):
    '''Method to get the x and y values from rawdata'''
    if not Measurement_time ==None:
        x = np.linspace(0,Measurement_time,len(data[:,0]))
    elif not samplerate == None:
        x = np.linspace(0,len(data[0,:])/samplerate,len(data[:,0]))
    else:
        raise AttributeError('Either Measurement_time or samplerate must not be None')
    y = ((data)*YMULT+y_off).T
    return x,y

def numinteg(x,y):
    '''simple Method for numeric integration'''
    return np.sum((x[1]-x[0])*y)

def number_of_electrons(x,y,R=50):
    '''simple Method for estimating the number of electrons passing the oszi'''
    return numinteg(x,y)*R/e

def plot(x,y,figsize = (10,5), index = False):
    '''plot for lazy people'''
    if (not index):
        x = np.linspace(1,len(y),len(y))
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(x,y)
    plt.show()

def mean_plot(y):
    fig, ax = plt.subplots(figsize = (10,5))
    ax.plot(np.mean(y, axis = 0))
    plt.show()

def hist(waveforms, ped_min=60, ped_max=180, sig_min=200, sig_max=400, histo_range= None, plot = True, name = None):
    ped_sig_ratio = (ped_max - ped_min) / (sig_max - sig_min)
    pedestals = np.sum(waveforms[:, ped_min:ped_max], axis=1)
    charges = (np.sum(waveforms[:, sig_min:sig_max], axis=1) - pedestals * ped_sig_ratio)
    if plot:
        fig, ax = plt.subplots(figsize= (10,5))
        ax.hist(charges, range=None, bins=200, log=True)
        plt.show()
        if not name == None :
            fig.savefig(name)
    return np.histogram(charges, range = histo_range, bins = 200)

def fit(x,y, p0 = [], plot = True):
    x = x[:-1]
    p_opt, cov = curve_fit(doppel_gauss,x,y)
    if plot:
        plt.plot(x,y)
        plt.plot(x,doppel_gauss(x,*p_opt))
        plt.semilogy()
        plt.show()
    return p_opt, cov

def transit_time_spread(waveforms,threshold=0.01):
    k = []
    for i in range(len(waveforms[0,:])):
        for j in range(len(waveforms[:,i])):
            if waveforms[j,i] > threshold:
                k.append(j)
                break
    return k
def transit_time_spread_Testdaten(waveforms,threshold=0.008):
    k = []
    for i in range(len(waveforms[:,0])):
        for j in range(len(waveforms[i,:])):
            if waveforms[i,j] < threshold:
                k.append(j)
                break
    return k

def log_transit_spread(SN,n,N,bins,binwidth,p0,cov):
    name = 'Laser_Transit_Spread/log_transit_time_spread_{}.txt'.format(SN)
    f = open(name, 'a')
    text = 'Date = {8},\n n_triggerd = {0}\n N = {1}\n Number Photoelektrons = {2}\n Histparameter:\n bins = {3}, binwidth = {4},\n Fitparameter: mu[ns], sigma[ns], Ampl= {5},\n Delta_mu, Delta_sigma, delta_Ampl = {6},\n cov =\n {7}\n\n'.format(n,N,-np.log(1-n/N),bins,binwidth,p0,np.sqrt(np.diag(cov)),cov,datetime.now(),SN)
    f.write(text)
    f.close()
