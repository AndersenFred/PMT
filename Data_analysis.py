import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.constants
import bisect
from peeemtee.pmt_resp_func import ChargeHistFitter
e = scipy.constants.e
from scipy.optimize import curve_fit as curve_fit

def doppel_gauss(x,mu_1,sig_1,ampl_1,mu_2,sig_2,ampl_2):
    return ampl_1/(np.sqrt(2*np.pi)*sig_1)*np.exp(-(x-mu_1)**2/(2*sig_1**2))\
    +ampl_2/(np.sqrt(2*np.pi)*sig_2)*np.exp(-(x-mu_2)**2/(2*sig_2**2))
def gauss(x,mu,sigma,a):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def read_rawdata_from_file(filepath):
    retval = dict()
    with h5py.File(filepath, 'r') as file:
        retval['samplerate'] = float(file.attrs[u'samplerate'])
        retval['measurement_time'] = float(file.attrs[u'measurement_time'])
        retval['y_off'] = float(file.attrs[u'y_off'])
        retval['YMULT'] = float(file.attrs[u'YMULT'])
        retval['HV'] = float(file.attrs[u'HV'])
        retval['data'] = np.asarray(file['waveforms'])
    return retval

def save_rawdata_to_file(filepath, data, Measurement_time, y_off, YMULT, samplerate, HV):
    with h5py.File('{}.h5'.format(filepath),'w') as file:
        file.attrs[u'samplerate'] = samplerate
        file.attrs[u'measurement_time'] = Measurement_time
        file.attrs[u'y_off'] = y_off
        file.attrs[u'YMULT'] = YMULT
        file.attrs[u'HV'] = HV
        file.create_dataset('waveforms', data = data)

def values(retval = None, filename = None):
    '''Method for reading Measured values. From retval and filename one has to be none '''
    if(retval == None and filename == None):
        print('either retval or filename must not be none')
        return
    elif filename != None and retval != None:
        print('either retval or filename must not be none')
        return
    elif retval == None and filename != None:
        retval = read_rawdata_from_file(filename)
    samplerate = retval['samplerate']
    measurement_time = retval['measurement_time']
    y_off = retval['y_off']
    YMULT = retval['YMULT']
    HV = retval['HV']
    data = retval['data']
    return data, measurement_time, y_off, YMULT, samplerate, HV

def x_y_values(data, Measurement_time, y_off, YMULT, samplerate = None):
    x = np.linspace(0,Measurement_time,len(data[0,:]))
    y = ((data)*YMULT+y_off).T
    return x,y
def numinteg(x,y):
    return np.sum((x[1]-x[0])*y)

def number_of_electrons(x,y,R=50):
    return numinteg(x,y)*R/e

def plot(x,y,figsize = (10,5), index = False):
    if (not index):
        x = np.linspace(1,len(y),len(y))
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(x,y)
    plt.show()

def mean_plot(y):
    fig, ax = plt.subplots(figsize = (10,5))
    ax.plot(np.mean(y, axis = 0))
    plt.show()

def hist(waveforms, ped_min=60, ped_max=300, sig_min=500, sig_max=1500, histo_range= None, plot = True, save = False, name = None):
    ped_sig_ratio = (ped_max - ped_min) / (sig_max - sig_min)
    pedestals = np.sum(waveforms[:, ped_min:ped_max], axis=1)
    charges = (np.sum(waveforms[:, sig_min:sig_max], axis=1) - pedestals * ped_sig_ratio)
    if plot:
        plt.hist(charges, range=None, bins=200, log=True)
        plt.show()
        if save:
            fig, ax = plt.subplots(figsize= (10,5))
            ax.hist(charges, range=None, bins=200, log=True)
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

def transit_time_speed(waveforms,threshold=0.01):
    k = []
    for i in range(len(waveforms[0,:])):
        for j in range(len(waveforms[:,i])):
            if waveforms[j,i] > threshold:
                k.append(j)
                break
    return k
