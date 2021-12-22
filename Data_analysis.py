import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.constants as c
e = c.e
def read_from_file(filepath):
    retval = dict()
    with h5py.File(filepath, 'r') as file:
        retval['samplerate'] = float(file.attrs[u'samplerate'])
        retval['measurement_time'] = float(file.attrs[u'measurement_time'])
        retval['y_off'] = float(file.attrs[u'y_off'])
        retval['YMULT'] = float(file.attrs[u'YMULT'])
        retval['HV'] = float(file.attrs[u'HV'])
        retval['data'] = np.asarray(file['waveforms'])
    return retval


def save_to_file(filepath, data, Measurement_time, y_off, YMULT, samplerate, HV):
    with h5py.File('{}.h5'.format(filepath),'w') as file:
        file.attrs[u'samplerate'] = samplerate
        file.attrs[u'measurement_time'] = Measurement_time
        file.attrs[u'y_off'] = y_off
        file.attrs[u'YMULT'] = YMULT
        file.attrs[u'HV'] = HV
        file.create_dataset('waveforms', data = data)

def numinteg(x,y):
    return np.sum((x[1]-x[0])*y)


def values(retval = None, filename = None):
    '''Method for reading Measured values. From retval and filename one has to be none '''
    if(retval == None and filename == None):
        print('Error 418 I\'m a teapod')
        return
    elif filename != None and retval != None:
        print('Error 418 I\'m a teapod')
        return
    elif retval == None and filename != None:
        retval = read_from_file(filename)
    samplerate = retval['samplerate']
    measurement_time = retval['measurement_time']
    y_off = retval['y_off']
    YMULT = retval['YMULT']
    HV = retval['HV']
    data = retval['data']
    return data, measurement_time, y_off, YMULT, samplerate, HV

def x_y_values(data, Measurement_time, y_off, YMULT, samplerate = None):
    x = np.linspace(0,Measurement_time,len(data[0,:]))
    y = (data-y_off)*YMULT
    return x,y

def number_of_electrons(x,y,R=50):
    return numinteg(x,y)*R/e

def plot(x,y, figsize = (10,5)):
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(x,y)
    plt.show()
