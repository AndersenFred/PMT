import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.optimize as opt
from datetime import datetime
import scipy.constants
from peeemtee.pmt_resp_func import ChargeHistFitter
e = scipy.constants.e
import time
from scipy.optimize import curve_fit as curve_fit


class WavesetReader:
    def __init__(self, h5_filename):
        self.filename = h5_filename
        self._wavesets = None
        self._keys = None

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
            try:
                fit_results = (f[key]['fit_results']['gain'][()],f[key]['fit_results']['nphe'][()],f[key]['fit_results']['gain_err'][()],f[key]['fit_results']['int_ranges'][()])
            except KeyError:
                fit_results = None
        return Waveset(raw_waveforms, v_gain, h_int,Measurement_time,y_off, fit_results = fit_results)
def linear(x,m,t):
    return m*x+t

class Waveset:
    def __init__(self, raw_waveforms, v_gain, h_int,Measurement_time, y_off, fit_results = None):
        self.raw_waveforms = raw_waveforms
        self.v_gain = v_gain
        self.h_int = h_int
        self.Measurement_time = Measurement_time
        self.samplerate = 1/h_int
        self.y_off = y_off
        self._waveforms = None
        self.fit_results = fit_results


    @property
    def waveforms(self):
        if self._waveforms is None:
            self._waveforms = self.raw_waveforms * self.v_gain+self.y_off
        return self._waveforms

    def zeroed_waveforms(self, baseline_min, baseline_max):
        return (self.waveforms.T- np.mean(self.waveforms[:, baseline_min:baseline_max], axis=1)).T

def save_rawdata_to_file( h5_filename, data, Measurement_time, y_off, YMULT, samplerate, HV):
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
    return i

def add_fit_results(h5_filename, HV, gain, nphe, gain_err, int_ranges):
    with h5py.File(h5_filename, "a") as f:
        fit_results = f.create_group(f"{HV}/fit_results")
        fit_results["nphe"] = nphe
        fit_results["gain"] = gain
        fit_results["gain_err"] = gain_err
        fit_results["int_ranges"] = int_ranges
        f.close()

def rewrite_fit_results(h5_filename, HV, gain, nphe, gain_err, int_ranges):
    with h5py.File(h5_filename, "r+") as f:
        fit_results = f[f"{HV}/fit_results"]
        del fit_results["nphe"]
        fit_results["nphe"] = nphe
        del fit_results["gain"]
        fit_results["gain"] = gain
        del fit_results["gain_err"]
        fit_results["gain_err"] = gain_err
        del fit_results["int_ranges"]
        fit_results["int_ranges"] = int_ranges
        f.close()

def x_y_values(data,  y_off, YMULT, Measurement_time = None, samplerate = None, h_int = None):
    '''Method to get the x and y values from rawdata'''
    if not Measurement_time == None:
        x = np.linspace(0,Measurement_time,len(data[0,:]))
    elif not samplerate == None:
        x = np.linspace(0,len(data[0,:])/samplerate,len(data[0,:]))
    elif not h_int == None:
        x = np.linspace(0,len(data[0,:])*h_int,len(data[0,:]))
    else:
        raise AttributeError('Either Measurement_time or samplerate or h_int must not be None')
    y = (data)*YMULT+y_off
    return x,y


def mean_plot(y, int_ranges = (60, 180, 200, 350)):
    fig, ax = plt.subplots(figsize = (10,5))
    ax.axvspan(int_ranges[0], int_ranges[1], facecolor="red", alpha=0.5)
    ax.axvspan(int_ranges[2], int_ranges[3], facecolor="green", alpha=0.5)
    plt.xlabel("time in sample")
    plt.ylabel("Voltage in V")
    plt.title("mean waveform")
    y_data = np.mean(y, axis = 0)
    y_std = np.std(y,axis=0)/np.sqrt(len(y))
    ax.plot(y_data)
    ax.fill_between(np.linspace(1,len(y_data),len(y_data)), y_data-y_std, y_data + y_std, color='gray', alpha=0.2)
    plt.show(block = False)

def hist_variable_sig_max(waveforms, ped_min, ped_max, sig_min, sig_max_start,h_int, interval = 10, number = 10):
    gains = []
    gain_errs = []
    sig_end = []
    for i in range(number):
        try:
            x,y, int_ranges = histogramm(waveforms, ped_min, ped_max, sig_min, sig_max_start+interval*i)
            gain, nphe, gain_err = hist_fitter(x,y,h_int, plot = False)
            print(gain)
            gains.append(gain)
            sig_end.append(sig_max_start+interval*i)
            gain_errs.append(gain_err)
        except ValueError:
            continue
        except TypeError:
            continue
    return np.array(gains), np.array(sig_end), np.array(gain_errs)

def hist_variable_sig_min(waveforms, ped_min, ped_max, sig_min_start, sig_max,h_int, interval = 10, number = 10):
    gains = []
    gain_errs = []
    sig_end = []
    for i in range(number):
        try:
            x,y, int_ranges = histogramm(waveforms, ped_min, ped_max, sig_min_start-interval*i, sig_max)
            gain, nphe, gain_err = hist_fitter(x,y,h_int, plot = False)
            print(gain)
            gains.append(gain)
            sig_end.append(sig_min_start-interval*i)
            gain_errs.append(gain_err)
        except ValueError:
            continue
        except TypeError:
            continue
    return np.array(gains), np.array(sig_end), np.array(gain_errs)


def hist_variable_values(waveforms, ped_min, ped_max, sig_min_start, sig_max_start,h_int, interval_sig_min = 10,interval_sig_max = 10, number_sig_min = 10, number_sig_max = 10):
    gains = np.zeros((number_sig_min, number_sig_max))
    sig_min = np.linspace(sig_min_start-number_sig_min*interval_sig_min, sig_min_start, number_sig_min)
    sig_max = np.linspace(sig_max_start, sig_max_start+interval_sig_max*number_sig_max, number_sig_max)
    for i in range(number_sig_min):
        for j in range(number_sig_max):
            try:
                x,y, int_ranges = histogramm(waveforms, ped_min, ped_max, sig_min_start-interval_sig_min*i, sig_max_start+interval_sig_max*j)
                gain, nphe, gain_err = hist_fitter(x,y,h_int, plot = False)
                gains[i,j] = gain
            except ValueError:
                gains[i,j] = None
                continue
            except TypeError:
                gains[i,j] = None
                continue
    return sig_min, sig_max, gains

def plot_hist_variable_values(sig_min, sig_max, gains, name):
    X,Y = np.meshgrid(sig_min, sig_max)
    fig,ax = plt.subplots(figsize = (10,5))
    p = ax.pcolormesh(X,Y,gains/1e6)
    plt.xlabel("signal min")
    plt.ylabel("signal_max")
    fig.colorbar(p, ax = ax)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()



def hist(waveforms, ped_min=0, ped_max= 100, sig_min= 190, sig_max=400, bins = 200, histo_range= None, plot = False, name = None,title = None):
    int_ranges = (ped_min, ped_max, sig_min, sig_max)
    mean_plot(waveforms,int_ranges)
    try:
        ped_min, ped_max, sig_min, sig_max = [int(i) for i in input('set integration ranges: <ped_min, ped_max, sig_min, sig_max>\n').split(', ')]
        if ped_min<0 or ped_max<0 or sig_min<0 or sig_max<0:
            raise ValueError
    except ValueError:
        print(f'ValueError: used integration ranges {int_ranges}')
    return histogramm(waveforms, ped_min, ped_max, sig_min, sig_max, bins, histo_range, plot, name, title)

def histogramm(waveforms, ped_min=0, ped_max= 100, sig_min= 190, sig_max=400, bins = 200, histo_range= None, plot = False, name = None,title = None):
    ped_sig_ratio = (ped_max - ped_min) / (sig_max - sig_min)
    pedestals = (np.sum(waveforms[:, ped_min:ped_max], axis=1)) * ped_sig_ratio
    charges = (np.sum(waveforms[:, sig_min:sig_max], axis=1))-pedestals
    if plot:
        fig, ax = plt.subplots(figsize= (10,5))
        range = (np.min(charges),np.max(charges))
        ax.hist(charges, range=range, bins=200, log=True,color = 'b')
        ax.hist(pedestals, range=range, bins=200, log=True,color = 'r')
        plt.xlabel("Number of Photoelektrons")
        plt.ylabel("Number of events")
        if not title == None:
            plt.title(title)
        plt.show()
        if not name == None:
            print(f'saving {name}')
            fig.savefig(name + '.pdf')
    hi, bin_edges = np.histogram(charges, range = histo_range, bins = bins)
    bin_edges = (bin_edges-(bin_edges[1])/2)[:-1]
    return hi, bin_edges, np.array([ped_min,ped_max,sig_min,sig_max])

def hist_fitter(hi, bin_edges, h_int, plot = True):#input has to be from hist()
    fitter = ChargeHistFitter()
    fitter.pre_fit(bin_edges, hi)
    fit_function = fitter.fit_pmt_resp_func(bin_edges,hi)
    fitter.opt_ped_values
    fitter.opt_spe_values
    if (plot):
        plt.semilogy(bin_edges, hi)
        plt.plot(bin_edges,fitter.opt_prf_values)
        plt.ylim(.1,1e5)
        plt.show(block = False)
    gain = fitter.popt_prf['spe_charge']*h_int/(50*e)
    nphe = fitter.popt_prf['nphe']
    gain_err = np.sqrt(fitter.pcov_prf['spe_charge', 'spe_charge'])*h_int/(50*e)
    return gain, nphe, gain_err

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
    for i in range(len(waveforms[:,0])):
        for j in range(len(waveforms[i:])):
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

def log_transit_spread(name,SN,n,N,bins,binwidth,p0,cov):
    name = '{}.txt'.format(name)
    f = open(name, 'a')
    text = 'Date = {8},\n n_triggerd = {0}\n N = {1}\n Number Photoelektrons = {2}\n Histparameter:\n bins = {3}, binwidth = {4},\n Fitparameter: mu[ns], sigma[ns], Ampl= {5},\n Delta_mu, Delta_sigma, delta_Ampl = {6},\n cov =\n {7}\n\n'.format(n,N,-np.log(1-n/N),bins,binwidth,p0,np.sqrt(np.diag(cov)),cov,datetime.now(),SN)
    f.write(text)
    f.close()


def analysis_complete_data(h5_filename,nom_manuf_hv,reanalyse= False, saveresults = True, nominal_gains = [3e6], SN = 'AB2363'):
    f = WavesetReader(h5_filename)
    hv = []
    gains = []
    nphes = []
    gain_errs = []
    int_range = []
    for key in f.wavesets:
        waveset = f[key]
        hv.append(int(key.split('_')[0]))
        if not reanalyse and not waveset.fit_results == None:
            waveset = f[key]
            gains.append(waveset.fit_results[0])
            h_int = waveset.h_int
            nphes.append(waveset.fit_results[1])
            gain_errs.append(waveset.fit_results[2])
            int_range.append(np.array(waveset.fit_results[3]))
        else:
            waveforms = waveset.waveforms
            h_int = waveset.h_int
            y,x, int_ranges = hist(waveforms, plot = True)
            int_range.append(int_ranges)
            gain, nphe, gain_err = hist_fitter(y,x,h_int)
            if reanalyse and saveresults:
                rewrite_fit_results(h5_filename = h5_filename, HV = key, gain = gain, nphe = nphe, gain_err = gain_err, int_ranges = int_ranges)
            elif saveresults:
                add_fit_results(h5_filename = h5_filename, HV = key, gain = gain, nphe = nphe, gain_err = gain_err, int_ranges = int_ranges)
            gains.append(gain)
            print(gain)
            nphes.append(nphe)
            gain_errs.append(gain_err)
    p_opt, cov = curve_fit(linear, np.log10(hv), np.log10(gains), sigma = np.log10(gain_errs))
    nominal_hvs = []
    for nominal_gain in nominal_gains:
        nominal_hvs.append(10**((np.log10(nominal_gain) - p_opt[1])/ p_opt[0]))
    #print(nominal_hvs)
    manuf_gain=10**(np.log10(nom_manuf_hv)*p_opt[0] +p_opt[1])
    fig, ax = plt.subplots(figsize = (10,5))
    for i in range(len(nominal_hvs)):
        ax.axhline(np.log10(nominal_gains[i]), color="black", ls="--")
        ax.axvline(np.log10(nominal_hvs[i]), color="black", ls="--")
    #gains,gain_errs = np.array(gains), np.array(gain_errs)
    #gain_err_plus = np.log10(gains+gain_errs)-np.log10(gains)
    #gain_err_minus = np.log10(gains-gain_errs)-np.log10(gains)
    ax.plot(np.log10(hv), np.log10(gains),'x')
    #print(np.log10(gains)-np.log10(gain_errs))
    hv_min=np.log10(min(hv)-50)
    hv_max=np.log10(max(hv)+50)
    xs = np.linspace(hv_min, hv_max, 1000)
    gain_min=hv_min*p_opt[0] +p_opt[1]
    gain_max=hv_max*p_opt[0] +p_opt[1]
    ax.vlines(x=np.log10(nom_manuf_hv), ymin=gain_min,ymax=np.log10(manuf_gain), colors="red", linestyles="dashed")
    plt.title("gainslope "+SN)
    ax.plot(xs, linear(xs, *p_opt))
    plt.axis([hv_min, hv_max, gain_min, gain_max])
    plt.xlabel("log10(HV)")
    plt.ylabel("log10(gain)")
    plt.show()
    err_nom_hv = err_hv(gain_nom = nominal_gain,gain = gains,gain_errs = gain_errs, hvs = hv)
    log_complete_data(name = h5_filename,hvs = hv,err_nom_hv = err_nom_hv, gains=gains, gain_errs = gain_errs, nphe = nphes,int_ranges = np.array(int_range),h_int = h_int, p_opt = p_opt, cov = cov,nominal_gain = nominal_gains, nominal_hv = nominal_hvs)
    print(f'nominal HV: {nominal_hvs} pm {err_nom_hv} for nominal gain: {nominal_gains}')
    return gains, nphes, hv, gain_errs

def err_hv(gain, gain_errs, gain_nom, hvs):
    p_opt, cov = curve_fit(linear,hvs, np.log10(gain)-np.log10(gain_nom), sigma = np.log10(gain_errs))
    return 10**np.sqrt(cov[1,1])

def log_complete_data(name, hvs, gains, gain_errs,err_nom_hv, nphe,int_ranges,h_int, p_opt, cov,nominal_gain, nominal_hv):
    name = '{}_log.txt'.format(name)
    f = open(name, 'a')
    int_time = (int_ranges[:,3]-int_ranges[:,2])*h_int*1e9
    gains = [np.format_float_scientific(i,4) for i in gains]
    gain_errs = [np.format_float_scientific(i,2) for i in gain_errs]

    text = f'Date: {datetime.now()}\n HV: {hvs}\n gains: {gains}\n gain_errs: {gain_errs}\n nphes: {np.round(nphe,2)}\n int ranges: {int_ranges};\n int_ranges in ns {int_ranges*h_int*1e9}\n integration time in ns {int_time}\n fit results: {np.round(p_opt,4)}\n cov: {np.round(cov,4)}\n nominal gain: {nominal_gain}\n nominal hv: {np.round(nominal_hv,0)}pm{np.round(err_nom_hv,2)}\n\n\n'
    f.write(text)
    f.close()
