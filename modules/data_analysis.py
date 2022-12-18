import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
from datetime import datetime
import scipy.constants
from modules.pmt_resp_func import ChargeHistFitter
from scipy.optimize import curve_fit as curve_fit
import os
import multiprocessing as mp
e = scipy.constants.e


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
            y_off = f[f"{key}/waveform_info/y_off"][()]
            try:

                fit_results = (f[key]['fit_results']['gain'][()], f[key]['fit_results']['nphe'][()], f[key]['fit_results']['gain_err'][()], f[key]['fit_results']['int_ranges'][()])
            except KeyError:
                fit_results = None
        return Waveset(raw_waveforms, v_gain, h_int, y_off = y_off, fit_results = fit_results)


def linear(x, m, t):
    return m*x+t


class Waveset:
    def __init__(self, raw_waveforms, v_gain, h_int, y_off, fit_results = None):
        self.raw_waveforms = raw_waveforms
        self.v_gain = v_gain
        self.h_int = h_int
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
        return (self.waveforms.T - np.mean(self.waveforms[:, baseline_min:baseline_max], axis=1)).T


def save_rawdata_to_file(h5_filename: str, data, measurement_time: float, y_off: float, YMULT: float, samplerate: float, HV: int)-> None:
    """
    saves rawdata to file

    Parameters
    ----------
    h5_filename: string
        folder an filename where to save the data
    data: np.array
        data to be saved
    measurement_time: float
        time that measurment took
    y_off: float
        offset to be added on the measurment
    YMULT: float
        conversion factor form raw data to data
    samplerate: float
        samplerate on which the measurment were taken
    HV: int or str
        voltage of measurment

    """
    f = h5py.File(h5_filename, 'a')
    try:
        f.create_dataset(f"{HV}/waveforms", data=data, dtype=np.int8)
        wf_info = f.create_group(f"{HV}/waveform_info")
    except ValueError:
        raise ValueError(f'There already exists a measurment with HV = {HV}')
    wf_info["h_int"] = 1/samplerate
    wf_info["v_gain"] =  YMULT
    wf_info["measurement_time"] = measurement_time
    wf_info["y_off"] = y_off
    f.close()

def add_fit_results(h5_filename: str, HV:int, gain:float, nphe:float, gain_err:float, int_ranges) -> None:
    """
    saves the fit results

    Parameters
    ----------
    h5_filename: string
        folder an filename where to save the data
    HV: int
        voltage on which data were taken
    gain: float
        resulting gain
    gain_err: float
        error on gain
    int_ranges: array
        integraton range
    """
    with h5py.File(h5_filename, "a") as f:
        fit_results = f.create_group(f"{HV}/fit_results")
        fit_results["nphe"] = nphe
        fit_results["gain"] = gain
        fit_results["gain_err"] = gain_err
        fit_results["int_ranges"] = int_ranges
        f.close()

def rewrite_fit_results(h5_filename, HV, gain, nphe, gain_err, int_ranges):
    """
    deletes old fit results and saves new

    Parameters
    ----------
    h5_filename: string
        folder an filename where to save the data
    HV: int
        voltage on which data were taken
    gain: float
        resulting gain
    gain_err: float
        error on gain
    int_ranges: array
        integraton range
    """
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

def x_y_values(data,  y_off:float, YMULT:float, measurement_time:float = None, samplerate:float = None, h_int:float = None):
    '''
    converges raw data to x and y data
    !!!Either Measurement_time or samplerate or h_int must not be None!!!

    Parameters
    ----------
    data: np.array
        data to be converged
    measurement_time: float, default: None
        time that measurment took
    y_off: float, default: None
        offset to be added on the measurment
    YMULT: float, default: None
        conversion factor form raw data to data
    samplerate: float, default: None
        samplerate on which the measurment were taken
    h_int: float, default: None
        horizontal interval of data, it is 1/samplerate

    Returns
    -------
    np.array, np.array
        x x-Data of input
        y y-Data of input
    '''
    if not measurement_time == None:
        x = np.linspace(0,measurement_time,len(data[0,:]))
    elif not samplerate == None:
        x = np.linspace(0,len(data[0,:])/samplerate,len(data[0,:]))
    elif not h_int == None:
        x = np.linspace(0,len(data[0,:])*h_int,len(data[0,:]))
    else:
        raise AttributeError('Either Measurement_time or samplerate or h_int must not be None')
    y = (data)*YMULT+y_off
    return x,y


def mean_plot(y, int_ranges = (60, 180, 200, 350), path = None):
    """
    plot mean of input with standard derivation

    Parameters
    ----------
    y: np.array
        input value to be plotted
    int_ranges: array, default: (60, 180, 200, 350)
        area to be highlighted
    """
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
    if not path == None:
        fig.savefig(path + '/mean_plot.pdf' )
        plt.close()
    else:
        plt.show(block = False)


def mp_var_int_range(waveforms, ped_min, ped_max, sig_min_start, sig_max_start,h_int, interval_sig_min,interval_sig_max, number_sig_min, number_sig_max, number_of_processes = os.cpu_count(), print_level = 0):
    """
    calculates the histogramm of charges with a variable integration borders
    Works on multiple threads

    Parameters
    ----------
    waveforms: np.array
        input array to calculate the autogram
    ped_min, ped_max: int
        the numimum and maximum values for pedestals
    sig_min_start: int
        start of Integration
    sig_max_start: int
        ending value for integraton
    h_int: float
        horizontal interval of data
    interval_sig_min, interval_sig_max: int, default: 10
        step between integrations
    number_sig_min: int
        number of integrations in sig_min direction
    number_sig_max: int
        number of integrations in sig_min direction
    number_of_processes: int default: os.cpu_count()
        number of created threads
    Returns
    -------
    np.array, np.array, np.array, np.array, np.array

        sig_min, sig_max, gains, gain_errs, nphes
    !!!sig_min, gains, gain_errs, nphes has always a lenth math.ceil(number_sig_min/number_of_processes)*number_of_processes
    """
    queue = mp.Queue()
    chunks = int(math.ceil(number_sig_min/number_of_processes))
    procs = []
    results = {}
    for i in range(number_of_processes):
        proc = mp.Process(target = hist_variable_values_mp, args = (queue, waveforms, ped_min, ped_max, sig_min_start-i*chunks,sig_max_start,h_int,i, interval_sig_min,interval_sig_max ,chunks,  number_sig_max, print_level))
        procs.append(proc)
        proc.start()
    for i in range(number_of_processes):
        results.update(queue.get())
    for i in procs:
        i.join()
    gains = np.array(results[0][0])
    nphes = np.array(results[0][2])
    gain_errs = np.array(results[0][1])
    sig_min = np.linspace(sig_min_start-chunks*number_of_processes*interval_sig_min, sig_min_start, chunks*number_of_processes)
    sig_max = np.linspace(sig_max_start, sig_max_start+interval_sig_max*number_sig_max, number_sig_max)
    for i in range(number_of_processes-1):
        i+=1
        gains = np.append(gains,results[i][0],axis = 0)
        nphes = np.append(nphes,results[i][2],axis = 0)
        gain_errs = np.append(gain_errs,results[i][1],axis = 0)
    return sig_min, sig_max, np.flip(gains, axis = 0), np.flip(nphes, axis = 0),np.flip(gain_errs, axis = 0)

def hist_variable_values_mp(queue, waveforms, ped_min, ped_max, sig_min_start, sig_max_start,h_int,p, interval_sig_min,interval_sig_max, number_sig_min, number_sig_max, print_level):
    """
    auxiliary method for mp_var_int_range

    Parameters
    ----------
    queue:
    waveforms: np.array
        input array to calculate the autogram
    ped_min, ped_max: int
        the numimum and maximum values for pedestals
    sig_min_start: int
        start of Integration
    sig_max_start: int
        ending value for integraton
    h_int: float
        horizontal interval of data
    interval_sig_min, interval_sig_max: int, default: 10
        step between integrations
    number_sig_min: int
        number of integrations in sig_min direction
    number_sig_max: int
        number of integrations in sig_min direction
    print_level: int
        0: quiet, 1,2,3: print fit details

    """
    gains = np.full((number_sig_min, number_sig_max),np.nan)
    nphes = np.full((number_sig_min, number_sig_max),np.nan)
    gain_errs = np.full((number_sig_min, number_sig_max),np.nan)
    for i in range(int(number_sig_min)):
        for j in range(int(number_sig_max)):
            try:
                x,y, int_ranges = histogramm(waveforms, ped_min, ped_max, sig_min_start-interval_sig_min*i, sig_max_start+interval_sig_max*j)
                gain, nphe, gain_err = hist_fitter(x,y,h_int, plot = False)
                if gain < 0:
                    continue
                if gain > 1e8:
                    print('Strange value occured at:')
                    print('sig_min = ', sig_min_start-interval_sig_min*i)
                    print('sig_max = ', sig_max_start+interval_sig_max*j)
                    print('gain = ', gain)
                    print(int_ranges)
                    continue
                gains[i,j] = gain
                nphes [i,j]= nphe
                gain_errs [i,j]=gain_err
            except ValueError:
                continue
            except TypeError:
                continue
    queue.put({p: (gains, gain_errs, nphes)})



def plot_hist_variable_values(sig_min, sig_max, gains, nphes, h_int ,gain_errs, name= None, nrows = 3, show = False, waveforms= None):
    if nrows != 3 and nrows !=4:
        raise ValueError(f'nrows has to be either 3 or 4. nrows = {nrows}')
    try:
        if (nrows == 4 and waveforms == None):
            raise ValueError(f'if nrows equals 4, waveforms must not ne None')
    except ValueError:
        pass
    X,Y = np.meshgrid(sig_min, sig_max)
    fig,ax = plt.subplots(figsize = (10,15), nrows = nrows , constrained_layout = True)
    secx = ax[0].secondary_xaxis('top', functions = (lambda x: x*h_int*1e9, lambda x: x/(h_int*1e9)))
    secx.set_xlabel('Integration start in ns')
    secy = ax[0].secondary_yaxis('right', functions = (lambda x: x*h_int*1e9, lambda x: x/(h_int*1e9)))
    secy.set_ylabel('Integration end in ns')
    p = ax[0].pcolormesh(X,Y,gains.T/1e6,shading='auto')
    ax[0].set_ylabel("Integration end in sample")
    fig.colorbar(p, label = r'gain $10^6$', orientation = "vertical", ax = ax[0])

    secy_2 = ax[1].secondary_yaxis('right', functions = (lambda x: x*h_int*1e9, lambda x: x/(h_int*1e9)))
    secy_2.set_ylabel('"Integration end in ns')
    secx_2 = ax[1].secondary_xaxis('top', functions = (lambda x: x*h_int*1e9, lambda x: x/(h_int*1e9)))
    p = ax[1].pcolormesh(X,Y,gain_errs.T/1e6,shading='auto')
    ax[1].set_ylabel("Integration end in sample")
    fig.colorbar(p, label = r'gain errors $10^6$', orientation = "vertical", ax = ax[1])

    secy_3 = ax[2].secondary_yaxis('right', functions = (lambda x: x*h_int*1e9, lambda x: x/(h_int*1e9)))
    secy_3.set_ylabel('Integration end in ns')
    ax[2].set_xlabel('Integration start in Sample')
    secx_3 = ax[2].secondary_xaxis('top', functions = (lambda x: x*h_int*1e9, lambda x: x/(h_int*1e9)))
    p = ax[2].pcolormesh(X,Y,nphes.T,shading='auto')
    ax[2].set_ylabel("Integration end in sample")
    fig.colorbar(p, label = r'nphes', orientation = "vertical", ax = ax[2])

    if nrows == 4:
        ax[3].set_title('Mean Plot')
        secy_3 = ax[3].secondary_xaxis('top', functions = (lambda x: x*h_int*1e9, lambda x: x/(h_int*1e9)))
        secy_3.set_xlabel('time in ns')
        ax[3].set_xlabel('time in sample')
        ax[3].set_ylabel('Voltage in V')
        ax[3].plot(np.mean(waveforms, axis = 0))
    if name != None:
        plt.savefig(name)
    if show:
        plt.show()

def hist(waveforms, ped_min=0, ped_max= 200, sig_min= 250, sig_max=500, bins = 200, histo_range= None, plot = True,path = None,):
    int_ranges = (ped_min, ped_max, sig_min, sig_max)
    try:
        if(plot):
            mean_plot(waveforms, int_ranges)
            ped_min, ped_max, sig_min, sig_max = [int(i) for i in input('set integration ranges: <ped_min, ped_max, sig_min, sig_max>\n').split(', ')]
        elif not path == None:
                mean_plot(waveforms, int_ranges, path = path)
        if ped_min<0 or ped_max<0 or sig_min<0 or sig_max<0:
            ped_min, ped_max, sig_min, sig_max = [int(i) for i in int_ranges]

    except ValueError:
        print(f'ValueError: used integration ranges {int_ranges}')
    return histogramm(waveforms, ped_min, ped_max, sig_min, sig_max, bins, histo_range)

def histogramm(waveforms, ped_min=0, ped_max= 100, sig_min= 190, sig_max=400, bins = 200, histo_range= None, mask = -0.5):
    ped_sig_ratio = (ped_max - ped_min) / (sig_max - sig_min)
    pedestals = (np.sum(waveforms[:, ped_min:ped_max], axis=1))
    charges = -(np.sum(waveforms[:, sig_min:sig_max], axis=1))+pedestals/ped_sig_ratio
    if mask != None:
        charges = charges[charges>mask]
    hi, bin_edges = np.histogram(charges, range = histo_range, bins = bins)
    new_bin_edges = (bin_edges-(bin_edges[1])/2)[:-1]
    return hi, new_bin_edges, np.array([ped_min,ped_max,sig_min,sig_max])

def hist_fitter(hi, bin_edges, h_int, plot = True,print_level = 0, valley = None, path = None, title = None):#input has to be from hist()
    fitter = ChargeHistFitter()
    fitter.pre_fit(bin_edges, hi, print_level = print_level, valley = valley)
    fit_function = fitter.fit_pmt_resp_func(bin_edges,hi, print_level = print_level)
    fitter.opt_ped_values
    if print_level:
        print('Fit quality: ',  fitter.quality_check(bin_edges, hi))
    if (plot or not path == None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(bin_edges, hi)
        ax.plot(bin_edges,fitter.opt_prf_values)
        if not title == None:
            plt.title(title)
        plt.xlabel("ADC Channel")
        plt.ylabel("Number of events")
        plt.ylim(.1,1e5)
        if plot:
            plt.show(block = True)
        if not path == None:
            fig.savefig(path + '/Histogramm_mit_Fit.pdf' )
        plt.close(fig)
    gain = fitter.popt_prf['spe_charge']*h_int/(50*e)
    if (gain < 0 or gain >1e7) and valley is None:
        return hist_fitter(hi = hi, bin_edges = bin_edges, h_int = h_int, plot = plot ,print_level = print_level, valley = bin_edges[np.argmin(hi[np.argmax(hi):int(len(hi)/4)])])
    nphe = fitter.popt_prf['nphe']
    gain_err = np.sqrt(fitter.pcov_prf['spe_charge', 'spe_charge'])*h_int/(50*e)
    if not path == None:
        np.savetxt(path + '/daten_histogramm.txt',np.array([bin_edges, hi]))
    return gain, nphe, gain_err



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


def analysis_complete_data(h5_filename, nom_manuf_hv = None,reanalyse= False, saveresults = True, nominal_gains = [3e6], SN = 'AB2363', plot = False, log = True):
    f = WavesetReader(h5_filename)
    hv = []
    gains = []
    nphes = []
    gain_errs = []
    int_range = []
    for key in f.wavesets:
        path = f'/media/pmttest/TOSHIBA EXT/Messdaten/KM3Net_{SN}/{key}'
        try:
            os.mkdir(f'/media/pmttest/TOSHIBA EXT/Messdaten/KM3Net_{SN}')
        except FileExistsError:
            pass
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        waveset = f[key]
        hv.append(int(key.split('_')[0]))
        if not reanalyse and not waveset.fit_results == None:
            waveset = f[key]
            gains.append(waveset.fit_results[0])
            h_int = waveset.h_int
            nphes.append(waveset.fit_results[1])
            gain_errs.append(waveset.fit_results[2])
            int_range.append(np.array(waveset.fit_results[3]))
            print(f'HV: {key}, gain: {waveset.fit_results[0]}, nphes: {waveset.fit_results[1]}')
        else:
            waveforms = waveset.waveforms
            h_int = waveset.h_int
            y,x, int_ranges = hist(waveforms, plot = False, path = path)
            int_range.append(int_ranges)
            gain, nphe, gain_err = hist_fitter(y,x,h_int, path = path, plot = plot, title = f'SN: {SN}, HV: {key}')
            if saveresults and not waveset.fit_results == None:
                rewrite_fit_results(h5_filename = h5_filename, HV = key, gain = gain, nphe = nphe, gain_err = gain_err, int_ranges = int_ranges)
            elif saveresults:
                print("Yes")
                add_fit_results(h5_filename = h5_filename, HV = key, gain = gain, nphe = nphe, gain_err = gain_err, int_ranges = int_ranges)
            gains.append(gain)
            nphes.append(nphe)
            gain_errs.append(gain_err)
    p_opt, cov = curve_fit(linear, np.log10(hv), np.log10(gains), sigma = np.log10(gain_errs))
    nominal_hvs = []
    for nominal_gain in nominal_gains:
        nominal_hvs.append(10**((np.log10(nominal_gain) - p_opt[1])/ p_opt[0]))
    #print(nominal_hvs)
    if plot:
        fig, ax = plt.subplots(figsize = (10,5))
        for i in range(len(nominal_hvs)):
            ax.axhline(np.log10(nominal_gains[i]), color="black", ls="--")
            ax.axvline(np.log10(nominal_hvs[i]), color="black", ls="--")
        ax.plot(np.log10(hv), np.log10(gains), 'x')
        hv_min=np.log10(min(hv)-50)
        hv_max=np.log10(max(hv)+50)
        xs = np.linspace(hv_min, hv_max, 1000)
        gain_min=hv_min*p_opt[0] +p_opt[1]
        gain_max=hv_max*p_opt[0] +p_opt[1]
        if not nom_manuf_hv == None:
            manuf_gain=10**(np.log10(nom_manuf_hv)*p_opt[0] +p_opt[1])
            ax.vlines(x=np.log10(nom_manuf_hv), ymin=gain_min,ymax=np.log10(manuf_gain), colors="red", linestyles="dashed")
        plt.title("gainslope "+SN)
        ax.plot(xs, linear(xs, *p_opt))
        plt.axis([hv_min, hv_max, gain_min, gain_max])
        plt.xlabel("log10(HV)")
        plt.ylabel("log10(gain)")
        plt.show()
        fig.savefig(f'/media/pmttest/TOSHIBA EXT/Messdaten/KM3Net_{SN}/gainslope_{SN}.pdf')
    err_nom_hv = err_hv(gain_nom = nominal_gain,gain = gains,gain_errs = gain_errs, hvs = hv)
    if log:
        name = f'/media/pmttest/TOSHIBA EXT/Messdaten/KM3Net_{SN}/log'
        log_complete_data(name = name,hvs = hv,err_nom_hv = err_nom_hv, gains=gains, gain_errs = gain_errs, nphe = nphes,int_ranges = np.array(int_range),h_int = h_int, p_opt = p_opt, cov = cov,nominal_gain = nominal_gains, nominal_hv = nominal_hvs)
    print(f'{SN}: nominal HV: {nominal_hvs} pm {err_nom_hv} for nominal gain: {nominal_gains}')
    return gains, nphes, hv, gain_errs

def err_hv(gain, gain_errs, gain_nom, hvs):
    p_opt, cov = curve_fit(linear,hvs, np.log10(gain)-np.log10(gain_nom), sigma = np.log10(gain_errs))
    return 10**np.sqrt(cov[1,1])

def log_complete_data(name, hvs, gains, gain_errs,err_nom_hv, nphe,int_ranges,h_int, p_opt, cov,nominal_gain, nominal_hv):
    name = '{}.txt'.format(name)
    f = open(name, 'a')
    int_time = (int_ranges[:,3]-int_ranges[:,2])*h_int*1e9
    gains = [np.format_float_scientific(i,4) for i in gains]
    gain_errs = [np.format_float_scientific(i,2) for i in gain_errs]

    text = f'Date: {datetime.now()}\n HV: {hvs}\n gains: {gains}\n gain_errs: {gain_errs}\n nphes: {np.round(nphe,2)}\n int ranges: {int_ranges};\n int_ranges in ns {int_ranges*h_int*1e9}\n integration time in ns {int_time}\n fit results: {np.round(p_opt,4)}\n cov: {np.round(cov,4)}\n nominal gain: {nominal_gain}\n nominal hv: {np.round(nominal_hv,0)}pm{np.round(err_nom_hv,2)}\n\n\n'
    f.write(text)
    f.close()



if __name__ == '__main__':
    pass
