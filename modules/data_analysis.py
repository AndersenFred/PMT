import time

import numpy as np
import h5py
from matplotlib import colors
import matplotlib.pyplot as plt
import math
from datetime import datetime
import scipy.constants
from modules.pmt_resp_func import ChargeHistFitter
from scipy import optimize as opt
import os
import multiprocessing as mp
from scipy import signal

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
                fit_results = (f[key]['fit_results']['gain'][()], f[key]['fit_results']['nphe'][()],
                               f[key]['fit_results']['gain_err'][()], f[key]['fit_results']['int_ranges'][()])
            except KeyError:
                fit_results = None
        return Waveset(raw_waveforms, v_gain, h_int, y_off=y_off, fit_results=fit_results)


class Waveset:
    def __init__(self, raw_waveforms, v_gain, h_int, y_off, fit_results=None):
        self.raw_waveforms = raw_waveforms
        self.v_gain = v_gain
        self.h_int = h_int
        self.samplerate = 1 / h_int
        self.y_off = y_off
        self._waveforms = None
        self.fit_results = fit_results

    @property
    def waveforms(self):
        if self._waveforms is None:
            self._waveforms = self.raw_waveforms * self.v_gain + self.y_off
        return self._waveforms

    def zeroed_waveforms(self, baseline_min, baseline_max):
        return (self.waveforms.T - np.mean(self.waveforms[:, baseline_min:baseline_max], axis=1)).T


def linear(x, m, t):
    return m * (x - t)


def save_rawdata_to_file(h5_filename: str, data, measurement_time: float, y_off: float, YMULT: float, samplerate: float,
                         HV: int) -> None:
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
    wf_info["h_int"] = 1 / samplerate
    wf_info["v_gain"] = YMULT
    wf_info["measurement_time"] = measurement_time
    wf_info["y_off"] = y_off
    f.close()


def add_fit_results(h5_filename: str, HV: int, gain: float, nphe: float, gain_err: float, int_ranges: list) -> None:
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
    int_range: array
        integraton range
    """
    with h5py.File(h5_filename, "a") as f:
        try:
            fit_results = f.create_group(f"{HV}/fit_results")
        except:
            fit_results = f[f"{HV}/fit_results"]
        try:
            del fit_results["nphe"]
        except KeyError as e:
            print(e)
        try:
            del fit_results["gain"]
        except KeyError as e:
            print(e)
        try:
            del fit_results["gain_err"]
        except KeyError as e:
            print(e)
        try:
            del fit_results["int_ranges"]
        except KeyError as e:
            print(e)
        fit_results["nphe"] = nphe
        fit_results["gain"] = gain
        fit_results["gain_err"] = gain_err
        fit_results["int_ranges"] = int_ranges
        f.close()


def x_y_values(data, y_off: float, YMULT: float, measurement_time: float = None, samplerate: float = None,
               h_int: float = None):
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
        x = np.linspace(0, measurement_time, len(data[0, :]))
    elif not samplerate == None:
        x = np.linspace(0, len(data[0, :]) / samplerate, len(data[0, :]))
    elif not h_int == None:
        x = np.linspace(0, len(data[0, :]) * h_int, len(data[0, :]))
    else:
        raise AttributeError('Either Measurement_time or samplerate or h_int must not be None')
    y = (data) * YMULT + y_off
    return x, y


def mean_plot(y, int_ranges=(60, 180, 200, 350), path=None):
    """
    plot mean of input with standard derivation

    Parameters
    ----------
    y: np.array
        input value to be plotted
    int_ranges: array, default: (60, 180, 200, 350)
        area to be highlighted
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(int_ranges[0], int_ranges[1], facecolor="red", alpha=0.5)
    ax.axvspan(int_ranges[2], int_ranges[3], facecolor="green", alpha=0.5)
    plt.xlabel("time in sample")
    plt.ylabel("Voltage in V")
    plt.title("mean waveform")
    y_data = np.mean(y, axis=0)
    y_std = np.std(y, axis=0) / np.sqrt(len(y))
    ax.plot(y_data)
    ax.fill_between(np.linspace(1, len(y_data), len(y_data)), y_data - y_std, y_data + y_std, color='gray', alpha=0.2)
    if not path == None:
        fig.savefig(path + '/mean_plot.pdf')
        plt.close()
    else:
        plt.show(block=False)


def mp_var_int_range(waveforms, ped_min, ped_max, sig_min_start, sig_max_start, h_int, interval_sig_min,
                     interval_sig_max, number_sig_min, number_sig_max, SN, HV, number_of_processes=os.cpu_count(),
                     print_level=0, ):
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
    sig_max: int
        ending value for integraton
    h_int: float
        horizontal interval of data
    interval: int, default: 10
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
    chunks = int(math.ceil(number_sig_min / number_of_processes))
    procs = []
    results = {}
    for i in range(number_of_processes):
        proc = mp.Process(target=hist_variable_values_mp, args=(
            queue, waveforms, ped_min, ped_max, sig_min_start - i * chunks, sig_max_start, h_int, i, interval_sig_min,
            interval_sig_max, chunks, number_sig_max, print_level, SN, HV))
        procs.append(proc)
        proc.start()
    for i in range(number_of_processes):
        results.update(queue.get())
    for i in procs:
        i.join()
    gains = np.array(results[0][0])
    nphes = np.array(results[0][2])
    gain_errs = np.array(results[0][1])
    sig_min = np.linspace(sig_min_start - chunks * number_of_processes * interval_sig_min, sig_min_start,
                          chunks * number_of_processes)
    sig_max = np.linspace(sig_max_start, sig_max_start + interval_sig_max * number_sig_max, number_sig_max)
    for i in range(number_of_processes - 1):
        i += 1
        gains = np.append(gains, results[i][0], axis=0)
        nphes = np.append(nphes, results[i][2], axis=0)
        gain_errs = np.append(gain_errs, results[i][1], axis=0)
    return sig_min, sig_max, np.flip(gains, axis=0), np.flip(nphes, axis=0), np.flip(gain_errs, axis=0)


def hist_variable_values_mp(queue, waveforms, ped_min, ped_max, sig_min_start, sig_max_start, h_int, p,
                            interval_sig_min, interval_sig_max, number_sig_min, number_sig_max, print_level, SN, HV):
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
    sig_max: int
        ending value for integraton
    h_int: float
        horizontal interval of data
    interval: int, default: 10
        step between integrations
    number_sig_min: int
        number of integrations in sig_min direction
    number_sig_max: int
        number of integrations in sig_min direction
    print_level: int
        0: quiet, 1,2,3: print fit details

    """
    gains = np.full((number_sig_min, number_sig_max), np.nan)
    nphes = np.full((number_sig_min, number_sig_max), np.nan)
    gain_errs = np.full((number_sig_min, number_sig_max), np.nan)
    for i in range(int(number_sig_min)):
        for j in range(int(number_sig_max)):
            try:
                hi, bin_edges, int_ranges = histogramm(waveforms, ped_min, ped_max,
                                                       sig_min_start - interval_sig_min * i,
                                                       sig_max_start + interval_sig_max * j)
                gain, nphe, gain_err = hist_fitter(hi, bin_edges, h_int, plot=False, valley=bin_edges[
                    np.argmin(hi[np.argmax(hi):int(len(hi) / 5)]) + np.argmax(hi)])
                if (gain < 0 or gain > 1e7 or nphe > 2 or gain_err > 1e6):
                    f = open('Strange values.txt', 'a')
                    f.write(
                        f'Strange value occured with gain = {gain}, nphes = {nphe} at HV = {HV}, SN = {SN} with sig_min = {sig_min_start - interval_sig_min * i}, sig_max = {sig_max_start + interval_sig_max * j}\n')
                    f.close()
                if gain < 0:
                    continue
                if gain > 1e8:
                    print('Strange value occured at:')
                    print('sig_min = ', sig_min_start - interval_sig_min * i)
                    print('sig_max = ', sig_max_start + interval_sig_max * j)
                    print('gain = ', gain)
                    print(int_ranges)
                    continue
                gains[i, j] = gain
                nphes[i, j] = nphe
                gain_errs[i, j] = gain_err
            except ValueError:
                continue
            except TypeError:
                continue
            except RuntimeError:
                continue
    queue.put({p: (gains, gain_errs, nphes)})


def var_sig_max(ped_min, ped_max, sig_min, sig_max_start, interval_sig_max, number_sig_max, SN,
                path='/media/pmttest/TOSHIBA EXT/Messdaten/', print_level=0):
    reader = WavesetReader(path + 'KM3Net_{}.h5'.format(SN))
    queue = mp.Queue()
    procs = []
    results = {}
    for HV in reader.wavesets:
        proc = mp.Process(target=var_sig_max_aux, args=(
            queue, reader[HV].waveforms, ped_min, ped_max, sig_min, sig_max_start, reader[HV].h_int, interval_sig_max,
            number_sig_max, print_level, HV, SN))
        procs.append(proc)
        proc.start()
    for i in range(len(reader.wavesets)):
        results.update(queue.get())
    for i in procs:
        i.join()
    return results


def var_sig_max_aux(queue, waveforms, ped_min, ped_max, sig_min, sig_max_start, h_int, interval_sig_max, number_sig_max,
                    print_level, HV, SN, path='/media/pmttest/TOSHIBA EXT/Messdaten/', plot=1):
    gains = np.full((number_sig_max), np.nan)
    nphes = np.full((number_sig_max), np.nan)
    gain_errs = np.full((number_sig_max), np.nan)
    sig_max = []
    f = lambda x: (x * (h_int * 1e9) % 2 == 0)
    for i in range(number_sig_max):
        try:
            hi, bin_edges, int_ranges = histogramm(waveforms, ped_min, ped_max, sig_min,
                                                   sig_max_start + interval_sig_max * i)
            gain, nphe, gain_err = hist_fitter(hi, bin_edges, h_int, plot=False,
                                               path=plot * f(sig_max_start + interval_sig_max * i) * (
                                                       path + f'KM3Net_{SN}/{HV}/sig_max_{sig_max_start + interval_sig_max * i}_'),
                                               print_level=print_level, valley=bin_edges[
                    np.argmin(hi[np.argmax(hi):int(len(hi) / 5)]) + np.argmax(hi)])
            if gain < 0:
                continue
            if gain > 1e8:
                print('Strange value occured at:')
                print('sig_max = ', sig_max_start + interval_sig_max * i)
                print('gain = ', gain)
                print(int_ranges)
                continue
            gains[i] = gain
            nphes[i] = nphe
            gain_errs[i] = gain_err
            sig_max.append(sig_max_start + interval_sig_max * i)
        except ValueError:
            continue
        except TypeError:
            continue
        except RuntimeError:
            continue
    queue.put({HV: (gains, gain_errs, nphes, np.array(sig_max))})


def plot_hist_variable_sig_max(results, SN, path='/media/pmttest/TOSHIBA EXT/Messdaten/KM3Net_{}.h5', sig_min=250):
    reader = WavesetReader(path.format(SN))
    for HV in results.keys():
        waveforms = reader[HV].waveforms
        h_int = reader[HV].h_int
        fig, ax = plt.subplots(figsize=(10, 15), nrows=4, constrained_layout=True)
        ax[0].set_title(f'Variating integration ending for SN: {SN}, HV: {HV}, with signal min: {sig_min}')
        ax[0].plot(results[HV][3], results[HV][0][~np.isnan(results[HV][0])] / 1e6)
        secx = ax[0].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
        secx.set_xlabel('Integration end in ns')
        ax[0].set_ylabel(r'Gain in $10^6$')
        ax[0].set_xlabel("Integration end in sample")

        ax[1].plot(results[HV][3], results[HV][1][~np.isnan(results[HV][1])] / 1e6)
        secx = ax[1].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
        secx.set_xlabel('Integration end in ns')
        ax[1].set_ylabel(r'Gain Error in $10^6$')
        ax[1].set_xlabel("Integration end in sample")

        ax[2].plot(results[HV][3], results[HV][2][~np.isnan(results[HV][2])])
        secx = ax[2].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
        secx.set_xlabel('Integration end in ns')
        ax[2].set_ylabel('Number of Photoelektrons')
        ax[2].set_xlabel("Integration end in sample")

        wf = waveforms[np.sum(waveforms, axis=1) * h_int / (50 * e) > 0.1]
        ax[3].plot(np.mean(wf, axis=0))
        ax[3].set_title('Mean Plot')
        secy_3 = ax[3].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
        secy_3.set_xlabel('time in ns')
        ax[3].set_xlabel('time in sample')
        ax[3].set_ylabel('Voltage in V')
        fig.savefig(path[:-3].format(SN) + f'/{HV}/variating sig max.pdf')
        plt.close()


def plot_hist_variable_values(sig_min, sig_max, gains, nphes, gain_errs, h_int, name=None, nrows=3, show=False,
                              waveforms=None, threshold=0.1):
    """
    plots the results of hist_variable_values with 3 or 4 rows
    if nrows == 3 there are 3 scatter-plots, if nrows == 4
    Parameters
    ----------
    sig_min: np.array
        the values used for sig_min with len = x
    sig_min: np.array
        the values used for sig_max with len = y
    gains, nphes, gain_errs: np.array with shape = (x,y)
        the values for gain, nphes and gain_errors
    h_int: int
        horizontal interval between points
    name: Str, default: None
        name where the plot has to be saved
    nrows: int, default: 3
        number of rows to be plotted. if nrows equals 4, waveforms must not ne None. with nrows equals 4 the mean_waveforms of signal will be in fourth plot.
        nrows has to be either 3 or 4
    show: bool, default: False
        determines if the plot will be shown
    waveforms: np.array, default: None
        waveforms to be plotten if nrows equals 4
    threshold: int, default: 0.1
        threshold to determin if a single waveform contains a signal
    Returns
    -------
    None
    """
    if nrows != 3 and nrows != 4:
        raise ValueError(f'nrows has to be either 3 or 4. nrows = {nrows}')
    try:
        if (nrows == 4 and waveforms == None):
            raise TypeError(f'if nrows equals 4, waveforms must not ne None')
    except ValueError:
        pass
    X, Y = np.meshgrid(sig_min, sig_max)
    fig, ax = plt.subplots(figsize=(10, 15), nrows=nrows, constrained_layout=True)
    secx = ax[0].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
    secx.set_xlabel('Integration start in ns')
    secy = ax[0].secondary_yaxis('right', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
    secy.set_ylabel('Integration end in ns')
    p = ax[0].pcolormesh(X, Y, gains.T / 1e6, shading='auto')
    ax[0].set_ylabel("Integration end in sample")
    fig.colorbar(p, label=r'gain $10^6$', orientation="vertical", ax=ax[0])

    secy_2 = ax[1].secondary_yaxis('right', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
    secy_2.set_ylabel('Integration end in ns')
    secx_2 = ax[1].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
    p = ax[1].pcolormesh(X, Y, gain_errs.T / 1e6, shading='auto')
    ax[1].set_ylabel("Integration end in sample")
    fig.colorbar(p, label=r'gain errors $10^6$', orientation="vertical", ax=ax[1])

    secy_3 = ax[2].secondary_yaxis('right', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
    secy_3.set_ylabel('Integration end in ns')
    ax[2].set_xlabel('Integration start in Sample')
    secx_3 = ax[2].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
    p = ax[2].pcolormesh(X, Y, nphes.T, shading='auto')
    ax[2].set_ylabel("Integration end in sample")
    fig.colorbar(p, label=r'nphes', orientation="vertical", ax=ax[2])

    if nrows == 4:
        ax[3].set_title('Mean Plot')
        secy_3 = ax[3].secondary_xaxis('top', functions=(lambda x: x * h_int * 1e9, lambda x: x / (h_int * 1e9)))
        secy_3.set_xlabel('time in ns')
        ax[3].set_xlabel('time in sample')
        ax[3].set_ylabel('Voltage in V')
        wf = waveforms[np.sum(waveforms, axis=1) * h_int / (50 * e) > threshold]
        ax[3].plot(np.mean(wf, axis=0))
        y_std = np.std(wf, axis=0) / np.sqrt(len(wf))
        wf = np.mean(wf, axis=0)
        ax[3].fill_between(np.linspace(0, len(wf), len(wf)), wf - y_std, wf + y_std, color='gray', alpha=0.2)
    if name != None:
        plt.savefig(name)
    if show:
        plt.show()
    else:
        plt.close()


def moving_average(data, window):
    weights = np.repeat(1.0, window) / window
    ma = np.convolve(data, weights, 'valid')
    return ma


def hist(waveforms, ped_min=0, ped_max=200, sig_min=250, sig_max=500, bins=200, histo_range=None, plot=True,
         path=None, ):
    int_ranges = (ped_min, ped_max, sig_min, sig_max)
    try:
        if (plot):
            mean_plot(waveforms, int_ranges)
            ped_min, ped_max, sig_min, sig_max = [int(i) for i in input(
                'set integration ranges: <ped_min, ped_max, sig_min, sig_max>\n').split(', ')]
        elif not path == None:
            mean_plot(waveforms, int_ranges, path=path)
        if ped_min < 0 or ped_max < 0 or sig_min < 0 or sig_max < 0:
            ped_min, ped_max, sig_min, sig_max = [int(i) for i in int_ranges]
    except ValueError:
        print(f'ValueError: used integration ranges {int_ranges}')
    return histogramm(waveforms, ped_min, ped_max, sig_min, sig_max, bins, histo_range)


def histogramm(waveforms, ped_min=0, ped_max=200, sig_min=250, sig_max=500, bins=200, range=None, plot=False, name=None,
               title=None, block=True):
    ped_sig_ratio = (ped_max - ped_min) / (sig_max - sig_min)
    pedestals = (np.sum(waveforms[:, ped_min:ped_max], axis=1))
    charges = -(np.sum(waveforms[:, sig_min:sig_max], axis=1)) + pedestals / ped_sig_ratio
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        range = (np.min(charges), np.max(charges))

        ax.hist(charges, range=range, bins=200, log=True, color='b')
        ax.hist(pedestals - np.mean(pedestals), range=range, bins=200, log=True, color='r')

        plt.xlabel("Number of Photoelektrons")
        plt.ylabel("Number of events")
        if not title == None:
            plt.title(title)
        plt.show(block=block)
        if not name == None:
            print(f'saving {name}.pdf')
            fig.savefig(name + '.pdf')
    hi, bin_edges = np.histogram(charges, range=range, bins=bins)
    bin_edges = (bin_edges - (bin_edges[1]) / 2)[:-1]
    return hi, bin_edges, np.array([ped_min, ped_max, sig_min, sig_max])


def hist_fitter(hi, bin_edges, h_int, plot=True, print_level=0, valley=None, mask=None):  # input has to be from hist()
    fitter = ChargeHistFitter()
    if mask != None:
        hi = hi[bin_edges > mask]
        bin_edges = bin_edges[bin_edges > mask]
    fitter.pre_fit(bin_edges, hi, print_level=print_level, valley=valley)
    fit_function = fitter.fit_pmt_resp_func(bin_edges, hi, print_level=print_level)
    # fitter.opt_ped_values
    # fitter.opt_spe_values
    if (plot):
        plt.semilogy(bin_edges, hi, label='Mesurments')
        plt.plot(bin_edges, fitter.opt_prf_values, label='Fit')
        if not valley == None:
            plt.axvline(valley, label='valley')
        plt.ylim(.1, 1e5)
        plt.legend
        plt.show(block=False)
    gain = fitter.popt_prf['spe_charge'] * h_int / (50 * e)
    nphe = fitter.popt_prf['nphe']
    if (gain < 0 or gain > 1e7 or nphe > 2) and valley is None:
        return hist_fitter(hi=hi, bin_edges=bin_edges, h_int=h_int, plot=plot, print_level=print_level,
                           valley=bin_edges[np.argmin(hi[np.argmax(hi):int(len(hi) / 5)]) + np.argmax(hi)])
    try:
        gain_err = np.sqrt(fitter.pcov_prf['spe_charge', 'spe_charge']) * h_int / (50 * e)
        if gain_err > 1e6 and valley == None:
            return hist_fitter(hi=hi, bin_edges=bin_edges, h_int=h_int, plot=plot, print_level=print_level,
                               valley=bin_edges[np.argmin(hi[np.argmax(hi):int(len(hi) / 5)]) + np.argmax(hi)])
    except TypeError:
        if valley == None:
            return hist_fitter(hi=hi, bin_edges=bin_edges, h_int=h_int, plot=plot, print_level=print_level,
                               valley=bin_edges[np.argmin(hi[np.argmax(hi):int(len(hi) / 5)]) + np.argmax(hi)])
        return None, None, None
    if gain < 0:
        raise TypeError(f'gain = {gain}: Fit did not work correctly')
    return gain, nphe, gain_err


def analysis_complete_data(h5_filename, reanalyse=False, saveresults=False, nominal_gains=[3e6], mask=-0.1, **kwargs):
    f = WavesetReader(h5_filename)
    hv = []
    gains = []
    nphes = []
    gain_errs = []
    int_range = []
    if not 'SN' in kwargs:
        kwargs['SN'] = None
    SN = kwargs['SN']
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
            y, x, int_ranges = histogramm(waveforms, plot=False, bins=200)
            int_range.append(int_ranges)
            if mask == None:
                mask = -np.inf
            mask_ = x > mask
            bin_edges = x[mask_]
            hi = y[mask_]
            avg_sig = moving_average(hi, 5)
            l = np.argmax(avg_sig)
            o = signal.find_peaks(avg_sig[l:], threshold=0)
            gain, nphe, gain_err = None, None, None
            try:
                valley = x[l + np.argmin(avg_sig[:o[0][0]])]
            except IndexError:
                valley = None
            if 'type' in kwargs:
                if kwargs['type'] == 'input':
                    while (True):
                        try:
                            SN = kwargs['SN']
                            plt.semilogy(bin_edges[2:-2], avg_sig)
                            plt.axvline(valley)
                            gain, nphe, gain_err = hist_fitter(hi, bin_edges, h_int, plot=True, valley=valley,
                                                               mask=None)
                            print(f'resulting gain for SN {SN} and HV {key}: {gain} pm {gain_err}, with nphe: {nphe}')
                        except:
                            plt.semilogy(bin_edges[2:-2], avg_sig)
                            plt.semilogy(x, y)
                            plt.axvline(valley)
                            plt.show(block=False)
                            print(f'Fit did not work with valley = {valley}')
                        try:
                            valley = float(input('valley:\n'))
                            plt.close()
                        except ValueError:
                            plt.close()
                            break
            else:
                while (True):
                    try:
                        gain, nphe, gain_err = hist_fitter(y, x, h_int, plot=False, valley=valley)
                        if nphe == None:
                            raise ValueError
                        break
                    except:
                        plt.close()
                        plt.semilogy(x[2:-2], avg_sig)
                        plt.axvline(valley)
                        plt.show(block=False)
                        valley = float(input('valley:\n'))
                        gain, nphe, gain_err = hist_fitter(y, x, h_int, plot=True, mask=0, valley=valley)

            if saveresults:
                add_fit_results(h5_filename=h5_filename, HV=key, gain=gain, nphe=nphe, gain_err=gain_err,
                                int_ranges=int_ranges)
            gains.append(gain)
            nphes.append(nphe)
            gain_errs.append(gain_err)
    p_opt, cov = opt.curve_fit(linear, np.log10(hv), np.log10(gains) - np.log10(3e6), sigma=np.log10(gain_errs))
    nominal_hvs = []
    for nominal_gain in nominal_gains:
        nominal_hvs.append(10 ** (p_opt[1]))
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(nominal_hvs)):
        ax.axhline(np.log10(nominal_gains[i]), color="black", ls="--")
        ax.axvline(np.log10(nominal_hvs[i]), color="black", ls="--")
    # gains,gain_errs = np.array(gains), np.array(gain_errs)
    # gain_err_plus = np.log10(gains+gain_errs)-np.log10(gains)
    # gain_err_minus = np.log10(gains-gain_errs)-np.log10(gains)
    ax.plot(np.log10(hv), np.log10(gains), 'x')
    hv_min = np.log10(min(hv) - 50)
    hv_max = np.log10(max(hv) + 50)
    xs = np.array([hv_min, hv_max])
    gain_min = linear(hv_min, *p_opt) + np.log10(3e6)
    gain_max = linear(hv_max, *p_opt) + np.log10(3e6)
    # ax.vlines(x=np.log10(nom_manuf_hv), ymin=gain_min,ymax=np.log10(manuf_gain), colors="red", linestyles="dashed")
    if kwargs['SN'] == None:
        kwargs['SN'] = ''
    plt.title("gainslope " + kwargs['SN'])

    ax.plot(xs, linear(xs, *p_opt) + np.log10(3e6))
    plt.axis([hv_min, hv_max, gain_min, gain_max])
    plt.xlabel("log10(HV)")
    plt.ylabel("log10(gain)")
    fig.savefig(h5_filename[:-3] + 'gainslope.pdf')
    if 'plot' in kwargs.values():
        plt.show()
    err_nom_hv = 10 ** p_opt[1] * np.sqrt(cov[1, 1]) * np.log(10)
    if reanalyse or saveresults:
        log_complete_data(name=h5_filename, hvs=hv, err_nom_hv=err_nom_hv, gains=gains, gain_errs=gain_errs, nphe=nphes,
                          int_ranges=np.array(int_range), h_int=h_int, p_opt=p_opt, cov=cov, nominal_gain=nominal_gains,
                          nominal_hv=nominal_hvs)
    delta_g = 3e6 * p_opt[0] / (10 ** p_opt[1]) * err_nom_hv

    print(
        f'SN: {SN}, nominal HV: {nominal_hvs} pm {err_nom_hv} for nominal gain: {nominal_gains}, delta g: {delta_g:.2e}')
    return gains, nphes, hv, gain_errs, nominal_hvs


def err_hv(gain, gain_errs, gain_nom, hvs):
    p_opt, cov = opt.curve_fit(linear, hvs, np.log10(gain) - np.log10(gain_nom), sigma=np.log10(gain_errs))
    return 10 ** np.sqrt(cov[1, 1])


def log_complete_data(name, hvs, gains, gain_errs, err_nom_hv, nphe, int_ranges, h_int, p_opt, cov, nominal_gain,
                      nominal_hv):
    name = '{}.txt'.format(name)
    f = open(name, 'a')
    int_time = (int_ranges[:, 3] - int_ranges[:, 2]) * h_int * 1e9
    gains = [np.format_float_scientific(i, 4) for i in gains]
    gain_errs = [np.format_float_scientific(i, 2) for i in gain_errs]

    text = f'Date: {datetime.now()}\n HV: {hvs}\n gains: {gains}\n gain_errs: {gain_errs}\n nphes: {np.round(nphe, 2)}\n int ranges: {int_ranges};\n int_ranges in ns {int_ranges * h_int * 1e9}\n integration time in ns {int_time}\n fit results: {np.round(p_opt, 4)}\n cov: {np.round(cov, 4)}\n nominal gain: {nominal_gain}\n nominal hv: {np.round(nominal_hv, 0)}pm{np.round(err_nom_hv, 2)}\n\n\n'
    f.write(text)
    f.close()


###################################
###################################
# Time over Threshold Measurements#
###################################
###################################


def save_tot_mess(h5_filename: str, data, measurement_time: float, y_off: float, YMULT: float,
                  samplerate: float) -> None:
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
    """
    f = h5py.File(h5_filename, 'a')
    f.create_dataset(f"tot/waveforms", data=data, dtype=np.int8)
    wf_info = f.create_group(f"tot/waveform_info")
    wf_info["samplerate"] = samplerate
    wf_info["v_gain"] = YMULT
    wf_info["measurement_time"] = measurement_time
    wf_info["y_off"] = y_off
    f.close()


def read_tot_mess(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        y = f[f"tot/waveforms"][:]
        samplerate = f[f"tot/waveform_info/samplerate"][()]
        mes_time = f[f"tot/waveform_info/measurement_time"][()]
        y_mul = f[f"tot/waveform_info/v_gain"][()]
        y_off = f[f"tot/waveform_info/y_off"][()]
    return to_data(y, y_off, y_mul), mes_time, samplerate


def write_tot_fit(h5_filename, nphe, p_opt_hist, cov_hist, p_opt_corr, cov_corr):
    with h5py.File(h5_filename, "a") as f:
        try:
            fit_results = f.create_group(f"tot/fit_results")
        except:
            fit_results = f[f"tot/fit_results"]
        try:
            del fit_results["nphe"]
            del fit_results["p_opt_hist"]
            del fit_results["cov_hist"]
            del fit_results["p_opt_corr"]
            del fit_results["cov_corr"]
        except KeyError as e:
            print(e)

        fit_results["nphe"] = nphe
        fit_results["p_opt_hist"] = p_opt_hist
        fit_results["cov_hist"] = cov_hist
        fit_results["p_opt_corr"] = p_opt_corr
        fit_results["cov_corr"] = cov_corr
        f.close()

def get_fit_results(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        fit_results = f[f"tot/fit_results"]
        nphe = fit_results["nphe"]
        p_opt_hist = fit_results["p_opt_hist"]
        cov_hist = fit_results["cov_hist"]
        p_opt_corr = fit_results["p_opt_corr"]
        cov_corr = fit_results["cov_corr"]
    return nphe, p_opt_hist, cov_hist, tot_hist, p_opt_corr, cov_corr
def to_data(y_raw, YOFF, YMU):
    y_value = []
    for i in range(len(YOFF)):
        y_value.append(y_raw[i] * YMU[i] + YOFF[i])
    return y_value


def threshold_in_hex(threshold=1.095):
    return hex(int(255 / 1.6 * threshold - 255 / 2))


def HV_in_Hex(SN=None, HV=None):
    def _to_hex(HV: float):
        HV = -abs(HV)
        return hex(int(-255 * (HV + 692) / 800))

    if (SN == None and HV == None) or ((not SN == None) and not (HV == None)):
        raise AttributeError("either SN or HV must not be None")
    if SN is None:
        return _to_hex(HV)
    else:
        print()
        return [_to_hex(i) for i in analysis_complete_data(SN, reanalyse=False, saveresults=False)[4]]


def get_tot(y_data, mes_time, threshold=0.150):
    digital = y_data[1] - y_data[0]
    digital += 0.150
    rising_edge = np.argmax(digital > threshold, axis=1)
    falling_edge = np.array([rising_edge[i] + np.argmax(digital[i, rising_edge[i]:] < threshold) for i in range(len(rising_edge))])
    tot = (falling_edge - rising_edge) * mes_time / np.shape(y_data)[2]
    for i in range(len(tot)):
        if rising_edge[i] == 0 or falling_edge[i] == rising_edge[i]:
            tot[i] = 0
    return tot


def get_tot_ampl(y_data, Measurement_time, threshold = 0.15):
    analog = y_data[2] - y_data[3]
    digital = y_data[1] - y_data[0]
    digital += 0.150
    rising_edge = np.argmax(digital > threshold, axis=1)
    falling_edge = np.array([rising_edge[i] + np.argmax(digital[i, rising_edge[i]:] < threshold) for i in range(len(rising_edge))])
    tot = (falling_edge - rising_edge) * Measurement_time / np.shape(y_data)[2]
    ampl = []
    for i in range(len(tot)):
        if rising_edge[i] == 0 or falling_edge[i] == rising_edge[i] or tot[i] == 0:
            tot[i] = 0
            ampl.append(0)
        else:
            ampl.append(np.max(analog[i, rising_edge[i]:falling_edge[i]]))
    return tot, ampl

def doppel_tail(y_data, time):
    tot, ampl = get_tot_ampl(y_data, time)
    tot = tot*1e9
    for i in range(len(tot)):
        if tot[i] < 5 and 0.01 < ampl[i]:
            digital = y_data[1] - y_data[0]
            analog = y_data[2] - y_data[3]
            fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
            ax[0].plot(digital[i])
            ax[1].plot(analog[i])
            plt.title(f"ampl: {ampl[i]:.2f}, tot: {tot[i]:.1f}, at index {i}")
            plt.show()


def normal(x, ampl, mu, sigma):
    return ampl * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2) / np.sqrt(2 * np.pi) / sigma


def log_vert(x, a, b, c):
    return a * np.log(b * x + c)


def tot_hist(y_data, Measurement_time, folder, SN):
    if not os.path.isdir(f"{folder}/ToT_{SN}"):
        os.mkdir(f"{folder}/ToT_{SN}")
    tot = get_tot(y_data, Measurement_time) * 1e9
    zerod = np.sum(tot == 0)
    print(f'NPHE: {1 - zerod / len(tot):.3f}')
    hi, bins = np.histogram(tot, bins=60, range=(0, 60))
    bins = bins[:-1]
    plt.semilogy(bins, hi)
    plt.show()

    fig, ax = plt.subplots()
    maximum = 5 + np.argmax(hi[5:])
    plus_max = np.min([14, len(bins) - maximum])
    # noinspection PyTupleAssignmentBalance
    p_opt, cov = opt.curve_fit(normal, bins[maximum - plus_max:maximum + plus_max],
                               hi[maximum - plus_max:maximum + plus_max],
                               p0=[10000, 25, 50], maxfev=50000)
    print(
        f'Mean ToT: {p_opt[1]:.2f} pm {np.sqrt(cov[1, 1]):.2f} with sigma: {p_opt[2]:.2f} pm {np.sqrt(cov[2, 2]):.2f},')
    ax.axvspan(bins[maximum - plus_max], bins[maximum + plus_max], facecolor="green", alpha=0.35)
    ax.semilogy(bins, hi)
    ax.set_ylim(1e-1)
    ax.plot(bins[maximum - plus_max:maximum + plus_max], normal(bins[maximum - plus_max:maximum + plus_max], *p_opt))

    plt.xlabel("time over threshold in ns")
    plt.ylabel("count")

    plt.show()
    # noinspection PyTupleAssignmentBalance
    p_opt, cov = opt.curve_fit(normal,
        bins[int((p_opt[1] - 2 * p_opt[2]) / bins[1]):int((p_opt[1] + 2 * p_opt[2]) / bins[1])+1],
        hi[int((p_opt[1] - 2 * p_opt[2]) / bins[1]):int((p_opt[1] + 2 * p_opt[2]) / bins[1])+1],
                               p0=p_opt)
    fig, ax = plt.subplots()
    ax.semilogy(bins, hi)
    ax.axvline(bins[int(p_opt[1] / bins[1])], color = "r")
    ax.axvspan(bins[int((p_opt[1] - 2 * p_opt[2]) / bins[1]) + 1], bins[int((p_opt[1] + 2 * p_opt[2]) / bins[1])],
               facecolor="green", alpha=0.35)
    ax.set_ylim(1e-1)
    ax.plot(bins[int((p_opt[1] - 2 * p_opt[2]) / bins[1]) + 1: 1 + int((p_opt[1] + 2 * p_opt[2]) / bins[1])],
            normal(bins[int((p_opt[1] - 2 * p_opt[2]) / bins[1]) + 1: 1 + int((p_opt[1] + 2 * p_opt[2]) / bins[1])], *p_opt))
    plt.xlabel("time over threshold in ns")
    plt.ylabel("count")
    plt.title(f"ToT for {SN}")
    fig.savefig(f"{folder}/ToT_{SN}/{SN}_tot_hist.pdf")
    plt.show()
    print(
        f'Mean ToT: {p_opt[1]:.2f} pm {np.sqrt(cov[1, 1]):.2f} with sigma: {p_opt[2]:.2f} pm {np.sqrt(cov[2, 2]):.2f},')
    return p_opt, cov, tot, 1 - zerod / len(tot)


def corr_plot(y_data, Measurement_time, folder, SN):
    if not os.path.isdir(f"{folder}/ToT_{SN}"):
        os.mkdir(f"{folder}/ToT_{SN}")
    tot, ampl = get_tot_ampl(y_data, Measurement_time)
    tot = tot*1e9
    fig, ax = plt.subplots()
    ax.plot(ampl, tot, 'x')
    plt.xlabel("max amplitude in V")
    plt.ylabel("tot time in ns")
    plt.show()
    fig.savefig(f"{folder}/ToT_{SN}/{SN}_corr_plot.pdf")
    fig, ax = plt.subplots()
    H, xedges, yedges = np.histogram2d(ampl, tot, bins=[30, 30])
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, H.T, cmap='Greys', norm=colors.LogNorm())
    fig.colorbar(im, ax=ax)

    xedges = xedges[:-1] + xedges[1]/2
    y = np.array([yedges[np.argmax(H[i])] for i in range(len(H))]) + yedges[1] / 2
    #y = y[0 < np.argmax(H, axis=1)]
    plt.plot(xedges, y, 'x')
    mask = yedges[1] < y
    y = y[mask]
    # noinspection PyTupleAssignmentBalance
    p_opt, cov = opt.curve_fit(log_vert, xedges[mask][:-4], y[:-4], p0=[11,20,-1], maxfev=10000)
    plt.plot(xedges, log_vert(xedges, *p_opt))
    plt.plot(xedges[mask], y)
    bor = np.nan_to_num(log_vert(xedges, *p_opt))

    pdp = [np.sum(H[:np.max([int(bor[i]-1), 0]), i], axis=0) for i in range(len(bor))]
    plt.xlabel("max amplitude in V")
    plt.ylabel("tot time in ns")
    plt.ylim(0)
    plt.show()
    fig.savefig(f"{folder}/ToT_{SN}/{SN}_corr_plot_hist.pdf")
    print(f'Percentage of partly delayed pulsed: {np.sum(pdp) * 100 / np.sum(H):.3f}%')
    return p_opt, cov


def log_tot(folder, SN, p_opt_hist, cov_hist, nphe, p_opt_corr, cov_corr):
    name = f'{folder}\\log_tot_{SN}.txt'
    f = open(name, 'a')
    text = f'Date: {datetime.now()}\n  nphes: {np.round(nphe, 3)}\n fit results hist: {np.round(p_opt_hist, 4)}\n cov: {np.round(cov_hist, 4)}\nfit results corr: {np.round(p_opt_corr, 4)}\n cov: {np.round(cov_corr, 4)} \n\n\n'
    f.write(text)
    f.close()

def tot_to_tex(folder):
    with open("tot_table.tex", "r") as f:
        to_be_written = "SN&NPHE&Mittlere Time over Threshold"
        for file in os.listdir(folder):
            nphe, p_opt_hist, cov_hist, tot_hist, p_opt_corr, cov_corr = get_fit_results(file)
        f.write(to_be_written)


def show_a_few_tot(filename):
    y_data, mes_time, samplerate = read_tot_mess(filename)
    digital = y_data[1] - y_data[0]
    analog = y_data[2]-y_data[3]
    fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
    for j in range(500):
        ax[0].plot(digital[j])
        ax[1].plot(analog[j])
    plt.show()

def tot_messurement(number_of_mesurements:int):
    pass

if __name__ == '__main__':
    pass
