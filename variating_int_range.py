import numpy as np
import matplotlib.pyplot as plt
import modules.data_analysis as data
import time
import os
from os import listdir, mkdir
from os.path import isfile, join

import requests
import datetime


if __name__ == '__main__':
    path = '/media/pmttest/TOSHIBA EXT/Messdaten/'
    print(f'Started at {str(datetime.datetime.now().time())[0:8]}')
    before_completed = time.time()
    SerialNumbers = []
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for name in files:
        SerialNumbers.append = name[7:-3]
    TOKEN = '???'#No one controlls my bot!!!
    chat_id = '???'
    for SN in SerialNumbers:
        try:
            mkdir(path + f'KM3Net_{SN}')
        except FileExistsError:
            pass
        h5_filename = path + '/KM3Net_{}.h5'.format(SN)
        reader = data.WavesetReader(h5_filename)
        for HV in reader.wavesets:
            try:
                mkdir(path + f'KM3Net_{SN}/{HV}')
            except FileExistsError:
                pass
            before = time.time()
            waveforms = reader[HV].waveforms
            h_int =  reader[HV].h_int
            message = f'Analysis for SN = {SN}, HV = {HV} started'
            print(message)
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
            requests.get(url)

            ped_min, ped_max, sig_min, sig_max = 0, 150, 270, 330

            sig_min, sig_max, gains, nphes, gain_errs = data.mp_var_int_range(waveforms, ped_min = ped_min, ped_max = ped_max,\
            sig_min_start = sig_min, sig_max_start = sig_max,h_int = h_int, interval_sig_min = 1,\
            interval_sig_max = 1, number_sig_min =200 , number_sig_max = 200, number_of_processes = os.cpu_count(), SN = SN, HV = HV)
            measurment_time = time.time()-before
            hours = int(measurment_time/3600)
            minutes = int((measurment_time-hours*3600)/60)
            seconds = int((measurment_time-hours*3600-minutes*60))
            print(f'computing time: {hours}h{minutes}m{seconds}s')
            try:
                data.plot_hist_variable_values(sig_min, sig_max, gains,nphes = nphes,gain_errs = gain_errs, h_int = h_int, nrows = 4,waveforms = waveforms, name = (path + f'KM3Net_{SN}/{HV}/var_int_range.pdf') )
            except PermissionError:
                message = f"Permission Error with SN = {SN}, HV = {HV}"
                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
                requests.get(url)
    measurment_time = time.time()-before_completed
    hours = int(measurment_time/3600)
    minutes = int((measurment_time-hours*3600)/60)
    seconds = int((measurment_time-hours*3600-minutes*60))


    message = f"Computing finished after {hours}h{minutes}m{seconds}s"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url)
