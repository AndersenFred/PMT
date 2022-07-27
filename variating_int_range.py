import numpy as np
import matplotlib.pyplot as plt
import Data_analysis as data
import time
before = time.time()
HV = '1170_0'
SN = 10221
h5_filename = 'Messdaten/KM3Net_{}.h5'.format(SN)

reader = data.WavesetReader(h5_filename)
#print(reader.wavesets)
waveforms = reader[HV].waveforms
h_int =  reader[HV].h_int
ped_min, ped_max, sig_min, sig_max = 750, 999, 240, 305

sig_min, sig_max, gains, nphes, gain_errs = data.hist_variable_values(waveforms, ped_min = ped_min, ped_max = ped_max,\
sig_min_start = sig_min, sig_max_start = sig_max,h_int = h_int, interval_sig_min = 1,\
interval_sig_max = 1, number_sig_min = 100, number_sig_max = 100)
measurment_time = time.time()-before
hours = int(measurment_time/3600)
minutes = int((measurment_time-hours*3600)/60)
seconds = int((measurment_time-hours*3600-minutes*60))
print(f'computing time: {hours}h{minutes}m{seconds}s')

data.plot_hist_variable_values(sig_min, sig_max, gains,nphes = nphes,gain_errs = gain_errs,h_int = h_int, name = f'variating_int_range/Variating_signal_int_ranges {SN}, HV = {HV}.pdf')
