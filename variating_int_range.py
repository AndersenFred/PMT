import numpy as np
import matplotlib.pyplot as plt
import Data_analysis as data


HV = '1170_0'
SN = 10221
h5_filename = 'Messdaten/KM3Net_{}.h5'.format(SN)

reader = data.WavesetReader(h5_filename)
#print(reader.wavesets)
waveforms = reader[HV].waveforms
h_int =  reader[HV].h_int
ped_min, ped_max, sig_min, sig_max = 750, 999, 190, 350
data.mean_plot(waveforms, int_ranges = (ped_min, ped_max, sig_min, sig_max))
gains, signal_max, gain_errs = data.hist_variable_sig_max(waveforms, ped_min=ped_min, ped_max= ped_max, sig_min= sig_min, sig_max_start=sig_max,h_int = h_int, number = 100, interval = 4)
fig, ax = plt.subplots(figsize = (10,5))
ax.errorbar(signal_max, gains/1e6,yerr = gain_errs/1e6, fmt = 'x', label = 'gains')
ax.legend()
plt.xlabel("signal max")
plt.ylabel("gain [1e6]")
plt.title(f'Variating signal max {SN}, HV = {HV}')
plt.show(block = False)
fig.savefig(f'variating_int_range/Variating signal max {SN}, HV = {HV}.pdf')

ped_min, ped_max, sig_min, sig_max = 750, 999, 250, 350
gains, signal_max, gain_errs = data.hist_variable_sig_max(waveforms, ped_min=ped_min, ped_max= ped_max, sig_min= sig_min, sig_max_start=sig_max,h_int = h_int, number = 5, interval = 4)
fig, ax = plt.subplots(figsize = (10,5))
ax.errorbar(signal_max, gains/1e6,yerr = gain_errs/1e6, fmt = 'x', label = 'gains')
ax.legend()
plt.xlabel("signal max")
plt.ylabel("gain [1e6]")
plt.title(f'Variating signal max {SN}, HV = {HV}')
plt.show()
fig.savefig(f'variating_int_range/Variating signal min {SN}, HV = {HV}.pdf')


ped_min, ped_max, sig_min, sig_max = 750, 999, 250, 350

sig_min, sig_max, gains = data.hist_variable_values(waveforms, ped_min = ped_min, ped_max = ped_max,\
sig_min_start = sig_min, sig_max_start = sig_max,h_int = h_int, interval_sig_min = 3,\
interval_sig_max = 5, number_sig_min = 50, number_sig_max = 50)
data.plot_hist_variable_values(sig_min, sig_max, gains, name = 'variating_int_range/Variating_signal_int_ranges {SN}, HV = {HV}.pdf')


