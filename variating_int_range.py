import numpy as np
import matplotlib.pyplot as plt
import Data_analysis as data


HV = '1130_0'
SN = 10221
h5_filename = 'Messdaten/KM3Net_{}.h5'.format(SN)

reader = data.WavesetReader(h5_filename)
print(reader.wavesets)
waveforms = reader[HV].waveforms
h_int =  reader[HV].h_int
gains, signal_max, gain_errs = data.hist_variable_values(waveforms, ped_min=750, ped_max= 999, sig_min= 190, sig_max_start=350,h_int = h_int, number = 50, interval = 1)
fig1, ax1 = plt.subplots(figsize = (10,5))
ax1.errorbar(signal_max, gains/1e6,yerr = gain_errs/1e6, fmt = 'x', label = 'gains')
ax1.legend()
plt.xlabel("signal max")
plt.ylabel("gain [1e6]")
plt.show()
