import Inst_control as inst
import time
import numpy as np
import matplotlib.pyplot as plt
import Data_analysis as data
import sys
try:
    SN = sys.argv[1]
    HV = int(sys.argv[2])
except IndexError:
    print('usage: python {} <serial number> <high voltage>'.format(sys.argv[0]))
    sys.exit()
except TypeError:
    print('usage: python {} <serial number> <high voltage>, high voltage must be an integer'.format(sys.argv[0]))
    sys.exit()
try:
    number = int(sys.argv[3])
except IndexError:
    number = 10000
except TypeError:
    number = 10000

SHR = inst.SHR(volt = HV)
SHR.voltage(HV)
SHR.output_on()
osci =  inst.Osci()
osci.write('HOR:FAST:STATE OFF')
osci.write('DISplay:WAVEform ON')
while (True):
    if 'y' == input(f'HV = {HV}\nCorrect HV and correct scale?(y)\n'):
        break
h5_filename = 'Messdaten/KM3Net_{}.h5'.format(SN)
y_values, Measurement_time, YOFF, YMU, samplerate = osci.messung(number, Measurement_time =8e-8,vertical_delay = 180e-9, samplerate = 12.5e9, Max_Ampl = 80e-3)
i = data.save_rawdata_to_file(h5_filename,y_values, Measurement_time, YOFF, YMU, samplerate, HV)
x,y = data.x_y_values(y_values, YOFF, YMU, Measurement_time, samplerate)
h_int = 1/samplerate
waveforms = y
y,x, int_ranges = data.hist(waveforms, plot = True)
gain, nphe, gain_err =  data.hist_fitter(y,x,h_int)
data.add_fit_results(h5_filename, f'{HV}_{i}', gain, nphe,gain_err, int_ranges)
print('Gain: ' ,gain, 'pm', gain_err)
print('Number Photoelektrons: ', nphe)
#data.analysis_complete_data('Messdaten/KM3Net_AB2363.h5', 1167)
