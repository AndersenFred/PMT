import modules.inst_control as inst
import time
import numpy as np
import matplotlib.pyplot as plt
import modules.data_analysis as data
import sys
import os
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

func_gen = inst.Funk_Gen()
func_gen.pulse(freq = 1e3, ampl = 1, off=0, width = 100e-9, channel = 1)
func_gen.on()
#SHR = inst.SHR(volt = HV)
#SHR.voltage(HV)
#SHR.output_on()
osci = inst.Osci()
time.sleep(1)
osci.write('HOR:FAST:STATE OFF')
osci.write('DISplay:WAVEform ON')
osci.write('ACQ:STATE RUN')
path = f'/media/pmttest/TOSHIBA EXT/Messdaten/KM3Net_{SN}'
try:
    os.mkdir(path)
except FileExistsError:
    pass
path += f'/{HV}'

try:
    os.mkdir(path)
except FileExistsError:
    pass

while (True):
    if 'y' == input(f'HV = {HV}\nCorrect HV and correct scale?(y)\n'):
        break

h5_filename = f'/media/pmttest/TOSHIBA EXT/Messdaten/KM3Net_{SN}.h5'
y_values, Measurement_time, YOFF, YMU, samplerate = osci.messung(number, Measurement_time = 8e-8, vertical_delay = 180e-9, samplerate = 12.5e9, Max_Ampl = 80e-3)
try:
    data.save_rawdata_to_file(h5_filename,y_values, Measurement_time, YOFF, YMU, samplerate, HV)
except ValueError:
    import h5py
    with h5py.File(h5_filename, "r+") as f:
        del f[f"{HV}"]
        f.close()
    data.save_rawdata_to_file(h5_filename, y_values, Measurement_time, YOFF, YMU, samplerate, HV)

x, y = data.x_y_values(y_values, YOFF, YMU, Measurement_time, samplerate)
h_int = 1/samplerate
waveforms = y
plot = False
if not HV % 100:
    plot = True
y, x, int_ranges = data.hist(waveforms, plot = plot, path = path)
gain, nphe, gain_err = data.hist_fitter(y,x,h_int, plot = plot, path = path, title = f'SN: {SN}, HV: {HV}')
print('Gain: ' ,gain, 'pm', gain_err)
if (gain < 3e6 and gain + gain_err > 3e6) or (gain > 3e6 and gain - gain_err < 3e6):
    print('Nominal Gain is within the error')
elif gain >3e6:
    print('Gains is too high')
elif gain < 3e6:
    print('Gain is too low')

print('Number Photoelektrons: ', nphe)
data.add_fit_results(h5_filename, f'{HV}', gain, nphe,gain_err, int_ranges)

reader = data.WavesetReader(h5_filename)
if number > 10000 or len(reader.wavesets) > 4:
    data.analysis_complete_data(h5_filename, reanalyse= True, saveresults = True, SN = sys.argv[1], plot = True)
    func_gen.off()
#data.analysis_complete_data('Messdaten/KM3Net_AB2363.h5', 1167)
