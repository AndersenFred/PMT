import Inst_control as inst
import time
import numpy as np
import matplotlib.pyplot as plt
import Data_analysis as data
import sys

try:
    SN = sys.argv[1]
    HV = sys.argv[2]
except IndexError:
    print('usage: python {} <serial number> <high voltage>'.format(sys.argv[0]))
    sys.exit()

SHR = inst.SHR(volt = HV)
SHR.voltage(HV)
SHR.output_on()
time.sleep(5)

h5_filename = 'KM3Net_{}'.format(SN)
osci =  inst.Osci()
y_values, Measurement_time, YOFF, YMU, samplerate = osci.messung(100, Measurement_time =2*10**-6, samplerate = 3.125e10, Max_Ampl = 10e-3)

x,y = data.x_y_values(y_values, Measurement_time, YOFF, YMU, samplerate)
data.save_to_file(h5_filename,y_values, Measurement_time, YOFF, YMU, samplerate, HV)
data, measurement_time, y_off, YMULT, samplerate, HV = data.values(filename = '{}.h5'.format(h5_filename))

SHR.output_off()
del SHR
del osci
