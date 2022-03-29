import Data_analysis as data
import peeemtee as pt
import numpy as np
filename = 'Messdaten/KM3Net_{}.h5'.format('AB2363')
#reader = data.WavesetReader(filename)
#waveset = reader["1200_4"]
#waveforms = waveset.waveforms
#h_int = waveset.h_int
filename = "/home/pmttest/Desktop/Frederik/Testdaten/BA0455.h5"
reader = pt.WavesetReader(filename)
waveset = reader["1088"]
waveforms = waveset.waveforms
h_int = waveset.h_int
y,x = data.hist(waveforms, external_value = True)
print(data.hist_fitter(y,x,h_int))
