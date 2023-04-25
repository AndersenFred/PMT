import time
import requests
import numpy as np
import modules.data_analysis as data
import matplotlib.pyplot as plt
import sys
from modules.inst_controll import Osci, Funk_Gen


SN = "2.15924"
folder = "D:\ToT_Messdaten"
filename = f"{folder}\{SN}.h5"
print(data.HV_in_Hex(SN=f"D:\Gain_Messdaten\KM3Net_{SN}.h5"))
osci = Osci()
osci.write('HOR:FAST:STATE OFF')
osci.write('DISplay:WAVEform ON')
osci.write('ACQ:STATE RUN')
#func_gen = Funk_Gen()
#func_gen.pulse(freq = 1e3, ampl = 2, off=0, width = 16-9, channel = 1)
#func_gen.on()
input("Correct scale and HV?")
#"https://api.telegram.org/bot5619479679:AAE2zq66Opudre_W0DhCeDFup_N7BszXIQg/sendMessage?chat_id=563209854&text="
token = '5619479679:AAE2zq66Opudre_W0DhCeDFup_N7BszXIQg'
chat_id = '563209854'
template = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text="
p = []
requests.get(template + f'Measurement  for PMT {SN} started')
try:
    for _ in range(4):
        y, mes_time, YOFF, YMU, samplerate = osci.messung(62500, samplerate=6.25e9, Measurement_time=160e-9,
                                                          chanels=['CH1', 'CH2', 'CH3', 'CH4'])
        p.append(y)
        time.sleep(2)
except:
    requests.get(template + f'Measurement failed ')
    sys.exit()
#func_gen.off()


y = np.concatenate(p, axis=1)
data.save_tot_mess(filename, y, mes_time, YOFF, YMU, samplerate)
requests.get(template + f'Measurement done ')

y = data.to_data(y, YOFF, YMU)
p_opt_hist, cov_hist, tot, nphe = data.tot_hist(y, mes_time, folder, SN)
p_opt_corr, cov_corr = data.corr_plot(y, mes_time, folder, SN)
data.write_tot_fit(filename, nphe, p_opt_hist, cov_hist, p_opt_corr, cov_corr)
data.save_first(y, mes_time, data.get_tot(y, mes_time), folder, SN)
data.log_tot(folder, SN, p_opt_hist, cov_hist, nphe, p_opt_corr, cov_corr)
