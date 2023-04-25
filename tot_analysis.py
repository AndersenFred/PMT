import os
import gc
import modules.data_analysis as data
import sys
import numpy as np
import matplotlib.pyplot as plt


SN = "2.15917"
folder = "C:\\Users\\P\\Desktop\\BA\\ToT_Messdaten"


tot_all = []
for file in os.listdir(folder):
    if ".h5" in file:
        try:
            SN = file.split(".h5")[0]
            print(f"ToT analysis for {SN}")
            filename = f"{folder}\\{file}"
            y, mes_time, samplerate = data.read_tot_mess(filename)
            tot, ampl = data.get_tot_ampl(y, mes_time)
            p_opt_hist, cov_hist, nphe = data.tot_hist(y, tot, folder, SN)
            p_opt_ampl, cov_ampl = data.ampl_hist(ampl, folder, SN)
            data.save_first(y, mes_time, tot, folder, SN)
            tot_all.append(p_opt_hist[1])
            p_opt_corr, cov_corr, pdp = data.corr_plot(folder, SN, tot, ampl)
            data.write_tot_fit(filename, nphe, p_opt_hist, cov_hist, p_opt_corr, cov_corr)
            data.log_tot(folder, SN, p_opt_hist, cov_hist, p_opt_ampl, cov_ampl, nphe, p_opt_corr, cov_corr, pdp)
            gc.collect()
        except RuntimeError as e:
            print(e)

fig, ax = plt.subplots()
ax.hist(tot_all, bins =11)
plt.xlabel("ToT in ns")
plt.ylabel("count")
plt.title(f"Time over Threshold")
plt.grid()
plt.tight_layout()
fig.savefig(f"{folder}/tot_all_hist.pdf")