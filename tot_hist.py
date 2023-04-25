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
            nphe, p_opt_hist, cov_hist, tot_hist, p_opt_corr, cov_corr = data.get_fit_results(filename)
            tot_all.append(p_opt_hist[1])
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
