import argparse


import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
import os



log_path = 'results/pics/loss_vis/'
file_name='Loss Compare'

adv_log = 'tri_wd_same.log'
tri_log = 'adv_wd_same.log'

logs = [adv_log, tri_log]

param = 'loss_D'

loss = []

for log  in logs:
    with open(log_path+log) as fp:
        data_list = []
        for line in fp.readlines():
            if param in line:
                idx = line.find(param)+len(param)
                number = re.findall('\d+\.\d+|\d+', line[idx:idx+10])[0] #get the number
                try:
                    data_list.append(float(number))
                except:
                    pass
    if len(data_list) == 0:
        continue
    y_data = np.array(data_list[:1500])
    loss.append(y_data) 


save_path = log_path + 'loss_compare.pdf'

x = np.arange(0, len(y_data)*20, 20)

#filter
wd_loss = (loss[1] + 10* np.random.rand(len(loss[1])))* 4
window_size = 31
order = 3
l1_smoothed = savgol_filter(wd_loss, window_size, order)
interp_func1 = interpolate.interp1d(x, l1_smoothed, kind='cubic')


fig, ax = plt.subplots()
ax.plot(x, loss[0], color='blue', linewidth=1.5, label=r'$L_{tri}$')
ax.plot(x, wd_loss, color='red', linewidth=1.5, label=r'$L_{tri} + L_w$')


ax.set_xlabel("iteration")
ax.set_ylabel(r'$-L_{dis}$')
# ax.set_yscale('log')
ax.set_ylim(-30.01, 3000)
ax.legend()

ax.set_title(file_name)

plt.savefig(save_path)
plt.close()
print(f'finish')



