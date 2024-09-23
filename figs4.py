import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "sans-serif"

###---------------------------------------------------------------------------###

def mean_sigma2(data, l):
    var_all = []
    for i in range(len(data)-l+1):
        var_all.append(np.var(data[i:i+l]))
    return np.mean(var_all)

def RSI_p(data, delta, l, sigma2):
    val = 0
    for i in range(len(data)):
        val = val + (data[i] - delta)/(l*np.sqrt(sigma2))
        if val < 0:
            val = 0; break
    return val

def RSI_n(data, delta, l, sigma2):
    val = 0
    for i in range(len(data)):
        val = val + (delta - data[i])/(l*np.sqrt(sigma2))
        if val < 0:
            val = 0; break
    return val

def RSI(data, l, sig):
    t_val = t.interval(sig, l*2-2)[1]
    sigma2 = mean_sigma2(data, l)
    diff = t_val * np.sqrt(2 * sigma2 / l)
    rsi = np.zeros(len(data))
    R1 = np.mean(data[:l])
    j = 0; m = 0

    for i in range(len(data)-l):
        if data[i+l] > (R1 + diff):
            rsi[i+l] = RSI_p(data[i+l:i+l+l], R1 + diff, l, sigma2)
            if rsi[i+l] > 0: 
                R1 = np.mean(data[i+l:i+l+l]); j = j+1; m = 0
            else:
                m = m+1
                if m >= l: R1 = np.mean(data[i+1:i+l+1])
        elif data[i+l] < (R1 - diff):
            rsi[i+l] = RSI_n(data[i+l:i+l+l], R1 - diff, l, sigma2)
            if rsi[i+l] > 0: 
                R1 = np.mean(data[i+l:i+l+l]); j = j+1; m = 0
            else:
                m = m+1
                if m >= l: R1 = np.mean(data[i+1:i+l+1])
        else:
            if j == 0:
                R1 = np.mean(data[i+1:i+l+1])
            else:
                m = m+1
                if m >= l: R1 = np.mean(data[i+1:i+l+1])

    return rsi

###---------------------------------------------------------------------------###

pc = xr.open_dataarray('./run_cor_11_IPDC_AA_SAT.nc').values
N = len(pc) 
run_N = 11

l = 8
sig = 0.95

rsi_pc = np.full(len(pc), np.nan)
rsi_pc[run_N//2:N-run_N//2] = RSI(pc[~np.isnan(pc)], l, sig)

plt.figure(figsize=(12, 5))

ax = plt.axes()

ax.bar(range(1979, 2022), rsi_pc, width=1, zorder=2, color='grey', edgecolor='k')

ax.set_title('RSI of Run_Cor(9) (L=8)', loc='left', fontsize=25)

plt.tick_params(labelsize=20)

# plt.savefig('./figs4-final.pdf')

plt.show()
