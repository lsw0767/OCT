import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

ORDER = 4

def func_4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x +e
def func_2(x, a, b, c):
    return a*x**2 + b*x +c
func = func_4 if ORDER==4 else func_2

K_index_num = 3
k_index_list = []

for i in range(K_index_num):
    f = open('data/OCT_K_linear/K-index_no.{}/K_index_{}.txt'.format(i+1, str(i+1).zfill(2))).readlines()

    idx = []
    for line in f:
        for val in line.split():
            idx.append(val)
    idx = np.round(np.asarray(idx, dtype=np.float32), 3)

    popt, pcov = curve_fit(func, np.arange(0, 2048), idx)

    plt.scatter(np.arange(0, 2048), idx, marker='.', s=1, c='red', label='k-index')
    plt.plot(np.arange(0, 2048), func(np.arange(0, 2048), *popt), label='fitted curve')
    plt.legend()
    plt.savefig('imgs/curve_fit_{}order_{}.jpg'.format(ORDER, str(i+1).zfill(2)))
    plt.close()

    k_index_list.append(popt)

k_index_list = np.asarray(k_index_list, dtype=np.float32)
k_index_scaler = [1e13, 1e9, 1e5, 1e1, 1e2]
print(k_index_list)
f = open('temp.txt', 'w')
for line in k_index_list:
    for val in line:
        f.write(str(val) + '\t')
    f.write('\n')
np.save('data/k_index_{}.npy'.format(ORDER), k_index_list)
k_index_list = k_index_list*k_index_scaler
print(k_index_list)
np.save('data/k_index_{}_scaled.npy'.format(ORDER), k_index_list)
