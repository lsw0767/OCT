import numpy as np
import os
import csv

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

ORDER = 4

def func_4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x +e
def func_2(x, a, b, c):
    return a*x**2 + b*x +c
func = func_4 if ORDER==4 else func_2

K_index_num = 5
k_index_list = []
curves = []

for i in range(K_index_num):
    idx = []
    f_list = os.listdir('data/OCT_K_linear/K-index_no.{}'.format(i+1))
    for file in f_list:
        if file.endswith('txt'):
            f = open(os.path.join('data/OCT_K_linear/K-index_no.{}'.format(i+1), file)).readlines()
            for line in f:
                for val in line.split():
                    idx.append(val)
        elif file.endswith('csv'):
            f = csv.reader(open(os.path.join('data/OCT_K_linear/K-index_no.{}'.format(i+1), file)))
            for line in f:
                idx.append(line[0])
            idx[0] = 0

    idx = np.asarray(idx, dtype=np.float32)
    popt, pcov = curve_fit(func, np.arange(0, 2048), idx)

    plt.scatter(np.arange(0, 2048), idx, marker='.', s=1, c='red', label='k-index')
    plt.plot(np.arange(0, 2048), func(np.arange(0, 2048), *popt), label='fitted curve')
    plt.legend()
    plt.savefig('data/imgs/curve_fit_{}order_{}.png'.format(ORDER, str(i+1).zfill(2)))
    plt.close()

    k_index_list.append(popt)
    curves.append(func(np.arange(0, 2048), *popt))

k_index_list = np.asarray(k_index_list, dtype=np.float32)
k_index_scaler = [1e11, 1e7, 1e3, 1e0, 1e0]
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

for idx in curves:
    plt.plot(np.arange(0, 2048), idx)
plt.savefig('data/imgs/coeffs.png')
plt.close()