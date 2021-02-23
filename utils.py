import numpy as np
import os
import io
from matplotlib import pyplot as plt


def postprocessing(data):
    fft = np.fft.fft(data)
    log = np.log(abs(fft)+1e-5)
    return log


def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass


def k_warping(signal, params):
    idx = []
    deci = []
    new_signal = []
    for i in range(len(signal)):
        val = 0.
        for j in range(len(params)):
            val += params[j]*i**(len(params)-1-j)
        idx.append(int(val))
        deci.append(val-int(val))

        new_pos = np.clip(i-idx[i], 1, 2047)
        new_signal.append((1-deci[i]) * signal[new_pos] + deci[i] * signal[new_pos-1])

    arr = np.asarray(new_signal)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def save_figs_to_arr(figs, converting=True):
    num_img = len(figs)
    f = plt.figure()

    subplot_idx = num_img*100+11
    for i in range(num_img):
        img = figs[i]
        if converting:
            img = postprocessing(img)
            img = img[:int(len(img)/2)]

        ax = plt.subplot(subplot_idx + i)
        ax.plot(img)

    buf = io.BytesIO()
    f.savefig(buf, format='raw')
    buf.seek(0)
    arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                     newshape=(int(f.bbox.bounds[3]), int(f.bbox.bounds[2]), -1))
    buf.close()
    plt.close(f)
    return arr[:, :, :3]


if __name__ == '__main__':
    k_warping(np.ones([32]), [0.1, 0.52, 0.257, 2, 0.521])