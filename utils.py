import numpy as np
import os
import io
from matplotlib import pyplot as plt


def postprocessing(data):
    fft = np.fft.fft(data)
    log = np.log(abs(fft))
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


def save_figs(train, target, model, path, step, converting=True, regression=False):
    train = train[0]
    logit = model([train])[0]
    target = target[0]
    if regression:
        k_index_scaler = [1e13, 1e9, 1e5, 1e1, 1e2]
        logit = k_warping(train, np.divide(logit, k_index_scaler))
        target = k_warping(train, np.divide(target, k_index_scaler))
    if converting:
        train = postprocessing(train)
        logit = postprocessing(logit)
        target = postprocessing(target)
        train = train[:int(len(train)/2)]
        logit = logit[:int(len(logit)/2)]
        target = target[:int(len(target)/2)]

    ax1 = plt.subplot(311)
    ax1.plot(train)
    ax2 = plt.subplot(312)
    ax2.plot(logit)
    ax3 = plt.subplot(313)
    ax3.plot(target)

    img_name = 'converted' + str(step) + '.jpg' if converting else 'signal' + str(step) + '.jpg'
    plt.savefig(os.path.join(path, img_name))
    plt.close()


def save_figs_to_arr(train, target, model, converting=True, regression=False):
    train = train[0]
    logit = model([train])[0]
    target = target[0]
    if regression:
        k_index_scaler = [1e13, 1e9, 1e5, 1e1, 1e2]
        logit = k_warping(train, np.divide(logit, k_index_scaler))
        target = k_warping(train, np.divide(target, k_index_scaler))
    if converting:
        train = postprocessing(train)
        logit = postprocessing(logit)
        target = postprocessing(target)
        train = train[:int(len(train)/2)]
        logit = logit[:int(len(logit)/2)]
        target = target[:int(len(target)/2)]

    f = plt.figure()
    ax1 = plt.subplot(311)
    ax1.plot(train)
    ax2 = plt.subplot(312)
    ax2.plot(logit)
    ax3 = plt.subplot(313)
    ax3.plot(target)

    buf = io.BytesIO()
    f.savefig(buf, format='raw')
    buf.seek(0)
    arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                     newshape=(int(f.bbox.bounds[3]), int(f.bbox.bounds[2]), -1))
    buf.close()
    plt.close(f)
    return arr



if __name__ == '__main__':
    k_warping(np.ones([32]), [0.1, 0.52, 0.257, 2, 0.521])