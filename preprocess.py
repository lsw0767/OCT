"""
OCT data input producer
load preprocessed .npy data with threading

"""
import numpy as np
import os
import tqdm
import queue


from multiprocessing import Process


def normalize(arr):
    return (arr - arr.min(axis=1, keepdims=True)) / (
                arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))


def convert2npy(data_list, mode, split_num=10, pid=0, index_no=0, use_shift=True):
    print('process start: ', mode, pid)

    signal_per_split = len(data_list)/split_num

    count = 0
    before = None
    after = None
    if pid%4==0:
        iterator = tqdm.tqdm(enumerate(data_list))
    else:
        iterator = enumerate(data_list)
    for i, item in iterator:
        temp = open(os.path.join(item[0], 'Before_K-linearization', item[1])).readlines()
        sample = [line.split() for line in temp]
        sample = np.asarray(sample, dtype=np.float32).transpose([1, 0])
        if use_shift:
            sample = np.reshape(sample, [-1])
            sample[:-4] = sample[4:]
            sample = np.reshape(sample, [1000, 2048])[:500]
        else:
            sample = sample[:500]
        sample = normalize(sample)
        if before is None:
            before = sample
        else:
            before = np.concatenate([before, sample])

        temp = open(os.path.join(item[0], 'After_K-linearization', item[1])).readlines()
        sample = [line.split() for line in temp]
        sample = np.asarray(sample, dtype=np.float32).transpose([1, 0])
        if use_shift:
            sample = np.reshape(sample, [-1])
            sample[:-4] = sample[4:]
            sample = np.reshape(sample, [1000, 2048])[:500]
        else:
            sample = sample[:500]
        sample = normalize(sample)
        if after is None:
            after = sample
        else:
            after = np.concatenate([after, sample])

        if (i+1)%signal_per_split==0:
            count+=1
            shuffle_idx = np.arange(len(after))
            if mode=='train':
                np.random.shuffle(shuffle_idx)
            before = np.expand_dims(np.asarray(before, dtype=np.float32)[shuffle_idx], -1)
            after = np.expand_dims(np.asarray(after, dtype=np.float32)[shuffle_idx], -1)
            data = np.concatenate([before, after], axis=-1)
            fname = '{}_{}.npy'.format(mode, str(count+split_num*pid).zfill(3))

            if pid % 3 == 0:
                print(fname, data.shape)
            np.save('/cache/preprocess_npy/index{}/'.format(index_no+1)+fname, data)

            after = None
            before = None


if __name__ == '__main__':
    index_list = ['K-index_no.{}'.format(i+4) for i in range(2)]
    train_sample = ['Sample{}'.format(i+1) for i in range(9)]
    train_list = []
    for index in index_list:
        temp = []
        for sample in train_sample:
            sig_list = os.listdir(os.path.join('data', 'OCT_K_linear', index, sample, 'After_K-linearization'))
            sig_list.sort()
            for sig in sig_list:
                assert sig.split('.')[-1]=='txt'
                sig_num = int(sig.split('_')[-1].split('.')[0])
                if sig_num>50:
                    continue
                temp.append([os.path.join('data', 'OCT_K_linear', index, sample), sig])

        shuffle_idx = np.arange(len(temp))
        np.random.shuffle(shuffle_idx)
        train_list.append(np.asarray(temp)[shuffle_idx])
    train_list = np.asarray(train_list)

    index_list = ['K-index_no.{}'.format(i+4) for i in range(2)]
    test_sample = ['Sample{}'.format(i+10) for i in range(1)]
    test_list = []
    for index in index_list:
        temp = []
        for sample in test_sample:
            sig_list = os.listdir(os.path.join('data', 'OCT_K_linear', index, sample, 'After_K-linearization'))
            sig_list.sort()
            for sig in sig_list:
                assert sig.split('.')[-1]=='txt'
                sig_num = int(sig.split('_')[-1].split('.')[0])
                if sig_num>50:
                    continue
                temp.append([os.path.join('data', 'OCT_K_linear', index, sample), sig])

        test_list.append(np.asarray(temp))
    test_list = np.asarray(test_list)
    process_queue = queue.Queue()

    train_process = 9
    test_process = 1

    train_idx = int(len(train_list[0])/train_process)
    test_idx = int(len(test_list[0])/test_process)
    print(train_list)
    print(train_idx, test_idx)

    for i in range(len(train_list)):
        for j in range(train_process):
            process_queue.put(
                Process(target=convert2npy, args=(train_list[i][train_idx*j:train_idx*(j+1)], 'train', 10, j, i+3, False))
            )
    for i in range(len(test_list)):
        for j in range(test_process):
            process_queue.put(
                Process(target=convert2npy, args=(test_list[i][test_idx*j:test_idx*(j+1)], 'test', 10, j, i+3, False))
            )

    parallel_process = 2
    iters = (train_process + test_process)*3 / parallel_process
    while process_queue.qsize()>0:
        p = []
        for i in range(parallel_process):
            p.append(process_queue.get())
            p[-1].start()
        for a in p:
            a.join()
