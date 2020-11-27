"""
OCT data input producer
load preprocessed .npy data with threading

"""
import numpy as np
import os
import tqdm

from multiprocessing import Process


def normalize(arr):
    return (arr - arr.min(axis=1, keepdims=True)) / (
                arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))


def convert2npy(data_list, mode, split_num=10, pid=0):
    print('process start: ', mode, pid)
    assert len(data_list)%split_num==0
    signal_per_split = len(data_list)/split_num

    count = 0
    before = None
    after = None
    if pid==2:
        iterator = tqdm.tqdm(enumerate(data_list))
    else:
        iterator = enumerate(data_list)
    for i, item in iterator:
        temp = open(os.path.join(item[0], 'Before_K-linearization', item[1])).readlines()
        sample = [line.split() for line in temp]
        sample = np.asarray(sample, dtype=np.float32).transpose([1, 0])
        sample = np.reshape(sample, [-1])
        sample[:-4] = sample[4:]
        sample = normalize(np.reshape(sample, [1000, 2048])[:999])
        if before is None:
            before = sample
        else:
            before = np.concatenate([before, sample])

        temp = open(os.path.join(item[0], 'After_K-linearization', item[1])).readlines()
        sample = [line.split() for line in temp]
        sample = np.asarray(sample, dtype=np.float32).transpose([1, 0])
        sample = np.reshape(sample, [-1])
        sample[:-4] = sample[4:]
        sample = normalize(np.reshape(sample, [1000, 2048])[:999])
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
            print(data.shape)
            np.save('/cache/preprocess_npy/3index/{}_{}.npy'.format(mode, count+split_num*pid), after)

            after = None
            before = None


if __name__ == '__main__':
    index_list = ['K-index_no.{}'.format(i+1) for i in range(3)]
    train_sample = ['Sample_no.{}'.format(i+1) for i in range(9)]
    train_list = []
    for index in index_list:
        for sample in train_sample:
            sig_list = os.listdir(os.path.join('data', 'OCT_K_linear', index, sample, 'After_K-linearization'))
            sig_list.sort()
            for sig in sig_list:
                assert sig.split('.')[-1]=='txt'
                train_list.append([os.path.join('data', 'OCT_K_linear', index, sample), sig])
    shuffle_idx = np.arange(len(train_list))
    np.random.shuffle(shuffle_idx)
    train_list = np.asarray(train_list)[shuffle_idx]

    index_list = ['K-index_no.{}'.format(i+1) for i in range(3)]
    test_sample = ['Sample_no.{}'.format(i+10) for i in range(1)]
    test_list = []
    for index in index_list:
        for sample in train_sample:
            sig_list = os.listdir(os.path.join('data', 'OCT_K_linear', index, sample, 'After_K-linearization'))
            sig_list.sort()
            for sig in sig_list:
                assert sig.split('.')[-1]=='txt'
                test_list.append([os.path.join('data', 'OCT_K_linear', index, sample), sig])

    p = []
    num_process = 1
    assert len(train_list)%num_process==0
    split_idx = int(len(train_list)/num_process)

    for i in range(num_process):
        p.append(Process(target=convert2npy, args=(train_list[split_idx*i:split_idx*(i+1)], 'train', 45, i)))
        p[i].start()
    p.append(Process(target=convert2npy, args=(test_list, 'test', 5)))
    p[-1].start()
    for i in range(num_process):
        p[i].join()

    p[-1].join()

