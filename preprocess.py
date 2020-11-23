"""
OCT data input producer
load preprocessed .npy data with threading

Todo:   integrate argparser over preprocess-input_producer-train_env
        cache memory handling
        use more process if possible
"""
import numpy as np
import os
import tqdm
import psutil

from subprocess import call, Popen, PIPE
from multiprocessing import Process

K_index_num = 3
sample_num = 10
test_sample_num = 1
signal_num = 10
split_num = 10


def convert2npy(k_index):
    idx = np.zeros([signal_num*sample_num, 2], dtype=np.int32)
    for m in range(signal_num*sample_num):
        idx[m, 0] = m//signal_num+1
        idx[m, 1] = m%signal_num
    train_idx = idx[:(sample_num-test_sample_num)*signal_num]
    test_idx = idx[(sample_num-test_sample_num)*signal_num:]
    print(train_idx.shape, test_idx.shape)
    np.random.shuffle(train_idx)

    for mode in ['train', 'test']:
        count = 0
        before = []
        after = []
        idx = train_idx if mode=='train' else test_idx
        for j in tqdm.tqdm(range(len(idx))):
            fname = os.listdir('data/OCT_K_linear/K-index_no.{}/Sample_no.{}/Before_K-linearization'.format(
                k_index, idx[j, 0]))
            fname = fname[0].split('0')[0]

            temp = open('data/OCT_K_linear/K-index_no.{}/Sample_no.{}/Before_K-linearization/{}{}.txt'.format(
                k_index, idx[j, 0], fname, str(idx[j, 1]).zfill(4)
            )).readlines()
            sample = []
            for line in temp:
                sample.append(line.split())
            sample = np.asarray(sample).transpose([1, 0])
            for x in sample:
                x = np.asarray(x, dtype=np.float16)
                before.append(x)

            temp = open('data/OCT_K_linear/K-index_no.{}/Sample_no.{}/After_K-linearization/{}{}.txt'.format(
                k_index, idx[j, 0], fname, str(idx[j, 1]).zfill(4)
            )).readlines()
            sample = []
            for line in temp:
                sample.append(line.split())
            sample = np.asarray(sample).transpose([1, 0])
            for x in sample:
                x = np.asarray(x, dtype=np.float16)
                after.append(x)

            count += 1
            if count%10==0:
                print(k_index, count)
            if count%(signal_num*sample_num/split_num)==0:
                shuffle_idx = np.arange(len(after))
                if mode=='train':
                    np.random.shuffle(shuffle_idx)
                after = np.asarray(after, dtype=np.float16)[shuffle_idx]
                before = np.asarray(before, dtype=np.float16)[shuffle_idx]
                np.save('data/preprocess_npy/{}/index_{}/after_{}.npy'.format(mode, k_index, int(count//(signal_num*sample_num/split_num))), after)
                np.save('data/preprocess_npy/{}/index_{}/before_{}.npy'.format(mode, k_index, int(count//(signal_num*sample_num/split_num))), before)

                after = []
                before = []

def drop_cache():
    # check = os.popen('top -b | grep buff/cache')
    # print(check)
    print(psutil.virtual_memory().cached)
    if psutil.virtual_memory().cached > 30000000000:
        # res = os.popen('sudo -S sysctl -w vm.drop_caches=3', 'w').write('010-4121-0767')
        # print(res)

        # os.popen('010-4121-0767')
        cmd1 = Popen(['echo', '010-4121-0767'], stdout=PIPE)
        cmd2 = Popen(['sudo', 'sysctl -w vm.drop_caches=3'], stdin=cmd1.stdout, stdout=PIPE)
        print(cmd2.stdout.read())
        # p.communicate('010-4121-0767\n')



if __name__ == '__main__':
    # with Pool(3) as p:
    #     p.map(convert2npy, [1, 2, 3])

    # drop_cache()

    p = []
    for i in range(3):
        p.append(Process(target=convert2npy, args=(i+1, )))
        p[i].start()
    for i in range(3):
        p[i].join()

