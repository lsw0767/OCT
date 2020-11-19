import numpy as np
import os
import queue
import time
import threading


class IP:
    def __init__(self, is_train=True, k_regression=True, order=4,
                 num_index=3, num_split=18, sample_per_split=5*1000):
        self.is_train = is_train
        self.k_regression = k_regression
        self.num_index = num_index
        self.num_split = num_split
        self.sample_per_split = sample_per_split

        path = 'data/preprocess_npy/train' if is_train else 'data/preprocess_npy/test'
        data_list = [[os.path.join(path, 'index_{}'.format(i+1), 'before_{}.npy'.format(j+1))
                       for j in range(num_split)] for i in range(num_index)]
        self.data_list = np.asarray(data_list)

        if k_regression:
            target_list = np.load('data/k_index_{}_scaled.npy'.format(order))
        else:
            target_list = [[os.path.join(path, 'index_{}'.format(i+1), 'after_{}.npy'.format(j+1))
                           for j in range(num_split)] for i in range(num_index)]
        self.target_list = np.asarray(target_list)

        self.batch_index = 0
        self.load_counter = 0

        self.train_queue = [queue.Queue() for _ in range(num_index)]
        self.target_queue = [queue.Queue() for _ in range(num_index)]

        self.idx = np.arange(0, self.num_split)

    def _init_queue(self, shuffle=True, batch_per_class=32):
        # print('subthread start')
        while True:
            while self.target_queue[0].qsize()<batch_per_class*10:
                if self.batch_index%self.num_split==0:
                    self.batch_index = self.batch_index//self.num_split

                if shuffle:
                    np.random.shuffle(self.idx)
                train = [np.load([self.data_list[i, self.idx] for i in range(self.num_index)][i][self.batch_index]) for i in range(self.num_index)]
                # temp: [3, sample_per_split, 2048, 1000]
                for i in range(self.num_index):
                    # self.train_queue[i].append(sample for sample in train[i])
                    for sample in train[i]:
                        self.train_queue[i].put(sample)

                if not self.k_regression:
                    target = [np.load([self.target_list[i, self.idx] for i in range(self.num_index)][i][self.batch_index]) for i in range(self.num_index)]
                    for i in range(self.num_index):
                        # self.target_queue[i].append(sample for sample in target[i])
                        for sample in target[i]:
                            self.target_queue[i].put(sample)
                else:
                    for i in range(self.num_index):
                        for _ in range(self.sample_per_split):
                            self.target_queue[i].put(self.target_list[i])
                self.batch_index = (self.batch_index+1)%self.num_split
            time.sleep(1)

    @staticmethod
    def _normalize(arr):
        return (arr - arr.min(axis=1, keepdims=True)) / (
                    arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))

    def init_producer(self, shuffle=True, batch_per_class=32):
        t = threading.Thread(target=self._init_queue, args=(shuffle, batch_per_class))
        t.daemon = True
        t.start()
        time.sleep(3)

        def produce():
            #     time.sleep(1)
            batch_data = np.asarray([self.train_queue[i].get() for i in range(self.num_index) for _ in range(min(batch_per_class, self.target_queue[0].qsize()))], dtype=np.float32)
            batch_target = np.asarray([self.target_queue[i].get() for i in range(self.num_index) for _ in range(min(batch_per_class, self.target_queue[0].qsize()))], dtype=np.float32)
            batch_data = self._normalize(batch_data)
            if not self.k_regression:
                batch_target = self._normalize(batch_target)
            return batch_data, batch_target

        return produce


if __name__ == '__main__':
    ip = IP(is_train=False, num_split=1)
    producer = ip.init_producer()
    while 1:
        batch_train, batch_target = producer()
        if len(batch_target)==0:
            break
        print(batch_train.shape, batch_target.shape)