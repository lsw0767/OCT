"""
OCT data input producer
load preprocessed .npy data with threading

Todo:   integrate argparser over preprocess-input_producer-train_env
"""
import numpy as np
import os
import time
import threading
import queue


class IP:
    def __init__(self, path, is_train=True, return_index=False, num_index=5):
        self.path = path
        self.is_train = is_train
        self.return_index = return_index
        self.num_index = num_index
        self.batch_per_class = 0

        self.data_list = []
        for i in range(num_index):
            mode = 'train' if is_train else 'test'
            data_list = os.listdir(os.path.join(path, 'index{}'.format(i+1)))
            data_list.sort()
            data_list = [s for s in data_list if mode in s]
            self.data_list.append(data_list)
        self.index_list = np.load('data/k_index_4_scaled.npy') if return_index else None

        self.data_queue = queue.Queue()

    def _init_queue(self, batch_per_class):
        idx = 0
        while True:
            if self.data_queue.qsize()<100*batch_per_class:
                if self.is_train:
                    idx = np.random.randint(0, len(self.data_list[0]))
                temp = np.asarray([np.load(os.path.join(self.path, 'index{}'.format(i+1), self.data_list[i][idx])) for i in range(self.num_index)])
                shape = temp.shape
                if len(shape)<=1:
                    continue
                total_batch = shape[1]//batch_per_class
                temp = np.reshape(temp[:, :total_batch*batch_per_class], [shape[0], total_batch, batch_per_class, shape[-2], shape[-1]])

                for i in range(total_batch):
                    mini_batch = np.concatenate([temp[j, i] for j in range(shape[0])])
                    self.data_queue.put(mini_batch)

                idx = (idx+1)%len(self.data_list)
            time.sleep(0.1)

    @staticmethod
    def _normalize(arr):
        return (arr - arr.min(axis=1, keepdims=True)) / (
                    arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))

    def init_producer(self, batch_per_class=32):
        self.batch_per_class = batch_per_class
        t = threading.Thread(target=self._init_queue, args=(batch_per_class, ))
        t.daemon = True
        t.start()

    def produce(self):
        temp = self.data_queue.get()
        batch_data = temp[:, :, 0]
        batch_target = temp[:, :, 1]
        if self.return_index:
            batch_index = [self.index_list[i] for i in range(self.num_index) for _ in range(self.batch_per_class)]
            return batch_data, batch_target, batch_index
        else:
            return batch_data, batch_target


if __name__ == '__main__':
    ip = IP(path='/cache/preprocess_npy', is_train=True, return_index=False)
    ip.init_producer(32)
    count = 0
    while 1:
        batch_train, batch_target = ip.produce()
        count+=1
        if count%1000==0:
            print(count)
