"""
OCT data input producer
load preprocessed .npy data with threading

Todo:   integrate argparser over preprocess-input_producer-train_env
        memory usage should be tested on training situation
"""
import numpy as np
import os
import time
import threading
import queue


class IP:
    def __init__(self, path, is_train=True, return_index=False, num_index=3):
        self.path = path
        self.is_train = is_train
        self.return_index = return_index
        self.num_indx = num_index

        mode = 'train' if is_train else 'test'
        data_list = os.listdir(path)
        data_list.sort()
        self.data_list = [s for s in data_list if mode in s]
        self.index_list = np.load('data/k_index_4_scaled.npy') if return_index else None

        self.data_queue = queue.Queue()

    def _init_queue(self, batch_per_class):
        idx = 0
        while True:
            if self.data_queue.qsize()<1000*batch_per_class:
                # print('reload')
                if self.is_train:
                    idx = np.random.randint(0, len(self.data_list))
                temp = np.load(os.path.join(self.path, self.data_list[idx]))
                shape = temp.shape
                total_batch = shape[0]//batch_per_class
                temp = np.reshape(temp[:total_batch*batch_per_class], [total_batch, batch_per_class, shape[-2], shape[-1]])

                for batch in temp:
                    self.data_queue.put(batch)

                idx = (idx+1)%len(self.data_list)
            time.sleep(0.1)

    @staticmethod
    def _normalize(arr):
        return (arr - arr.min(axis=1, keepdims=True)) / (
                    arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))

    def init_producer(self, batch_per_class=32):
        t = threading.Thread(target=self._init_queue, args=(batch_per_class, ))
        t.daemon = True
        t.start()

    def produce(self):
        temp = self.data_queue.get()
        batch_data = temp[:, :, 0]
        batch_target = temp[:, :, 1]

        return batch_data, batch_target


if __name__ == '__main__':
    ip = IP(path='/cache/preprocess_npy/3index', is_train=True)
    ip.init_producer(100)
    producer = ip.produce
    count = 0
    while 1:
        batch_train, batch_target = producer()
        count+=1
        if count%1000==0:
            print(count)

