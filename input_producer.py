"""
OCT data input producer
load preprocessed .npy data with threading

Todo:   integrate argparser over preprocess-input_producer-train_env
"""
import numpy as np
import os
import time

import multiprocessing


class IP:
    def __init__(self, path, is_train=True, return_after=True, target_idx=[]):
        self.path = path
        self.is_train = is_train
        self.return_after = return_after
        self.target_idx = target_idx
        self.batch_per_class = 0
        self.num_to_read = 2

        self.data_list = []
        if is_train:
            before_list = [[os.path.join(
                'index{}'.format(str(i).zfill(2)),
                'sample{}'.format(str(j).zfill(2)),
                'before_{}.npy'.format(str(k+1).zfill(2))
            ) for j in [1, 2, 3, 4, 5, 7, 8, 9] for k in range(self.num_to_read)] for i in self.target_idx]
            after_list = [[os.path.join(
                'index{}'.format(str(i).zfill(2)),
                'sample{}'.format(str(j).zfill(2)),
                'after_{}.npy'.format(str(k+1).zfill(2))
            ) for j in [1, 2, 3, 4, 5, 7, 8, 9] for k in range(self.num_to_read)] for i in self.target_idx]
        else:
            before_list = [[os.path.join(
                'index{}'.format(str(i).zfill(2)),
                'sample{}'.format(str(j).zfill(2)),
                'before_{}.npy'.format(str(k+1).zfill(2))
            ) for j in [6] for k in range(self.num_to_read)] for i in self.target_idx]
            after_list = [[os.path.join(
                'index{}'.format(str(i).zfill(2)),
                'sample{}'.format(str(j).zfill(2)),
                'after_{}.npy'.format(str(k+1).zfill(2))
            ) for j in [6] for k in range(self.num_to_read)] for i in self.target_idx]
        self.before_list = np.asarray(before_list)
        self.after_list = np.asarray(after_list)

        self.index_list = np.load('data/k_index_scaled.npy')[np.asarray(self.target_idx)-1]
        self.data_queue = multiprocessing.Queue()

    def _init_queue(self, batch_per_class, queue):
        shape = self.index_list.shape
        index_list = np.tile(self.index_list, [1, batch_per_class]).reshape([shape[0]*batch_per_class, shape[1]])
        while True:
            if queue.qsize()<100*batch_per_class:
                idx = np.random.randint(0, len(self.before_list[0]))
                before = np.asarray([np.load(os.path.join(self.path, self.before_list[i][idx])) for i in range(len(self.target_idx))])
                shape = before.shape
                total_batch = shape[1]//batch_per_class
                before = np.reshape(before[:, :total_batch*batch_per_class], [shape[0], total_batch, batch_per_class, shape[-1]])
                before = before.transpose([1, 0, 2, 3]).reshape([total_batch, -1, shape[-1]])

                if self.return_after:
                    after = np.asarray([np.load(os.path.join(self.path, self.after_list[i][idx])) for i in range(len(self.target_idx))])
                    after = np.reshape(after[:, :total_batch*batch_per_class], [shape[0], total_batch, batch_per_class, shape[-1]])
                    after = after.transpose([1, 0, 2, 3]).reshape([total_batch, -1, shape[-1]])

                for i in range(total_batch):
                    before_batch = self._normalize(before[i])
                    after_batch = self._normalize(after[i])
                    if self.return_after:
                        queue.put([before_batch, after_batch, index_list])
                    else:
                        queue.put([before_batch, after_batch])

            time.sleep(0.1)

    @staticmethod
    def _normalize(arr):
        return (arr - arr.min(axis=1, keepdims=True)) / (
                    arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))

    def init_producer(self, batch_per_class=32):
        self.batch_per_class = batch_per_class
        p = multiprocessing.Process(target=self._init_queue, args=(batch_per_class, self.data_queue), daemon=True)
        p.start()

    def produce(self):
        return self.data_queue.get()


if __name__ == '__main__':
    ip = IP(path='/cache/preprocess', is_train=True, target_idx=[7, 8, 9])
    ip.init_producer(10)
    count = 0
    while 1:
        batch_train, batch_target, batch_index = ip.produce()
        count+=1
        if count%1000==0:
            print(batch_train.shape)
            print(batch_target.shape)
            print(batch_index.shape)
