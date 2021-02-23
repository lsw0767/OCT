"""
OCT data preprocessor
generate npy files that contain a bundle of signals

"""

import numpy as np
import os
import tqdm


def normalize(arr):
    return (arr - arr.min(axis=1, keepdims=True)) / (
                arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))


def postprocessing(arr):
    np.log10(abs(np.fft.fft(arr))+1e-5, out=arr)
    arr_len = int(arr.shape[1]/2)
    return arr[:, :arr_len]


def shift(arr, len=4):
    shape = arr.shape
    arr = np.reshape(arr, [-1])
    arr[:-len] = arr[len:]
    arr = np.reshape(arr, [shape[0], shape[1]])
    return arr


class Preprocessor:
    def __init__(self, data_path, save_path, target_index, num_sigs, max_sigs=100, ops=None):
        self.data_path = data_path
        self.save_path = save_path
        self.num_sigs = num_sigs
        self.max_sigs = max_sigs
        self.ops = ops
        self.target_index = target_index

        self.read_count = 0
        self.before_buf = np.zeros([num_sigs*1000, 2048], dtype=np.float16)
        self.after_buf = np.zeros([num_sigs*1000, 2048], dtype=np.float16)

    def run(self, mode):
        for i in self.target_index:
            target = 'K-index_no.{}'.format(str(i).zfill(2))
            sample_list = os.listdir(os.path.join(self.data_path, target))
            sample_list.sort()

            new_list = []
            for sample in sample_list:
                if not ('txt' in sample or 'csv' in sample or '10' in sample):
                    new_list.append(sample)

            for s, sample in enumerate(new_list):
                before_list = self._get_target_files(
                    os.path.join(self.data_path, target, sample, 'Before_K-linearization2'), mode
                )

                after_list = self._get_target_files(
                    os.path.join(self.data_path, target, sample, 'After_K-linearization'), mode
                )

                full_path = os.path.join(
                    self.save_path, 'index{}'.format(str(i).zfill(2)), 'sample{}'.format(str(s + 1).zfill(2))
                )
                print(full_path)
                self.read_count = 0
                for j in tqdm.tqdm(range(len(before_list))):
                    name = os.path.join(self.data_path, target, sample, 'Before_K-linearization2', before_list[j])
                    success = self._read_and_merge(name, mode)
                    if not success:
                        continue

                    name = os.path.join(self.data_path, target, sample, 'After_K-linearization', after_list[j])
                    success = self._read_and_merge(name, mode)
                    if not success:
                        continue

                    self.read_count += 1
                    if self.read_count%self.num_sigs==0:
                        # bundle = np.asarray([self.before_buf, self.after_buf])
                        self._mkdir(full_path)
                        np.save(os.path.join(full_path, 'before_'+str((j+1)//self.num_sigs).zfill(2)), self.before_buf)
                        np.save(os.path.join(full_path, 'after_'+str((j+1)//self.num_sigs).zfill(2)), self.after_buf)

                    if self.read_count==self.max_sigs:
                        break

    def _get_target_files(self, path, mode):
        temp = os.listdir(path)
        temp.sort()

        final_list = []
        for item in temp:
            if mode in item:
                final_list.append(item)
        return final_list

    def _read_and_merge(self, path, mode):
        # try:
        if mode is 'txt':
            temp = open(path).readlines()
            temp = [line.split() for line in temp]
            temp = np.asarray(temp, dtype=np.float32)
            np.transpose(temp, [1, 0], out=temp)
        elif mode is 'bin':
            # temp = open(path, 'rb')
            temp = np.fromfile(path, np.double)
            temp = np.reshape(temp, [1000, 2048])
            temp.astype(np.float16)
        else:
            raise Exception('Not supported type')
        # except:
        #     return False

        for ops in self.ops:
            temp = ops(temp)

        if 'Before' in path:
            self.before_buf[self.read_count%self.num_sigs*1000:(self.read_count%self.num_sigs+1)*1000] = temp
        else:
            self.after_buf[self.read_count%self.num_sigs*1000:(self.read_count%self.num_sigs+1)*1000] = temp
        return True

    def _mkdir(self, path):
        names = os.path.abspath(path).split('/')
        full_path = '/'

        for name in names:
            try:
                full_path = os.path.join(full_path, name)
                os.mkdir(full_path)
            except:
                pass


if __name__ == '__main__':
    # Preprocessor('data/OCT_K_linear', 'data/preprocess_fft', [1, 2, 3, 4, 5], 50, ops=[shift]).run('txt')
    # with Before~

    Preprocessor('data/OCT_K_linear', '/cache/preprocess', [6, 10, 11], 50, ops=[]).run('bin')
    # with Before~2
