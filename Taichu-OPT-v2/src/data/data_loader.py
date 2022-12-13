# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

""" dataloader """

import queue
import threading
import traceback

class DataLoader:
    """ DataLoader """

    def __init__(self, dataset, batch_sampler, collate_fn, is_train=True, drop_last=True, device_num=256, full_batch=False, num_workers=16):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collat_fn = collate_fn
        self.is_train = is_train
        self.drop_last = drop_last

        self.batch_indices = iter(self.batch_sampler)

        if self.is_train:
            self.lock = threading.Lock()
            self.num_workers = num_workers
            self.queue = queue.Queue(maxsize=self.num_workers*2)
            for _ in range(self.num_workers):
                self._start_thread()

    def _start_thread(self):
        thread = threading.Thread(target=self.get_batch_worker_thread, args=())
        thread.setDaemon(True)
        thread.start()

    def get_batch_worker_thread(self):
        while True:
            try:
                data = self.get_batch_worker()
                self.queue.put(data)
            except Exception as e:
                traceback.print_exc()

    def get_batch_worker(self):

        self.lock.acquire()
        try:
            indices = next(self.batch_indices)
        except StopIteration:
            self.batch_indices = iter(self.batch_sampler)
            indices = next(self.batch_indices)
            print("Run out of data, start another epoch", flush=True)
        self.lock.release()

        data = [self.dataset[idx] for idx in indices]
        data = self.collat_fn(data)

        return data

    def __iter__(self):
        self.batch_indices = iter(self.batch_sampler)
        return self

    def __next__(self):
        if self.is_train:
            data = self.queue.get()
        else:
            indices = next(self.batch_indices)
            data = [self.dataset[idx] for idx in indices]
            data = self.collat_fn(data)
        return data

