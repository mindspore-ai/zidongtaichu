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

from mindspore.train.callback._callback import Callback
import logging
import time, os
import mindspore
import mindspore.nn as nn
import numpy as np
from PIL import Image
from mindspore.communication import init, get_rank, get_group_size

class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, steps_size):
        super(TimeMonitor, self).__init__()
        self.epoch_time = time.time()
        self.steps_size = steps_size

    def step_begin(self, run_context):
        self.epoch_time = time.time()

    def step_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        cb_params = run_context.original_args()
        # TrainOneStepWithLossScaleCell returns tuple while TrainOneStepCell returns loss directly
        loss = cb_params.net_outputs[0].asnumpy()
        overflow = cb_params.net_outputs[3]
        scale = cb_params.net_outputs[2]

        step_seconds = epoch_seconds / 1000

        # print("per step time: {:5.3f} ms".format(step_seconds), flush=True)
        if overflow:
            logging.warning("Epoch: {}, Step: {}, Step Time: {} sec, Total Loss: {}, Overflow: {}, Scale: {}."
                            .format(cb_params.cur_epoch_num, (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                                    str(step_seconds)[:5], str(loss)[:6], overflow, scale))
        else:
            logging.warning("Epoch: {}, Step: {}, Step Time: {} sec, Total Loss: {}."
                            .format(cb_params.cur_epoch_num, (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                            str(step_seconds)[:5], str(loss)[:6]))

class TimeMonitor_ValidOnlyLoss(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, steps_size, validnet, validset, exp_path, logger):
        super(TimeMonitor_ValidOnlyLoss, self).__init__()
        self.epoch_time = time.time()
        self.steps_size = steps_size
        self.validnet = validnet
        self.validset = validset
        self.exp_path = exp_path
        self.file_logger=logger

    def step_begin(self, run_context):
        self.epoch_time = time.time()

    def step_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        cb_params = run_context.original_args()
        # TrainOneStepWithLossScaleCell returns tuple while TrainOneStepCell returns loss directly
        loss = cb_params.net_outputs[0].asnumpy()
        scale = cb_params.net_outputs[2]

        step_seconds = epoch_seconds / 1000

        # print("per step time: {:5.3f} ms".format(step_seconds), flush=True)
        logging_str = f"Epoch: {cb_params.cur_epoch_num}, Step: {(cb_params.cur_step_num - 1) % cb_params.batch_num + 1}, " \
                      f"Step Time: {str(step_seconds)[:5]} sec, Total Loss: {str(loss)[:6]}, Scale: {scale}."
        logging.warning(logging_str)
        self.file_logger.info(logging_str)

        rec_losses = nn.Loss()
        rec_losses.clear()
        self.validnet.set_train(False)
        for i, vdata in enumerate(self.validset.create_dict_iterator()):
            rec_loss = self.validnet(vdata['image'])
            rec_losses.update(rec_loss)
            if i % 10 == 0:
                logging.info(f"   Valid Iter {i} with  rec loss {rec_losses.eval()}...")
        logging_str = f"Validation epoch-{cb_params.cur_epoch_num} step-{(cb_params.cur_step_num - 1) % cb_params.batch_num + 1}  " \
                      f"finished with avg rec loss {rec_losses.eval()}..."
        logging.info(logging_str)
        self.file_logger.info(logging_str)
        # logging.info(f"====================Finish Validation====================")
        self.validnet.set_train(True)

class TimeMonitorV2(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, steps_size, validnet, validset, exp_path, logger):
        super(TimeMonitorV2, self).__init__()
        self.epoch_time = time.time()
        self.steps_size = steps_size
        self.validnet = validnet
        self.validset = validset
        self.exp_path = exp_path
        self.file_logger=logger

    def step_begin(self, run_context):
        self.epoch_time = time.time()

    def step_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        cb_params = run_context.original_args()
        # TrainOneStepWithLossScaleCell returns tuple while TrainOneStepCell returns loss directly
        loss = cb_params.net_outputs[0].asnumpy()
        scale = cb_params.net_outputs[2]
        step_seconds = epoch_seconds / 1000
        self.file_logger.info("Epoch: {}, Step: {}, Step Time: {} sec, Total Loss: {}, Scale: {}."
                              .format(cb_params.cur_epoch_num, (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                                      str(step_seconds)[:5], str(loss)[:6], scale))

        # Output Validation result
        xs, xrecs = [], []
        rec_losses, vq_losses = nn.Loss(), nn.Loss()
        rec_losses.clear()
        vq_losses.clear()
        self.validnet.set_train(False)
        for i, vdata in enumerate(self.validset.create_dict_iterator()):
            rec_loss, vq_loss, x, x_rec = self.validnet(vdata['image'])
            rec_losses.update(rec_loss)
            vq_losses.update(vq_loss)
            xs.append(x.asnumpy())
            xrecs.append(x_rec.asnumpy())
            if i % 100 == 0:
                print(f"   Valid Iter {i} with  rec loss {rec_loss}, vq loss {vq_loss}...")
            if i == 500:
                break
        self.file_logger.info(
            f"Validation epoch-{cb_params.cur_epoch_num} step-{(cb_params.cur_step_num - 1) % cb_params.batch_num + 1} "
            f"finished with avg rec loss {rec_losses.eval()}, "
            f"avg vq loss {vq_losses.eval()}...")
        show_x = np.concatenate(xs, axis=0)
        n, c, h, w = show_x.shape
        n_w, n_h = 16, min(n // 16, 4)
        show_x = show_x[:n_h*n_w]
        show_x = show_x.reshape((n_h, n_w, c, h, w)).transpose((0, 3, 1, 4, 2))
        show_x = show_x.reshape((n_h * h, n_w * w,c)) * 255.
        show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')

        show_xrec = np.concatenate(xrecs, axis=0)
        show_xrec = np.clip(show_xrec, a_min=0, a_max=1)
        n, c, h, w = show_xrec.shape
        n_w, n_h = 16, min(n // 16, 4)
        show_xrec = show_xrec[:n_h * n_w]
        show_xrec = show_xrec.reshape((n_h, n_w, c, h, w)).transpose((0, 3, 1, 4, 2))
        show_xrec = show_xrec.reshape((n_h * h, n_w * w, c)) * 255.
        show_xrec = Image.fromarray(show_xrec.astype(np.uint8)).convert('RGB')

        try:
            if get_rank() == 0:
                save_name = f'epoch{cb_params.cur_epoch_num}_' \
                            f'step{(cb_params.cur_step_num - 1) % cb_params.batch_num + 1}.jpg'
                show_x.save(os.path.join(self.exp_path, 'gt_img', save_name))
                show_xrec.save(os.path.join(self.exp_path, 'rec_img', save_name))
        except Exception as e:
            self.file_logger.info(f"Fail to save show_x, show_xrec for epoch{cb_params.cur_epoch_num}, "
                                  f"step{(cb_params.cur_step_num - 1)} with exception {e}.")
            pass

        try:
            if get_rank() == 0:
                save_name = f'Model_epoch{cb_params.cur_epoch_num}_step{(cb_params.cur_step_num - 1)}.ckpt'
                mindspore.save_checkpoint(self.validnet._model, os.path.join(self.exp_path, save_name))
                self.file_logger.info(f"Save checkpoint: {save_name} in {self.exp_path}.")
        except Exception as e:
            save_name = f'Model_epoch{cb_params.cur_epoch_num}_step{(cb_params.cur_step_num - 1)}.ckpt'
            self.file_logger.info(f"Fail to save checkpoint {save_name} with exception {e}.")
        pass

        # logging.info(f"====================Finish Validation====================")
        self.validnet.set_train(True)