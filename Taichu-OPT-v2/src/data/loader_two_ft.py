# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""loader"""

import time
from collections import defaultdict
import numpy as np
from src.config import config as C
from .data_loader import DataLoader
from .loader_two import task2id_two
from mindspore.communication.management import get_rank

data_column_two_ft = [
    'input_ids',
    'position_ids',
    'attention_mask',
    'txt_mask',
    'txt_label_mask',
    'itm_target',
    'attn_masks_text',
    'attn_masks_img',
    'images',
    'images_rand',
    'input_ids_mask',
    'txt_gts',
    'txt_gts_mask',
    'taskId'
]

class MetaLoaderTwoFt():
    """ wraps multiple data loaders """

    def __init__(self, loaders, datalen, accum_steps=1, task_num=3):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.name2iter_copy = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)

        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.accum_steps = accum_steps
        self.step_cnt = 0
        try:
            print(f"rank {get_rank()} task_num ===> {self.task_num}")
        except Exception as e:
            print(e)
            print(f"rank 0 task_num ===> {self.task_num}")
        self.task_index_list = np.arange(self.task_num)
        self.all_ids = []

    def init_iter(self, task_name):
        self.name2iter[task_name] = iter(self.name2loader[task_name])

    def return_ids(self):
        return self.all_ids

    def get_batch_params(self, batch):
        """ get_batch_params """

        batch = defaultdict(lambda: None, batch)

        input_ids = batch['input_ids']              # 文本
        position_ids = batch['position_ids']        # 位置
        attention_mask = batch['attn_masks']        # 文本token+图片patch的 mask
        txt_mask = batch['txt_mask']                # mlm的label的mask
        txt_label_mask = batch['txt_label_mask']    # mlm的label
        itm_target = batch['targets']               # itm的label
        attn_masks_text = batch['attn_masks_text']  # 文本token的mask
        attn_masks_img = batch['attn_masks_img']    # 图片patch的mask
        images = batch['images']                    # 图片
        images_rand = batch['images_rand']          # itm的采样图片
        input_ids_mask = batch['input_ids_mask']    # mlm的输入mask

        # for finetune task
        txt_gts = batch['txt_gts']                  # ground truth 文本，
        txt_gts_mask = batch['txt_gts_mask']        # ground truth 文本的mask

        return (input_ids, position_ids, attention_mask, txt_mask,
                txt_label_mask, itm_target,
                attn_masks_text, attn_masks_img, images,
                images_rand, input_ids_mask,
                txt_gts, txt_gts_mask)

    def get_batch_check(self, input_ids, position_ids, attention_mask, txt_mask,
                txt_label_mask, itm_target,
                attn_masks_text, attn_masks_img, images,
                images_rand, input_ids_mask,
                txt_gts, txt_gts_mask):
        """ get_batch_check """
        if(images is None):
            raise Exception("image is empty")

        bs = len(images)

        if input_ids is None:
            input_ids = np.zeros((bs, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if position_ids is None:
            position_ids = np.zeros((1, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if attention_mask is None:
            attention_mask = np.zeros((bs, C.MAX_IMG_TEXT_LEN)).astype(np.int32)
        if txt_mask is None:
            txt_mask = np.zeros((bs, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if txt_label_mask is None:
            txt_label_mask = np.zeros((bs, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if itm_target is None:
            itm_target = np.zeros((bs, )).astype(np.int32)

        if attn_masks_text is None:
            attn_masks_text = np.zeros((bs, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if attn_masks_img is None:
            attn_masks_img = np.zeros((bs, C.MAX_IMG_LEN)).astype(np.int32)
        if images_rand is None:
            images_rand = images
        if input_ids_mask is None:
            input_ids_mask = np.zeros((bs, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        
        if txt_gts is None:
            txt_gts = np.zeros((bs, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if txt_gts_mask is None:
            txt_gts_mask = np.zeros((bs, C.MAX_FULL_TEXT_LEN)).astype(np.int32)
        
        return (input_ids, position_ids, attention_mask, txt_mask,
                txt_label_mask, itm_target,
                attn_masks_text, attn_masks_img, images,
                images_rand, input_ids_mask,
                txt_gts, txt_gts_mask)

    def get_batch(self, batch, task):
        """ get_batch """

        params = self.get_batch_params(batch)
        params = self.get_batch_check(*params)
        taskId = np.array([task2id_two[task]]).astype(np.int32)
        output = (*params, taskId)

        return output

    def __getitem__(self, index):

        start_time = time.time()

        task_index = self.task_index_list[self.step_cnt]
        local_task = self.sampling_pools[task_index]

        iter_ = self.name2iter[local_task]

        name = local_task
        is_load = False

        while not is_load:
            try:
                batch = next(iter_)
                is_load = True
            except StopIteration:
                print("============EPOCH END=============", flush=True)
                self.init_iter(local_task)
                print("cost init iter time :", time.time() - start_time, flush=True)
                iter_ = self.name2iter[local_task]
            except Exception as e:
                print(e)

        task = name.split('_')[0]
        # Prepare Data
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        output = self.get_batch(batch, task)

        # if self.print_time:
        #     print("============index: {}, taskname: {}, costtime: {}".format(index, task, time.time() - start_time))
        self.step_cnt = (self.step_cnt + 1) % self.task_num

        return output

    def __len__(self):
        return self.datalen