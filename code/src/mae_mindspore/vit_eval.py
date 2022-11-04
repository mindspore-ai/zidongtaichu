# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import argparse
import numpy as np

import mindspore as ms
import tqdm
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import init, get_group_size, get_rank

from src.vit import VitEval
from src.logger import get_logger
from src.helper import parse_with_config, str2bool

from mindspore import ops as P
import mindspore.dataset.vision.c_transforms as C
from mindspore.dataset.vision.utils import Inter

from PIL import Image

def trans_image(image_path):

    interpolation = "BILINEAR"
    resize = 224
    image_size = 224

    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]


    if hasattr(Inter, interpolation):
        interpolation = getattr(Inter, interpolation)
    else:
        interpolation = Inter.BILINEAR
        print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))


    trans = [
        # C.Decode(),
        C.Resize(resize, interpolation=interpolation),
        C.CenterCrop(image_size),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    for tran in trans:
        image = tran(image)

    return image

def context_init(args):
    np.random.seed(args.seed)
    set_seed(args.seed)
    rank_id = 0
    device_num = 1

    if args.use_parallel:
        init()
        device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
        rank_id = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num), flush=True)
        args.context["device_id"] = device_id
        context.set_context(**args.context)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            device_num=device_num,
            gradients_mean=True)
    else:
        device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
        args.context["device_id"] = device_id
        context.set_context(**args.context)

    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)
    return rank_id, device_num

def encode(file):

    img_path, npz_path = file

    image = trans_image(img_path)
    image = np.expand_dims(image, 0)

    image = ms.Tensor(image)

    return image, npz_path


def main(args):
    local_rank, device_num = context_init(args)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.ckpt_save_dir, rank=args.local_rank)
    args.logger.info("model config: {}".format(args))

    net = VitEval(batch_size=args.batch_size, patch_size=args.patch_size,
              image_size=args.train_image_size, dropout=args.dropout,
              num_classes=args.num_classes, **args.model_config)

    # load finetune ckpt
    if os.path.exists(args.finetune_ckpt):
        params_dict = load_checkpoint(args.finetune_ckpt)
        net_not_load = net.init_weights(params_dict)
        args.logger.info(f"===============net_not_load================{net_not_load}")


    # sdn = P.StandardNormal(seed=2)
    # image = sdn((1, 3, 224, 224))

    all_files = []

    image_train_path = args.image_path
    save_train_path = args.npz_save_path

    print(image_train_path)
    print(save_train_path)

    for file in os.listdir(image_train_path):
        all_files.append((image_train_path + file,
                          save_train_path + file + ".npz"))

    # image_val_path = "/store0/images/mscoco/val2014/"
    # save_val_path = "/store0/image_feat/mscoco/val2014/"
    # for file in os.listdir(image_val_path):
    #     all_files.append((image_val_path + file,
    #                       save_val_path + file.replace(".jpg", ".npz")))

    # pool = multiprocessing.Pool(8)
    # images = pool.imap(encode, all_files, 25)
    # pbar = tqdm.tqdm(total=len(all_files))
    # for image, npz_path in images:
    #     tokens = net(image).asnumpy()[0][:, 0:]
    #     np.savez(npz_path, feat=tokens)
    #     pbar.update(1)

    for img_path, npz_path in tqdm.tqdm(all_files):

        try:
            image = trans_image(img_path)
        except Exception as e:
            print(e)
            continue

        image = np.expand_dims(image, 0)

        image = ms.Tensor(image)
        tokens = net(image).asnumpy()[0][:, 1:]
        np.savez(npz_path, feat=tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', default="./config/vit-base-p32.yaml",
        help='YAML config files')
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use_parallel.")
    parser.add_argument("--per_step_size", default=2, type=int, help="per_step_size.")
    parser.add_argument("--image_path", default="/store0/images/cc3m/0/training/", type=str, help="per_step_size.")
    parser.add_argument("--npz_save_path", default="/store0/image_feat/cc3m/training/", type=str, help="per_step_size.")
    args_ = parse_with_config(parser)

    main(args_)
