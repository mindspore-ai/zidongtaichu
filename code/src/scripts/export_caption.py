import os
import numpy as np
import argparse

from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor, export, float32, int64

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from config.config import IMG_DIM, AUDIO_DIM
from src.model_mindspore.caption_ms import UniterThreeForPretrainingForCapFinetuneInf


def load_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print(f"start loading ckpt:{ckpt_file}")
    params_dict = load_checkpoint(ckpt_file)
    if params_dict:
        new_params_dict = {}
        for key in params_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_params_dict[key_new] = params_dict[key]
            new_params_dict[key] = params_dict[key]
        param_not_load = load_param_into_net(net, new_params_dict)
        print("param not load:", param_not_load)
    print(f"end loading ckpt:{ckpt_file}")

def export_network(config, ckpt_file, mindir_file, file_format):
    
    print("load network")
    net = UniterThreeForPretrainingForCapFinetuneInf(config, IMG_DIM, AUDIO_DIM, full_batch=False)
    load_ckpt(net, ckpt_file)

    image = Tensor(np.array(np.random.randn(1, 197, 3072), dtype=np.float32), float32)
    img_pos_feat = Tensor(np.expand_dims(np.arange(0, 197, dtype=np.int64), axis=0), int64)
    attn_masks = Tensor(np.ones((1, 197), dtype=np.int64), int64)
    gather_index = Tensor(np.expand_dims(np.arange(0, 197, dtype=np.int64), axis=0), int64)
    print("mindir file:", mindir_file)
    export(net, image, img_pos_feat, attn_masks, gather_index, file_name=mindir_file, file_format=file_format)
    print("export finished")

def main(ops):
    print("init environment")
    device_id = int(os.getenv('DEVICE_ID') if os.getenv('DEVICE_ID') else 0)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    
    export_network(ops.model_config, ops.ckpt_file, ops.output_file, ops.file_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default="config/caption/cross_modal_encoder_base.json", type=str, help='model config file')
    parser.add_argument('--ckpt_file', required=True, default="", type=str, help='check point file')
    parser.add_argument('--output_file', default="opt_caption", type=str, help='output file')
    parser.add_argument('--file_format', default="MINDIR", type=str, help='output file format')
    
    args = parser.parse_args()
    main(args)