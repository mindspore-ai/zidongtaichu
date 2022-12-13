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
import sys
import json
import yaml
import time
import argparse
import numpy as np
from PIL import Image
from os.path import join
from pytorch_pretrained_bert import BertTokenizer

import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

sys.path.append('./')
from src.tools.logger import LOGGER
from src.tools.misc import parse_with_config
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM
from src.data.generator import get_batch_data_t2i_eval
from src.vqvae_mindspore.src.utils.get_model import get_model
from src.data.pretrain_three_data import create_three_dataloaders
from src.model_mindspore.t2i import UniterThreeForPretrainingForT2IfinetuneInf

bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']
bad_endings += ['the']

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val


def get_text_ids(tokenizer, txt):
    ws = tokenizer.tokenize(txt)
    txt_tokens = tokenizer.convert_tokens_to_ids(ws)
    return txt_tokens

def main(opts):
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    local_rank = rank_id
    print('local_rank:{}, device id:{}'.format(local_rank, device_id))

    context.set_context(mode=context.PYNATIVE_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(max_call_depth=1000000)
    context.set_context(reserve_class_name_in_scope=False)

    # load datasets
    device_num = 1
    rank = 0
    opts.rank = rank
    opts.val_batch_size = 1
    test_loader, _ = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, False, opts, device_num)

    # load vqvae
    assert os.path.exists(args.vae_config), f"{args.vae_config} must exists!"
    with open(args.vae_config, 'r') as f:
        vae_opt = yaml.load(f)
    vae_net = get_model(vae_opt['model'])
    vae_param_dict = load_checkpoint(args.vae_ckpt)
    load_param_into_net(vae_net, vae_param_dict)
    vae_net.set_train(mode=False)
    print(f"=====Successfully load vqvae ckpt: {args.vae_ckpt} from cfg {args.vae_config}.")

    # load opt model
    net_without_loss = UniterThreeForPretrainingForT2IfinetuneInf(opts.model_config, img_dim=opts.img_dim,
                                                                   img_label_dim=IMG_LABEL_DIM,
                                                                   audio_dim=opts.audio_dim,
                                                                   audio_label_dim=AUDIO_LABEL_DIM,
                                                                   use_txt_out=opts.use_txt_out,
                                                                   use_video=opts.use_video,
                                                                   full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                                   args=opts)
    net_without_loss.set_train(mode=False)
    ckpt_file = opts.ckpt_file
    print("=====loading ckpt file: ", ckpt_file)
    assert ckpt_file != ""
    if ckpt_file == "":
        modified_params_dict = None
    else:
        params_dict = load_checkpoint(ckpt_file)

        modified_params_dict = {}
        for k, v in params_dict.items():
            if 't2i_output.tfm_decoder' in k:
                modified_k = k.replace('t2i_output.tfm_decoder', 't2i_output.tfm_decoder.decoder.tfm_decoder')
                v.name = v.name.replace('t2i_output.tfm_decoder', 't2i_output.tfm_decoder.decoder.tfm_decoder')
                modified_v = v
                modified_params_dict[modified_k] = modified_v
            else:
                modified_params_dict[k] = v
    if modified_params_dict:
        net_not_load = load_param_into_net(net_without_loss, modified_params_dict)
        print("===============net_not_load================", net_not_load)

    if args.inf_txt_file is None:
        validate_id_fromloader(net_without_loss, vae_net, test_loader, opts, ckpt_file=ckpt_file)
    else:
        validate_id_interactive(model=net_without_loss, vae_model=vae_net, txt_file=args.inf_txt_file, opts=opts, ckpt_file=ckpt_file)


def validate_id_interactive(model, vae_model, txt_file, opts, ckpt_file):
    """
         validate_id
    """
    LOGGER.info("start running T2I Inference through INTERACTIVE...")

    # generate_path = os.path.join(opts.output_dir, 'gen_samples')
    generate_path = opts.output_dir
    save_path = join(generate_path, f'generation_by_{os.path.basename(ckpt_file)[:-5]}')
    # assert os.path.exists(save_path) is False, f"Current CKPT has generated!"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'tokens'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)

    # load txt tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        './datasets/txt2img/mscoco/cocodata_zh/bert-base-chinese-vocab.txt', do_lower_case=False)

    # captions
    id2cap = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            id, cap = line.split('##')
            cap = cap.strip()
            id2cap.append((id, cap))

    # generate image for each cap
    for (cid, curcap) in id2cap:
        save_name = "{}_{}_{}".format(cid, curcap, time.strftime("%Y%m%d%H%M", time.localtime()))
        print(f"generation for {cid} is saved in {save_path} as {save_name} with caption: {curcap}.")

        stime = time.time()
        input_ids = np.array(get_text_ids(tokenizer, curcap), dtype=np.int32)[:50]
        attn_mask = np.ones(input_ids.shape[0], dtype=np.int32)
        input_ids = np.pad(input_ids, (0, 50-input_ids.shape[0]), constant_values=(0, 0)).reshape(1, 50) # pad
        position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)
        attn_mask = np.pad(attn_mask, (0, 50 - attn_mask.shape[0]), constant_values=(0, 0)).reshape(1, 50)  # pad
        out_size, batch_size = attn_mask.shape[1], attn_mask.shape[0]
        gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

        print("===input_ids: ", input_ids)
        print("===attn_mask: ", attn_mask)

        input_ids, position_ids = Tensor(input_ids), Tensor(position_ids)
        attn_mask, gather_index = Tensor(attn_mask), Tensor(gather_index)

        seq = model(input_ids, position_ids, None, None, None,
                    None, attn_mask, gather_index, None, None,
                    None, None, None, None, None,
                    None, None, None, None, None,
                    None, None, None, None, None,
                    None, None, None, None, None, None)
        seq = seq.asnumpy()

        print(seq.shape, seq)
        print("============Total time: ", time.time() - stime)
        np.save(os.path.join(save_path, 'tokens', save_name + '.npy'), np.array(seq) - 1)

        try:
            seq = seq[0, :opts.image_token_len] - 1
            s = int(np.sqrt(opts.image_token_len))
            ctokens = seq.reshape((1, s, s))
            xrec = vae_model.get_xrec_from_codes(mindspore.Tensor(ctokens))
            xrec = mindspore.ops.clip_by_value(xrec, clip_value_min=Tensor(0),
                                               clip_value_max=Tensor(1))
            xrec = xrec.asnumpy()[0]
            show_x = np.transpose(xrec, (1, 2, 0)) * 255.
            show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
            show_x.save(os.path.join(save_path, 'images', save_name + '.png'))
        except:
            print(f"Fail to save png file: {os.path.join(save_path, 'images', save_name + '.png')}.")


def validate_id_fromloader(model, vae_model, test_loader, opts, ckpt_file):
    """
     validate_id
    """
    LOGGER.info("start running T2I Inference through TESTSET...")

    # generate_path = os.path.join(opts.output_dir, 'gen_samples')
    generate_path = opts.output_dir
    save_path = join(generate_path, f'generation_by_{os.path.basename(ckpt_file)[:-5]}')
    # assert os.path.exists(save_path) is False, f"Current CKPT has generated!"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'tokens'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)

    # captions
    id2cap = json.load(open('/store0/dataset/coco_data/coco_trans_captions.json', 'r'))

    for batch in test_loader:
        ids = batch['ids']
        assert len(ids) == 1 # batch size == 1
        save_name = ids[0].split('/')[-1].split('.')[0]
        # curcap = np.random.choice(id2cap[str(int(save_name.split('_')[-1]))])
        curcap = id2cap[str(int(save_name.split('_')[-1]))][0]
        for c in ['/', '\\', ' ']:
            curcap.replace(c, ',')
        print(f"generation for {ids} is saved in {save_path} as {save_name}, random caption: {curcap}.")

        stime = time.time()
        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
         audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks,
         taskId) = get_batch_data_t2i_eval(batch)

        seq = model(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                    audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                    txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                    mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                    txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                    txt_masks, img_token_gts, img_token_masks,
                    taskId)
        seq = seq.asnumpy()

        print(seq.shape, seq)
        print("============Total time: ", time.time() - stime)
        np.save(os.path.join(save_path, 'tokens', save_name+f'_{curcap}.npy'), np.array(seq) - 1)

        try:
            seq = seq[0, :opts.image_token_len] - 1
            s=int(np.sqrt(opts.image_token_len))
            ctokens = seq.reshape((1, s, s))
            xrec = vae_model.get_xrec_from_codes(mindspore.Tensor(ctokens))
            xrec = mindspore.ops.clip_by_value(xrec, clip_value_min=Tensor(0),
                                     clip_value_max=Tensor(1))
            xrec = xrec.asnumpy()[0]
            show_x = np.transpose(xrec, (1, 2, 0)) * 255.
            show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
            show_x.save(os.path.join(save_path, 'images', save_name+f'_{curcap}.png'))
        except:
            print(f"Fail to save png file: {os.path.join(save_path, 'images', save_name+f'_{curcap}.png')}.")


def process_gt_gile(gt_path, gt_processed_path):
    """
    process_gt_gile
    """
    src = json.load(open(gt_path))
    tgt = {}
    tgt['annotations'] = []
    for k, v in src.items():
        while len(k) < 6:
            #

            k = '0' + k
        for vs in v:
            js = {'image_id': k, 'caption': vs, 'id': k}
            tgt['annotations'].append(js)
    print(len(tgt['annotations']))
    json.dump(tgt, open(gt_processed_path, 'w'))


def process_predict_gile(predict_path, predict_processed_path):
    src = json.load(open(predict_path))
    tgt = []
    for i in src:
        v = {}
        v['image_id'] = i['image_id']
        v['caption'] = ''.join(i['caption'].split(' '))
        tgt.append(v)
    print(len(tgt))
    json.dump(tgt, open(predict_processed_path, 'w'))



def str2bool(b):
    if b.lower() in ["false"]:
        return False
    # elif b.lower() in ["true"]:
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default="/home/work/user-job-dir/uniter-three/config/" +
                        "pretrain_three_modal_txt_img_audio_config.json",
                        help='JSON config files')
    parser.add_argument("--start_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=120000, type=int,
                        help="The decay step.")
    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=512, type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048, type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True, type=str2bool, help='use txt out')

    parser.add_argument('--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument('--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument('--name_audio', default="audio2len_three.json", type=str, help='use audio out')
    parser.add_argument('--output_dir', default="", type=str, help='use audio out')

    parser.add_argument("--init_loss_scale", default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument("--load_ckpt", default=True, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps", default=5000, type=int, help="")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--ckpt_file', default="", type=str, help='use txt out')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument('--sink_size', default=2, type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")
    parser.add_argument('--use_vit', default=False, type=str2bool, help='use txt out')

    # args for vqvae
    parser.add_argument('--vae_config', default=False, type=str)
    parser.add_argument('--vae_ckpt', default=False, type=str)

    # reference file
    parser.add_argument('--inf_txt_file', default=None, type=str)

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
