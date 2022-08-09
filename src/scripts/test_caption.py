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
import argparse
import json
import os
from os.path import join
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from src.data.generator import get_batch_data_captioneval
from src.model_mindspore.caption_ms import UniterThreeForPretrainingForCapFinetuneEval
from src.data.pretrain_three_data import create_three_dataloaders
from src.tools.misc import parse_with_config
from src.tools.aic_caption.pycxevalcap.eval import COCOEvalCap
from src.tools.aic_caption.pycxtools.coco import COCO


bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']
bad_endings += ['the']

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)

def load_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print(f"start loading ckpt:{ckpt_file}")
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        new_param_dict = {}
        for key in param_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_param_dict[key_new] = param_dict[key]
            new_param_dict[key] = param_dict[key]
        param_not_load = load_param_into_net(net, new_param_dict)
        print("param not load:", param_not_load)
    print(f"end loading ckpt:{ckpt_file}")

def main(opts):
    res_dir = join(opts.output_dir, 'eval')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_name = opts.ckpt_file.split('/')[-1].replace(".ckpt", ".json")
    res_path = join(res_dir, res_name)
    print("result file:", res_path)
    if os.path.exists(res_path):
        eval_result = compute_metric(opts.caption_eval_gt, res_path, opts.cut)
        json.dump(eval_result, open(res_path.replace('.json', '_metric.json'), 'w'))
        print(eval_result)
        return

    device_id = int(os.getenv('DEVICE_ID'))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])  # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    local_rank = rank_id
    print('local_rank:{}, device id:{}'.format(local_rank, device_id))
    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    device_num = 1
    rank = 0
    opts.rank = rank

    test_loaders, _ = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, False,
                                            opts, device_num=device_num)
    test_loader = test_loaders['ftCap']

    net_without_loss = UniterThreeForPretrainingForCapFinetuneEval(opts.model_config, img_dim=opts.img_dim,                                                
                                                                   audio_dim=opts.audio_dim,
                                                                   use_txt_out=opts.use_txt_out,
                                                                   use_video=opts.use_video,
                                                                   full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                                   args=opts,
                                                                   beam_width=opts.beam_width)
    load_ckpt(net_without_loss, opts.ckpt_file.strip())

    validate_td(net_without_loss, test_loader, opts, res_path)


def validate_td(model, test_loader, opts, res_path):
    """
     validate_td
    """
    print("start running Text Decoder validation...")

    vocab = json.load(open(opts.vocab_path))
    predictions = []
    split = ''
    cap_idx = 0
    total = 0
    for batch in test_loader:
        ids = batch['ids']
        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
            audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
            txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
            mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
            ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
            txt_masks, img_token_gts, img_token_masks,images, images_mask,
            taskId) = get_batch_data_captioneval(batch)
            
        seq = model(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                    audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                    txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                    mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                    ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                    txt_masks, img_token_gts, img_token_masks,images, images_mask,
                    taskId)
        total += seq.shape[0]
        seq = seq[:, 0, 1:]
        print("already_processed: ", total)
        
        seq = seq.asnumpy()
        sents = decode_sequence(vocab, seq, split=split)
        for k, sent in enumerate(sents):
            image_id = ids[k].split('.jpg')[0][-6:]
            entry = {'image_id': image_id, 'caption': sent}
            print("image_id:{} caption:{}".format(image_id, sent))
            predictions.append(entry)
        
    print(cap_idx)
    print(len(predictions))
    json.dump(predictions, open(res_path, "w"))
    eval_result = compute_metric(opts.caption_eval_gt, res_path, opts.cut)
    json.dump(eval_result,open(res_path.replace('.json','_metric.json'),'w'))
    print(eval_result)


def process_gt_file(gt_path, gt_processed_path):
    """
    process_gt_gile
    """
    src = json.load(open(gt_path))
    tgt = {}
    tgt['annotations'] = []
    for k, v in src.items():
        while len(k) < 6:
            k = '0' + k
        for vs in v:
            js = {'image_id': k, 'caption': vs, 'id': k}
            tgt['annotations'].append(js)
    print(len(tgt['annotations']))
    json.dump(tgt, open(gt_processed_path, 'w'))

def compute_metric(gt_path, predict_path, cut):
    """
    compute_metric
    """
    gt_processed_path = gt_path.split('.json')[-2] + '_processed' + '.json'
    if not os.path.exists(gt_processed_path):
        process_gt_file(gt_path, gt_processed_path)
    coco = COCO(gt_processed_path, cut=cut)
    cocoRes = coco.loadRes(predict_path, cut=cut)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    return cocoEval.eval

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq, split=' '):
    """
    decode_sequence
    """
    N = seq.shape[0]
    D = seq.shape[1]
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + split
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j - 1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words) + flag])
        out.append(txt.replace(' ##', ''))
    return out

def str2bool(b):
    if b.lower() in ["false"]:
        return False
    # elif b.lower() in ["true"]:
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='JSON config files')
    parser.add_argument('--output_dir', default="", type=str, help='use audio out')
    parser.add_argument('--ckpt_file', default="", type=str, help='use txt out')
    parser.add_argument('--cut', default=True, type=str2bool, help='use txt out')
    
    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=1024, type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048, type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True, type=str2bool, help='use txt out')

    parser.add_argument('--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument('--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument('--name_audio', default="audio2len_three.json", type=str, help='use audio out')

    parser.add_argument("--init_loss_scale", default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of data.')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument('--sink_size', default=0, type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")
    parser.add_argument("--audio_preprocess_config", type=str)
    parser.add_argument('--beam_width', default=1, type=int, help='use audio out')
    parser.add_argument('--use_vit', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_patch', default=True, type=str2bool, help='use txt out')
    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
