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
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import sys
sys.path.append('.')
from src.tools.aic_caption.pycxevalcap.eval import COCOEvalCap
from src.tools.aic_caption.pycxtools.coco import COCO
from src.data.generator import get_batch_data_vqa_eval, get_batch_data
from src.model_mindspore.vqa import UniterThreeForPretrainingForVQAFinetuneEval
from src.data.pretrain_three_data import create_three_dataloaders
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM
from src.tools.logger import LOGGER
from src.tools.misc import parse_with_config
from src.config.config import *

bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']
bad_endings += ['the']

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)

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
        new_params_dict["uniter.img_embeddings.img_linear.weight"] = new_params_dict["feat_regress.weight"]
        new_params_dict["uniter.audio_embeddings.audio_linear.weight"] = new_params_dict["audio_feat_regress.weight"]
        new_params_dict["uniter.embeddings.word_embeddings.embedding_table"] = new_params_dict["cls.predictions.decoder.weight"]
        param_not_load = load_param_into_net(net, new_params_dict)
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
        eval_result = compute_metric(opts.vqa_eval_gt, res_path, opts.cut)
        json.dump(eval_result, open(res_path.replace('.json', '_metric.json'), 'w'))
        print(eval_result)
        return

    device_id = int(os.getenv('DEVICE_ID'))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:]) 
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
    test_loader = test_loaders['vqa']
    net_without_loss = UniterThreeForPretrainingForVQAFinetuneEval(opts.model_config, img_dim=IMG_DIM,
                                                                img_label_dim=IMG_LABEL_DIM,
                                                                audio_dim=AUDIO_DIM, audio_label_dim=AUDIO_LABEL_DIM,
                                                                use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                                full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                                is_parallel=opts.use_parallel, args=opts)

    if opts.ckpt_file == "":
        print("no ckpt_file input")
        return
    load_ckpt(net_without_loss, opts.ckpt_file.strip())
    
    validate_td(net_without_loss, test_loader, opts, res_path)


def validate_td(model, test_loader, opts, res_path):
    """
     validate_td
    """
    LOGGER.info("start running Text Decoder validation...")

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
            txt_masks, img_token_gts, img_token_masks, images, images_mask,
            taskId) = get_batch_data_vqa_eval(batch)
        
        seq = model(input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt,
                img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target,
                audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target, ma_neg_index, ma_neg_sample,
                mr_neg_index, mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks, images, images_mask,
                taskId)

        total += seq.shape[0]
        seq = seq[:, 0, 1:]
        print("already_processed: ", total)

        seq = seq.asnumpy()
        sents = decode_sequence(vocab, seq, split=split)
        for k, sent in enumerate(sents):
            key = ids[k]
            entry = {'question_id': key, 'answer': sent}
            predictions.append(entry)
    print(cap_idx)
    print(len(predictions))

    json.dump(predictions, open(res_path, "w"))

    eval_result = compute_metric(opts.vqa_eval_gt, res_path, opts.cut)
    json.dump(eval_result, open(res_path.replace('.json', '_metric.json'), 'w'))
    print(eval_result)


def process_gt_file(gt_path, gt_processed_path):
    """
    process_gt_file
    """
    src = json.load(open(gt_path))
    src = src["val"]
    tgt = {}
    tgt['annotations'] = []
    for item in src:
        js = {'image_id': item['question_id'], 'caption': item['Answer'], 'id': item['question_id']}
        tgt['annotations'].append(js)
    print(len(tgt['annotations']))
    json.dump(tgt, open(gt_processed_path, 'w'))


def process_predict_file(predict_path, predict_processed_path):
    src = json.load(open(predict_path))
    tgt = []
    for i in src:
        v = {}
        v['image_id'] = i['question_id']
        v['caption'] = ''.join(i['answer'].split(' '))
        if v['caption'] == "":
            v['caption'] =  "\u3002"
        tgt.append(v)
    print(len(tgt))
    json.dump(tgt, open(predict_processed_path, 'w'))


def compute_metric(gt_path, predict_path, cut):
    """
    compute_metric
    """
    gt_processed_path = gt_path.split('.json')[-2] + '_processed' + '.json'
    predict_processed_path = predict_path.split('.json')[-2] + '_processed' + '.json'
    if not os.path.exists(gt_processed_path):
        process_gt_file(gt_path, gt_processed_path)

    if not os.path.exists(predict_processed_path):
        process_predict_file(predict_path, predict_processed_path)

    coco = COCO(gt_processed_path, cut=cut)
    cocoRes = coco.loadRes(predict_processed_path, cut=cut)
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
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='JSON config files')
    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--cut', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of data.')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")

    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")
    parser.add_argument('--ckpt_file', default="", type=str, help='use txt out')
    parser.add_argument('--output_dir', default="", type=str, help='use audio out')
    parser.add_argument("--mode", default="val", type=str)

    args = parse_with_config(parser)

    main(args)
