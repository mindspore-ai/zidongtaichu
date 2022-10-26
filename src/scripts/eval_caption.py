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
import time
import argparse
import json
import time
import numpy as np
from pathlib2 import Path
import mindspore as ms
from mindspore import context, Model, Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import load_checkpoint, load_param_into_net

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from src.config.config import *
from src.data.generator import get_batch_data_captioneval
from src.data.utils import pad_sequence
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.caption_ms import UniterThreeForPretrainingForCapFinetuneEval
from src.data.pretrain_three_data import create_three_dataloaders
from src.tools.misc import parse_with_config, set_random_seed
from src.tools.aic_caption.pycxevalcap.eval import COCOEvalCap
from src.tools.aic_caption.pycxtools.coco import COCO

def init_env(opts):
    """ init_env """
    if opts.use_parallel:
        device_id = int(os.getenv('DEVICE_ID'))
        print('device_id:{}'.format(device_id))
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    else:
        device_num = 1
        device_id = int(os.getenv('DEVICE_ID'))
        rank = 0
        rank_id = 0
        opts.rank = rank
    local_rank = rank_id
    print('local_rank:{}, device id:{}'.format(local_rank, device_id))
    profiling_path = os.path.join(opts.output_dir, f'cache/{local_rank}-graphs/')
    if not os.path.exists(profiling_path):
        Path(profiling_path).mkdir(parents=True, exist_ok=True)
    time.sleep(1)
    save_graphs_path = os.path.join(profiling_path, "graph")
    if not os.path.exists(save_graphs_path):
        Path(save_graphs_path).mkdir(parents=True, exist_ok=True)

    strategy_ckpt_save_file = save_graphs_path + \
                              "strategy" + str(local_rank) + ".ckpt"

    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"
    os.system('ulimit -s 102400')
    set_random_seed(opts.seed)

    print('local_rank:{}, device id:{} start to run...'.format(
        local_rank, device_id), flush=True)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(max_device_memory="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    if opts.use_parallel:
        init()
        print("start init")

        device_num = get_group_size()
        rank = get_rank()
        opts.rank = rank
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        
        parallel_optimizer_config = {"gradient_accumulation_shard": True}
        optimizer_weight_shard_size = opts.optimizer_shard_size
        print("===========optimizer_weight_shard_size============", optimizer_weight_shard_size)
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=opts.full_batch,
            enable_alltoall=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=opts.enable_parallel_optimizer,
            strategy_ckpt_save_file=strategy_ckpt_save_file,
            parallel_optimizer_config=parallel_optimizer_config,
            optimizer_weight_shard_size=optimizer_weight_shard_size
        )
        
        ParallelConfig.mp = opts.tensor_shard_size
        ParallelConfig.dp = device_num // ParallelConfig.mp
        ParallelConfig.optimizer_shard = opts.enable_parallel_optimizer
        if opts.full_batch:
            opts.val_batch_size = int(opts.val_batch_size * ParallelConfig.dp)
        print((f"=====device_num:{device_num} dp:{ParallelConfig.dp} mp:{ParallelConfig.mp} "
               f"enable_optimizer_shard:{ParallelConfig.optimizer_shard} op:{opts.optimizer_shard_size} " 
               f"val_batch_size:{opts.val_batch_size}"))
    else:
        device_num = 1
        print(f"val_batch_size:{opts.val_batch_size}")

    # init dataset
    test_loader, _ = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, False,
                                            opts, device_num=device_num)
    dataset = test_loader['ftCap']

    return rank_id, dataset

def load_pretrain_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print('start loading pretrain ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        new_param_dict = {}
        for key in param_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_param_dict[key_new] = param_dict[key]
            else:
                new_param_dict[key] = param_dict[key]
        param_dict = new_param_dict
        param_dict["uniter.img_embeddings.img_linear.weight"] = param_dict["feat_regress.weight"]
        param_dict["uniter.audio_embeddings.audio_linear.weight"] = param_dict["audio_feat_regress.weight"]
        param_dict["uniter.embeddings.word_embeddings.embedding_table"] = param_dict["cls.predictions.decoder.weight"]
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load pretrain ckpt finished:', ckpt_file)

def load_finetune_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print('start loading finetune ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        new_param_dict = {}
        for key in param_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_param_dict[key_new] = param_dict[key]
            new_param_dict[key] = param_dict[key]
        param_dict = new_param_dict
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load finetune ckpt finished:', ckpt_file)

def load_distributed_pretrain_ckpt(net, rank_id, ckpt_size, ckpt_path, ckpt_name):
    if not ckpt_path or not ckpt_name:
        return
    ckpt_id = rank_id % ckpt_size
    folder = "rank_" + str(ckpt_id)
    ckpt_file = os.path.join(ckpt_path, folder, ckpt_name)
    if not os.path.exists(ckpt_file):
        print(f"ckpt_file:{ckpt_file} does not exists")
        return
    print('start loading distributed pretrain ckpt:', ckpt_file)  
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        new_param_dict = {}
        for key in param_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_param_dict[key_new] = param_dict[key]
            else:
                new_param_dict[key] = param_dict[key]
        param_dict = new_param_dict
        param_dict["uniter.img_embeddings.img_linear.weight"] = param_dict["feat_regress.weight"]
        param_dict["uniter.audio_embeddings.audio_linear.weight"] = param_dict["audio_feat_regress.weight"]
        param_dict["uniter.embeddings.word_embeddings.embedding_table"] = param_dict["cls.predictions.decoder.weight"]
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load distributed pretrain ckpt finished:', ckpt_file)

def load_distributed_finetune_ckpt(net, rank_id, ckpt_size, ckpt_path, ckpt_name):
    if not ckpt_path or not ckpt_name:
        return
    ckpt_id = rank_id % ckpt_size
    folder = "rank_" + str(ckpt_id)
    ckpt_file = os.path.join(ckpt_path, folder, ckpt_name)
    if not os.path.exists(ckpt_file):
        print(f"ckpt_file:{ckpt_file} does not exists")
        return
    print('start loading distributed finetune ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        new_param_dict = {}
        for key in param_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_param_dict[key_new] = param_dict[key]
            else:
                new_param_dict[key] = param_dict[key]
        param_dict = new_param_dict
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load distributed finetune ckpt finished:', ckpt_file)

def create_fake_input(batch_size):
    img_feat = Tensor(np.random.rand(batch_size, MAX_IMG_LEN, PATCH_SIZE ** 2 * 3), ms.float32)
    img_pos_feat = Tensor(np.expand_dims(np.arange(0, MAX_IMG_LEN, dtype=np.int32), axis=0), ms.int32)
    attention_mask = Tensor(pad_sequence(np.ones((batch_size, MAX_IMG_LEN), dtype=np.int32), 
                                         batch_first=True, padding_value=0, max_lens=MAX_IMG_LEN), ms.int32)
    gather_index = Tensor(np.repeat(np.expand_dims(np.arange(0, MAX_IMG_LEN, dtype=np.int32), axis=0), 
                                    batch_size, axis=0), ms.int32)
    return (img_feat, img_pos_feat, attention_mask, gather_index)

def main(opts):
    res_dir = os.path.join(opts.output_dir, 'eval')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if opts.pretrain_ckpt_file:
        ckpt_name = opts.pretrain_ckpt_file
    if opts.finetune_ckpt_file:
        ckpt_name = opts.finetune_ckpt_file
    if opts.pretrain_ckpt_name:
        ckpt_name = opts.pretrain_ckpt_name
    if opts.finetune_ckpt_name:
        ckpt_name = opts.finetune_ckpt_name
    res_name = os.path.splitext(os.path.split(ckpt_name)[-1])[0] + ".json"
    res_path = os.path.join(res_dir, res_name)
    print("result file:", res_path)
    if os.path.exists(res_path):
        eval_result = compute_metric(opts.caption_eval_gt, res_path, opts.cut)
        json.dump(eval_result, open(res_path.replace('.json', '_metric.json'), 'w'))
        print(eval_result)
        return
 
    (rank_id, dataset) = init_env(opts)
    
    print("start initializing model...")
    net_without_loss = UniterThreeForPretrainingForCapFinetuneEval(opts.model_config, 
                                                                   img_dim=IMG_DIM, audio_dim=AUDIO_DIM,
                                                                   use_txt_out=opts.use_txt_out, use_video=opts.use_video, 
                                                                   full_batch=opts.full_batch, use_moe=opts.use_moe, args=opts)
    
    model = Model(net_without_loss)
    
    # load ckpt
    load_pretrain_ckpt(net_without_loss, opts.pretrain_ckpt_file)
    load_finetune_ckpt(net_without_loss, opts.finetune_ckpt_file)
    print("start building model...")
    (img_feat, img_pos_feat, attention_mask, gather_index) = create_fake_input(opts.val_batch_size)
    model.predict(img_feat, img_pos_feat, attention_mask, gather_index)
    load_distributed_pretrain_ckpt(net_without_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.pretrain_ckpt_name)
    load_distributed_finetune_ckpt(net_without_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.finetune_ckpt_name)

    print("start validating...")
    validate_td(model, dataset, opts, res_path)

def validate_td(model, val_ds, opts, res_path):
    """
     validate_td
    """
    print("start running Text Decoder validation...")
    
    vocab = json.load(open(opts.vocab_path))
    predictions = []
    split = ''
    total = 0
    time_start = time.time()
    for batch in val_ds:
        ids = batch['ids']
        (_, _, img_feat, img_pos_feat, _, _, attention_mask, gather_index, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _, _, _,_, _, _) = get_batch_data_captioneval(batch)
        time_batch_start = time.time()
        seq = model.predict(img_feat, img_pos_feat, attention_mask, gather_index)
        
        batch = seq.shape[0]
        total += batch
        seq = seq[:, 0, 1:]
        seq = seq.asnumpy()
        sents = decode_sequence(vocab, seq, split=split)
        for k, sent in enumerate(sents):
            image_id = ids[k].split('.jpg')[0][-6:]
            entry = {'image_id': image_id, 'caption': sent}
            print("image_id:{} caption:{}".format(image_id, sent))
            predictions.append(entry)
        
        print("alreadyprocessed: ", total)
        print("batch time: ", time.time() - time_batch_start)
    
    time_end = time.time()
    total_time = time_end - time_start
    print(total)
    
    print(f"total time cost {total_time}, per batch time {total_time / total * batch}")
    
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
    bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']
    bad_endings += ['the']
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
    project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "../..")
    print('project_root:', project_root)
    print('process id:', os.getpid())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='JSON config files')
    parser.add_argument('--output_dir', default="", type=str, help='output directory')
    
    parser.add_argument('--cut', default=True, type=str2bool)
    parser.add_argument('--beam_width', default=1, type=int)

    parser.add_argument('--use_parallel', default=True,
                        type=str2bool, help='use parallel')
    parser.add_argument('--use_txt_out', default=False,
                        type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False,
                        type=str2bool, help='use video')
    parser.add_argument("--use_moe", default=False,
                        type=bool, help="use moe")
    
    parser.add_argument("--pretrain_ckpt_file", default=None,
                        type=str, help="pretrain ckpt file path")
    parser.add_argument("--finetune_ckpt_file", default=None,
                        type=str, help="finetune ckpt file path")
    parser.add_argument("--ckpt_size", default=32,
                        type=int, help="distribute ckpt nums")
    parser.add_argument("--ckpt_path", default=None,
                        type=str, help="distribute ckpt path")
    parser.add_argument("--pretrain_ckpt_name", default=None,
                        type=str, help="distribute pretrain ckpt name")
    parser.add_argument("--finetune_ckpt_name", default=None,
                        type=str, help="distribute finetune ckpt name")
    
    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    print(args)
    main(args)
