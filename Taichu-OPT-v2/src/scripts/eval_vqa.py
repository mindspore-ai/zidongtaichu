import argparse
import json
import os
import numpy as np
import copy
from collections import defaultdict
from os.path import join
from pathlib2 import Path

import sys
sys.path.append("./")

import src.config.config as C
from src.tools.logger import LOGGER
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.tools.misc import parse_with_config, set_random_seed
from src.data.pretrain_three_data import build_dataloader_ft
from src.vqa.vqa_ms import UniterTwoForVqaForEval
from src.vqa.vqa_data import ImgDataEval, VqaDataset, vqa_collate
from src.data.data_three import ImgData, TxtData

from src.tools.aic_caption.pycxevalcap.eval import COCOEvalCap
from src.tools.aic_caption.pycxtools.coco import COCO

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_group_size, get_rank

bad_endings = []

def get_batch_data_vqaeval(batch):
    """ get_batch_data_vqaeval """

    for key, val in batch.items():
        if isinstance(val, np.ndarray):
            if val.dtype == np.int64:
                batch[key] = val.astype(np.int32)

    for key, value in batch.items():
        if key in ['input_ids', 'position_ids', 'attn_masks_text', 'images']:
            batch[key] = Tensor(value)

    # text input
    input_ids = batch.get('input_ids', None)
    position_ids = batch.get('position_ids', None)
    attn_masks_text = batch.get('attn_masks_text', None)
    # original image
    images = batch.get('images', None)

    taskId = None

    return (input_ids, position_ids, attn_masks_text, images, taskId)

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

def process_gt_file(gt_path, gt_processed_path):
    """
    process_gt_gile
    """
    src = json.load(open(gt_path))['val']
    tgt = {}
    tgt['annotations'] = []
    for data in src:
        js = {'image_id': data['question_id'], 'caption': data['Answer'], 'id': data['question_id']}
        tgt['annotations'].append(js)
    print(len(tgt['annotations']))
    json.dump(tgt, open(gt_processed_path, 'w'))

def process_predict_file(predict_path, predict_processed_path):
    """
    process_predict_gile
    """
    res = []
    res_l = json.load(open(predict_path))
    for ins in res_l:
        tmp = {}
        ques = ins['question_id']
        ans = ins['answer']
        tmp['image_id'] = ques
        tmp['caption'] = ans
        res.append(tmp)
    json.dump(res, open(predict_processed_path,'w'))

def compute_metric(gt_path, predict_path):
    """
    compute_metric
    """
    gt_processed_path = gt_path.split('.json')[-2] + '_processed' + '.json'
    if not os.path.exists(gt_processed_path):
        process_gt_file(gt_path, gt_processed_path)
    predict_processed_path = predict_path.split('.json')[-2] + '_processed' + '.json'
    if not os.path.exists(predict_processed_path):
        process_predict_file(predict_path, predict_processed_path)
    coco = COCO(gt_processed_path, cut=True)
    cocoRes = coco.loadRes(predict_processed_path, cut=True)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    return cocoEval.eval

def str2bool(b):
    if b.lower() in ["false"]:
        return False
    return True

def init_config(opts):

    C.USE_LARGE_DATA = getattr(opts, 'use_large_data', False)

    C.IMG_DIM = getattr(opts, 'img_dim', 768)
    C.IMG_SIZE = opts.image_size
    C.IMG_PATCH_SIZE = opts.patch_size

    C.MAX_TEXT_LEN = opts.text_len - 2
    C.MAX_FULL_TEXT_LEN = opts.text_len
    C.MAX_TEXT_GTS_LEN = opts.text_len - 1

    C.MAX_IMG_LEN = (C.IMG_SIZE // C.IMG_PATCH_SIZE)**2 + 1
    C.MAX_IMG_TEXT_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN
    C.MAX_FULL_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN + C.MAX_AUDIO_LEN

    print(f"USE_LARGE_DATA:{C.USE_LARGE_DATA}")
    print(f"IMG_DIM:{C.IMG_DIM} IMG_SIZE:{C.IMG_SIZE} IMG_PATCH_SIZE:{C.IMG_PATCH_SIZE}")
    print(f"MAX_IMG_LEN:{C.MAX_IMG_LEN} MAX_IMG_TEXT_LEN:{C.MAX_IMG_TEXT_LEN}  MAX_FULL_LEN:{C.MAX_FULL_LEN}")

def pad_last_bacth(pad_bacth, batch):
    size_pad = len(pad_bacth['ids'])
    size = len(batch['ids'])
    if(size < size_pad):
        for key, val in batch.items():
            print(key)
            if(key in ['input_ids','attn_masks_text']):
                batch[key] = np.concatenate([val,pad_bacth[key][0:size_pad-size,:]], axis=0)
            if(key in ['images']):
                batch[key].extend(pad_bacth[key][0:size_pad-size])

        return batch, size
    else:
        return batch, size

def validate_td(model, test_loader, opts, res_path):
    """
     validate_td
    """
    print("start running Text Decoder validation...")

    vocab = json.load(open(opts.vocab_path))
    predictions = []
    split = ''
    total = 0
    pad_bacth = None
    for batch in test_loader:
        if pad_bacth is None:
            pad_bacth = copy.deepcopy(batch)
        batch, seq_sz = pad_last_bacth(pad_bacth, batch)

        ids = batch['ids']
        (input_ids, position_ids, attn_masks_text, images, _) = get_batch_data_vqaeval(batch)
        seq = model(input_ids, position_ids, attn_masks_text, images)

        total += seq_sz
        seq = seq[:seq_sz, 0, 1:]
        print("already_processed: ", total)
        
        seq = seq.asnumpy()
        sents = decode_sequence(vocab, seq, split=split)
        for k, sent in enumerate(sents):
            question_id = ids[k]
            entry = {'question_id': question_id, 'answer': sent}
            print("question_id:{} answer:{}".format(question_id, sent))
            predictions.append(entry)
        
    print(len(predictions))
    json.dump(predictions, open(res_path, "w"))
    print('finish generete caption')

    print("start computing metrics")
    eval_result = compute_metric(opts.gt_path, res_path)
    json.dump(eval_result, open(res_path.replace('.json', '_metric.json'), 'w'))
    print(eval_result)

def init_env(opts):
    """ init_env """
    # get the device and rank info
    if opts.use_parallel:
        device_id = int(os.getenv('DEVICE_ID'))
        LOGGER.info(f'device_id:{device_id}')
        rank_id_str = os.getenv('RANK_ID', '0')
        # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        LOGGER.info(f'rank_id:{rank_id} rank_id str:{rank_id_str}')
    else:
        device_num = 1
        device_id = int(os.getenv('DEVICE_ID'))
        rank = 0
        rank_id = 0
        opts.rank = rank

    # set the log file and output file
    LOGGER.info(f'output_dir: {opts.output_dir}')
    log_dir = os.path.join(opts.output_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    local_rank = rank_id
    LOGGER.info(f'local_rank:{local_rank}, device id:{device_id}')
    profiling_path = os.path.join(opts.output_dir, f'cache/{local_rank}-graphs/')
    if not os.path.exists(profiling_path):
        Path(profiling_path).mkdir(parents=True, exist_ok=True)
    save_graphs_path = os.path.join(profiling_path, "graph")
    if not os.path.exists(save_graphs_path):
        Path(save_graphs_path).mkdir(parents=True, exist_ok=True)

    strategy_ckpt_save_file = save_graphs_path + \
                              "strategy" + str(local_rank) + ".ckpt"

    # set some system info
    set_random_seed(opts.seed)
    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"
    os.system('ulimit -s 102400')
    LOGGER.info(f'local_rank:{local_rank}, device id:{device_id} start to run...')

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(max_device_memory="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    # set parallel mode to data parallel, dp is data parallel ways, which is device num
    if opts.use_parallel:
        init()
        LOGGER.info("start init")

        device_num = get_group_size()
        rank = get_rank()
        opts.rank = rank
        LOGGER.info(f"device_id is {device_id}, rank_id is {rank}, device_num is {device_num}")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num)
    else:
        device_num = 1

    # model parallel = 1
    ParallelConfig.mp = 1
    ParallelConfig.optimizer_shard = False

    # set the dataset and adapt the parallel params
    ParallelConfig.dp = device_num // ParallelConfig.mp

    return local_rank, rank_id, strategy_ckpt_save_file, device_id, device_num

def main(opts):
    init_config(opts)
    
    (local_rank, rank_id, strategy_ckpt_save_file, device_id, device_num) = init_env(opts)

    res_dir = join(opts.output_dir, 'eval')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_name = opts.ckpt_file.split('/')[-1].replace(".ckpt", ".json")
    # res_name = 'rank_' + str(rank_id) + '_' + res_name
    res_path = join(res_dir, res_name)
    print("result file:", res_path)
    if os.path.exists(res_path):
        print('already have results')

        print("start computing metrics")
        eval_result = compute_metric(opts.gt_path, res_path)
        json.dump(eval_result, open(res_path.replace('.json', '_metric.json'), 'w'))
        print(eval_result)

        print("finish compute metrics")

        return

    dset = opts.val_datasets[0]
    txt_db = TxtData(dset['db'][0])
    img_db = ImgDataEval(dset['img'][0], opts)
    dataset = VqaDataset(opts.ids_val_path, txt_db, img_db)
    test_loader = build_dataloader_ft(dataset, vqa_collate, False, opts, device_num)

    net_without_loss = UniterTwoForVqaForEval(opts.model_config, args=opts)

    load_ckpt(net_without_loss, opts.ckpt_file)

    validate_td(net_without_loss, test_loader, opts, res_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='',
                        help='JSON config files')
    parser.add_argument('--cut', default=True, type=str2bool, help='')
    parser.add_argument("--start_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=120000, type=int,
                        help="The decay step.")
    parser.add_argument('--use_txt_out', default=False,
                        type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False,
                        type=str2bool, help='use video')
    parser.add_argument('--use_patch', default=False,
                        type=str2bool, help='use patch')
    parser.add_argument('--path_size', default=32,
                        type=int, help='path size')
    parser.add_argument('--use_parallel', default=True,
                        type=str2bool, help='use parallel')
    parser.add_argument('--data_type', default=2,
                        type=int, help='data type')
    parser.add_argument('--use_data_fix', default=True,
                        type=str2bool, help='use data fix')
    parser.add_argument('--use_mask_fix', default=True,
                        type=str2bool, help='use mask fix')
    parser.add_argument('--name_txt', default="id2len_three.json",
                        type=str, help='txt id2len file')
    parser.add_argument('--name_img', default="img2len_three.json",
                        type=str, help='img img2len file')
    parser.add_argument('--name_audio', default="audio2len_three.json",
                        type=str, help='audio audio2len file')
    parser.add_argument("--init_loss_scale",
                        default=65536, type=float, help="init loss scale")
    parser.add_argument("--loss_scale_factor", default=2,
                        type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000,
                        type=float, help="scale window")
    parser.add_argument("--ckpt_file", default=None,
                        type=str, help="ckpt file path to load")
    parser.add_argument("--vit_ckpt_file", default=None,
                        type=str, help="vit ckpt file path to load")
    parser.add_argument("--save_checkpoint_steps",
                        default=2000, type=int, help="save checkpoint steps")
    parser.add_argument("--epochs", default=100,
                        type=int, help="epochs")
    parser.add_argument('--sink_size', default=0,
                        type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False,
                        type=bool, help="use full batch")
    parser.add_argument("--use_moe", default=False,
                        type=bool, help="use moe")
    parser.add_argument("--is_two", default=False,
                        type=bool, help="two model")
    parser.add_argument("--use_pipeline", default=False,
                        type=bool, help="use pipeline")
    args = parse_with_config(parser)

    main(args)
