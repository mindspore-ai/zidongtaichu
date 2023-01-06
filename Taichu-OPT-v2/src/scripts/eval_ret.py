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

import os
import sys
sys.path.append("./")

import argparse
import json
import mindspore
from src.config import config as C
import mindspore as ms
from mindspore import context, ops
from mindspore.train.model import Model
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model_mindspore.parallel_transformer import ParallelConfig
from mindspore.ops import operations as P
from src.model_mindspore.retrieval_ms import UniterThreeForRet, UniterThreeForITM
from src.tools.logger import LOGGER, add_log_to_file
from src.tools.misc import parse_with_config, set_random_seed
import numpy as np
from src.data.pretrain_three_data import build_naive_dataloader
from src.data.re_dataset import Re_Eval_Img_Dataset, Re_Eval_Txt_Dataset, itmTwo_collate
from pathlib2 import Path

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)

def init_config(opts):

    C.IMG_DIM = getattr(opts, 'img_dim', 768)
    C.IMG_SIZE = opts.image_size
    C.IMG_PATCH_SIZE = opts.patch_size

    C.MAX_IMG_LEN = (C.IMG_SIZE // C.IMG_PATCH_SIZE)**2 + 1
    C.MAX_IMG_TEXT_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN
    C.MAX_FULL_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN + C.MAX_AUDIO_LEN

    print(f"IMG_SIZE:{C.IMG_SIZE} IMG_PATCH_SIZE:{C.IMG_PATCH_SIZE}")
    print(f"MAX_IMG_LEN:{C.MAX_IMG_LEN} MAX_IMG_TEXT_LEN:{C.MAX_IMG_TEXT_LEN}  MAX_FULL_LEN:{C.MAX_FULL_LEN}")


def init_env(opts):

    """ init_env """
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

    log_dir = os.path.join(opts.output_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    add_log_to_file(os.path.join(log_dir, f"log_{rank_id}.txt"))
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

    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"

    os.system('ulimit -s 102400')
    set_random_seed(opts.seed)

    LOGGER.info(f'local_rank:{local_rank}, device id:{device_id} start to run...')


    if opts.use_parallel:
        context.set_context(mode=context.GRAPH_MODE,
                            save_graphs=False,
                            save_graphs_path=save_graphs_path,
                            device_target="Ascend",
                            device_id=device_id)
        context.set_context(max_device_memory="30GB")
        context.set_context(reserve_class_name_in_scope=False)
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
        context.set_context(mode=context.GRAPH_MODE,
                            save_graphs=False,
                            save_graphs_path=save_graphs_path,
                            device_target="Ascend",
                            device_id=device_id)
        context.set_context(max_device_memory="30GB")
        context.set_context(reserve_class_name_in_scope=False)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num)
        LOGGER.info(f"device_id is {device_id}, device_num is {device_num}")

    ParallelConfig.mp = 1
    ParallelConfig.optimizer_shard = False
    ParallelConfig.dp = device_num // ParallelConfig.mp
    return local_rank, rank_id, strategy_ckpt_save_file, device_id, device_num

def get_batch_data_ret(batch):
    """ get_batch_data """
    for key, value in batch.items():
        batch[key] = Tensor(value)
    input_ids = batch['input_ids']
    position_ids = batch['position_ids']
    attention_mask = batch['attn_masks']
    attn_masks_text = batch['attn_masks_text']
    images = batch['images']
    return (input_ids, position_ids, attention_mask, attn_masks_text, images)

def load_ckpt(net_with_grads, ckpt_file):
    if not ckpt_file:
        return
    LOGGER.info(f'load ckpt: {ckpt_file}')
    params_dict = load_checkpoint(ckpt_file)
    if params_dict:
        # new_params_dict = {}
        # for key in params_dict.keys():
        #     if key.find("txt_output.tfm_decoder") >= 0:
        #         key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
        #         new_params_dict[key_new] = params_dict[key]
        #     new_params_dict[key] = params_dict[key]
        # new_params_dict["uniter.img_embeddings.img_linear.weight"] = new_params_dict["feat_regress.weight"]
        # new_params_dict["uniter.audio_embeddings.audio_linear.weight"] = new_params_dict["audio_feat_regress.weight"]
        # new_params_dict["uniter.embeddings.word_embeddings.embedding_table"] = new_params_dict["cls.predictions.decoder.weight"]
        net_not_load = load_param_into_net(net_with_grads, params_dict)
        print("===============net_not_load================", net_not_load)
    # print("init model......................................")
    # net_with_grads.init_output()
    print('load ckpt:', ckpt_file)


def main(opts):

    init_config(opts)

    # init
    (local_rank, rank_id, strategy_ckpt_save_file, device_id, device_num) = init_env(opts)
    # eval
    ids = json.load(open(opts.ids_val_path, 'r'))
    LOGGER.info(f"retrieval dataset's length is: {len(ids)}")

    ann_file = opts.ids_val_path
    image_root = opts.val_datasets[0]['img'][0]

    LOGGER.info(f"ann_file: {ann_file}")
    LOGGER.info(f"image_root: {image_root}")

    img_dataset = Re_Eval_Img_Dataset(ann_file, image_root)
    iloader = build_naive_dataloader(img_dataset, itmTwo_collate, batch_size=10, device_num=device_num)

    txt_dataset = Re_Eval_Txt_Dataset(ann_file, image_root)
    tloader = build_naive_dataloader(txt_dataset, itmTwo_collate, batch_size=10, device_num=device_num)

    net_itc = UniterThreeForRet(opts.model_config, opts)
    load_ckpt(net_itc, opts.ckpt_file)

    if opts.use_parallel:
        model_itc = Model(net_itc)
    else:
        net_itc.set_train(False)
        model_itc = net_itc

    k_test = getattr(opts, 'k_test', 128)
    LOGGER.info(f"k_test: {k_test}")

    opts.train_batch_size = k_test
    net_itm = UniterThreeForITM(opts.model_config, opts)
    load_ckpt(net_itm, opts.ckpt_file)

    if opts.use_parallel:
        model_itm = Model(net_itm)
    else:
        net_itm.set_train(False)
        model_itm = net_itm

    img_embeds, img_feats, _ = get_feature(model_itc, img_dataset, iloader, opts, rank_id, type='img')
    LOGGER.info(f"{img_embeds.shape}, {img_feats.shape}")

    txt_embeds, txt_feats, attn_txts = get_feature(model_itc, txt_dataset, tloader, opts, rank_id, type='txt')
    LOGGER.info(f"{txt_embeds.shape}, {txt_feats.shape}")

    bmm = ops.MatMul(transpose_a=False, transpose_b=True)
    score_i2t = bmm(img_feats.astype(mindspore.float16), txt_feats.astype(mindspore.float16))
    score_t2i = score_i2t.transpose()

    score_matrix_i2t = np.zeros((img_feats.shape[0], txt_embeds.shape[0])).astype(np.float32)
    score_matrix_i2t.fill(-100.0)
    score_matrix_i2t = Tensor(score_matrix_i2t)

    for i, sims in enumerate(score_i2t):

        topk_sim, topk_idx = ops.TopK(sorted=True)(sims, k_test)
        img_embed_full = P.BroadcastTo((k_test, -1, -1))(img_embeds[i])
        text_embed_full = txt_embeds[topk_idx]

        attn_txts_full = attn_txts[topk_idx].astype(ms.int32)
        attn_img_full = Tensor(np.ones((img_embed_full.shape[0], img_embed_full.shape[1])), ms.int32)
        attention_mask = ops.Concat(axis=1)([attn_img_full, attn_txts_full])

        if opts.use_parallel:
            score = model_itm.predict(text_embed_full, img_embed_full, attention_mask, attn_txts_full)[:, 1]
        else:
            score = model_itm(text_embed_full, img_embed_full, attention_mask, attn_txts_full)[:, 1]

        score_matrix_i2t[i, topk_idx] = score

        if i % 100 == 0:
            LOGGER.info(f"score_matrix_i2t {i} / {len(score_i2t)}")


    score_matrix_t2i = np.zeros((txt_embeds.shape[0], img_feats.shape[0])).astype(np.float32)
    score_matrix_t2i.fill(-100.0)
    score_matrix_t2i = Tensor(score_matrix_t2i)

    for i, sims in enumerate(score_t2i):

        topk_sim, topk_idx = ops.TopK(sorted=True)(sims, k_test)
        img_embed_full = img_embeds[topk_idx]
        text_embed_full = P.BroadcastTo((k_test, -1, -1))(txt_embeds[i])
        attn_txts_full = P.BroadcastTo((k_test, -1))(attn_txts[i]).astype(ms.int32)

        attn_img_full = Tensor(np.ones((img_embed_full.shape[0], img_embed_full.shape[1])), ms.int32)
        attention_mask = ops.Concat(axis=1)([attn_img_full, attn_txts_full])

        if opts.use_parallel:
            score = model_itm.predict(text_embed_full, img_embed_full, attention_mask, attn_txts_full)[:, 1]
        else:
            score = model_itm(text_embed_full, img_embed_full, attention_mask, attn_txts_full)[:, 1]

        score_matrix_t2i[i, topk_idx] = score

        if i % 100 == 0:
            LOGGER.info(f"score_matrix_t2i {i} / {len(score_t2i)}")

    np_score_i2t_itm = score_matrix_i2t.asnumpy()
    np_score_t2i_itm = score_matrix_t2i.asnumpy()

    LOGGER.info(opts.ckpt_file)

    LOGGER.info("results")
    res_itm = itm_eval(np_score_i2t_itm, np_score_t2i_itm, img_dataset.txt2img, img_dataset.img2txt)
    LOGGER.info(res_itm)

    return
    
def get_feature(model, dataset, loader, opts, rank, type):
    '''get feature'''
    LOGGER.info(type + " start feature extract...")
    n_ex = 0
    embs = []
    feats = []
    attn_txts = []
    for batch in loader:
        (input_ids, position_ids, attention_mask,
         attn_masks_text, images) = get_batch_data_ret(batch)
        if type=='img':
            if opts.use_parallel:
                _, _, attn_txt, emb, feat = model.predict(input_ids, position_ids, attn_masks_text, images)
            else:
                _, _, attn_txt, emb, feat = model(input_ids, position_ids, attn_masks_text, images)
        elif type=='txt':
            if opts.use_parallel:
                emb, feat, attn_txt, _, _ = model.predict(input_ids, position_ids, attn_masks_text, images)
            else:
                emb, feat, attn_txt, _, _ = model(input_ids, position_ids, attn_masks_text, images)
        n_ex += emb.shape[0]
        if rank == 0 and n_ex % 100 == 0:
            LOGGER.info(f"[{n_ex}/{len(dataset)}]")
        embs.append(emb)
        feats.append(feat)
        attn_txts.append(attn_txt)
        if n_ex>=len(dataset):
            break
    embs = ops.Concat(axis=0)(embs)
    feats = ops.Concat(axis=0)(feats)
    attn_txts = ops.Concat(axis=0)(attn_txts)
    return embs[:len(dataset)], feats[:len(dataset)], attn_txts[:len(dataset)]

def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    '''itm eval'''
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    tr100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    tr128 = 100.0 * len(np.where(ranks < 128)[0]) / len(ranks)

    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        
    ir50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    ir100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    ir128 = 100.0 * len(np.where(ranks < 128)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r50': tr50,
                    'txt_r100': tr100,
                    'txt_r128': tr128,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r50': ir50,
                    'img_r100': ir100,
                    'img_r128': ir128,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def str2bool(b):
    if b.lower() in ["false"]:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default="/mnt/sfs_turbo/simplify/uniter-three/config/train_retrieval_config_vit448.json",
                        help='JSON config files')
    parser.add_argument("--eval_only", default=False, type=str2bool,
                        help="eval only?")
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
    # parser.add_argument('--audio_dim', default=1024,
    #                     type=int, help='audio dim')
    # parser.add_argument('--img_dim', default=2048,
    #                     type=int, help='img dim')
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
    parser.add_argument("--save_checkpoint_steps",
                        default=5000, type=int, help="save checkpoint steps")
    parser.add_argument("--epochs", default=10,
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
