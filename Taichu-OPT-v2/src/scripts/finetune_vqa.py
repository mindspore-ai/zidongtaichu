import os
import sys
sys.path.append("./")
import argparse

from pathlib2 import Path
from mindspore import context
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn import TrainOneStepCell
from mindspore.train.model import Model
from mindspore import ops as P
from mindspore.nn import ResizeBilinear as nn_ResizeBilinear
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype

from src.config import config as C
from src.tools.logger import LOGGER, add_log_to_file
from src.tools.misc import parse_with_config, set_random_seed
from src.tools.monitor import LossMonitor
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.vqa.vqa_ms import UniterTwoForVqaWithLoss
from src.model_mindspore.utils import LearningRate
from src.model_mindspore.optim_ms import build_optimizer
from src.model_mindspore.cell_wrapper import ParallelTrainOneStepWithLossScaleCell
from src.data import create_dataset

def init_config(opts):

    C.USE_LARGE_DATA = getattr(opts, 'use_large_data', False)

    C.IMG_DIM = getattr(opts, 'img_dim', 768)
    C.IMG_SIZE = opts.image_size
    C.IMG_PATCH_SIZE = opts.patch_size
    C.MAX_IMG_LEN = (C.IMG_SIZE // C.IMG_PATCH_SIZE)**2 + 1

    C.MAX_TEXT_LEN = opts.text_len - 2
    C.MAX_FULL_TEXT_LEN = opts.text_len
    C.MAX_TEXT_GTS_LEN = opts.text_len - 1

    C.MAX_IMG_TEXT_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN
    C.MAX_FULL_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN + C.MAX_AUDIO_LEN

    print(f"USE_LARGE_DATA:{C.USE_LARGE_DATA}")
    print(f"IMG_DIM:{C.IMG_DIM} IMG_SIZE:{C.IMG_SIZE} IMG_PATCH_SIZE:{C.IMG_PATCH_SIZE}")
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

    LOGGER.info(f'output_dir: {opts.output_dir}')
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

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(max_device_memory="30GB")
    context.set_context(reserve_class_name_in_scope=False)


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

    ParallelConfig.mp = 1
    ParallelConfig.optimizer_shard = False

    ds = create_dataset(opts, device_num=device_num)
    dataset_size = ds.get_dataset_size()
    LOGGER.info(f"=====dataset size: {dataset_size}")
    if opts.sink_size > 0:
        new_epoch = opts.epochs * dataset_size // opts.sink_size
        callback_size = opts.sink_size
    else:
        new_epoch = opts.epochs
        callback_size = 1

    ParallelConfig.dp = device_num // ParallelConfig.mp

    LOGGER.info(f"=====device_num:{device_num} dp:{ParallelConfig.dp} mp:{ParallelConfig.mp} "
                f"train_batch_size:{opts.train_batch_size} val_batch_size:{opts.val_batch_size}")

    return local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num, new_epoch, ds

def interpolate_pos_embed(pos_embed_checkpoint, img_size, patch_size, num_extra_tokens=1):
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = img_size // patch_size

    if orig_size != new_size:
        extra_tokens = pos_embed_checkpoint[:num_extra_tokens, :]
        pos_tokens = pos_embed_checkpoint[num_extra_tokens:, :]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size)
        pos_tokens = P.transpose(pos_tokens, (0,3,1,2))
        resize_bilinear = nn_ResizeBilinear()
        pos_tokens = resize_bilinear(pos_tokens,(new_size, new_size),align_corners=False)
        pos_tokens = P.transpose(pos_tokens,(0,2,3,1)).reshape(new_size * new_size, embedding_size)
        new_pos_embed = P.Concat(axis=0)((extra_tokens, pos_tokens))
        return new_pos_embed
    else:
        return pos_embed_checkpoint

def load_ckpt(net_with_loss, ckpt_file):
    if not ckpt_file or len(ckpt_file) == 0:
        return
    print('begin load ckpt:', ckpt_file)
    params_dict = load_checkpoint(ckpt_file)
    if params_dict:
        new_params_dict = {}
        for key in params_dict.keys():
            if(key == 'uniter.img_embeddings.vit.pos_embed.embedding_table'):
                new_pos_emb = interpolate_pos_embed(params_dict[key],C.IMG_SIZE, C.IMG_PATCH_SIZE)
                new_pos_emb = Parameter(Tensor(new_pos_emb, mstype.float32), name='uniter.img_embeddings.vit.pos_embed.embedding_table', requires_grad=True)
                new_params_dict[key] = new_pos_emb
                print('load resized vit pos_embed')
                continue
            if(key == 'uniter.img_embeddings.vit.position_ids'):
                print('skip vit position_ids')
                continue
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_params_dict[key_new] = params_dict[key]
            new_params_dict[key] = params_dict[key]
        new_params_dict["uniter.embeddings.word_embeddings.embedding_table"] = new_params_dict["cls.predictions.decoder.weight"]
        net_not_load = load_param_into_net(net_with_loss, new_params_dict)
        print("===============net_not_load================", net_not_load)
    
    print('finish load ckpt:', ckpt_file)

def numel(shape):
    total = 1
    for val in shape:
        total *= val
    return total

def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output

def main(opts):

    # init
    init_config(opts)

    (local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num,
     new_epoch, ds) = init_env(opts)

    net_with_loss = UniterTwoForVqaWithLoss(opts.model_config, opts)

    load_ckpt(net_with_loss, opts.ckpt_file)

    if rank_id == 0:
        LOGGER.info("model have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.get_parameters())))
        LOGGER.info("model text have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.uniter.embeddings.get_parameters())))
        LOGGER.info("model img have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.uniter.img_embeddings.get_parameters())))
        LOGGER.info("model cross have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.uniter.encoder.get_parameters())))

    net_with_loss = _VirtualDatasetCell(net_with_loss)

    lr = LearningRate(opts.start_learning_rate,
                      opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)
    if opts.use_parallel:
        net_with_grads = ParallelTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                           scale_sense=update_cell, enable_global_norm=True,
                                                           clip_norm=opts.grad_norm, parallel_config=ParallelConfig)
    else:
        net_with_grads = TrainOneStepCell(net_with_loss, optimizer)

    # all cards will save ckpty
    save_steps = opts.save_checkpoint_steps
    ckpt_dir = os.path.join(opts.output_dir, "train/ckpt", f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    if rank_id == 0:

        config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                     keep_checkpoint_max=10,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix="OPT-VQA",
                                     directory=ckpt_dir,
                                     config=config_ck)
        callback = [TimeMonitor(callback_size), LossMonitor(callback_size, opts.is_two)]
        callback.append(ckpoint_cb)
    else:
        callback = []

    model = Model(net_with_grads)

    print('=====start training...=====')
    model.train(new_epoch, ds, callbacks=callback, dataset_sink_mode=False, sink_size=callback_size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='',
                        help='JSON config files')
    parser.add_argument('--output_dir', default='',
                        help='output_dir')
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
