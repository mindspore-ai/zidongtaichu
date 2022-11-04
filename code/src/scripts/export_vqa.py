import os
import sys
import argparse
import numpy as np

from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor, export, float32, int64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from config.config import IMG_DIM, IMG_LABEL_DIM, AUDIO_DIM, AUDIO_LABEL_DIM
from src.model_mindspore.vqa import UniterThreeForPretrainingForVQAFinetuneInf


def pad_sequence(sequences, batch_first=True, padding_value=0.0, max_lens=-1):
    """pad_sequence"""
    lens = [len(x) for x in sequences]
    if max_lens == -1:
        max_lens = max(lens)

    padded_seq = []
    for x in sequences:
        pad_width = [(0, max_lens - len(x))]
        padded_seq.append(np.pad(x, pad_width, constant_values=(padding_value, padding_value)))

    sequences = np.stack(padded_seq, axis=0 if batch_first else 1)
    return sequences


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
        new_params_dict["uniter.embeddings.word_embeddings.embedding_table"] = new_params_dict[
            "cls.predictions.decoder.weight"]
        param_not_load = load_param_into_net(net, new_params_dict)
        print("param not load:", param_not_load)
    print(f"end loading ckpt:{ckpt_file}")


def export_network(config, ckpt_file, mindir_file, file_format):
    print("load network")
    net = UniterThreeForPretrainingForVQAFinetuneInf(config, IMG_DIM, IMG_LABEL_DIM, AUDIO_DIM, AUDIO_LABEL_DIM,
                                                     full_batch=False)
    load_ckpt(net, ckpt_file)

    input_ids = Tensor(np.array(np.random.randn(1, 50), dtype=np.float32), int64)
    position_ids = Tensor(np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0), int64)
    image = Tensor(np.array(np.random.randn(1, 197, 3072), dtype=np.float32), float32)
    img_pos_feat = Tensor(np.expand_dims(np.arange(0, 197, dtype=np.int64), axis=0), int64)
    attn_masks = Tensor(np.ones((1, 247), dtype=np.int64), int64)
    gather_index = Tensor(np.expand_dims(np.arange(0, 247, dtype=np.int64), axis=0), int64)
    print("mindir file:", mindir_file)
    export(net, input_ids, position_ids, image, img_pos_feat, attn_masks, gather_index, file_name=mindir_file,
           file_format=file_format)
    print("export finished")


def main(ops):
    print("init environment")
    device_id = int(os.getenv('DEVICE_ID') if os.getenv('DEVICE_ID') else 0)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)

    export_network(ops.model_config, ops.ckpt_file, ops.output_file, ops.file_format)


# example
# python export_vqa.py --model_config xxx/config/vqa/cross_modal_encoder_base.json --ckpt_file xxx/OPT_vqa-xxx.ckpt --output_file xxx/opt_vqa
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default="config/vqa/cross_modal_encoder_base.json", type=str,
                        help='model config file')
    parser.add_argument('--ckpt_file', required=True, default="", type=str, help='check point file')
    parser.add_argument('--output_file', default="opt_vqa", type=str, help='output file')
    parser.add_argument('--file_format', default="MINDIR", type=str, help='output file format')

    args = parser.parse_args()
    main(args)
