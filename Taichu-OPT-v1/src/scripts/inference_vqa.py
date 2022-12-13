import os
import json
import time
import logging
import argparse
import numpy as np
from PIL import Image
from mindspore import context, Tensor, float32, int64
from mindspore.dataset.vision.utils import Inter
import mindspore.dataset.vision.c_transforms as C
from mindspore import load_checkpoint, load_param_into_net
from transformers import BertTokenizer

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))

from src.config.config import IMG_DIM, IMG_LABEL_DIM, AUDIO_DIM, AUDIO_LABEL_DIM
from src.model_mindspore.caption_ms import UniterThreeForPretrainingForCapFinetuneInf
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


def pad_tensors(tensors, lens=None, pad=0, max_len=-1):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    if max_len == -1:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    output = np.zeros((bs, max_len, hid), dtype=dtype)
    if pad:
        output.fill(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output[i, :l, ...] = t
    return output


def preprocess(image_path, text, bert_base_chinese_vocab):
    image_size = 448
    patch_size = 32

    resize = image_size
    image_size = image_size
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    interpolation = "BILINEAR"
    if hasattr(Inter, interpolation):
        interpolation = getattr(Inter, interpolation)
    else:
        interpolation = Inter.BILINEAR
        print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))
    trans = [
        C.Resize(resize, interpolation=interpolation),
        C.CenterCrop(image_size),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    for tran in trans:
        image = tran(image)

    p = patch_size
    channels, h, w = image.shape
    x = np.reshape(image, (channels, h // p, p, w // p, p))
    x = np.transpose(x, (1, 3, 0, 2, 4))
    patches = np.reshape(x, ((h // p) * (w // p), channels * p * p))
    img_pos_feat = np.arange(patches.shape[0] + 1)
    attn_masks = np.ones(img_pos_feat.shape[0], dtype=np.int64)

    img_feat = Tensor(pad_tensors([patches, ], [196], max_len=197))
    img_pos_feat = Tensor(np.stack([img_pos_feat, ], axis=0))
    attn_masks = Tensor(pad_sequence([attn_masks, ], batch_first=True, padding_value=0, max_lens=247))
    out_size = attn_masks.shape[1]
    batch_size = attn_masks.shape[0]
    gather_index = Tensor(np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0))

    tokenizer = BertTokenizer.from_pretrained(bert_base_chinese_vocab)
    question_tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    input_ids = Tensor(pad_sequence([input_ids, ], batch_first=True, padding_value=0, max_lens=50), int64)
    position_ids = Tensor(np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0), int64)

    return input_ids, position_ids, img_feat, img_pos_feat, attn_masks, gather_index


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
        out.append(txt.replace(' ##', ''))
    return out


def postprocess(sequence, vocab):
    return decode_sequence(vocab, sequence[:, 0, 1:].asnumpy(), split='')


def inference_one(net, image_path, text, vocab, bert_base_chinese_vocab):
    inputs_id, position_ids, img_feat, img_pos_feat, attn_masks, gather_index = preprocess(image_path, text,
                                                                                           bert_base_chinese_vocab)
    seq = net(inputs_id, position_ids, img_feat, img_pos_feat, attn_masks, gather_index)
    output = postprocess(seq, vocab)
    return output


def inference_directory(net, image_dir, text, vocab, bert_base_chinese_vocab):
    image_list = os.listdir(image_dir)
    for image_path in image_list:
        if os.path.splitext(image_path)[-1] in (".jpg", ".png"):
            image = os.path.join(image_dir, image_path)
            print(image)
            last_time = time.time()
            inference_output = inference_one(net, image, text, vocab, bert_base_chinese_vocab)
            print(time.time() - last_time)
            print(inference_output)


def inference_list_json(net, image_dir, list_file, text, vocab, bert_base_chinese_vocab):
    with open(list_file, "r") as f:
        image_list = json.load(f)
    for image_path in image_list:
        image_path = image_path
        last_time = time.time()
        inference_output = inference_one(net, os.path.join(image_dir, image_path), text, vocab, bert_base_chinese_vocab)
        print(image_path, time.time() - last_time)
        print(inference_output)


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


def main(opts):
    print("init environment")
    device_id = int(os.getenv('DEVICE_ID') if os.getenv('DEVICE_ID') else 0)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)

    print("load network")
    net = UniterThreeForPretrainingForVQAFinetuneInf(opts.model_config, IMG_DIM, IMG_LABEL_DIM, AUDIO_DIM,
                                                     AUDIO_LABEL_DIM, full_batch=False, beam_width=opts.beam_width)
    load_ckpt(net, opts.ckpt_file)

    vocab = json.load(open(opts.vocab_path))
    bert_base_chinese_vocab = opts.bert_base_chinese_vocab

    if os.path.isfile(opts.inference_list):
        inference_list_json(net, opts.inference_dir, opts.inference_list, opts.inference_question, vocab,
                            bert_base_chinese_vocab)
    elif os.path.isdir(opts.inference_dir):
        inference_directory(net, opts.inference_dir, opts.inference_question, vocab, bert_base_chinese_vocab)


# python inference_vqa.py  --ckpt_file=xxx/OPT_vqa-11_16473.ckpt --inference_dir xxx/deploy_vqa/opt_vqa/test/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config',
                        default="config/vqa/cross_modal_encoder_base.json",
                        type=str,
                        help='model config file')
    parser.add_argument('--ckpt_file', required=True, default="", type=str, help='check point file')
    parser.add_argument('--beam_width', default="4", type=int, help='beam search width')
    parser.add_argument('--bert_base_chinese_vocab',
                        default="dataset/vqa/bert-base-chinese-vocab.txt",
                        type=str, help='vocab file')
    parser.add_argument('--vocab_path',
                        default="dataset/vqa/ids_to_tokens_zh.json",
                        type=str, help='vocab file')
    parser.add_argument('--inference_dir', required=True, default="", type=str, help='inference directory')
    parser.add_argument('--inference_list', default="", type=str, help='inference list file')
    parser.add_argument('--inference_question', default="这张图片描述的是什么？", type=str, help='inference list file')
    args = parser.parse_args()

    main(args)
