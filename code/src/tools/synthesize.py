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
"""fastspeech2 synthesize"""
import re
import os
import argparse
from string import punctuation
import yaml
import json
import mindspore.context as context
import numpy as np
from g2p_en import G2p
from pypinyin import pinyin, Style

from fastspeech2_ms.utils.tools import synth_samples
from fastspeech2_ms.text import text_to_sequence
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from model_mindspore.pretrain_ms import UniterThreeForPretrainingForAdEval
from fastspeech2_ms import hifigan

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context.set_context(mode=context.PYNATIVE_MODE,
                    device_target="Ascend",
                    device_id=0)

def get_model(opts):

    device_id = int(os.getenv('DEVICE_ID'))

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    ckpt_file = opts.ckpt_file
    print(ckpt_file)
    if ckpt_file == "":
        modified_params_dict = None
    else:
        params_dict = load_checkpoint(ckpt_file)

        modified_params_dict = {}
        for k, v in params_dict.items():
            if 'txt_output.tfm_decoder' in k:
                modified_k = k.replace('txt_output.tfm_decoder', 'txt_output.tfm_decoder.decoder.tfm_decoder')
                v.name = v.name.replace('txt_output.tfm_decoder', 'txt_output.tfm_decoder.decoder.tfm_decoder')
                modified_v = v
                modified_params_dict[modified_k] = modified_v
            else:
                modified_params_dict[k] = v

    net_without_loss = UniterThreeForPretrainingForAdEval(opts.model_config, full_batch=opts.full_batch,
                                                           use_moe=opts.use_moe, opts=opts)

    if modified_params_dict:
        net_not_load = load_param_into_net(net_without_loss, modified_params_dict)
        print("===============net_not_load================", net_not_load)


    return net_without_loss

def get_vocoder(speaker):
    '''
    mindspore;done;useful
    '''
    # name = "HiFi-GAN"
    with open("fastspeech2_ms/hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    if speaker == "LJSpeech":
        ckpt = load_checkpoint("fastspeech2_ms/hifigan/generator_LJSpeech.ckpt")
    elif speaker == "universal":
        ckpt = load_checkpoint("fastspeech2_ms/hifigan/generator_universal.ckpt")
    else:
        raise Exception("error speaker")
    load_param_into_net(vocoder, ckpt, strict_load=True)
    return vocoder


def read_lexicon(lex_path):
    """read_lexicon"""
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    """preprocess_english"""
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    """preprocess_mandarin"""
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, configs, vocoder, batchs, control_values):
    """synthesize"""
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        # batch = to_device(batch, device)
        # with torch.no_grad():
        # Forward
        output = model(
            *(batch[2:]),
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control
        )
        synth_samples(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
            train_config["path"]["result_path"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    # Get model
    model = get_model(args)

    # Load vocoder
    vocoder = get_vocoder(model_config)

    ids = raw_texts = [args.text[:100]]
    speakers = np.array([args.speaker_id])
    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([preprocess_english(args.text, preprocess_config)])
    elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, vocoder, batchs, control_values)

if __name__ == "__main__":
    main()
