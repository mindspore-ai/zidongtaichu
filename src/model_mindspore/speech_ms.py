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
import yaml

import mindspore.nn as nn
import mindspore.ops as ops

from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.model_config import UniterConfig
from src.model_mindspore.model_ms import UniterThreeModelAudio

from src.fastspeech2_mindspore.model.fastspeech2 import FastSpeech2ThreeV3
from src.fastspeech2_mindspore.model.loss import FastSpeech2ThreeV3Loss

class UniterThreeForPretrainingForAdWithLoss(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, full_batch=True, use_moe=False, opts=None):
        super(UniterThreeForPretrainingForAdWithLoss, self).__init__()
        parallel_config = ParallelConfig()
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        self.uniter = UniterThreeModelAudio(config, parallel_config, use_moe)

        # Audio Generator
        group_for_loss = 2
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False

        preprocess_config = yaml.load(open(opts.audio_preprocess_config, "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open(opts.audio_model_config, "r"), Loader=yaml.FullLoader)
        self.audio_output = FastSpeech2ThreeV3(preprocess_config, model_config, config.hidden_size)
        self.audio_crit = FastSpeech2ThreeV3Loss(preprocess_config, model_config)

        self.reduce_sum = ops.ReduceSum()
        self.add = ops.Add()

        # sequence_output: (batch_size, seq_len, hidden_size) 56, 155, 768

    def generate_audio(self, sequence_output, mel_targets, duration_targets,
                       speakers, texts, src_lens, mel_lens, audio_max_text_len, audio_max_mel_len,
                       pitch_targets, energy_targets):
        """
        generate_audio
        """
        input_data = sequence_output

        # input_data,
        # speakers,
        # texts,
        # src_lens,
        # max_src_len,
        # mels = None,
        # mel_lens = None,
        # max_mel_len = None,
        # p_targets = None,
        # e_targets = None,
        # d_targets = None,
        # p_control = 1.0,
        # e_control = 1.0,
        # d_control = 1.0,

        mel_predictions, postnet_mel_predictions, p_predictions, e_predictions, log_duration_predictions, _, \
        src_masks, mel_masks, src_lens, mel_lens = \
            self.audio_output(input_data, speakers, texts, src_lens, audio_max_text_len, mel_targets, mel_lens,
                              audio_max_mel_len,
                              pitch_targets, energy_targets, duration_targets)

        total_loss = \
            self.audio_crit(mel_targets, src_masks, mel_masks, duration_targets,
                            pitch_targets, energy_targets, mel_predictions,
                            postnet_mel_predictions, log_duration_predictions,
                            p_predictions, e_predictions)

        return total_loss

    def construct(self, input_ids, position_ids, attention_mask,
                  mel_targets, duration_targets, speakers, texts, src_lens, mel_lens,
                  audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets):
        """Construct Function"""
        sequence_output, moe_loss = self.uniter(input_ids, position_ids, attention_mask)

        loss = self.generate_audio(sequence_output, mel_targets, duration_targets, speakers, texts, src_lens,
                                   mel_lens, audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets)

        final_loss = self.reduce_sum(loss)

        return final_loss


class UniterThreeForPretrainingForAdEval(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, full_batch=True, use_moe=False, opts=None):
        super(UniterThreeForPretrainingForAdEval, self).__init__()
        parallel_config = ParallelConfig()
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        self.uniter = UniterThreeModelAudio(config, parallel_config, use_moe)

        # Audio Generator
        group_for_loss = 2
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False

        preprocess_config = yaml.load(open(opts.audio_preprocess_config, "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open(opts.audio_model_config, "r"), Loader=yaml.FullLoader)
        self.audio_output = FastSpeech2ThreeV3(preprocess_config, model_config, config.hidden_size)
        self.audio_crit = FastSpeech2ThreeV3Loss(preprocess_config, model_config)

        self.reduce_sum = ops.ReduceSum()
        self.add = ops.Add()

        # sequence_output: (batch_size, seq_len, hidden_size) 56, 155, 768

    def generate_audio_eval(self, sequence_output, speakers, texts, src_lens, audio_max_text_len):
        """generate audio"""

        # generate audio
        input_data = sequence_output

        _, postnet_mel_predictions, _, _, _, _, \
        _, _, src_lens, mel_lens = \
            self.audio_output(input_data, speakers, texts, src_lens, audio_max_text_len)

        return postnet_mel_predictions, mel_lens

    def construct(self, input_ids, position_ids, attention_mask,
                  mel_targets, duration_targets, speakers, texts, src_lens, mel_lens,
                  audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets):
        """Construct Function"""
        sequence_output, moe_loss = self.uniter(input_ids, position_ids, attention_mask)

        # speakers, texts, src_lens, audio_max_text_len
        postnet_mel_predictions, mel_lens = self.generate_audio_eval(sequence_output, speakers, texts, src_lens,
                                                                     audio_max_text_len)

        return postnet_mel_predictions, mel_lens