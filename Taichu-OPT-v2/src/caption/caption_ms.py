import src.config.config as C
import copy
import numpy as np

from src.model_mindspore.model_ms import freeze_net
from src.model_mindspore.pretrain_two_ms import UniterTwoForPretrainingWithLoss
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.beam_search import BeamSearchDecoder, TileBeam
from src.config.model_config import UniterConfig
from src.config.transformer_config import transformer_net_cfg as cfg

from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

from src.tools.logger import LOGGER

class BertAttentionCausalMask(Cell):
    r"""
    Get the attention and causal masks
    Args:
        config(parallel_config): the parallel configure
    Inputs:
        input_mask: the mask indicating whether each position is a valid input with (batch_size, seq_length), attention mask
    Outputs:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length) causal mask * attention mask
    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, parallel_config=ParallelConfig):
        super(BertAttentionCausalMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.multiply = P.Mul().shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))

    def construct(self, input_mask):
        """
        Generate the mask matrix.
        """
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        # [bs, seq_length, seq_length]
        causal_mask = self.get_causal_mask(input_shape[1])
        attention_causal_mask = self.multiply(attention_mask,causal_mask)
        return attention_causal_mask

    def get_causal_mask(self, seq_length):
        ones = np.ones(shape=(seq_length, seq_length))
        tril_ones = Tensor(np.tril(ones).reshape(1, seq_length, seq_length), dtype=mstype.float32)
        return tril_ones

class SeqCrossEntropy(nn.Cell):
    """ CrossEntropy """

    def __init__(self, parallel_config, vocab_size):
        super(SeqCrossEntropy, self).__init__()
        self.onehot = ops.OneHot().shard(((parallel_config.dp, 1), (), ()))
        self.log_softmax = ops.LogSoftmax().shard(((parallel_config.dp, 1),))
        self.label_smoothing = 0.1
        self.vocab_size = vocab_size
        self.on_value = Tensor(float(1 - self.label_smoothing), mstype.float32)
        self.off_value = Tensor(self.label_smoothing / float(self.vocab_size - 1), mstype.float32)
        self.last_idx = (-1,)
        self.mul = ops.Mul().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.reduce_sum = ops.ReduceSum().shard(((parallel_config.dp, 1),))
        self.neg = ops.Neg().shard(((parallel_config.dp,),))
        self.mul1 = ops.Mul().shard(((parallel_config.dp,), (parallel_config.dp,)))
        self.sum = ops.ReduceSum().shard(((1,),))
        self.div = ops.RealDiv().shard(((), ()))
        self.eps = 1e-7
        self.add = ops.Add().shard(((), ()))
    def construct(self, logits, label, attention_mask):
        """ construct 
            logits : bs * seq * vocab
            label : bs * seq (* 1)
        """
        attention_mask = attention_mask.astype(mstype.float32)
        flat_label = label.view(label.shape[0],-1)
        # bs*seq*vocab
        onehot_label = self.onehot(flat_label, logits.shape[-1], self.on_value, self.off_value)
        # bs*seq*vocab
        log_logits = self.log_softmax(logits)
        # bs*seq
        per_example_loss = self.neg(
            self.reduce_sum(self.mul(log_logits, onehot_label), self.last_idx))
        mask_loss = self.mul1(per_example_loss, attention_mask)
        total_loss = self.sum(mask_loss)
        valid_token = self.sum(attention_mask.view(-1))
        valid_token = self.add(valid_token, self.eps)
        loss = self.div(total_loss, valid_token)

        return loss

class UniterTwoForCaptionWithLoss(UniterTwoForPretrainingWithLoss):

    def __init__(self, config, args=None):
        super().__init__(config, args)

        freeze_net(self.vision_proj)
        freeze_net(self.text_proj)
        freeze_net(self.clip_loss)
        freeze_net(self.itm_head_two)
        
        config = UniterConfig.from_json_file(config)
        vocab_size = config.vocab_size

        parallel_config = ParallelConfig()

        # 修改 text embedding 使用的mask, 在construct时生效
        self.uniter.embeddings.attention_mask = BertAttentionCausalMask(parallel_config)
        self.uniter.encoder.attention_mask = BertAttentionCausalMask(parallel_config)
        
        # 设置loss
        self.seq_loss = SeqCrossEntropy(parallel_config, vocab_size)

    def construct(self, input_ids, position_ids, attention_mask, txt_mask, txt_label_mask, itm_target,
                    attn_masks_text, attn_masks_img, images, images_rand, input_ids_mask,
                    txt_gts, txt_gts_mask, taskId):
        """
            Construct Function
            Inputs:
                inputs_ids:     txt ids
                position_ids:   txt pos ids
                txt_gts:        txt ground truth
                txt_masks:      inputs and gts's mask
                images:         image
        
        """
        if not self.full_batch:
            taskId = self.stride_slice_1(taskId, (0,), (1,), (1,))
            position_ids = self.stride_slice_2(position_ids, (0, 0), (1, C.MAX_FULL_TEXT_LEN), (1, 1))

        # 获取两个模态的embedding
        image_embed = self.get_img_feat(images)
        text_embed = self.get_txt_feat(input_ids, position_ids, attn_masks_text)
        # 融合生成, bs * seq * hidden
        sequence_output, _ = self.uniter.feat_fusion(text_embed, image_embed, None, attn_masks_text) 
        # 复用MLM进行生成 bs * seq * vocab
        prediction_scores = self.cls(sequence_output)
        # 计算loss 
        loss = self.seq_loss(prediction_scores, txt_gts, attn_masks_text)

        return loss

class UniterTwoForCaptionForEval(UniterTwoForPretrainingWithLoss):
    def __init__(self, config, args=None):
        
        full_batch = args.full_batch
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.use_pipeline = args.use_pipeline
        config.batch_size = args.val_batch_size * args.beam_width
        config.seq_length = C.MAX_IMG_TEXT_LEN
        config.patch_size = C.IMG_PATCH_SIZE
        config.train_image_size = C.IMG_SIZE

        super().__init__(config, args)

        parallel_config = ParallelConfig()

        # 无需修改 mask, 可以使用已生成的结果
        # self.uniter.embeddings.attention_mask = BertAttentionCausalMask(parallel_config)
        # self.uniter.encoder.attention_mask = BertAttentionCausalMask(parallel_config)

        self.tile_beam = TileBeam(beam_width=args.beam_width)
        self.log_softmax = ops.LogSoftmax().shard(((parallel_config.dp, 1),))
        self.ones_like = P.OnesLike()
        self.zeros = P.Zeros()
        self.cat_mask = P.Concat(axis=-1)
        self.cat_inp = P.Concat(axis=-1)
        self.pos_full = Tensor(np.expand_dims(np.arange(0, C.MAX_FULL_TEXT_LEN, dtype=np.int64), 0))

        self.beam_width = args.beam_width
        self.batch_size = args.val_batch_size

        self.generator = BeamSearchDecoder(
            batch_size=self.batch_size,
            seq_length= None,
            vocab_size=config.vocab_size,
            decoder=self.generate_step,
            beam_width=args.beam_width,
            max_decode_length = C.MAX_FULL_TEXT_LEN,
            sos_id=0, 
            eos_id=0)
        
    # def set_clip_loss(self, config, parallel_config):
    #     pass
    
    def generate_step(self, input_ids, image_embed, image_mask=None):
        '''
            input_ids : batch*beam, seq
            image_embed : batch*beam, seq, dim
        '''
        seq_len = input_ids.shape[1]
        txt_masks = self.ones_like(input_ids)
        zero_masks = self.zeros((input_ids.shape[0],C.MAX_FULL_TEXT_LEN),mstype.int32)
        txt_masks = self.cat_mask((txt_masks,zero_masks))
        txt_masks = txt_masks[::, :C.MAX_FULL_TEXT_LEN]
        input_ids = self.cat_inp((input_ids,zero_masks))
        input_ids = input_ids[::, :C.MAX_FULL_TEXT_LEN]

        text_embed = self.get_txt_feat(input_ids, self.pos_full, txt_masks)

        # 融合生成, [batch*beam, max_seq, hidden]
        sequence_output, _ = self.uniter.feat_fusion(text_embed, image_embed, None, txt_masks)
        # 生成token分数
        prediction_scores = self.cls(sequence_output)
        prediction_scores = prediction_scores[::, seq_len-1, ::]
        log_probs = self.log_softmax(prediction_scores)
        return log_probs

    def construct(self, images):

        # 获取图像模态的embedding
        image_embed = self.get_img_feat(images)
        tile_image_embed = self.tile_beam(image_embed)
        # 生成文本
        predict_ids = self.generator(tile_image_embed, None)

        return predict_ids