import src.config.config as C
import copy
import numpy as np

from src.model_mindspore.model_ms import freeze_net
from src.model_mindspore.pretrain_two_ms import UniterTwoForPretrainingWithLoss
from src.config.model_config import UniterConfig

from src.config.transformer_config import transformer_net_cfg as cfg
from src.model_mindspore.transformer_model import TransformerModel
from src.model_mindspore.loss import TransformerTrainingLoss
from src.model_mindspore.parallel_transformer import ParallelConfig

from mindspore.ops import operations as P

class UniterTwoForVqaWithLoss(UniterTwoForPretrainingWithLoss):

    def __init__(self, config, args=None):
        super().__init__(config, args)

        freeze_net(self.vision_proj)
        freeze_net(self.text_proj)
        freeze_net(self.clip_loss)
        freeze_net(self.itm_head_two)

        config = UniterConfig.from_json_file(config)

        parallel_config = ParallelConfig()
        fusion_group_backbone = parallel_config.fusion_group
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group

        self.txt_cfg = copy.deepcopy(cfg)
        self.txt_cfg.vocab_size = config.vocab_size
        self.txt_cfg.batch_size = args.train_batch_size
        self.txt_cfg.seq_length = C.MAX_FULL_TEXT_LEN

        self.txt_output = TransformerModel(self.txt_cfg, True, config.hidden_size, parallel_config, fg_backbone, False)
        self.td_crit = TransformerTrainingLoss(self.txt_cfg, parallel_config)

    def construct(self, input_ids, position_ids, attention_mask, txt_mask, txt_label_mask, itm_target,
                    attn_masks_text, attn_masks_img, images, images_rand, input_ids_mask,
                    txt_gts, txt_gts_mask, taskId):
        """
            Construct Function
            Inputs:
                inputs_ids:     txt ids
                position_ids:   txt pos ids
                txt_gts:        txt ground truth
                txt_gts_mask:   gts's mask
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

        loss = self.generate_text(sequence_output, attn_masks_text, txt_gts, txt_gts_mask)

        return loss

    def generate_text(self, sequence_output, att_masks, txt_gts, txt_gts_mask):
        # generate text
        txt_out = self.txt_output(sequence_output, att_masks, txt_gts, txt_gts_mask)
        loss = self.td_crit(txt_out, txt_gts, txt_gts_mask)
        return loss


class UniterTwoForVqaForEval(UniterTwoForPretrainingWithLoss):

    def __init__(self, config, args=None):

        full_batch = args.full_batch
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.use_pipeline = args.use_pipeline
        config.batch_size = args.val_batch_size
        config.seq_length = C.MAX_IMG_TEXT_LEN
        config.patch_size = C.IMG_PATCH_SIZE
        config.train_image_size = C.IMG_SIZE

        super().__init__(config, args)

        # super().__init__(config, args)

        # config = UniterConfig.from_json_file(config)
        # config.full_batch = full_batch
        # config.use_pipeline = args.use_pipeline
        # config.batch_size = args.val_batch_size
        # config.seq_length = C.MAX_TEXT_LEN


        parallel_config = ParallelConfig()
        fusion_group_backbone = parallel_config.fusion_group
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        
        self.txt_cfg = copy.deepcopy(cfg)
        self.txt_cfg.batch_size = args.val_batch_size
        self.txt_cfg.beam_width = args.beam_width
        self.txt_cfg.vocab_size = config.vocab_size
        self.txt_cfg.seq_length = C.MAX_FULL_TEXT_LEN

        self.txt_output = TransformerModel(self.txt_cfg, False, config.hidden_size, parallel_config, fg_backbone, False)
        
    def set_clip_loss(self, config, parallel_config):
        pass

    def construct(self, input_ids, position_ids, attn_masks_text, images):
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
            position_ids = self.stride_slice_2(position_ids, (0, 0), (1, C.MAX_FULL_TEXT_LEN), (1, 1))

        # 获取两个模态的embedding

        image_embed = self.get_img_feat(images)
        text_embed = self.get_txt_feat(input_ids, position_ids, attn_masks_text)

        sequence_output, moe_loss = self.uniter.feat_fusion(text_embed, image_embed, None, attn_masks_text)
        
        output = self.generate_text_eval(sequence_output, attn_masks_text)

        return output

    def generate_text_eval(self, sequence_output, att_masks):
        txt_out = self.txt_output(sequence_output, att_masks)
        return txt_out