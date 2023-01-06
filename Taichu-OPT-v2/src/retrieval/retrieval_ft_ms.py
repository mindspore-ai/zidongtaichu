import src.config.config as C
import mindspore as ms
from src.model_mindspore.pretrain_two_ms import UniterTwoForPretrainingWithLoss

class UniterTwoForRetItcItmWithLoss(UniterTwoForPretrainingWithLoss):

    def __init__(self, config, args=None):
        super().__init__(config, args)

    def construct(self, input_ids, position_ids, attention_mask, txt_mask, txt_label_mask, itm_target,
                    attn_masks_text, attn_masks_img, images, images_rand, input_ids_mask,
                    txt_gts, txt_gts_mask, taskId):
        """Construct Function"""
        if not self.full_batch:
            taskId = self.stride_slice_1(taskId, (0,), (1,), (1,))
            position_ids = self.stride_slice_2(position_ids, (0, 0), (1, C.MAX_FULL_TEXT_LEN), (1, 1))

        image_embed = self.get_img_feat(images)
        text_embed = self.get_txt_feat(input_ids, position_ids, attn_masks_text)
        pool_feat_text, pool_feat_image = self.get_text_image_feat(text_embed, image_embed)

        itc_loss = self.clip_loss(pool_feat_text, pool_feat_image)

        itm_loss = self.forward_itmHard_local_three(input_ids, position_ids, images,
                                              attn_masks_text, attn_masks_img,
                                              attention_mask,
                                              image_embed, text_embed,
                                              pool_feat_text, pool_feat_image)

        loss = self.concat((itc_loss.view(1, ), (itm_loss).view(1, )))
        final_loss = self.reduce_sum(loss)

        return final_loss

class UniterTwoForRetItcExport(UniterTwoForPretrainingWithLoss):
    def __init__(self, config, args=None):
        super().__init__(config, args)

    def construct(self, input_ids, position_ids, attn_masks_text, images):
        """Construct Function"""
        if not self.full_batch:
            position_ids = self.stride_slice_2(position_ids, (0, 0), (1, C.MAX_FULL_TEXT_LEN), (1, 1))

        image_embed = self.get_img_feat(images)
        text_embed = self.get_txt_feat(input_ids, position_ids, attn_masks_text)

        return image_embed, text_embed

class UniterTwoForRetItmExport(UniterTwoForPretrainingWithLoss):

    def __init__(self, config, args=None):
        super().__init__(config, args)

    def construct(self, image_embed, text_embed, attention_mask, attn_masks_text):
        """Construct Function"""

        sequence_output, _ = self.uniter.feat_fusion(text_embed, image_embed, attention_mask, attn_masks_text)
        pooled_output = self.uniter.get_first_token(sequence_output)
        itm_scores = self.itm_head_two(pooled_output)
        itm_scores = itm_scores.astype(ms.float32)

        return itm_scores