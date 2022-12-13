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

import mindspore
from src.model_mindspore.pretrain_two_ms import UniterTwoForPretrainingWithLoss

class UniterThreeForRet(UniterTwoForPretrainingWithLoss):
    """ UniterThreeForRet """

    def __init__(self, config, args):
        super().__init__(config, args)

    def get_txt_emb_and_feat(self, input_ids, position_ids, attn_masks_text):
        txt_emb = self.get_txt_feat(input_ids, position_ids, attn_masks_text)
        feat_text = self.uniter.get_first_token(txt_emb)
        feat_text = self.text_proj(feat_text)
        feat_text = self.normalize(feat_text)
        return txt_emb, feat_text

    def get_img_emb_and_feat(self, images):
        img_emb = self.get_img_feat(images)
        feat_img = self.uniter.get_first_token(img_emb)
        feat_img = self.vision_proj(feat_img)
        feat_img = self.normalize(feat_img)
        return img_emb, feat_img

    def construct(self, input_ids, position_ids, attn_masks_text, images):

        img_emb = self.get_img_feat(images)
        txt_emb = self.get_txt_feat(input_ids, position_ids, attn_masks_text)
        feat_text, feat_img = self.get_text_image_feat(txt_emb, img_emb)

        return txt_emb, feat_text, attn_masks_text, img_emb, feat_img


class UniterThreeForITM(UniterTwoForPretrainingWithLoss):
    """ UNITER pretraining """

    def __init__(self, config, args):
        super().__init__(config, args)

    def construct(self, text_embed, image_embed, attention_mask, att_text):

        sequence_output, moe_loss = self.uniter.feat_fusion(text_embed, image_embed, attention_mask, att_text)
        pooled_output = self.uniter.get_first_token(sequence_output)
        itm_scores = self.itm_head_two(pooled_output)
        itm_scores = itm_scores.astype(mindspore.float32)

        return itm_scores

class UniterThreeForCnClip(UniterTwoForPretrainingWithLoss):
    """ UNITER pretraining """

    def __init__(self, config, args):
        super().__init__(config, args)

    def get_txt_emb_and_feat(self, input_ids, attn_masks_text):
        txt_emb = self.uniter.embeddings(input_ids, attn_masks_text)
        txt_emb = self.uniter.get_first_token(txt_emb)
        txt_emb = self.text_proj(txt_emb)
        txt_emb = self.normalize(txt_emb)
        return txt_emb

    def get_img_emb_and_feat(self, images):
        img_emb = self.uniter.img_embeddings.get_feat(images)
        img_emb = self.uniter.get_first_token(img_emb)
        img_emb = self.vision_proj(img_emb)
        img_emb = self.normalize(img_emb)
        return img_emb

    def construct(self, input_ids, attn_masks_text, images):

        txt_emb = self.get_txt_emb_and_feat(input_ids, attn_masks_text)
        img_emb = self.get_img_emb_and_feat(images)

        return txt_emb, img_emb