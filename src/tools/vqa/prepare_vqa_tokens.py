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
from transformers import BertTokenizer
import json

with open('dataset/vqa/txt/FM-IQA.json','r') as f:
    data = json.load(f)

train_data = data["train"]
val_data = data["val"]

val_token_ids = {}
train_token_ids = {}

tokenizer = BertTokenizer.from_pretrained("dataset/vqa/bert-base-chinese-vocab.txt")

# 提取训练集数据的token
for data in train_data:

    txt_dump = {}

    question_tokens = tokenizer.tokenize(data['Question'])
    print(question_tokens)
    input_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    txt_dump['input_ids'] = input_ids

    answer_tokens = tokenizer.tokenize(data['Answer'])
    print(answer_tokens)
    answer_ids = tokenizer.convert_tokens_to_ids(answer_tokens)
    txt_dump['answer_ids'] = answer_ids

    train_token_ids[data['question_id']] = txt_dump

# 提取验证集数据的token
for data in val_data:

    txt_dump = {}

    question_tokens = tokenizer.tokenize(data['Question'])
    print(question_tokens)
    input_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    txt_dump['input_ids'] = input_ids

    answer_tokens = tokenizer.tokenize(data['Answer'])
    print(answer_tokens)
    answer_ids = tokenizer.convert_tokens_to_ids(answer_tokens)
    txt_dump['answer_ids'] = answer_ids

    val_token_ids[data['question_id']] = txt_dump
   

with open('val_token_ids.json','w') as f:
    json.dump(val_token_ids,f)

with open('train_token_ids.json','w') as f:
    json.dump(train_token_ids,f)
