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
""" config """

USE_LARGE_DATA = False

IMG_SIZE = 224
IMG_PATCH_SIZE = 32
MAX_IMG_LEN = (IMG_SIZE//IMG_PATCH_SIZE)**2 + 1

MAX_TEXT_LEN = 48
MAX_FULL_TEXT_LEN = 50

MAX_DEFAULT_LEN = 50
MAX_AUDIO_LEN = 50

MAX_IMG_TEXT_LEN = MAX_FULL_TEXT_LEN + MAX_IMG_LEN
MAX_FULL_LEN = MAX_FULL_TEXT_LEN + MAX_IMG_LEN + MAX_AUDIO_LEN

IMG_TOKEN_SIZE = 8192
IMG_TOKEN_LEN = 64

MAX_TEXT_GTS_LEN = 29
MAX_IMG_GTS_LEN = 63

MAX_MEL_LEN = 1289
MAX_SRC_LEN = 89

IMG_FEAT_DIM = 3072
IMG_DIM = 768
IMG_LABEL_DIM = 1601
BUCKET_SIZE = 8192

AUDIO_DIM = 1024
AUDIO_LABEL_DIM = 1600

MASK_SIZE=2
N_NEGATIVES=10