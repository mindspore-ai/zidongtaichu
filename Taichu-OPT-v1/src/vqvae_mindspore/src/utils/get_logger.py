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

import sys
import logging
from mindspore import context
from mindspore.communication import init, get_rank, get_group_size

class Logger():
    def __init__(self, logging_file, name, isopen, ifstdout=True):
        if isopen is True:
            # create a logger
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            
            if ifstdout is True:
                handler1 = logging.StreamHandler(sys.stdout)
                handler1.setLevel(logging.DEBUG)
                formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
                handler1.setFormatter(formatter)
                logger.addHandler(handler1)

            handler2 = logging.FileHandler(logging_file)
            handler2.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
            handler2.setFormatter(formatter)
            logger.addHandler(handler2)

            self.logger = logger
        self.isopen = isopen

    def info(self, message):
        if self.isopen:
            self.logger.info(message)

    def warning(self, message):
        if self.isopen:
            self.logger.warning(message)


def get_logger(logging_file=None, name=None, ifstdout=True):
    if not hasattr(get_logger, 'Logger'):
        assert logging_file is not None, "Logger hasn't been created."
        get_logger.Logger = Logger(logging_file, name, True, ifstdout=ifstdout)
    return get_logger.Logger

def get_logger_dist(logging_file=None, name=None, ifstdout=True):
    if not hasattr(get_logger, 'Logger'):
        assert logging_file is not None, "Logger hasn't been created."
        if get_rank() == 0:
            get_logger.Logger = Logger(logging_file, name, True, ifstdout=ifstdout)
        else:
            get_logger.Logger = Logger(None, None, False, ifstdout=False)
    return get_logger.Logger