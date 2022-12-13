"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc utilities
"""
import json
import sys
import random
import numpy as np

class NoOp:
    """ useful for distributed training No-Ops """

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def parse_with_config(parser):
    """Parse With Config"""
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def set_random_seed(seed):
    """Set Random Seed"""
    random.seed(seed)
    np.random.seed(seed)


class Struct:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
