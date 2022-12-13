"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
from os.path import exists, join

def make_dir_safe(dirs):
    if not exists(dirs):
        os.makedirs(dirs, exist_ok=True)


def save_training_meta(args):
    """save training meta"""
    if args.rank > 0:
        return

    make_dir_safe(args.output_dir)
    make_dir_safe(join(args.output_dir, 'log'))
    make_dir_safe(join(args.output_dir, 'ckpt'))

    with open(join(args.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    model_config = json.load(open(args.model_config))
    with open(join(args.output_dir, 'log', 'model.json'), 'w') as writer:
        json.dump(model_config, writer, indent=4)
