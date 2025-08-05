from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import yaml
from models.SLIP_supervisor import SLIPSupervisor
import torch
import random


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, yaml.FullLoader)
        supervisor = SLIPSupervisor(**supervisor_config)
        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default="./dataset/LA/LA_crime.yaml", type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main(args)
