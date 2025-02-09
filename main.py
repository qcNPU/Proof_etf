import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    # parser.add_argument('--config', type=str, default='./exps/cifar.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/imgnetr.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/imgnet100.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/tinyimagenet.json', help='Json file of settings.')
    parser.add_argument('--config', type=str, default='./exps/food.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/imgnetsub.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/cub.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/car.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/air.json', help='Json file of settings.')
    return parser

import numpy as np
def getmeanstd():
    AB_results =[89.48,89.56,89.14,89.40,89.26,89.34,88.40,88.74,89.70,89.08]
    ABmean,ABstd = np.mean(AB_results),np.std(AB_results)
    print(f"{ABmean:.2f}±{ABstd:.2f}")


if __name__ == '__main__':
    main()
    # getmeanstd()