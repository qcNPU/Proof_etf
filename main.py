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
    # parser.add_argument('--config', type=str, default='./exps/cifar224.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/ncscmp.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/imgnet100.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/ncscmp.json', help='Json file of settings.')
    parser.add_argument('--config', type=str, default='./exps/cifar.json', help='Json file of settings.')
    parser.add_argument('--setting', type=str, default='proofncscmp', help='Json file of settings.')
    # parser.add_argument('--setting', type=str, default='proof', help='Json file of settings.')
    parser.add_argument('--gen_proto_mode', type=str, default='kmeans++', help='Json file of settings.')
    parser.add_argument('--init_cls', type=int, default=10, help='Json file of settings.')
    parser.add_argument('--increment', type=int, default=10, help='Json file of settings.')
    parser.add_argument('--proto_num', type=int, default=5, help='Json file of settings.')
    parser.add_argument('--ncloss', type=int, default=5, help='Json file of settings.')
    parser.add_argument('--scmploss', type=int, default=3, help='Json file of settings.')
    parser.add_argument('--tuned_epoch', type=int, default=10, help='Json file of settings.')
    parser.add_argument('--train_templates', type=str, default='one',choices=['one','all'], help='Json file of settings.')
    parser.add_argument('--convnet_type', type=str, default='clip', help='Json file of settings.')
    parser.add_argument('--target_choose', type=str, default='reselect',choices=['reselect','fix'], help='Json file of settings.')
    parser.add_argument('--target_match', type=str, default='cosine',choices=['cosine','random'], help='Json file of settings.')
    parser.add_argument('--text_optimize', type=str, default='loop',choices=['loop','optimize'], help='Json file of settings.')
    parser.add_argument('--save_proto', type=bool, default=False, help='Json file of settings.')
    parser.add_argument('--optimize_feat', type=str, default='textimage',choices=['textimage','text','image'], help='Json file of settings.')
    parser.add_argument('--proto_select', type=str, default='max',choices=['max','maxmin','minmax', 'min'], help='Json file of settings.')
    parser.add_argument('--prediction', type=str, default='pmvmtm',choices=['pmvmtm','pm','vm', 'tm','pmvm','pmtm','vmtm'], help='Json file of settings.')
    parser.add_argument('--lossteam', type=str, default='pmtmncscmp',choices=['pmtmncscmp', 'pm','tm','nc','scmp', 'pmtm','pmnc','pmscmp',
                                                                          'tmnc','tmscmp','ncscmp','pmtmnc', 'pmtmscmp','pmncscmp','tmncscmp'], help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/ncscmp.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/cub.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/car.json', help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='./exps/air.json', help='Json file of settings.')
    return parser

import numpy as np
def getmeanstd():
    avg_results =[88.232, 87.77799999999999, 88.225]
    # avg_results = [avg_results[i] for i in [0,1,3]]
    # avg_results = [avg_results[i] for i in [1,2,8]]
    A_mean,A_std = np.mean(avg_results),np.std(avg_results)
    print(f"{A_mean:.2f}±{A_std:.2f}")
    # print(f"{A_mean}±{A_std}")
    last_results =[77.03, 76.72, 76.65]
    # last_results = [last_results[i] for i in [0,1,3]]
    # last_results = [last_results[i] for i in [1,2,8]]
    ABmean,ABstd = np.mean(last_results),np.std(last_results)
    # print(f"{ABmean}±{ABstd}")
    print(f"{ABmean:.2f}±{ABstd:.2f}")


if __name__ == '__main__':
    main()
    # getmeanstd()
    # getSub()