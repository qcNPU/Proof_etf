import sys
import logging
import copy

import numpy as np
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
logging.basicConfig(level=logging.INFO)


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    A_results = []
    AB_results = []
    for_results = []
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        A_,AB,avg_forgetting = _train(args)
        A_results.append(A_)
        AB_results.append(AB)
        for_results.append(avg_forgetting)
    A_mean,A_std = np.mean(A_results),np.std(A_results)
    ABmean,ABstd = np.mean(AB_results),np.std(AB_results)
    formean,forstd = np.mean(for_results),np.std(for_results)

    print(f"A_:{A_mean:.2f}±{A_std:.2f},results={A_results},\n"
          f"AB:{ABmean:.2f}±{ABstd:.2f},results={AB_results},\n"
          f"forget:{formean:.2f}±{forstd:.2f}, results={for_results}")


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_{}".format(args["model_name"], args["dataset"],
        init_cls, args["increment"], args["prefix"], args["seed"],args["convnet_type"],args["setting"])
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    # _set_random(1)#默认就是1 1无效
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args["dataset"],args["shuffle"],args["seed"],args["init_cls"],args["increment"], )
    model = factory.get_model(args["model_name"], args)
    model.save_dir=logs_name

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    zs_seen_curve, zs_unseen_curve, zs_harmonic_curve, zs_total_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}
    acc_history=[]
    for task in range(data_manager.nb_tasks):
        print("All params: {}".format(count_parameters(model._network)))
        print(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        acc,task_acc = model.incremental_train(data_manager)  #acc是总正确/总数取2位小数；task_acc是每个task训练完后在所有seen task上的准确率
        acc_history.append(task_acc)
        # cnn_accy, nme_accy = model.eval_task()
        # cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = model.eval_task()
        model.after_task()

       
        # print("CNN: {}".format(cnn_accy["grouped"]))
        #
        cnn_curve["top1"].append(acc)
        print("pertask acc_history:", acc_history)
        print("afterTask mean:",cnn_curve["top1"],", final avg:",sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        # cnn_curve["top1"].append(cnn_accy["top1"])
        # cnn_curve["top5"].append(cnn_accy["top5"])
        #
        # print("CNN top1 curve: {}".format(cnn_curve["top1"]))
        # print("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

        # print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        # print("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    # forgetting_rates, avg_forgetting = compute_forgetting_rates(acc_history)
    # print("final forgetting pertask:",forgetting_rates,",avg_forgetting",avg_forgetting)
    if args["save_proto"] and args["dataset"] != 'imagenetr':
        model.get_last_proto()

    return np.mean(cnn_curve["top1"]),cnn_curve["top1"][-1],0


def compute_forgetting_rates(acc_history):
    """
    根据 acc_history 计算各任务的遗忘率及平均遗忘率。

    参数：
        acc_history: list of lists，其中 acc_history[j] 是在学习完第 j 个任务后，
                     得到的各个任务的准确率列表。例如：
                     acc_history[0] = [acc(T0)]
                     acc_history[1] = [acc(T0), acc(T1)]
                     ...
    返回：
        forgetting_rates: list，其中每个元素为任务 i 的遗忘率（i = 0,..., N-2）
        avg_forgetting: 平均遗忘率
    """
    N = len(acc_history)
    forgetting_rates = []

    # 对于每个任务 T_i，从其出现的阶段 i 到最终阶段 N-1，计算最佳准确率与最终准确率之差
    for i in range(N - 1):
        # 收集任务 i 在阶段 i 到阶段 N-1 的准确率
        task_accs = [acc_history[j][i] for j in range(i, N)]
        best_acc = max(task_accs)
        final_acc = acc_history[-1][i]
        forgetting = best_acc - final_acc
        forgetting_rates.append(forgetting)

    avg_forgetting = sum(forgetting_rates) / len(forgetting_rates) if forgetting_rates else 0.0
    return forgetting_rates, avg_forgetting


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        print("{}: {}".format(key, value))
