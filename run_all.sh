#!/bin/bash

# 主脚本：按顺序将任务分配到 4 个 GPU 组，每组最多并行 3 个任务

# 定义所有任务（格式：命令:日志文件）
ALL_TASKS=(
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pm:result/cifar_proofncscmp_one_pm.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tm:result/cifar_proofncscmp_one_tm.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam nc:result/cifar_proofncscmp_one_nc.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam scmp:result/cifar_proofncscmp_one_scmp.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmtm:result/cifar_proofncscmp_one_pmtm.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmnc:result/cifar_proofncscmp_one_pmnc.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmscmp:result/cifar_proofncscmp_one_pmscmp.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tmnc:result/cifar_proofncscmp_one_tmnc.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tmscmp:result/cifar_proofncscmp_one_tmscmp.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam ncscmp:result/cifar_proofncscmp_one_ncscmp.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmtmnc:result/cifar_proofncscmp_one_pmtmnc.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmtmscmp:result/cifar_proofncscmp_one_pmtmscmp.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmncscmp:result/cifar_proofncscmp_one_pmncscmp.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tmncscmp:result/cifar_proofncscmp_one_tmncscmp.log"
    #
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pm:result/imgnetsub_proofncscmp_one_pm.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tm:result/imgnetsub_proofncscmp_one_tm.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam nc:result/imgnetsub_proofncscmp_one_nc.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam scmp:result/imgnetsub_proofncscmp_one_scmp.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmtm:result/imgnetsub_proofncscmp_one_pmtm.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmnc:result/imgnetsub_proofncscmp_one_pmnc.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmscmp:result/imgnetsub_proofncscmp_one_pmscmp.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tmnc:result/imgnetsub_proofncscmp_one_tmnc.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tmscmp:result/imgnetsub_proofncscmp_one_tmscmp.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam ncscmp:result/imgnetsub_proofncscmp_one_ncscmp.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmtmnc:result/imgnetsub_proofncscmp_one_pmtmnc.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmtmscmp:result/imgnetsub_proofncscmp_one_pmtmscmp.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam pmncscmp:result/imgnetsub_proofncscmp_one_pmncscmp.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 10 --increment 10 --lossteam tmncscmp:result/imgnetsub_proofncscmp_one_tmncscmp.log"
    #
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam pm:result/tinyimagenet_proofncscmp_one_pm.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam tm:result/tinyimagenet_proofncscmp_one_tm.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam nc:result/tinyimagenet_proofncscmp_one_nc.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam scmp:result/tinyimagenet_proofncscmp_one_scmp.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam pmtm:result/tinyimagenet_proofncscmp_one_pmtm.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam pmnc:result/tinyimagenet_proofncscmp_one_pmnc.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam pmscmp:result/tinyimagenet_proofncscmp_one_pmscmp.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam tmnc:result/tinyimagenet_proofncscmp_one_tmnc.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam tmscmp:result/tinyimagenet_proofncscmp_one_tmscmp.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam ncscmp:result/tinyimagenet_proofncscmp_one_ncscmp.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam pmtmnc:result/tinyimagenet_proofncscmp_one_pmtmnc.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam pmtmscmp:result/tinyimagenet_proofncscmp_one_pmtmscmp.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam pmncscmp:result/tinyimagenet_proofncscmp_one_pmncscmp.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 20 --increment 20 --lossteam tmncscmp:result/tinyimagenet_proofncscmp_one_tmncscmp.log"
)

# 初始化 4 个 GPU 组的任务列表
declare -a GPU0_TASKS GPU1_TASKS GPU2_TASKS GPU3_TASKS

# 按顺序将任务分配到 4 个 GPU 组（轮询分配）
for idx in "${!ALL_TASKS[@]}"; do
    gpu_id=$(((idx+0) % 3))
    case $gpu_id in
        0) GPU0_TASKS+=("${ALL_TASKS[$idx]}") ;;
        1) GPU1_TASKS+=("${ALL_TASKS[$idx]}") ;;
        2) GPU2_TASKS+=("${ALL_TASKS[$idx]}") ;;
#        3) GPU3_TASKS+=("${ALL_TASKS[$idx]}") ;;
    esac
done

# 启动所有 GPU 组的任务
./run_gpu_group.sh 0 "${GPU0_TASKS[@]}" &
./run_gpu_group.sh 1 "${GPU1_TASKS[@]}" &
./run_gpu_group.sh 2 "${GPU2_TASKS[@]}" &
#./run_gpu_group.sh 3 "${GPU3_TASKS[@]}" &

# 等待所有 GPU 组完成
wait
echo "所有任务已完成"