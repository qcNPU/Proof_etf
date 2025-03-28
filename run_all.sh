#!/bin/bash

# 主脚本：按顺序将任务分配到 4 个 GPU 组，每组最多并行 3 个任务

# 定义所有任务（格式：命令:日志文件）
ALL_TASKS=(
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 50 --increment 10:result/cifar_ncscmp_one_50-10.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 5 --increment 5:result/cifar_ncscmp_one_0-5.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --init_cls 20 --increment 20:result/cifar_ncscmp_one_0-20.log"
    #
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 50 --increment 10:result/imgnetsub_ncscmp_one_50-10.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 5 --increment 5:result/imgnetsub_ncscmp_one_0-5.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --init_cls 20 --increment 20:result/imgnetsub_ncscmp_one_0-20.log"
    #
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 40 --increment 40:result/tinyimagenet_ncscmp_one_0-40.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 100 --increment 20:result/tinyimagenet_ncscmp_one_100-20.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp --init_cls 10 --increment 10:result/tinyimagenet_ncscmp_one_0-10.log"
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