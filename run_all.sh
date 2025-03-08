#!/bin/bash

# 主脚本：按顺序将任务分配到 4 个 GPU 组，每组最多并行 3 个任务

# 定义所有任务（格式：命令:日志文件）
ALL_TASKS=(
    # 1. 4个数据集的component ablation
    "python -u main.py --config ./exps/cifar.json --setting proofnc:result/cifar_nc_553_102e-3_all.log"
    "python -u main.py --config ./exps/cifar.json --setting proofscmp:result/cifar_scmp_553_102e-3_all.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp:result/cifar_ncscmp_553_102e-3_all.log"

    "python -u main.py --config ./exps/imgnetr.json --setting proofnc:result/imgnetr_nc_553_102e-3_all.log"
    "python -u main.py --config ./exps/imgnetr.json --setting proofscmp:result/imgnetr_scmp_553_102e-3_all.log"
    "python -u main.py --config ./exps/imgnetr.json --setting proofncscmp:result/imgnetr_ncscmp_553_102e-3_all.log"

    "python -u main.py --config ./exps/imgnetsub.json --setting proofnc:result/imgnetsub_nc_553_101e-3_all.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofscmp:result/imgnetsub_scmp_553_101e-3_all.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp:result/imgnetsub_ncscmp_553_101e-3_all.log"

    "python -u main.py --config ./exps/tinyimagenet.json --setting proofnc:result/tinyimagenet_nc_553_101e-3_all.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofscmp:result/tinyimagenet_scmp_553_101e-3_all.log"
    "python -u main.py --config ./exps/tinyimagenet.json --setting proofncscmp:result/tinyimagenet_ncscmp_553_101e-3_all.log"

    "python -u main.py --config ./exps/food.json --setting proofnc:result/food_nc_553_101e-3_all.log"
    "python -u main.py --config ./exps/food.json --setting proofscmp:result/food_scmp_553_101e-3_all.log"
    "python -u main.py --config ./exps/food.json --setting proofncscmp:result/food_ncscmp_553_101e-3_all.log"
    # 2. cifar和imagenetsub的 nc scmp 的loss系数、prototypeNum 消融   todo 第一部分用all跑完之后，后面的都用one跑,最好的就是上面跑的
    # 2.1 nc loss
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --ncloss 1 --train_templates one:result/cifar_ncscmp_5-nc1-3_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --ncloss 3 --train_templates one:result/cifar_ncscmp_5-nc3-3_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --ncloss 4 --train_templates one:result/cifar_ncscmp_5-nc4-3_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --ncloss 6 --train_templates one:result/cifar_ncscmp_5-nc6-3_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --ncloss 7 --train_templates one:result/cifar_ncscmp_5-nc7-3_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --ncloss 10 --train_templates one:result/cifar_ncscmp_5-nc10-3_102e-3_one.log"

    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --ncloss 1 --train_templates one:result/imgnetsub_ncscmp_5-nc1-3_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --ncloss 3 --train_templates one:result/imgnetsub_ncscmp_5-nc3-3_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --ncloss 4 --train_templates one:result/imgnetsub_ncscmp_5-nc4-3_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --ncloss 6 --train_templates one:result/imgnetsub_ncscmp_5-nc6-3_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --ncloss 7 --train_templates one:result/imgnetsub_ncscmp_5-nc7-3_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --ncloss 10 --train_templates one:result/imgnetsub_ncscmp_5-nc10-3_101e-3_one.log"
    # 2.2. scmp loss
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --scmploss 1 --train_templates one:result/cifar_ncscmp_55-scmp1_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --scmploss 2 --train_templates one:result/cifar_ncscmp_55-scmp2_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --scmploss 4 --train_templates one:result/cifar_ncscmp_55-scmp4_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --scmploss 5 --train_templates one:result/cifar_ncscmp_55-scmp5_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --scmploss 7 --train_templates one:result/cifar_ncscmp_55-scmp7_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --scmploss 10 --train_templates one:result/cifar_ncscmp_55-scmp10_102e-3_one.log"

    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --scmploss 1 --train_templates one:result/imgnetsub_ncscmp_55-scmp1_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --scmploss 2 --train_templates one:result/imgnetsub_ncscmp_55-scmp2_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --scmploss 4 --train_templates one:result/imgnetsub_ncscmp_55-scmp4_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --scmploss 5 --train_templates one:result/imgnetsub_ncscmp_55-scmp5_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --scmploss 7 --train_templates one:result/imgnetsub_ncscmp_55-scmp7_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --scmploss 10 --train_templates one:result/imgnetsub_ncscmp_55-scmp10_101e-3_one.log"
    # 2.3 prototype num
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --proto_num 2 --train_templates one:result/cifar_ncscmp_proto2-53_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --proto_num 3 --train_templates one:result/cifar_ncscmp_proto3-53_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --proto_num 4 --train_templates one:result/cifar_ncscmp_proto4-53_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --proto_num 6 --train_templates one:result/cifar_ncscmp_proto6-53_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --proto_num 7 --train_templates one:result/cifar_ncscmp_proto7-53_102e-3_one.log"
    "python -u main.py --config ./exps/cifar.json --setting proofncscmp --proto_num 10 --train_templates one:result/cifar_ncscmp_proto10-53_102e-3_one.log"

    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --proto_num 2 --train_templates one:result/imgnetsub_ncscmp_proto2-53_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --proto_num 3 --train_templates one:result/imgnetsub_ncscmp_proto3-53_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --proto_num 4 --train_templates one:result/imgnetsub_ncscmp_proto4-53_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --proto_num 6 --train_templates one:result/imgnetsub_ncscmp_proto6-53_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --proto_num 7 --train_templates one:result/imgnetsub_ncscmp_proto7-53_101e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncscmp --proto_num 10 --train_templates one:result/imgnetsub_ncscmp_proto10-53_101e-3_one.log"

    # 单prototype（Mean与cluster对比）
    "python -u main.py --config ./exps/cifar.json --setting proofncsc --train_templates one:result/cifar_ncsc_553_102e-3_one.log"
    "python -u main.py --config ./exps/imgnetsub.json --setting proofncsc --train_templates one:result/imgnetsub_ncsc_553_101e-3_one.log"
)

# 初始化 4 个 GPU 组的任务列表
declare -a GPU0_TASKS GPU1_TASKS GPU2_TASKS GPU3_TASKS

# 按顺序将任务分配到 4 个 GPU 组（轮询分配）
for idx in "${!ALL_TASKS[@]}"; do
    gpu_id=$((idx % 4))
    case $gpu_id in
        0) GPU0_TASKS+=("${ALL_TASKS[$idx]}") ;;
        1) GPU1_TASKS+=("${ALL_TASKS[$idx]}") ;;
        2) GPU2_TASKS+=("${ALL_TASKS[$idx]}") ;;
        3) GPU3_TASKS+=("${ALL_TASKS[$idx]}") ;;
    esac
done

# 启动所有 GPU 组的任务
./run_gpu_group.sh 0 "${GPU0_TASKS[@]}" &
./run_gpu_group.sh 1 "${GPU1_TASKS[@]}" &
./run_gpu_group.sh 2 "${GPU2_TASKS[@]}" &
./run_gpu_group.sh 3 "${GPU3_TASKS[@]}" &

# 等待所有 GPU 组完成
wait
echo "所有任务已完成"