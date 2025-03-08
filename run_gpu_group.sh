#!/bin/bash
# 子脚本：在指定 GPU 上运行一组任务，组内最多并行 3 个任务

if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_id> <task1> <task2> ..."
    exit 1
fi

GPU_ID=$1
shift
TASKS=("$@")
MAX_JOBS=3  # 每个 GPU 最大并行任务数

# 遍历任务列表，动态控制并发
for task in "${TASKS[@]}"; do
    # 拆分命令和日志文件
    IFS=':' read -r CMD LOGFILE <<< "$task"

    # 执行任务（绑定到指定 GPU）
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup bash -c "$CMD" > "$LOGFILE" 2>&1 &

    # 控制并发：如果当前任务数 >= MAX_JOBS，等待直到有空闲
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
done

# 等待组内所有任务完成
wait
echo "GPU $GPU_ID 的任务组已完成"