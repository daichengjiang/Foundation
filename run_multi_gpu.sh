#!/bin/bash

# ==================== 配置区域 ====================
TASKS=(
    "0   raptor0"
    "0   raptor1"
    "0   raptor2"
    "0   raptor3"
    "0   raptor4"
    "0   raptor5"
    "1   raptor6"
    "1   raptor7"
    "1   raptor8"
    "1   raptor9"
    "1   raptor10"
    "1   raptor11"
    "2   raptor12"
    "2   raptor13"
    "2   raptor14"
    "2   raptor15"
    "2   raptor16"
    "2   raptor17"
    "3   raptor18"
    "3   raptor19"
    "3   raptor20"
    "3   raptor21"
    "3   raptor22"
    "3   raptor23"
    "4   raptor24"
    "4   raptor25"
    "4   raptor26"
    "4   raptor27"
    "4   raptor28"
    "4   raptor29"
    "5   raptor30"
    "5   raptor31"
    "5   raptor32"
    "5   raptor33"
    "5   raptor34"
    "5   raptor35"
)

NUM_TEACHERS=1000
START_ID=0
HEADLESS=true

# 创建一个专门存放日志的文件夹
LOG_DIR="logs_training"
mkdir -p "$LOG_DIR"
# ====================================================

echo "=================================================="
echo "Starting Multi-Task Training Manager (Clean Output)"
echo "Total Tasks: ${#TASKS[@]}"
echo "Logs are being saved to: ./$LOG_DIR/"
echo "=================================================="

for task in "${TASKS[@]}"; do
    read -r gpu_id timestamp <<< "$task"
    
    # 为每个任务定义一个独立的日志文件名，例如 logs_training/raptor0_gpu0.log
    LOG_FILE="$LOG_DIR/${timestamp}_gpu${gpu_id}.log"
    
    echo "[Launcher] Launching -> GPU: $gpu_id | Timestamp: $timestamp | Log: $LOG_FILE"
    
    # 使用 > $LOG_FILE 2>&1 将该任务的所有输出定向到文件，末尾加 & 放后台
    python foundation/rsl_rl/train_teacher_c5.py \
        --start_id "$START_ID" \
        --num_teachers "$NUM_TEACHERS" \
        --gpu_id "$gpu_id" \
        --timestamp "$timestamp" \
        ${HEADLESS:+--headless} > "$LOG_FILE" 2>&1 &
    
    sleep 5
done

echo "=================================================="
echo "All tasks dispatched cleanly to background!"
echo "Your terminal is now free. Use 'tail -f logs_training/raptor0_gpu0.log' to monitor a specific task."
echo "=================================================="

wait
echo "All training tasks completed successfully!"

# 看终端日志
# tail -f logs_training/raptor0_gpu0.log

# 杀死所有进程
# pkill -9 -f train_teacher

# 复制日志到本地
# scp -P 52647 -r "ai.user@222.173.29.146:/home/ai.user/dcj/Foundation/logs/rsl_rl/multi_teachers/raptor*" /home/nv/Foundation/logs/rsl_rl/multi_teachers/
