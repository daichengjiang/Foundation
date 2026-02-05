#!/bin/bash

echo "=================================================="
echo "🚀 Starting Automated Data Collection (Randomized)"
echo "Target: Collecting data until stopped..."
echo "=================================================="

success_count=0
fail_count=0

while true; do
    # 生成一个随机种子
    SEED=$RANDOM
    
    echo ""
    echo "[$(date '+%H:%M:%S')] Starting new iteration with SEED: $SEED"
    
    # 核心修改：在这里动态拼接 --seed 参数
    python generate_pid_dataset.py --task teacher --headless --use_pid --seed $SEED
    
    if [ $? -eq 0 ]; then
        echo "✅ Iteration finished successfully."
        ((success_count++))
    else
        echo "⚠️ Iteration crashed or failed."
        ((fail_count++))
        sleep 2
    fi
    
    echo "--------------------------------------------------"
    echo "Session Stats: Completed: $success_count | Crashes: $fail_count"
    echo "--------------------------------------------------"
    
    sleep 1
done