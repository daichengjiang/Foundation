#!/bin/bash

# ================= 配置区域 =================
# 目标收集数量
TARGET=10000
# CSV 文件名 (必须与 Python 代码默认值或参数一致)
CSV_FILE="teacher_dynamics.csv"
CURRENT_SEED=$RANDOM
# Python 运行命令 (建议加上 --output_csv 确保文件名一致，并加上 --num_envs 提高单次效率)
CMD="python foundation/rsl_rl/sample_teacher.py \
    --task teacher \
    --checkpoint logs/rsl_rl/multi_teachers/2026-01-20_10-14-56/teacher_0000/best_model.pt \
    --headless \
    --output_csv $CSV_FILE \
    --num_envs 20000 \
    --seed $CURRENT_SEED"
# ===========================================

echo "开始循环收集数据，目标: $TARGET 条..."

while true; do
    # 1. 计算当前 CSV 行数
    if [ -f "$CSV_FILE" ]; then
        # wc -l 计算行数，减 1 是因为有标题行(Header)
        # 如果文件存在但为空，处理一下防止报错
        NUM_LINES=$(wc -l < "$CSV_FILE")
        if [ "$NUM_LINES" -gt 0 ]; then
            COUNT=$((NUM_LINES - 1))
        else
            COUNT=0
        fi
    else
        COUNT=0
    fi

    # 2. 显示进度
    echo "--------------------------------------------------"
    echo "当前进度: $COUNT / $TARGET"
    echo "--------------------------------------------------"

    # 3. 检查是否达成目标
    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "✅ 已达成 $TARGET 条数据收集目标！脚本结束。"
        break
    fi

    # 4. 运行 Python 脚本
    echo "启动新一轮仿真..."
    $CMD

    # 5. 休息几秒，让 GPU 显存完全释放，并防止文件读写冲突
    echo "等待显存释放..."
    sleep 3
done