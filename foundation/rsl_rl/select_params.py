import os
import sys
import json
import itertools
import subprocess
import pandas as pd
import numpy as np
import time

def main():
    # 获取当前脚本所在的绝对路径目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 train_teacher_single.py 的绝对路径
    train_script_path = os.path.join(current_dir, "train_teacher_single.py")

    # 1. 定义参数空间
    hidden_dims_opts = [[128, 128, 128], [64, 64, 64]]
    entropy_opts = [0.01, 0.002, 0.0002]
    schedule_opts = ["adaptive", "fixed"]
    epochs_opts = [1, 4]

    # 生成所有组合 (共 24 种)
    combinations = list(itertools.product(hidden_dims_opts, entropy_opts, schedule_opts, epochs_opts))
    
    print(f"Total combinations to test: {len(combinations)}")
    print(f"Target Training Script: {train_script_path}")
    
    results = []

    # 2. 循环执行训练
    for idx, (dims, ent, sched, epochs) in enumerate(combinations):
        
        # 构造 Run ID (无空格，便于文件名处理)
        run_id = f"Dims{dims[0]}_Ent{ent}_Sch{sched}_Ep{epochs}"
        dims_arg = list(map(str, dims))
        
        print(f"\n[{idx+1}/{len(combinations)}] Starting Run: {run_id}")
        
        # 定义临时报告文件路径
        report_file = os.path.join(current_dir, f"temp_report_{run_id}.json")
        
        # 构造 subprocess 命令
        cmd = [
            sys.executable, train_script_path,
            "--task", "teacher", # 确保这是正确的 task name (对应 teacher_env.py)
            "--num_envs", "4000",        
            "--headless",                
            "--max_iterations", "800",   # 跑 850 轮，覆盖 700-800 的统计区间
            
            # WandB 参数
            "--logger", "wandb",
            "--wandb_project", "Quadcopter_Teacher_Search",
            
            # 搜索参数
            "--override_hidden_dims", *dims_arg,
            "--override_entropy", str(ent),
            "--override_schedule", sched,
            "--override_num_learning_epochs", str(epochs),
            "--run_name_suffix", run_id
        ]
        
        # 设置环境变量，告诉 teacher_env.py 把结果写到哪里
        env_vars = os.environ.copy()
        env_vars["TEACHER_REWARD_PATH"] = report_file
        
        try:
            # 阻塞执行
            subprocess.run(cmd, env=env_vars, check=True)
            
            # 3. 读取结果
            if os.path.exists(report_file):
                with open(report_file, "r") as f:
                    data = json.load(f)
                
                # [修改点] 对应 teacher_env.py 实际写入的 Key
                # teacher_env.py 已经计算好了 700-800 轮的均值，直接读取即可，不需要再切片
                record = {
                    "Hidden Dims": str(dims),
                    "Entropy": ent,
                    "Schedule": sched,
                    "Num Epochs": epochs,
                    "Run Name": run_id,
                    
                    # 直接读取 Float 值，如果 Key 不存在则填 NaN
                    "rew_position": data.get("position", np.nan),
                    "rew_orientation": data.get("orientation", np.nan),
                    "rew_action_smooth": data.get("action_smooth", np.nan)
                }
                
                # 简单的打印检查
                print(f"   -> Result: Pos={record['rew_position']:.4f}, Ori={record['rew_orientation']:.4f}")

                results.append(record)
                os.remove(report_file)
            else:
                print(f"Error: Report file {report_file} was not generated.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running combination {run_id}: {e}")
        except KeyboardInterrupt:
            print("Experiment interrupted by user.")
            break

    # 4. 生成报表
    print("\n" + "="*50)
    print("FINAL RESULTS (Avg over steps 700-800)")
    print("="*50)
    
    if results:
        df = pd.DataFrame(results)
        
        # 调整列顺序，好看一点
        cols = ["Hidden Dims", "Entropy", "Schedule", "Num Epochs", 
                "rew_position", "rew_orientation", "rew_action_smooth", "Run Name"]
        # 确保列存在
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df)
        
        csv_name = os.path.join(current_dir, f"param_selection_results_{int(time.time())}.csv")
        df.to_csv(csv_name, index=False)
        print(f"\nResults saved to {csv_name}")
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()