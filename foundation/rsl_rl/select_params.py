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
        
        # 构造易读的 Run ID
        run_id = f"Dims{dims[0]}_Ent{ent}_Sch{sched}_Ep{epochs}"
        dims_arg = list(map(str, dims))
        
        print(f"\n[{idx+1}/{len(combinations)}] Starting Run: {run_id}")
        
        report_file = os.path.join(current_dir, f"temp_report_{run_id}.json")
        
        # 构造 subprocess 命令
        cmd = [
            sys.executable, train_script_path,
            "--task", "teacher",
            "--num_envs", "4000",        
            "--headless",                
            "--max_iterations", "800",
            # [新增] 激活 WandB 的关键参数
            "--logger", "wandb",
            "--wandb_project", "Quadcopter_Teacher_Search", # 确保都在同一个项目下
            # 参数覆盖
            "--override_hidden_dims", *dims_arg,
            "--override_entropy", str(ent),
            "--override_schedule", sched,
            "--override_num_learning_epochs", str(epochs),
            "--run_name_suffix", run_id
        ]
        
        env_vars = os.environ.copy()
        env_vars["TEACHER_REWARD_PATH"] = report_file
        
        try:
            # 阻塞执行
            subprocess.run(cmd, env=env_vars, check=True)
            
            # 3. 读取结果 (这部分保持不变)
            if os.path.exists(report_file):
                with open(report_file, "r") as f:
                    data = json.load(f)
                
                target_keys = ["rew_action_smooth", "rew_orientation", "rew_position"] 
                
                record = {
                    "Hidden Dims": str(dims),
                    "Entropy": ent,
                    "Schedule": sched,
                    "Num Epochs": epochs,
                    "Run Name": run_id
                }
                
                start_idx = 700
                end_idx = 800
                
                for key in target_keys:
                    if key in data and isinstance(data[key], list):
                        valid_end = min(len(data[key]), end_idx)
                        if valid_end > start_idx:
                            segment = data[key][start_idx:valid_end]
                            record[key] = np.mean(segment)
                        else:
                            record[key] = np.nan
                    else:
                        print(f"Warning: Key '{key}' not found.")
                        record[key] = np.nan

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