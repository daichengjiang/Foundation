import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import time

# ================= 配置区域 =================

# 1. 基准系数 (Baseline)
BASELINE_COEFS = {
    "reward_coef_position_cost": 1.0,
    "reward_coef_orientation_cost": 0.2,
    "reward_coef_d_action_cost": 0.5,
    "reward_coef_termination_penalty": 100.0,
    "reward_constant": 1.5
}

# 2. 测试范围
TEST_RANGES = {
    "reward_coef_position_cost": [0.5, 0.8, 1.0, 1.5, 2.0],
    "reward_coef_d_action_cost": [0.1, 0.3, 0.5, 0.8, 1.2],
    "reward_coef_orientation_cost": [0.05, 0.1, 0.2, 0.4],
    # "reward_constant": [1.0, 1.5, 2.0]
}

# 3. 固定 PPO 参数 (请填入您筛选出的最优值)
FIXED_PPO_PARAMS = {
    "override_hidden_dims": ["128", "128", "128"],
    "override_entropy": "0.005",  
    "override_schedule": "adaptive",
    "override_num_learning_epochs": "1"
}

# ===========================================

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(current_dir, "train_teacher_single.py")

    tasks = []
    for param_name, values in TEST_RANGES.items():
        for val in values:
            current_config = BASELINE_COEFS.copy()
            current_config[param_name] = val
            task_info = {
                "tested_param": param_name,
                "tested_value": val,
                "config": current_config,
                "run_id": f"Test_{param_name.replace('reward_coef_', '')}_{val}"
            }
            tasks.append(task_info)

    print(f"Total experiments: {len(tasks)}")
    results = []

    for idx, task in enumerate(tasks):
        run_id = task["run_id"]
        cfg = task["config"]
        
        print(f"\n[{idx+1}/{len(tasks)}] Testing {task['tested_param']} = {task['tested_value']}")
        
        report_file = os.path.join(current_dir, f"temp_report_{run_id}.json")
        
        cmd = [
            sys.executable, train_script_path,
            "--task", "teacher", 
            "--num_envs", "4000",        
            "--headless",                
            "--max_iterations", "850",   
            "--logger", "wandb",
            "--wandb_project", "Quadcopter_Reward_Search",
            
            # PPO 参数
            "--override_hidden_dims", *FIXED_PPO_PARAMS["override_hidden_dims"],
            "--override_entropy", FIXED_PPO_PARAMS["override_entropy"],
            "--override_schedule", FIXED_PPO_PARAMS["override_schedule"],
            "--override_num_learning_epochs", FIXED_PPO_PARAMS["override_num_learning_epochs"],
            
            # 奖励系数
            "--reward_coef_position_cost", str(cfg["reward_coef_position_cost"]),
            "--reward_coef_orientation_cost", str(cfg["reward_coef_orientation_cost"]),
            "--reward_coef_d_action_cost", str(cfg["reward_coef_d_action_cost"]),
            "--reward_coef_termination_penalty", str(cfg["reward_coef_termination_penalty"]),
            "--reward_constant", str(cfg["reward_constant"]),
            
            "--run_name_suffix", run_id
        ]
        
        env_vars = os.environ.copy()
        env_vars["TEACHER_REWARD_PATH"] = report_file
        
        try:
            subprocess.run(cmd, env=env_vars, check=True)
            
            if os.path.exists(report_file):
                with open(report_file, "r") as f:
                    data = json.load(f)
                
                # 1. 获取原始奖励 (Raw Reward)
                raw_pos = data.get("position", np.nan)
                raw_ori = data.get("orientation", np.nan)
                raw_smooth = data.get("action_smooth", np.nan)
                
                # 2. 获取当前使用的系数 (Coefficients)
                coef_pos = cfg["reward_coef_position_cost"]
                coef_ori = cfg["reward_coef_orientation_cost"]
                coef_smooth = cfg["reward_coef_d_action_cost"]
                
                # 3. 计算归一化指标 (Normalized Metric = Raw / Coef)
                # 这代表了纯粹的物理表现：位置误差项、姿态误差项、动作抖动项
                # 注意：如果系数为0，则设为NaN
                norm_pos = raw_pos / coef_pos if coef_pos != 0 else np.nan
                norm_ori = raw_ori / coef_ori if coef_ori != 0 else np.nan
                norm_smooth = raw_smooth / coef_smooth if coef_smooth != 0 else np.nan
                
                record = {
                    "Tested Param": task["tested_param"],
                    "Value": task["tested_value"],
                    
                    # 记录两套数据
                    "Metric_Pos (Norm)": norm_pos,       # <-- 看这个来选参数！
                    "Metric_Smooth (Norm)": norm_smooth, # <-- 看这个来选参数！
                    "Metric_Ori (Norm)": norm_ori,
                    
                    "Raw_Pos": raw_pos,
                    "Raw_Smooth": raw_smooth,
                    
                    "Run Name": run_id,
                    **cfg 
                }
                
                print(f"   -> Result (Norm): Pos={norm_pos:.4f}, Smooth={norm_smooth:.4f}")
                results.append(record)
                os.remove(report_file)
            else:
                print(f"Error: Report file {report_file} was not generated.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running {run_id}: {e}")
        except KeyboardInterrupt:
            break

    print("\n" + "="*50)
    print("FINAL RESULTS (Normalized Metrics)")
    print("="*50)
    
    if results:
        df = pd.DataFrame(results)
        
        # 优先展示归一化后的 Metric
        cols = ["Tested Param", "Value", "Metric_Pos (Norm)", "Metric_Smooth (Norm)", "Metric_Ori (Norm)"]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df)
        
        csv_name = os.path.join(current_dir, f"coef_selection_results_{int(time.time())}.csv")
        df.to_csv(csv_name, index=False)
        print(f"\nResults saved to {csv_name}")
        print("提示：请依据 'Metric_...' 列进行筛选，数值越大（越接近0）表示物理性能越好。")

if __name__ == "__main__":
    main()