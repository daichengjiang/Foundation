import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import numpy as np
import os
import argparse

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"找不到文件: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return df

def generate_target_drone_samples(target_mass, target_arm, target_twr, num_samples=100):
    """根据输入的三个已知参数，生成目标无人机的衍生参数样本批次"""
    samples = []
    for _ in range(num_samples):
        r_t2i = np.random.uniform(40, 1200)
        total_thrust = target_twr * 9.81 * target_mass
        tau = total_thrust * np.sqrt(2) * target_arm
        Ixx = tau / r_t2i
        Iyy = Ixx
        Izz = Ixx * 1.832

        motor_tau_up = np.random.uniform(0.03, 0.1)
        motor_tau_down = np.random.uniform(0.03, 0.3)
        kappa = np.random.uniform(0.005, 0.05)

        samples.append({
            'mass': target_mass,
            'arm_length': target_arm,
            'Ixx': Ixx,
            'Iyy': Iyy,
            'Izz': Izz,
            'twr': target_twr,
            'motor_tau_up': motor_tau_up,
            'motor_tau_down': motor_tau_down,
            'kappa': kappa,
            'Type': 'Target Drone'
        })
    return pd.DataFrame(samples)

def plot_pairplot(df, save_path, is_combined=False):
    print(f"正在绘制散点图矩阵 (保存至 {os.path.basename(save_path)})...")
    cols_to_plot = ['mass', 'arm_length', 'twr', 'motor_tau_up', 'motor_tau_down', 'kappa']
    sns.set_theme(style="whitegrid")
    
    if is_combined:
        palette = {'Survivor': '#1f77b4', 'Target Drone': '#d62728'}
        g = sns.pairplot(df, 
                         vars=cols_to_plot,
                         hue='Type',
                         palette=palette,
                         diag_kind=None, 
                         plot_kws={'alpha': 0.5, 's': 15, 'edgecolor': 'none'},
                         corner=True)
        g.fig.suptitle("Teacher Dynamics Survival vs. Target Real Drone", y=1.02, fontsize=16)
    else:
        g = sns.pairplot(df[cols_to_plot], 
                         diag_kind=None, 
                         plot_kws={'alpha': 0.5, 's': 15, 'edgecolor': 'none', 'color': '#1f77b4'},
                         corner=True)
        g.fig.suptitle("Teacher Dynamics Survival Distribution & Correlations", y=1.02, fontsize=16)
                     
    # 彻底隐身对角线上的空方框
    for i in range(len(cols_to_plot)):
        ax = g.axes[i, i]
        if ax is not None:
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, left=False)
                     
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parallel_coordinates(df, save_path, is_combined=False):
    print(f"正在绘制平行坐标图 (保存至 {os.path.basename(save_path)})...")
    cols_to_plot = ['mass', 'arm_length', 'Ixx', 'twr', 'motor_tau_up', 'motor_tau_down', 'kappa']
    
    # 【修复核心】：严格限制只提取数值列和分类列，防止 Pandas 误将多余的字符串列画到坐标轴上
    cols_to_keep = cols_to_plot + ['Type']
    df_norm = df[cols_to_keep].copy()
    
    # 全局归一化
    for col in cols_to_plot:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5

    plt.figure(figsize=(12, 6))
    
    if is_combined:
        df_surv = df_norm[df_norm['Type'] == 'Survivor']
        df_tgt = df_norm[df_norm['Type'] == 'Target Drone']
        
        if not df_surv.empty:
            parallel_coordinates(df_surv, 'Type', color=['#1f77b4'], alpha=0.1)
        if not df_tgt.empty:
            parallel_coordinates(df_tgt, 'Type', color=['#d62728'], alpha=0.4, linewidth=2)
            
        plt.title("Parallel Coordinates: Surviving Teachers vs. Target Real Drone", fontsize=14)
        
        # 整理图例，避免重复
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    else:
        # 此时 df_norm 干净得只剩下数值列和 Type，直接用 Type 作为分类标签
        parallel_coordinates(df_norm, 'Type', color=['#1f77b4'], alpha=0.1)
        plt.title("Parallel Coordinates of Surviving Teachers (Normalized 0-1)", fontsize=14)
        plt.legend().set_visible(False)

    plt.xticks(rotation=45)
    plt.ylabel("Normalized Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_distributions_with_bounds(df_survivor, save_path, target_params=None):
    print(f"正在绘制单变量分布图 (保存至 {os.path.basename(save_path)})...")
    
    bounds = {
        'mass': (0.02, 5.0),
        'twr': (1.5, 5.0),
        'motor_tau_up': (0.03, 0.1),
        'motor_tau_down': (0.03, 0.3),
        'kappa': (0.005, 0.05)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (col, (b_min, b_max)) in enumerate(bounds.items()):
        ax = axes[i]
        
        # 1. 仅绘制 Survivor 的分布
        sns.histplot(df_survivor[col], kde=False, ax=ax, color='skyblue', stat='density', bins=40, label='Actual Survival')
        
        # 2. 绘制采样边界
        ax.axvline(b_min, color='red', linestyle='--', linewidth=2, label='Sample Bounds')
        ax.axvline(b_max, color='red', linestyle='--', linewidth=2)
        
        # 3. 绘制理论分布线
        x_theory = np.linspace(b_min, b_max, 500)
        if col == 'mass':
            a = np.cbrt(b_min)
            b = np.cbrt(b_max)
            y_theory = (1.0 / (b - a)) * (1.0 / 3.0) * np.power(x_theory, -2/3)
        else:
            pdf_val = 1.0 / (b_max - b_min)
            y_theory = np.full_like(x_theory, pdf_val)
        ax.plot(x_theory, y_theory, color='green', linestyle='-', linewidth=2.5, label='Ideal Sampling PDF')
        
        # 4. 如果包含目标实机参数，画出橙色标记线
        if target_params and col in target_params:
            ax.axvline(target_params[col], color='darkorange', linestyle='-', linewidth=3, label='Target Drone Position')
        
        ax.set_title(f'Distribution of {col}')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="teacher_dynamics.csv", help="Path to the teacher_dynamics.csv file")
    
    parser.add_argument("--target_mass", type=float, default=None, help="目标实机的已知质量 (kg)")
    parser.add_argument("--target_arm", type=float, default=None, help="目标实机的已知轴距 (m)")
    parser.add_argument("--target_twr", type=float, default=None, help="目标实机的已知推重比")
    parser.add_argument("--num_samples", type=int, default=100, help="要为目标实机生成的随机分布样本数")
    
    args = parser.parse_args()

    csv_filepath = args.csv_path
    save_directory = os.path.dirname(os.path.abspath(csv_filepath))

    df = load_data(csv_filepath)
    if df is not None and not df.empty:
        print(f"成功加载训练数据，共 {len(df)} 条记录。\n")
        df['Type'] = 'Survivor' 
        
        # ==========================================
        # 阶段一：始终生成 3 张纯净的原始数据图
        # ==========================================
        print(">>> 阶段 1/2: 生成纯净版原始训练分布图...")
        plot_distributions_with_bounds(df, os.path.join(save_directory, "dynamics_distributions.png"), target_params=None)
        plot_pairplot(df, os.path.join(save_directory, "dynamics_pairplot.png"), is_combined=False)
        plot_parallel_coordinates(df, os.path.join(save_directory, "dynamics_parallel.png"), is_combined=False)
        
        # ==========================================
        # 阶段二：如果输入了实机参数，生成 3 张叠加了实机数据的对比图
        # ==========================================
        if args.target_mass is not None and args.target_arm is not None and args.target_twr is not None:
            print("\n>>> 阶段 2/2: 检测到实机参数，生成带有 Target Drone 标记的对比图...")
            target_params = {
                'mass': args.target_mass,
                'arm_length': args.target_arm,
                'twr': args.target_twr
            }
            
            df_target = generate_target_drone_samples(args.target_mass, args.target_arm, args.target_twr, args.num_samples)
            combined_df = pd.concat([df, df_target], ignore_index=True)
            
            plot_distributions_with_bounds(df, os.path.join(save_directory, "dynamics_distributions_with_target.png"), target_params)
            plot_pairplot(combined_df, os.path.join(save_directory, "dynamics_pairplot_with_target.png"), is_combined=True)
            plot_parallel_coordinates(combined_df, os.path.join(save_directory, "dynamics_parallel_with_target.png"), is_combined=True)
            
            print("\n任务完成！共生成了 6 张图表。")
        else:
            print("\n未提供实机目标参数(--target_mass, --target_arm, --target_twr)。\n任务完成！共生成了 3 张图表。")
            
        print(f"所有图片均已保存在目录: {save_directory}")
    else:
        print("数据为空或文件格式错误。")