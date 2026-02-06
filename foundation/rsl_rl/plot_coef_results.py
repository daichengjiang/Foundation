import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

def get_latest_csv():
    """自动寻找当前目录下最新的系数筛选结果CSV文件"""
    list_of_files = glob.glob('coef_selection_results_1770364072.csv') 
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def plot_coef_analysis():
    # 1. 加载数据
    csv_file = get_latest_csv()
    if csv_file is None:
        print("Error: 当前目录下没有找到 coef_selection_results_*.csv 文件。")
        return

    print(f"Reading data from: {csv_file}")
    df = pd.read_csv(csv_file)

    # 2. 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 获取所有测试过的参数名称
    tested_params = df['Tested Param'].unique()
    
    # 定义要绘制的三个核心指标 (归一化后的)
    metrics = [
        {
            'col': 'Metric_Pos (Norm)', 
            'name': 'Position Accuracy', 
            'desc': 'Norm. Position Reward (Closer to 0 is Better)',
            'color': 'tab:blue'
        },
        {
            'col': 'Metric_Smooth (Norm)', 
            'name': 'Action Smoothness', 
            'desc': 'Norm. Smoothness Reward (Closer to 0 is Better)',
            'color': 'tab:green'
        },
        {
            'col': 'Metric_Ori (Norm)', 
            'name': 'Orientation Stability', 
            'desc': 'Norm. Orientation Reward (Closer to 0 is Better)',
            'color': 'tab:orange'
        }
    ]

    # 3. 为每个测试参数绘制趋势图
    for param in tested_params:
        # 筛选出只包含当前参数测试的数据
        subset = df[df['Tested Param'] == param].sort_values(by='Value')
        
        if subset.empty:
            continue
            
        # 创建画布：1行3列
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        short_param_name = param.replace('reward_coef_', '')
        fig.suptitle(f'Impact of {short_param_name}', fontsize=16, fontweight='bold')

        for i, metric in enumerate(metrics):
            ax = axes[i]
            y_col = metric['col']
            
            # 绘制折线图
            sns.lineplot(
                data=subset, 
                x='Value', 
                y=y_col, 
                ax=ax, 
                marker='o', 
                markersize=8, 
                color=metric['color'],
                linewidth=2
            )
            
            # 标注具体的数值
            for _, row in subset.iterrows():
                # 防止文字重叠，交替显示
                ax.text(
                    row['Value'], 
                    row[y_col], 
                    f"{row[y_col]:.1f}", 
                    color='black', 
                    ha='center', 
                    va='bottom' if i%2==0 else 'top',
                    fontsize=9
                )

            ax.set_title(metric['name'], fontsize=12)
            ax.set_ylabel(metric['desc'])
            ax.set_xlabel(f"Coefficient Value: {short_param_name}")
            
            # 如果是 reward_constant，可能只有几个离散点，强制设为整数刻度看起来更舒服
            # if 'constant' in param:
            #     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        save_name = f"analysis_{short_param_name}.png"
        plt.savefig(save_name)
        print(f"Saved figure: {save_name}")
        plt.show()

        # 4. 特殊处理：如果是 d_action_cost，额外画一张 Trade-off 散点图
        if 'd_action_cost' in param:
            plt.figure(figsize=(8, 6))
            plt.title(f"Trade-off: Position vs Smoothness ({short_param_name})", fontsize=14)
            
            # 绘制散点
            sns.scatterplot(
                data=subset, 
                x='Metric_Pos (Norm)', 
                y='Metric_Smooth (Norm)', 
                hue='Value', 
                palette='viridis', 
                s=100,
                legend='full'
            )
            
            # 连线表示趋势
            plt.plot(subset['Metric_Pos (Norm)'], subset['Metric_Smooth (Norm)'], color='gray', linestyle='--', alpha=0.5)
            
            # 标注系数点
            for _, row in subset.iterrows():
                plt.text(
                    row['Metric_Pos (Norm)'], 
                    row['Metric_Smooth (Norm)'], 
                    f" Coef={row['Value']}", 
                    fontsize=9
                )

            plt.xlabel("Position Metric (Norm) -> Higher is Better")
            plt.ylabel("Smoothness Metric (Norm) -> Higher is Better")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            tradeoff_save_name = f"tradeoff_{short_param_name}.png"
            plt.savefig(tradeoff_save_name)
            print(f"Saved trade-off figure: {tradeoff_save_name}")
            plt.show()

if __name__ == "__main__":
    plot_coef_analysis()