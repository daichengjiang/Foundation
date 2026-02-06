import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_param_effects():
    # 1. 读取数据
    csv_file = "param_selection_results_1770325356.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {csv_file}，请确保它在当前目录下。")
        return

    # 2. 数据预处理：简化 Hidden Dims 的标签显示
    # 将 "[128, 128, 128]" 简化为 "128x3"， "[64, 64, 64]" 简化为 "64x3"
    df['Hidden Dims'] = df['Hidden Dims'].apply(lambda x: '128x3' if '128' in x else '64x3')

    # 定义要分析的4个变量
    params = [
        {'col': 'Hidden Dims', 'name': 'Hidden Dimensions', 'type': 'cat'},
        {'col': 'Entropy',     'name': 'Entropy Coefficient', 'type': 'num'},
        {'col': 'Schedule',    'name': 'Learning Rate Schedule', 'type': 'cat'},
        {'col': 'Num Epochs',  'name': 'Num Learning Epochs', 'type': 'num'}
    ]

    # 定义3个奖励指标
    metrics = [
        {'col': 'rew_position',      'title': 'Position Reward (Higher is better)'},
        {'col': 'rew_orientation',   'title': 'Orientation Reward (Higher is better)'},
        {'col': 'rew_action_smooth', 'title': 'Action Smooth Reward (Higher is better)'}
    ]

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 3. 循环绘制4张图
    for param in params:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Impact of {param["name"]}', fontsize=16, fontweight='bold')
        
        x_col = param['col']
        
        for i, metric in enumerate(metrics):
            y_col = metric['col']
            ax = axes[i]
            
            # 使用 Pointplot 展示均值和置信区间（默认为95% CI）
            # 这能很好地展示在其他参数变化时，当前参数的"平均效应"
            sns.pointplot(
                data=df, 
                x=x_col, 
                y=y_col, 
                ax=ax, 
                capsize=.1, 
                errorbar='sd',  # 误差棒显示标准差，以此体现该参数下的波动情况
                color='b' if i==0 else ('g' if i==1 else 'r'),
                markers='o',
                linestyles='-'
            )
            
            # 如果是数值型x轴（如Entropy），也可以考虑用 lineplot，但 pointplot 对离散值更通用
            
            ax.set_title(metric['title'])
            ax.set_xlabel(param['name'])
            ax.set_ylabel("Reward Value")
            
            # 如果 Entropy 是对数分布，可以考虑把 x 轴设为 log (可选)
            # if param['col'] == 'Entropy':
            #     ax.set_xscale('log') 

        plt.tight_layout()
        
        # 保存图片
        filename = f"effect_{param['col'].replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        print(f"Saved figure: {filename}")
        
        # 显示图片
        plt.show()

if __name__ == "__main__":
    plot_param_effects()