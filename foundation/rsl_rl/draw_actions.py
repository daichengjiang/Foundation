import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_actions(csv_path):
    """简单读取CSV并绘制动作图表"""
    
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"数据形状: {df.shape}")
    
    # 创建时间轴
    time = [i * 0.01 for i in range(len(df))]  # 假设10ms间隔
    
    # 绘制4个子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    actions = ['roll', 'pitch', 'yaw', 'thrust']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (action, color) in enumerate(zip(actions, colors)):
        ax = axes[i//2, i%2]
        ax.plot(time, df[action], color=color, linewidth=1)
        ax.set_title(action.capitalize())
        ax.set_xlabel('时间 (s)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(csv_path.replace('.csv', '_plot.png'), dpi=200)
    plt.show()
    
    print(f"图片已保存到: {csv_path.replace('.csv', '_plot.png')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python simple_plot.py <csv文件路径>")
        sys.exit(1)
    
    plot_actions(sys.argv[1])