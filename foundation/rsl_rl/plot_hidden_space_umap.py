import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from datetime import datetime
import os
import warnings

# 忽略 UMAP 可能会报的一些关于 Numba 性能的警告
warnings.filterwarnings('ignore')

try:
    import umap
except ImportError:
    raise ImportError("未找到 umap-learn 库。请运行 'pip install umap-learn' 安装。")

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Plot and Save RNN hidden states from multiple trajectory NPZ files using UMAP.")
    parser.add_argument("--data_paths", nargs='+', type=str, required=True, help="一个或多个 trajectory_data.npz 的路径")
    parser.add_argument("--labels", nargs='+', type=str, default=None, help="每条轨迹在图例中的名字")
    parser.add_argument("--min_traj_time", type=float, default=None, help="绘制轨迹的起始时间(秒)")
    parser.add_argument("--max_traj_time", type=float, default=None, help="绘制轨迹的结束时间(秒)")
    parser.add_argument("--interval", type=float, default=0.5, help="打点的时间间隔(秒), 默认0.5s")
    parser.add_argument("--min_mark_time", type=float, default=None, help="标点开始的最小时间(秒)")
    parser.add_argument("--max_mark_time", type=float, default=None, help="标点最多打到第几秒")
    parser.add_argument("--dt", type=float, default=0.01, help="推算时间的仿真步长(秒)")
    parser.add_argument("--no_show", action="store_true", help="如果设置，只保存图片不弹窗显示")
    
    # [新增] UMAP 专属高级调参选项 (通常使用默认值即可)
    parser.add_argument("--n_neighbors", type=int, default=30, help="UMAP 局部邻域大小 (越大看全局，越小看局部)")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP 点的最小间距 (越小簇越紧密，越大越分散)")
    args = parser.parse_args()

    num_files = len(args.data_paths)
    print(f"[INFO] 准备加载 {num_files} 个轨迹文件...")

    if args.labels is not None and len(args.labels) != num_files:
        print("[WARNING] --labels 的数量与 --data_paths 不一致！将使用默认名字。")
        args.labels = None

    # 2. 批量加载数据
    all_h_states = []
    all_time_arrs = []
    
    for path in args.data_paths:
        data = np.load(path)
        if 'hidden_states' not in data.files:
            print(f"[ERROR] 文件 {path} 中未找到 'hidden_states'！跳过。")
            continue
            
        h_states = data['hidden_states']
        all_h_states.append(h_states)
        
        if 'timestamps' in data.files:
            all_time_arrs.append(data['timestamps'])
        else:
            all_time_arrs.append(np.arange(h_states.shape[0]) * args.dt)

    if len(all_h_states) == 0:
        print("[ERROR] 没有加载到任何有效数据，退出。")
        return

    # 3. 全局 UMAP 降维
    print(f"[INFO] 正在拟合统一的 UMAP 全局非线性流形空间 (数据量大时可能需要十几秒)...")
    combined_h = np.vstack(all_h_states)
    
    # 初始化 UMAP 降维器
    # 设定 random_state=42 保证可复现性（每次跑出来的形状是一样的，方便调整颜色和标点）
    reducer = umap.UMAP(n_components=2, 
                        random_state=42, 
                        n_neighbors=args.n_neighbors, 
                        min_dist=args.min_dist)
    
    # 拟合全局空间
    reducer.fit(combined_h)
    print("[INFO] UMAP 拟合完成！正在绘制轨迹...")

    # 4. 准备绘图
    fig, ax = plt.subplots(figsize=(12, 9))
    cmaps = ['viridis', 'plasma', 'cool', 'autumn', 'winter', 'Wistia']
    first_lc = None 

    for i, (h_states, time_arr) in enumerate(zip(all_h_states, all_time_arrs)):
        label = args.labels[i] if args.labels else f"Traj {i+1}"
        cmap_name = cmaps[i % len(cmaps)]
        
        # 将当前轨迹投影到拟合好的 UMAP 空间中
        h_2d = reducer.transform(h_states)
        x_full = h_2d[:, 0]
        y_full = h_2d[:, 1]

        mask = np.ones(len(time_arr), dtype=bool)
        if args.min_traj_time is not None: mask &= (time_arr >= args.min_traj_time)
        if args.max_traj_time is not None: mask &= (time_arr <= args.max_traj_time)

        x_plot, y_plot, time_plot = x_full[mask], y_full[mask], time_arr[mask]

        if len(x_plot) < 2:
            continue

        points = np.array([x_plot, y_plot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(time_plot[0], time_plot[-1])
        lc = LineCollection(segments, cmap=cmap_name, norm=norm, linewidth=2.5, alpha=0.8)
        lc.set_array(time_plot[:-1])
        ax.add_collection(lc)
        
        if first_lc is None: first_lc = lc

        ax.scatter(x_plot[0], y_plot[0], color='red', marker='o', s=80, zorder=5)
        ax.scatter(x_plot[-1], y_plot[-1], color='black', marker='*', s=180, zorder=5)

        mid_color = plt.get_cmap(cmap_name)(0.6) 
        ax.plot([], [], color=mid_color, label=label, linewidth=3)

        mark_start = args.min_mark_time if args.min_mark_time is not None else max(args.interval, time_plot[0])
        mark_end = args.max_mark_time if args.max_mark_time is not None else time_plot[-1]

        if mark_start <= mark_end:
            target_times = np.arange(mark_start, mark_end + 1e-5, args.interval)
            marker_indices = []
            for target_t in target_times:
                idx = (np.abs(time_plot - target_t)).argmin()
                if np.abs(time_plot[idx] - target_t) <= args.interval / 2.0:
                    marker_indices.append(idx)
            
            marker_indices = sorted(list(set(marker_indices)))

            if len(marker_indices) > 0:
                ax.scatter(x_plot[marker_indices], y_plot[marker_indices], 
                           color='white', edgecolor='black', marker='o', s=40, zorder=6)
                
                # 动态颜色标注
                cmap = lc.get_cmap()
                norm = lc.norm
                
                for idx in marker_indices:
                    t_val = time_plot[idx]
                    label_color = cmap(norm(t_val))
                    
                    ax.annotate(f"{t_val:.2f}s", (x_plot[idx], y_plot[idx]), 
                                textcoords="offset points", xytext=(7, 7), 
                                fontsize=8, fontweight='bold', 
                                color=label_color, 
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65), 
                                zorder=7)

    if num_files == 1 and first_lc is not None:
        cbar = fig.colorbar(first_lc, ax=ax)
        cbar.set_label('Time (s)', rotation=270, labelpad=15, fontweight='bold')

    ax.autoscale() 
    
    # 标题更新为 UMAP
    ax.set_title(f'RNN Hidden Space Dynamics Manifold (UMAP 2D Projection)', fontsize=15, fontweight='bold')
    
    # UMAP 没有解释方差的百分比概念，所以坐标轴直接命名为 Dimension
    ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.4)
    
    start_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Start Node')
    end_marker = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=12, label='End Node')
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([start_marker, end_marker])
    ax.legend(handles=handles, loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # 自动保存图片逻辑 (改名加上了 umap 前缀)
    base_dir = os.path.dirname(os.path.abspath(args.data_paths[0]))
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S")
    file_name = f"umap_manifold_{timestamp}.png"
    save_path = os.path.join(plot_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] 图表已成功保存至: {save_path}\n")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()