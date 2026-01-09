import os

from isaacsim import SimulationApp
import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom, Sdf, Gf, Vt
import math

def height_to_color(z_values, colormap='turbo'):
    """
    根据Z轴高度生成颜色渐变
    
    Args:
        z_values: Z坐标数组
        colormap: 颜色映射类型 ('turbo', 'rainbow', 'jet', 'plasma', 'inferno', 'ocean', 'terrain')
    
    Returns:
        colors: RGB颜色数组 (N, 3)
    """
    # 归一化Z值到[0, 1]
    z_min, z_max = np.min(z_values), np.max(z_values)
    if z_max == z_min:
        z_norm = np.zeros_like(z_values)
    else:
        z_norm = (z_values - z_min) / (z_max - z_min)
    
    colors = np.zeros((len(z_values), 3))
    
    if colormap == 'turbo':
        # Google Turbo 配色（高对比度，人眼友好）
        # 简化版本的turbo colormap
        for i, t in enumerate(z_norm):
            if t < 0.2:
                s = t / 0.2
                colors[i] = [0.19 + s * 0.11, 0.07 + s * 0.43, 0.48 + s * 0.22]
            elif t < 0.4:
                s = (t - 0.2) / 0.2
                colors[i] = [0.30 + s * 0.42, 0.50 + s * 0.35, 0.70 - s * 0.25]
            elif t < 0.6:
                s = (t - 0.4) / 0.2
                colors[i] = [0.72 + s * 0.20, 0.85 + s * 0.10, 0.45 - s * 0.20]
            elif t < 0.8:
                s = (t - 0.6) / 0.2
                colors[i] = [0.92 + s * 0.06, 0.95 - s * 0.20, 0.25 - s * 0.15]
            else:
                s = (t - 0.8) / 0.2
                colors[i] = [0.98 - s * 0.26, 0.75 - s * 0.45, 0.10 - s * 0.08]
    
    elif colormap == 'jet':
        # 经典Jet配色（深蓝->青->绿->黄->红）
        for i, t in enumerate(z_norm):
            if t < 0.125:
                colors[i] = [0, 0, 0.5 + t * 4]
            elif t < 0.375:
                s = (t - 0.125) / 0.25
                colors[i] = [0, s, 1]
            elif t < 0.625:
                s = (t - 0.375) / 0.25
                colors[i] = [s, 1, 1 - s]
            elif t < 0.875:
                s = (t - 0.625) / 0.25
                colors[i] = [1, 1 - s, 0]
            else:
                s = (t - 0.875) / 0.125
                colors[i] = [1 - s * 0.5, 0, 0]
    
    elif colormap == 'plasma':
        # Matplotlib plasma配色（深蓝紫->粉->黄）
        for i, t in enumerate(z_norm):
            r = 0.05 + 0.95 * (0.13 + 0.87 * t)**2
            g = 0.02 + 0.98 * (0.23 + 0.77 * (t**0.5))**2 * (1 - t * 0.2)
            b = 0.85 - 0.75 * t
            colors[i] = [r, g, b]
    
    elif colormap == 'inferno':
        # Matplotlib inferno配色（黑->紫->红->黄）
        for i, t in enumerate(z_norm):
            r = t ** 0.5
            g = (t ** 2) * 0.8
            b = max(0, (0.5 - t) * 2) if t < 0.5 else 0
            colors[i] = [r, g, b]
    
    elif colormap == 'ocean':
        # 海洋配色（深蓝->浅蓝->青绿）
        for i, t in enumerate(z_norm):
            r = t * 0.3
            g = 0.2 + t * 0.7
            b = 0.5 + t * 0.5
            colors[i] = [r, g, b]
    
    elif colormap == 'terrain':
        # 地形配色（蓝->绿->黄->棕->白）
        for i, t in enumerate(z_norm):
            if t < 0.2:  # 水域
                s = t / 0.2
                colors[i] = [0, 0.3 + s * 0.4, 0.8 + s * 0.2]
            elif t < 0.4:  # 低地
                s = (t - 0.2) / 0.2
                colors[i] = [s * 0.2, 0.7 + s * 0.3, 0.2 * (1 - s)]
            elif t < 0.6:  # 草地
                s = (t - 0.4) / 0.2
                colors[i] = [0.2 + s * 0.5, 1 - s * 0.3, 0]
            elif t < 0.8:  # 山地
                s = (t - 0.6) / 0.2
                colors[i] = [0.7 + s * 0.2, 0.7 - s * 0.3, 0.3 - s * 0.3]
            else:  # 雪山
                s = (t - 0.8) / 0.2
                colors[i] = [0.9 + s * 0.1, 0.9 + s * 0.1, 0.9 + s * 0.1]
    
    elif colormap == 'rainbow':
        # 传统彩虹色
        for i, t in enumerate(z_norm):
            if t < 0.25:
                s = t / 0.25
                colors[i] = [0, s, 1]
            elif t < 0.5:
                s = (t - 0.25) / 0.25
                colors[i] = [0, 1, 1-s]
            elif t < 0.75:
                s = (t - 0.5) / 0.25
                colors[i] = [s, 1, 0]
            else:
                s = (t - 0.75) / 0.25
                colors[i] = [1, 1-s, 0]
    
    return colors.astype(np.float32)

def load_pcd_as_points(stage, prim_path, pcd_path,
                       point_size=0.01, fallback_color=(0.0, 0.6, 1.0),
                       rx=0, ry=0, rz=0, tx=0, ty=0, tz=0,
                       use_height_color=True, colormap='turbo'):
    """
    把一个PCD加载为USD Points prim（无下采样）
    
    Args:
        stage: USD stage
        prim_path: USD路径
        pcd_path: PCD文件路径
        point_size: 点大小
        fallback_color: 默认颜色
        rx, ry, rz: 旋转角度（度）
        tx, ty, tz: 平移
        use_height_color: 是否使用高度颜色渐变
        colormap: 颜色映射类型 ('turbo', 'jet', 'plasma', 'inferno', 'ocean', 'terrain', 'rainbow')
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise RuntimeError(f"[PCD] zero points: {pcd_path}")
    N = pts.shape[0]

    # 颜色
    has_color = pcd.has_colors() and (len(pcd.colors) == N)
    cols = np.asarray(pcd.colors, dtype=np.float32) if has_color else None

    # 旋转
    if (rx, ry, rz) != (0, 0, 0):
        rxr, ryr, rzr = np.deg2rad([rx, ry, rz])
        cx, sx = np.cos(rxr), np.sin(rxr)
        cy, sy = np.cos(ryr), np.sin(ryr)
        cz, sz = np.cos(rzr), np.sin(rzr)
        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]])
        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                       [sz, cz, 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        pts = pts @ R.T

    # 平移
    pts = pts + np.array([tx, ty, tz])

    # 根据高度生成颜色
    if use_height_color:
        cols = height_to_color(pts[:, 2], colormap)
        print(f"[PCD] 使用 '{colormap}' 高度颜色渐变")
        print(f"[PCD] Z范围: {pts[:, 2].min():.3f} ~ {pts[:, 2].max():.3f} m")

    print(f"[PCD] 加载 {pcd_path}")
    print(f"[PCD] 点数: {N:,} (无下采样)")

    # 转成Points Prim
    points_prim = UsdGeom.Points.Define(stage, prim_path)
    
    # 转换为USD数据类型
    usd_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts.astype(np.float32)]
    points_prim.CreatePointsAttr(usd_points)
    
    # 宽度属性
    points_prim.CreateWidthsAttr([point_size] * N)

    # 颜色属性
    if cols is not None and len(cols) == N:
        usd_colors = [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in cols]
        points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(usd_colors)
    else:
        points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*fallback_color)])

    print(f"[PCD] ✓ 成功加载到 {prim_path}")
    return N


if __name__ == "__main__":
    simulation_app = SimulationApp(launch_config={"headless": False})
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    
    # 替换成你的PCD路径
    pcd_file = "16easy.pcd"
    
    # 可选配色方案：'turbo', 'jet', 'plasma', 'inferno', 'ocean', 'terrain', 'rainbow'
    colormap_choice = 'turbo'  # 推荐使用turbo，对比度高且美观
    
    load_pcd_as_points(
        stage, 
        "/World/PCDPoints", 
        pcd_file,
        point_size=0.02,  # 调整点大小
        tx=0.0, 
        ty=0.0, 
        tz=0.0,
        use_height_color=True, 
        colormap=colormap_choice
    )
    
    print(f"\n点云已加载到Isaac Sim中")
    print(f"配色方案: {colormap_choice}")
    print("按Ctrl+C退出...")
    
    # 保持程序运行
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n程序退出")
    
    simulation_app.close()