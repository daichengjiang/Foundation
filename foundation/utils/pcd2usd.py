import os

from isaacsim import SimulationApp
import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom, Sdf, Gf, Vt
import math

def height_to_color(z_values, colormap='rainbow'):
    """
    根据Z轴高度生成颜色渐变
    
    Args:
        z_values: Z坐标数组
        colormap: 颜色映射类型 ('rainbow', 'hot', 'cool', 'viridis')
    
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
    
    if colormap == 'rainbow':
        # 彩虹色：蓝->青->绿->黄->红
        for i, t in enumerate(z_norm):
            if t < 0.25:
                # 蓝到青
                s = t / 0.25
                colors[i] = [0, s, 1]
            elif t < 0.5:
                # 青到绿
                s = (t - 0.25) / 0.25
                colors[i] = [0, 1, 1-s]
            elif t < 0.75:
                # 绿到黄
                s = (t - 0.5) / 0.25
                colors[i] = [s, 1, 0]
            else:
                # 黄到红
                s = (t - 0.75) / 0.25
                colors[i] = [1, 1-s, 0]
    
    elif colormap == 'hot':
        # 热力图：黑->红->黄->白
        for i, t in enumerate(z_norm):
            if t < 0.33:
                s = t / 0.33
                colors[i] = [s, 0, 0]
            elif t < 0.66:
                s = (t - 0.33) / 0.33
                colors[i] = [1, s, 0]
            else:
                s = (t - 0.66) / 0.34
                colors[i] = [1, 1, s]
    
    elif colormap == 'cool':
        # 冷色调：青->品红
        colors[:, 0] = z_norm          # R: 0->1
        colors[:, 1] = 1 - z_norm      # G: 1->0
        colors[:, 2] = 1               # B: 1
    
    elif colormap == 'viridis':
        # 类似matplotlib的viridis色彩
        for i, t in enumerate(z_norm):
            # 简化的viridis近似
            r = 0.267004 + t * (0.993248 - 0.267004)
            g = 0.004874 + t * (0.906157 - 0.004874) 
            b = 0.329415 + t * (0.143936 - 0.329415)
            colors[i] = [r, g, b]
    
    return colors.astype(np.float32)

def load_pcd_as_points(stage, prim_path, pcd_path,
                       point_size=0.01, fallback_color=(0.0, 0.6, 1.0),
                       res=0.05, rx=0, ry=0, rz=0, tx=0, ty=0, tz=0,
                       use_height_color=True, colormap='rainbow'):
    """
    把一个PCD加载为USD Points prim
    
    Args:
        use_height_color: 是否使用高度颜色渐变
        colormap: 颜色映射类型
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise RuntimeError(f"[PCD] zero points: {pcd_path}")
    N_in = pts.shape[0]

    # 颜色
    has_color = pcd.has_colors() and (len(pcd.colors) == N_in)
    cols = np.asarray(pcd.colors, dtype=np.float32) if has_color else None

    # 简单体素下采样
    grid = np.floor(pts / res).astype(np.int64)
    lin = grid[:, 0] * 134217757 + grid[:, 1] * 134217767 + grid[:, 2]
    _, first_idx = np.unique(lin, return_index=True)
    pts = pts[first_idx]
    cols = cols[first_idx] if has_color else None
    N_out = pts.shape[0]

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
        print(f"[PCD] 使用{colormap}高度颜色渐变，Z范围: {pts[:, 2].min():.2f} ~ {pts[:, 2].max():.2f}")

    print(f"[PCD] {pcd_path} N_in={N_in} -> voxel({res:.3f}) -> N_out={N_out}")

    # 转成Points Prim
    points_prim = UsdGeom.Points.Define(stage, prim_path)
    
    # 修复：使用正确的USD数据类型
    # 将numpy数组转换为Gf.Vec3f数组
    usd_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts.astype(np.float32)]
    points_prim.CreatePointsAttr(usd_points)
    
    # 宽度属性
    points_prim.CreateWidthsAttr([point_size] * N_out)

    # 颜色属性
    if cols is not None and len(cols) == N_out:
        usd_colors = [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in cols]
        points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(usd_colors)
    else:
        points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*fallback_color)])

    return N_out


if __name__ == "__main__":
    simulation_app = SimulationApp(launch_config={"headless": False})
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    # 替换成你的PCD路径
    pcd_file = "16easy.pcd"
    load_pcd_as_points(stage, "/World/PCDPoints", pcd_file,
                       point_size=math.sqrt(3) * 0.1, res=0.05, tx=0.0, ty=0.0, tz=0.0,
                       use_height_color=True, colormap='rainbow')
    
    print("点云已加载到Isaac Sim中，按Ctrl+C退出...")
    
    # 保持程序运行
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("程序退出")
    
    simulation_app.close()