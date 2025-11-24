import numpy as np
import heapq
from collections import deque
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')


class TerrainRasterMap:
    """
    3D terrain raster map that divides space into cubes and calculates shortest path distances to goal.
    """
    
    def __init__(self, map_min, map_max, resolution=0.5, collision_radius=0.09):
        """
        Initialize the raster map.
        
        Args:
            map_min: [x_min, y_min, z_min] - minimum bounds of the map
            map_max: [x_max, y_max, z_max] - maximum bounds of the map
            resolution: size of each cube (meters)
            collision_radius: radius for collision checking with obstacles
        """
        self.map_min = np.array(map_min, dtype=float)
        self.map_max = np.array(map_max, dtype=float)
        self.resolution = resolution
        self.collision_radius = collision_radius
        
        # Calculate grid dimensions
        self.grid_size = ((self.map_max - self.map_min) / resolution).astype(int)
        
        # Initialize grids
        self.occupancy_grid = np.zeros(self.grid_size, dtype=bool)  # True = occupied
        self.distance_grid = np.full(self.grid_size, np.inf, dtype=float)  # Distance to goal
        self.parent_grid = None  # Store parent grid positions for shortest path
        self.esdf_grid = None
        self.esdf_gradient_grid = None
        self.goal_position = None
        self.dilated_indices = None  # Store indices of dilated obstacles
    
    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid indices."""
        world_pos = np.asarray(world_pos)
        
        # Handle single position (1D array) or multiple positions (2D array)
        if world_pos.ndim == 1:
            # Single position
            grid_pos = ((world_pos - self.map_min) / self.resolution).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_size) - 1)
            return grid_pos
        else:
            # Multiple positions
            grid_pos = ((world_pos - self.map_min) / self.resolution).astype(int)
            grid_pos = np.clip(grid_pos, [0, 0, 0], np.array(self.grid_size) - 1)
            return grid_pos
        
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates (cube center)."""
        grid_pos = np.asarray(grid_pos)
        
        # Handle single position or multiple positions
        world_pos = self.map_min + (grid_pos + 0.5) * self.resolution
        return world_pos
    
    def is_valid_grid_pos(self, grid_pos):
        """Check if grid position(s) are within bounds."""
        grid_pos = np.asarray(grid_pos)
        
        if grid_pos.ndim == 1:
            # Single position
            return (np.all(grid_pos >= 0) and 
                    np.all(grid_pos < self.grid_size))
        else:
            # Multiple positions
            return (np.all(grid_pos >= 0, axis=1) & 
                    np.all(grid_pos < self.grid_size, axis=1))
        
    def bresenham_3d_single(self, start_grid, end_grid):
        """
        3D Bresenham line algorithm for a single ray.
        
        Args:
            start_grid: (3,) numpy array of start grid coordinates
            end_grid: (3,) numpy array of end grid coordinates
        
        Returns:
            voxels: list of (x, y, z) tuples representing traversed voxels
        """
        start_grid = np.asarray(start_grid, dtype=int)
        end_grid = np.asarray(end_grid, dtype=int)
        
        # Calculate differences and absolute values
        dx = abs(end_grid[0] - start_grid[0])
        dy = abs(end_grid[1] - start_grid[1])
        dz = abs(end_grid[2] - start_grid[2])
        
        # Determine step directions
        sx = 1 if start_grid[0] < end_grid[0] else -1
        sy = 1 if start_grid[1] < end_grid[1] else -1
        sz = 1 if start_grid[2] < end_grid[2] else -1
        
        # Find the major axis (axis with largest change)
        if dx >= dy and dx >= dz:
            # X is the major axis
            return self._bresenham_3d_x_major(start_grid, end_grid, dx, dy, dz, sx, sy, sz)
        elif dy >= dx and dy >= dz:
            # Y is the major axis
            return self._bresenham_3d_y_major(start_grid, end_grid, dx, dy, dz, sx, sy, sz)
        else:
            # Z is the major axis
            return self._bresenham_3d_z_major(start_grid, end_grid, dx, dy, dz, sx, sy, sz)

    def _bresenham_3d_x_major(self, start, end, dx, dy, dz, sx, sy, sz):
        """Bresenham 3D with X as major axis."""
        voxels = []
        x, y, z = start[0], start[1], start[2]
        
        err_y = dx // 2
        err_z = dx // 2
        
        for _ in range(dx + 1):
            # Check if current voxel is within grid bounds
            if (0 <= x < self.grid_size[0] and 
                0 <= y < self.grid_size[1] and 
                0 <= z < self.grid_size[2]):
                voxels.append((x, y, z))
            
            if x == end[0]:
                break
                
            err_y -= dy
            err_z -= dz
            
            if err_y < 0:
                err_y += dx
                y += sy
                
            if err_z < 0:
                err_z += dx
                z += sz
                
            x += sx
        
        return voxels

    def _bresenham_3d_y_major(self, start, end, dx, dy, dz, sx, sy, sz):
        """Bresenham 3D with Y as major axis."""
        voxels = []
        x, y, z = start[0], start[1], start[2]
        
        err_x = dy // 2
        err_z = dy // 2
        
        for _ in range(dy + 1):
            # Check if current voxel is within grid bounds
            if (0 <= x < self.grid_size[0] and 
                0 <= y < self.grid_size[1] and 
                0 <= z < self.grid_size[2]):
                voxels.append((x, y, z))
            
            if y == end[1]:
                break
                
            err_x -= dx
            err_z -= dz
            
            if err_x < 0:
                err_x += dy
                x += sx
                
            if err_z < 0:
                err_z += dy
                z += sz
                
            y += sy
        
        return voxels

    def _bresenham_3d_z_major(self, start, end, dx, dy, dz, sx, sy, sz):
        """Bresenham 3D with Z as major axis."""
        voxels = []
        x, y, z = start[0], start[1], start[2]
        
        err_x = dz // 2
        err_y = dz // 2
        
        for _ in range(dz + 1):
            # Check if current voxel is within grid bounds
            if (0 <= x < self.grid_size[0] and 
                0 <= y < self.grid_size[1] and 
                0 <= z < self.grid_size[2]):
                voxels.append((x, y, z))
            
            if z == end[2]:
                break
                
            err_x -= dx
            err_y -= dy
            
            if err_x < 0:
                err_x += dz
                x += sx
                
            if err_y < 0:
                err_y += dz
                y += sy
                
            z += sz
        
        return voxels
    
    def bresenham_3d_batch(self, start_world, end_world_array):
        """
        Batch process multiple rays using Bresenham 3D algorithm.
        
        Args:
            start_world: (3,) world coordinates of ray start point
            end_world_array: (N, 3) world coordinates of ray end points
        
        Returns:
            all_voxels: list of lists, each containing voxel coordinates for one ray
            valid_rays: list of booleans indicating which rays have valid voxels
        """
        start_world = np.asarray(start_world)
        end_world_array = np.asarray(end_world_array)
        
        if end_world_array.ndim == 1:
            end_world_array = end_world_array.reshape(1, -1)
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_world)
        end_grid_array = self.world_to_grid(end_world_array)
        
        all_voxels = []
        valid_rays = []
        
        for end_grid in end_grid_array:
            # Skip if start and end are the same
            if np.array_equal(start_grid, end_grid):
                all_voxels.append([])
                valid_rays.append(False)
                continue
            
            # Get voxels for this ray
            voxels = self.bresenham_3d_single(start_grid, end_grid)
            
            if len(voxels) > 0:
                all_voxels.append(voxels)
                valid_rays.append(True)
            else:
                all_voxels.append([])
                valid_rays.append(False)

        return all_voxels, valid_rays
    
    def bresenham_3d_batch_with_occupancy(self, start_world, end_world_array):
        """
        Batch process multiple rays using Bresenham 3D algorithm and classify voxels by occupancy.
        
        Args:
            start_world: (3,) world coordinates of ray start point
            end_world_array: (N, 3) world coordinates of ray end points
        
        Returns:
            occupied_voxels: list of lists, each containing occupied voxel grid coordinates for one ray
            free_voxels: list of lists, each containing free voxel grid coordinates for one ray
            valid_rays: list of booleans indicating which rays have valid voxels
        """
        start_world = np.asarray(start_world)
        end_world_array = np.asarray(end_world_array)
        
        if end_world_array.ndim == 1:
            end_world_array = end_world_array.reshape(1, -1)
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_world)
        end_grid_array = self.world_to_grid(end_world_array)
        
        occupied_voxels = []
        free_voxels = []
        valid_rays = []
        
        for end_grid in end_grid_array:
            # Skip if start and end are the same
            if np.array_equal(start_grid, end_grid):
                occupied_voxels.append([])
                free_voxels.append([])
                valid_rays.append(False)
                continue
            
            # Get voxels for this ray
            voxels = self.bresenham_3d_single(start_grid, end_grid)
            
            if len(voxels) > 0:
                # Classify voxels by occupancy status
                ray_occupied_voxels = []
                ray_free_voxels = []
                
                for voxel in voxels:
                    x, y, z = voxel
                    is_occupied = self.occupancy_grid[x, y, z]
                    
                    if is_occupied:
                        ray_occupied_voxels.append(voxel)
                    else:
                        ray_free_voxels.append(voxel)
                
                occupied_voxels.append(ray_occupied_voxels)
                free_voxels.append(ray_free_voxels)
                valid_rays.append(True)
            else:
                occupied_voxels.append([])
                free_voxels.append([])
                valid_rays.append(False)

        return occupied_voxels, free_voxels, valid_rays

    def bresenham_3d_batch_with_truncation(self, start_world, end_world_array):
        """
        Batch process multiple rays with early termination at first obstacle.
        Returns only free space voxels up to the first collision.
        
        Args:
            start_world: (3,) world coordinates of ray start point
            end_world_array: (N, 3) world coordinates of ray end points
        
        Returns:
            free_voxels: list of lists, each containing free voxel grid coordinates before first obstacle
            first_obstacle_voxels: list of first obstacle voxel coordinates (or None if no obstacle)
            valid_rays: list of booleans indicating which rays have valid voxels
        """
        start_world = np.asarray(start_world)
        end_world_array = np.asarray(end_world_array)
        
        if end_world_array.ndim == 1:
            end_world_array = end_world_array.reshape(1, -1)
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_world)
        end_grid_array = self.world_to_grid(end_world_array)
        
        free_voxels = []
        first_obstacle_voxels = []
        valid_rays = []
        
        for end_grid in end_grid_array:
            # Skip if start and end are the same
            if np.array_equal(start_grid, end_grid):
                free_voxels.append([])
                first_obstacle_voxels.append(None)
                valid_rays.append(False)
                continue
            
            # Get voxels for this ray
            voxels = self.bresenham_3d_single(start_grid, end_grid)
            
            if len(voxels) > 0:
                ray_free_voxels = []
                first_obstacle = None
                
                # Process voxels in order until we hit an obstacle
                for voxel in voxels:
                    x, y, z = voxel
                    is_occupied = self.occupancy_grid[x, y, z]
                    
                    if is_occupied:
                        # Hit first obstacle, stop processing this ray
                        first_obstacle = voxel
                        break
                    else:
                        # Add to free voxels
                        ray_free_voxels.append(voxel)
                
                free_voxels.append(ray_free_voxels)
                first_obstacle_voxels.append(first_obstacle)
                valid_rays.append(True)
            else:
                free_voxels.append([])
                first_obstacle_voxels.append(None)
                valid_rays.append(False)

        return free_voxels, first_obstacle_voxels, valid_rays

    def build_occupancy_grid_from_kdtree(self, obstacle_kdtree):
        """Build occupancy grid using KDTree for efficient spatial queries."""
        self.occupancy_grid.fill(False)
        
        # Use unified KDTree approach for all grid sizes
        self.occupancy_grid = build_occupancy_grid_kdtree(
            self.map_min, self.grid_size, self.resolution, 
            self.collision_radius, obstacle_kdtree.data
        )
        # self.occupancy_grid[:, :, 0] = True
        # self.occupancy_grid[:, :, -1] = True
        # self.occupancy_grid[0, :, :] = True
        # self.occupancy_grid[-1, :, :] = True
        # self.occupancy_grid[:, 0, :] = True
        # self.occupancy_grid[:, -1, :] = True

    def build_occupancy_grid_from_points(self, obstacle_points):
        self.occupancy_grid.fill(False)
        
        self.occupancy_grid = build_occupancy_grid_from_points(
            self.map_min, self.grid_size, self.resolution,
            self.collision_radius, obstacle_points
        )
        # self.occupancy_grid[:, :, 0] = True
        # self.occupancy_grid[:, :, -1] = True
        # self.occupancy_grid[0, :, :] = True
        # self.occupancy_grid[-1, :, :] = True
        # self.occupancy_grid[:, 0, :] = True
        # self.occupancy_grid[:, -1, :] = True

    def dilate_existing_obstacles(self, kernel_size):
        """
        Dilate existing obstacles in the occupancy grid using a spherical kernel.
        
        Args:
            kernel_size: radius of dilation in grid cells
        """
        from scipy.ndimage import binary_dilation
        
        # Create spherical kernel
        x, y, z = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
        distances = np.sqrt(x**2 + y**2 + z**2)
        kernel = distances <= kernel_size

        occupancy_grid_before = self.occupancy_grid.copy()
        # Apply dilation
        self.occupancy_grid = binary_dilation(self.occupancy_grid, structure=kernel)
        dilated_mask = self.occupancy_grid & (~occupancy_grid_before)
        self.dilated_indices = np.argwhere(dilated_mask)
        
        # If distance grid computed, warn about recalculation
        if self.goal_position is not None:
            print("[WARNING] Dilation applied, distance grid may need recalculation.")

    def compute_distances_to_goal(self, goal_world_pos, connectivity=26):
        """Compute shortest path distances using optimized Dijkstra's algorithm."""
        self.goal_position = np.array(goal_world_pos)
        goal_grid_pos = self.world_to_grid(goal_world_pos)
        
        # Check if goal is in obstacle, find nearest free space if needed
        if self.occupancy_grid[tuple(goal_grid_pos)]:
            goal_grid_pos = find_nearest_free_space(self.occupancy_grid, self.grid_size, goal_grid_pos)
        
        # Use optimized Dijkstra implementation
        self.distance_grid, self.parent_grid = dijkstra_algorithm_with_parent(
            self.occupancy_grid, self.grid_size, tuple(goal_grid_pos), connectivity, self.resolution
        )
    
    def compute_distance_to_goal_larger_dilation(self, goal_world_pos, connectivity=26, kernel_size=2):
        from scipy.ndimage import binary_dilation
        # Create spherical kernel
        x, y, z = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
        distances = np.sqrt(x**2 + y**2 + z**2)
        kernel = distances <= kernel_size
        occupancy_grid = binary_dilation(self.occupancy_grid, structure=kernel)

        self.goal_position = np.array(goal_world_pos)
        goal_grid_pos = self.world_to_grid(goal_world_pos)
        # Check if goal is in obstacle, find nearest free space if needed
        if occupancy_grid[tuple(goal_grid_pos)]:
            goal_grid_pos = find_nearest_free_space(occupancy_grid, self.grid_size, goal_grid_pos)
        
        # Use optimized Dijkstra implementation
        self.distance_grid, self.parent_grid = dijkstra_algorithm_with_parent(
            occupancy_grid, self.grid_size, tuple(goal_grid_pos), connectivity, self.resolution
        )

    def compute_esdf(self):
        """计算欧几里得符号距离场 (ESDF)"""
        from scipy.ndimage import distance_transform_edt
        
        print("正在计算ESDF...")
        
        # 检查是否有障碍物
        if not np.any(self.occupancy_grid):
            print("警告：没有检测到障碍物，ESDF将全为正无穷")
            self.esdf_grid = np.full(self.grid_size, np.inf)
            return
        
        # 检查是否全是障碍物
        if np.all(self.occupancy_grid):
            print("警告：全部都是障碍物，ESDF将全为负无穷")
            self.esdf_grid = np.full(self.grid_size, -np.inf)
            return
        
        # 计算到最近障碍物的距离（对自由空间）
        distance_to_obstacles = distance_transform_edt(
            ~self.occupancy_grid,  # 在自由空间中计算到边界的距离
            sampling=self.resolution
        )
        
        # 计算到最近自由空间的距离（对占用空间）
        distance_to_free = distance_transform_edt(
            self.occupancy_grid,   # 在占用空间中计算到边界的距离
            sampling=self.resolution
        )
        
        # 构建ESDF
        # 自由空间：正值（距离最近障碍物的距离）
        # 占用空间：负值（负的距离最近自由空间的距离）
        self.esdf_grid = np.where(
            self.occupancy_grid,
            -distance_to_free,      # 占用区域：负值
            distance_to_obstacles   # 自由区域：正值
        )
        
        # 打印统计信息
        occupied_esdf = self.esdf_grid[self.occupancy_grid]
        free_esdf = self.esdf_grid[~self.occupancy_grid]
        
        print(f"ESDF计算完成:")
        print(f"  占用区域数量: {np.sum(self.occupancy_grid)}")
        print(f"  自由区域数量: {np.sum(~self.occupancy_grid)}")
        
        if len(occupied_esdf) > 0:
            print(f"  占用区域ESDF范围: [{occupied_esdf.min():.3f}, {occupied_esdf.max():.3f}]")
        if len(free_esdf) > 0:
            print(f"  自由区域ESDF范围: [{free_esdf.min():.3f}, {free_esdf.max():.3f}]")
        
        print(f"  总体ESDF范围: [{self.esdf_grid.min():.3f}, {self.esdf_grid.max():.3f}]")
        
        # 验证ESDF的正确性
        zero_crossing = np.abs(self.esdf_grid) < self.resolution
        zero_count = np.sum(zero_crossing)
        print(f"  零值附近的网格数量: {zero_count} (应该在障碍物边界)")

    def debug_esdf_computation(self):
        """调试ESDF计算过程"""
        print("=== ESDF调试信息 ===")
        print(f"网格形状: {self.occupancy_grid.shape}")
        print(f"网格总数: {np.prod(self.occupancy_grid.shape)}")
        print(f"占用网格数: {np.sum(self.occupancy_grid)}")
        print(f"自由网格数: {np.sum(~self.occupancy_grid)}")
        print(f"占用率: {np.sum(self.occupancy_grid) / np.prod(self.occupancy_grid.shape) * 100:.2f}%")
        
        # 检查边界
        boundary_occupied = (
            np.any(self.occupancy_grid[0, :, :]) or np.any(self.occupancy_grid[-1, :, :]) or
            np.any(self.occupancy_grid[:, 0, :]) or np.any(self.occupancy_grid[:, -1, :]) or
            np.any(self.occupancy_grid[:, :, 0]) or np.any(self.occupancy_grid[:, :, -1])
        )
        print(f"边界是否有障碍物: {boundary_occupied}")
        
        # 检查连通性
        if hasattr(self, 'esdf_grid') and self.esdf_grid is not None:
            print(f"ESDF网格形状: {self.esdf_grid.shape}")
            print(f"ESDF数据类型: {self.esdf_grid.dtype}")
            print(f"ESDF值范围: [{self.esdf_grid.min():.6f}, {self.esdf_grid.max():.6f}]")
            print(f"ESDF唯一值数量: {len(np.unique(self.esdf_grid))}")
            
            # 检查是否有NaN或无穷值
            nan_count = np.sum(np.isnan(self.esdf_grid))
            inf_count = np.sum(np.isinf(self.esdf_grid))
            print(f"NaN值数量: {nan_count}")
            print(f"无穷值数量: {inf_count}")

    def batch_query_esdf(self, world_positions):
        """
        Batch query Euclidean Signed Distance Field (ESDF) for multiple world positions.
        """
        
        world_positions = np.asarray(world_positions)
        
        # 处理单点查询
        if world_positions.ndim == 1:
            world_positions = world_positions.reshape(1, -1)
            return_single = True
        else:
            return_single = False
        
        # 转换为网格坐标
        grid_positions = self.world_to_grid(world_positions)
        
        # 检查边界
        valid_mask = self.is_valid_grid_pos(grid_positions)
        
        # 初始化结果
        esdf_values = np.full(len(world_positions), np.inf, dtype=float)
        
        if np.any(valid_mask):
            valid_grid_pos = grid_positions[valid_mask]
            esdf_values[valid_mask] = self.esdf_grid[
                valid_grid_pos[:, 0],
                valid_grid_pos[:, 1], 
                valid_grid_pos[:, 2]
            ]
        
        return esdf_values[0] if return_single else esdf_values

    def trilinear_interpolate_esdf(self, world_pos):
        world_pos = np.asarray(world_pos)
        grid_coords = (world_pos - self.map_min) / self.resolution
        x, y, z = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
        z0 = np.floor(z).astype(int)
        z1 = z0 + 1

        # Clamp indices to grid bounds
        max_x, max_y, max_z = np.array(self.esdf_grid.shape) - 1
        x0 = np.clip(x0, 0, max_x)
        x1 = np.clip(x1, 0, max_x)
        y0 = np.clip(y0, 0, max_y)
        y1 = np.clip(y1, 0, max_y)
        z0 = np.clip(z0, 0, max_z)
        z1 = np.clip(z1, 0, max_z)

        xd = x - x0
        yd = y - y0
        zd = z - z0

        c000 = self.esdf_grid[x0, y0, z0]
        c001 = self.esdf_grid[x0, y0, z1]
        c010 = self.esdf_grid[x0, y1, z0]
        c011 = self.esdf_grid[x0, y1, z1]
        c100 = self.esdf_grid[x1, y0, z0]
        c101 = self.esdf_grid[x1, y0, z1]
        c110 = self.esdf_grid[x1, y1, z0]
        c111 = self.esdf_grid[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd

        grad = np.zeros_like(world_pos)
        grad[:, 2] = (c1 - c0) / self.resolution
        grad[:, 1] = ((c10 - c00) * (1 - zd) + (c11 - c01) * zd) / self.resolution
        grad[:, 0] = ((c100 - c000) * (1 - yd) * (1 - zd) + 
                      (c101 - c001) * (1 - yd) * zd + 
                      (c110 - c010) * yd * (1 - zd) + 
                      (c111 - c011) * yd * zd) / self.resolution
        return c, grad

    def trilinear_interpolate_esdf_gradient(self, world_pos):
        world_pos = np.asarray(world_pos)
        
        # 修复变量名错误
        if world_pos.ndim == 1:
            world_positions = world_pos.reshape(1, -1)
            return_single = True
        else:
            world_positions = world_pos  # 直接使用，不需要重新赋值
            return_single = False

        interpolated_gradients = np.zeros((len(world_positions), 3))

        # 对每个梯度分量进行插值
        for component in range(3):
            gradient_component = self.esdf_gradient_grid[:, :, :, component]
            interpolated_gradients[:, component] = trilinear_interpolate(
                gradient_component, world_positions, self.map_min, self.resolution
            )

        if return_single:
            return interpolated_gradients[0]
        else:
            return interpolated_gradients

    def set_custom_distance_for_invalid(self, value=-1.0):
        """Set the distance of occupied and unreachable points to a custom value."""
        mask = (self.occupancy_grid == True) | (self.distance_grid == np.inf)
        self.distance_grid[mask] = value
    
    def batch_query_distances(self, world_positions):
        """Batch query distances for multiple world positions."""
        return batch_query_distances(world_positions, self.distance_grid, self.map_min, self.grid_size, self.resolution)
    
    def check_positions_occupancy(self, world_positions):
        """Check if world positions are free and get their distances."""
        return check_positions_occupancy(world_positions, self.occupancy_grid, self.distance_grid, self.map_min, self.resolution)
    
    def save_to_file(self, filename):
        """Save raster map to file."""
        np.savez_compressed(filename,
                           occupancy_grid=self.occupancy_grid,
                           distance_grid=self.distance_grid,
                           parent_grid=self.parent_grid,
                           esdf_grid=self.esdf_grid,
                           esdf_gradient_grid=self.esdf_gradient_grid,
                           map_min=self.map_min,
                           map_max=self.map_max,
                           resolution=self.resolution,
                           collision_radius=self.collision_radius,
                           goal_position=self.goal_position,
                           dilated_indices=self.dilated_indices
                           )

    def load_from_file(self, filename):
        """Load raster map from file."""
        data = np.load(filename, allow_pickle=True)
        self.occupancy_grid = data['occupancy_grid']
        self.distance_grid = data['distance_grid']
        self.esdf_grid = data['esdf_grid'] if 'esdf_grid' in data else None
        self.esdf_gradient_grid = data['esdf_gradient_grid'] if 'esdf_gradient_grid' in data else None
        self.map_min = data['map_min']
        self.map_max = data['map_max']
        self.resolution = data['resolution']
        self.collision_radius = data['collision_radius']
        self.goal_position = data['goal_position']
        self.grid_size = self.occupancy_grid.shape
        self.dilated_indices = data.get('dilated_indices', None)

        # Load parent grid if available
        if 'parent_grid' in data and data['parent_grid'] is not None:
            self.parent_grid = data['parent_grid']
        else:
            self.parent_grid = None
    
    def trilinear_interpolate(self, world_pos):
        """Trilinear interpolation for smooth distance queries."""
        return trilinear_interpolate(self.distance_grid, world_pos, self.map_min, self.resolution)

    def get_next_step_towards_goal(self, world_positions):
        """Get next step towards goal for batch of positions."""
        world_positions = np.asarray(world_positions)
        if world_positions.ndim == 1:
            world_positions = world_positions.reshape(1, -1)
        grid_indices = self.world_to_grid(world_positions)
        next_points = []
        for idx in grid_indices:
            if not self.is_valid_grid_pos(idx):
                next_points.append(None)
                continue
            parent_idx = self.parent_grid[tuple(idx)]
            if parent_idx is None:
                next_points.append(self.grid_to_world(idx))
            else:
                next_points.append(self.grid_to_world(parent_idx))
        return np.array(next_points)
    
    def get_dilated_positions(self):
        """Get positions of dilated obstacles."""
        if self.dilated_indices is None:
            return np.empty((0, 3))
        return np.array([self.grid_to_world(idx) for idx in self.dilated_indices])
    
    def get_occupied_positions(self):
        """Get positions of occupied cubes."""
        occupied_indices = np.argwhere(self.occupancy_grid)
        if len(occupied_indices) == 0:
            return np.empty((0, 3))
        
        # Vectorized conversion to world coordinates
        world_positions = self.map_min + (occupied_indices + 0.5) * self.resolution
        return world_positions
    
    def get_free_positions(self):
        """Get positions of free cubes."""
        free_indices = np.argwhere(~self.occupancy_grid)
        if len(free_indices) == 0:
            return np.empty((0, 3))
        
        # Vectorized conversion to world coordinates
        world_positions = self.map_min + (free_indices + 0.5) * self.resolution
        return world_positions

    def get_free_and_occupied_positions_with_distances(self):
        """Get positions and distances for both free and occupied cubes."""
        # Get all grid indices
        all_indices = np.indices(self.grid_size).reshape(3, -1).T
        all_positions = self.map_min + (all_indices + 0.5) * self.resolution
        
        # Get occupancy status for all positions
        occupancy_flat = self.occupancy_grid.ravel()
        distance_flat = self.distance_grid.ravel()
        
        # Separate free and occupied
        free_mask = ~occupancy_flat
        occupied_mask = occupancy_flat
        
        return {
            'free_positions': all_positions[free_mask],
            'free_distances': distance_flat[free_mask],
            'occupied_positions': all_positions[occupied_mask],
            'occupied_distances': distance_flat[occupied_mask]
        }

    def extract_path_to_goal(self, start_world_pos):
        """
        Extract the shortest path from start position to goal using parent grid.
        
        Args:
            start_world_pos: [x, y, z] world coordinates of start position
        
        Returns:
            path: List of world coordinates representing the path from start to goal
            path_length: Total length of the path
        """
        if self.parent_grid is None:
            raise ValueError("Distance grid not computed. Call compute_distances_to_goal() first.")
        
        # Convert start position to grid coordinates
        start_grid_pos = self.world_to_grid(start_world_pos)
        
        # Check if start position is valid and reachable
        if not self.is_valid_grid_pos(start_grid_pos):
            return []
        
        if self.occupancy_grid[tuple(start_grid_pos)]:
            # Find nearest free space if start is in obstacle
            start_grid_pos = find_nearest_free_space(self.occupancy_grid, self.grid_size, start_grid_pos)
        
        # Check if position is reachable (distance is not infinite)
        if self.distance_grid[tuple(start_grid_pos)] == np.inf:
            return []
        
        # Extract path by following parent pointers
        path_grid = []
        current_pos = tuple(start_grid_pos)
        
        while True:
            path_grid.append(current_pos)
            
            # Get parent position
            parent_idx = self.parent_grid[current_pos]
            
            # If we've reached the goal (parent points to itself)
            if parent_idx == -1:
                break
            
            # Convert flat index back to 3D coordinates
            parent_pos = np.unravel_index(parent_idx, self.grid_size)
            
            # If parent is the same as current (goal reached)
            if parent_pos == current_pos:
                break
            
            current_pos = parent_pos
        
        # Convert grid coordinates to world coordinates
        path_world = [self.grid_to_world(pos) for pos in path_grid]
        
        return path_world
    
def trilinear_interpolate(grid, world_pos, map_min, resolution):
    """Trilinear interpolation for a 3D grid at given world positions."""
    world_pos = np.asarray(world_pos)
    grid_coords = (world_pos - map_min) / resolution
    x, y, z = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    # Clamp indices to grid bounds
    max_x, max_y, max_z = np.array(grid.shape) - 1
    x0 = np.clip(x0, 0, max_x)
    x1 = np.clip(x1, 0, max_x)
    y0 = np.clip(y0, 0, max_y)
    y1 = np.clip(y1, 0, max_y)
    z0 = np.clip(z0, 0, max_z)
    z1 = np.clip(z1, 0, max_z)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c000 = grid[x0, y0, z0]
    c001 = grid[x0, y0, z1]
    c010 = grid[x0, y1, z0]
    c011 = grid[x0, y1, z1]
    c100 = grid[x1, y0, z0]
    c101 = grid[x1, y0, z1]
    c110 = grid[x1, y1, z0]
    c111 = grid[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c


def build_occupancy_grid_kdtree(map_min, grid_size, resolution, collision_radius, obstacle_points):
    """Build occupancy grid using KDTree for efficient spatial queries."""
    if len(obstacle_points) == 0:
        return np.zeros(grid_size, dtype=bool)
    
    kdtree = KDTree(obstacle_points)
    gx, gy, gz = np.indices(grid_size)
    centers = np.stack([
        map_min[0] + (gx + 0.5) * resolution,
        map_min[1] + (gy + 0.5) * resolution,
        map_min[2] + (gz + 0.5) * resolution
    ], axis=-1).reshape(-1, 3)
    
    neighbors = kdtree.query_ball_point(centers, r=collision_radius)
    occupied = np.array([len(n) > 0 for n in neighbors], dtype=bool)
    return occupied.reshape(grid_size)

def build_occupancy_grid_from_points(map_min, grid_size, resolution, collision_radius, obstacle_points):
    """Build occupancy grid by converting obstacle points to grid coordinates."""
    if len(obstacle_points) == 0:
        return np.zeros(grid_size, dtype=bool)
    
    occupancy_grid = np.zeros(grid_size, dtype=bool)
    
    # Convert world coordinates to grid coordinates
    map_min = np.array(map_min)
    grid_coords = ((obstacle_points - map_min) / resolution).astype(int)
    
    # Clip to valid grid bounds
    grid_coords = np.clip(grid_coords, [0, 0, 0], np.array(grid_size) - 1)
    
    # Method 1: Simple point marking (fastest)
    occupancy_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]] = True
    
    return occupancy_grid
    

def dijkstra_algorithm_with_parent(occupancy_grid, grid_size, goal_pos, connectivity, resolution):
    """Efficient Dijkstra's algorithm with parent tracking."""
    flat_size = np.prod(grid_size)
    distance_grid = np.full(flat_size, np.inf, dtype=np.float64)
    parent_grid = np.full(flat_size, -1, dtype=np.int64)
    visited = np.zeros(flat_size, dtype=bool)
    
    def ravel(x, y, z):
        return x * grid_size[1] * grid_size[2] + y * grid_size[2] + z
    
    def unravel(idx):
        x = idx // (grid_size[1] * grid_size[2])
        y = (idx % (grid_size[1] * grid_size[2])) // grid_size[2]
        z = idx % grid_size[2]
        return x, y, z
    
    goal_idx = ravel(*goal_pos)
    distance_grid[goal_idx] = 0.0
    parent_grid[goal_idx] = goal_idx
    heap = [(0.0, goal_idx)]
    
    if connectivity == 6:
        offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        costs = [resolution] * 6
    else:
        offsets = [(dx,dy,dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==0 and dy==0 and dz==0)]
        costs = [np.sqrt(dx*dx+dy*dy+dz*dz)*resolution for dx,dy,dz in offsets]
    
    occ_flat = occupancy_grid.ravel()
    
    while heap:
        min_dist, idx = heapq.heappop(heap)
        if visited[idx]:
            continue
        visited[idx] = True
        x, y, z = unravel(idx)
        
        for i, (dx, dy, dz) in enumerate(offsets):
            nx, ny, nz = x+dx, y+dy, z+dz
            if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and 0 <= nz < grid_size[2]):
                nidx = ravel(nx, ny, nz)
                if occ_flat[nidx] or visited[nidx]:
                    continue
                new_distance = min_dist + costs[i]
                if new_distance < distance_grid[nidx]:
                    distance_grid[nidx] = new_distance
                    parent_grid[nidx] = idx
                    heapq.heappush(heap, (new_distance, nidx))
    
    return distance_grid.reshape(grid_size), parent_grid.reshape(grid_size)


def find_nearest_free_space(occupancy_grid, grid_size, occupied_pos):
    """Find nearest free space using BFS."""
    queue = deque([tuple(occupied_pos)])
    visited = np.zeros(grid_size, dtype=bool)
    offsets = [(dx,dy,dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==0 and dy==0 and dz==0)]
    
    while queue:
        x, y, z = queue.popleft()
        if visited[x, y, z]:
            continue
        visited[x, y, z] = True
        
        if (0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2] and 
            not occupancy_grid[x, y, z]):
            return np.array([x, y, z], dtype=int)
        
        for dx, dy, dz in offsets:
            nx, ny, nz = x+dx, y+dy, z+dz
            if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and 0 <= nz < grid_size[2] and 
                not visited[nx, ny, nz]):
                queue.append((nx, ny, nz))
    
    return np.array(occupied_pos, dtype=int)


def batch_query_distances(world_positions, distance_grid, map_min, grid_size, resolution):
    """Batch query distances for multiple world positions."""
    idxs = ((world_positions - map_min) / resolution).astype(int)
    idxs = np.clip(idxs, [0,0,0], np.array(grid_size)-1)
    return distance_grid[idxs[:,0], idxs[:,1], idxs[:,2]]


def check_positions_occupancy(world_positions, occupancy_grid, distance_grid, map_min, resolution):
    """Check if world positions are free and get their distances."""
    world_positions = np.asarray(world_positions)
    if len(world_positions.shape) == 1:
        world_positions = world_positions.reshape(1, -1)
        return_single = True
    else:
        return_single = False

    grid_size = occupancy_grid.shape
    grid_positions = ((world_positions - map_min) / resolution).astype(int)
    valid_positions = (
        (grid_positions[:, 0] >= 0) & (grid_positions[:, 0] < grid_size[0]) &
        (grid_positions[:, 1] >= 0) & (grid_positions[:, 1] < grid_size[1]) &
        (grid_positions[:, 2] >= 0) & (grid_positions[:, 2] < grid_size[2])
    )

    is_free_vector = np.zeros(len(world_positions), dtype=bool)
    free_distances = np.full(len(world_positions), np.inf)

    if np.any(valid_positions):
        valid_grid_pos = grid_positions[valid_positions]
        valid_is_free = ~occupancy_grid[
            valid_grid_pos[:, 0],
            valid_grid_pos[:, 1],
            valid_grid_pos[:, 2]
        ]
        is_free_vector[valid_positions] = valid_is_free
        free_distances[valid_positions] = distance_grid[
            valid_grid_pos[:, 0],
            valid_grid_pos[:, 1],
            valid_grid_pos[:, 2]
        ]

    if return_single:
        return is_free_vector[0], free_distances[0]
    return is_free_vector, free_distances