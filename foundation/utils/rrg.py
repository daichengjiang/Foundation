import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict, deque

class TerrainRRGMap:
    def __init__(self, occ_kdtree, map_min, map_max, step_size=0.5, neighbor_radius=1.0, 
                 collision_radius=0.2, step=0.1, goal_bias=0.1):
        self.occ_kdtree = occ_kdtree
        self.map_min = np.array(map_min)
        self.map_max = np.array(map_max)
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.collision_radius = collision_radius
        self.step = step
        self.goal_bias = goal_bias  # Probability to sample goal directly
        self.nodes = []
        self.edges = defaultdict(list)  # More efficient edge storage
        self.node_kdtree = None
        self.dimension = len(map_min)  # Works for both 2D and 3D

    def is_collision_free(self, p1, p2):
        """Optimized collision checking with early termination"""
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        
        # Handle case where p1 and p2 are the same point
        if dist == 0:
            # Check collision at the single point
            return self.occ_kdtree.query(p1, distance_upper_bound=self.collision_radius)[0] >= self.collision_radius
        
        direction = vec / dist
        steps = np.arange(0, dist + self.step, self.step)
        
        for d in steps:
            interp = p1 + direction * d
            # Early termination if collision found
            if self.occ_kdtree.query(interp, distance_upper_bound=self.collision_radius)[0] < self.collision_radius:
                return False
        return True

    def sample_point(self, goal):
        """Biased sampling with goal biasing"""
        if np.random.random() < self.goal_bias:
            return np.array(goal)
        sampled = np.random.uniform(low=self.map_min, high=self.map_max, size=self.dimension)
        
        # Ensure no NaN or inf values
        if np.any(np.isnan(sampled)) or np.any(np.isinf(sampled)):
            return np.array(goal)  # Fallback to goal if sampling fails
        
        return sampled

    def validate_inputs(self, start, goal):
        """Validate input parameters to prevent runtime errors"""
        start = np.array(start)
        goal = np.array(goal)
        
        # Check for NaN or inf values
        if np.any(np.isnan(start)) or np.any(np.isinf(start)):
            raise ValueError("Start point contains NaN or inf values")
        if np.any(np.isnan(goal)) or np.any(np.isinf(goal)):
            raise ValueError("Goal point contains NaN or inf values")
        
        # Check if within bounds
        if np.any(start < self.map_min) or np.any(start > self.map_max):
            raise ValueError("Start point is outside map bounds")
        if np.any(goal < self.map_min) or np.any(goal > self.map_max):
            raise ValueError("Goal point is outside map bounds")
        
        # Check if start and goal are the same
        if np.linalg.norm(start - goal) < 1e-6:
            raise ValueError("Start and goal points are identical")
        
        return start, goal

    def extract_shortest_path(self, start_idx, goal_idx):
        """BFS回溯最短路径，返回节点索引序列"""
        queue = deque([start_idx])
        visited = {start_idx: None}
        while queue:
            current = queue.popleft()
            if current == goal_idx:
                break
            for neighbor in self.edges[current]:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        # 回溯路径
        if goal_idx not in visited:
            return []
        path = [goal_idx]
        while path[-1] != start_idx:
            path.append(visited[path[-1]])
        path.reverse()
        return path

    def extract_shortest_path_dijkstra(self, start_idx, goal_idx):
        """Dijkstra算法提取欧氏距离最短路径，返回节点索引序列"""
        import heapq
        n = len(self.nodes)
        dist = {i: float('inf') for i in range(n)}
        prev = {i: None for i in range(n)}
        dist[start_idx] = 0
        heap = [(0, start_idx)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == goal_idx:
                break
            if d > dist[u]:
                continue
            for v in self.edges[u]:
                weight = np.linalg.norm(self.nodes[u] - self.nodes[v])
                if dist[v] > dist[u] + weight:
                    dist[v] = dist[u] + weight
                    prev[v] = u
                    heapq.heappush(heap, (dist[v], v))
        # 回溯路径
        if dist[goal_idx] == float('inf'):
            return []
        path = [goal_idx]
        while path[-1] != start_idx:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def plan(self, start, goal, max_iter=1000):
        # Validate inputs
        start, goal = self.validate_inputs(start, goal)
        
        self.nodes = [start]
        self.edges = defaultdict(list)
        self.node_kdtree = KDTree([start])
        self.shortest_path = []
        self.all_shortest_paths = dict()
        goal_idx = None
        
        for _ in range(max_iter):
            # Biased sampling
            rand_point = self.sample_point(goal)
            
            # Find nearest node
            d, idx = self.node_kdtree.query(rand_point)
            nearest = self.nodes[idx]
            
            # Steer towards random point
            direction = rand_point - nearest
            dist = np.linalg.norm(direction)
            
            # Skip if the random point is too close to the nearest node
            if dist < 1e-6:  # Very small threshold to avoid numerical issues
                continue
                
            if dist > self.step_size:
                direction = direction / dist * self.step_size
                new_point = nearest + direction
            else:
                new_point = rand_point.copy()
            
            # 限制new_point在矩形边界内
            if np.any(new_point < self.map_min) or np.any(new_point > self.map_max):
                continue
            
            # Skip if collision
            if not self.is_collision_free(nearest, new_point):
                continue
            
            # Skip if new_point is too close to any existing node (prevent duplicates)
            if self.node_kdtree.query(new_point)[0] < 1e-6:
                continue
                
            # Add new node
            new_idx = len(self.nodes)
            self.nodes.append(new_point)
            self.edges[idx].append(new_idx)
            self.edges[new_idx].append(idx)  # 保证最近邻连接为双向
            self.node_kdtree = KDTree(self.nodes)  # Rebuild KDTree
            
            # Connect to neighbors
            neighbor_indices = self.node_kdtree.query_ball_point(new_point, self.neighbor_radius)
            for n_idx in neighbor_indices:
                if n_idx != new_idx and self.is_collision_free(self.nodes[n_idx], new_point):
                    self.edges[n_idx].append(new_idx)
                    self.edges[new_idx].append(n_idx)
            
            # Check goal condition
            goal_distance = np.linalg.norm(new_point - goal)
            if goal_distance <= self.step_size and goal_distance > 1e-6 and self.is_collision_free(new_point, goal):
                goal_idx = len(self.nodes)
                self.nodes.append(goal)
                self.edges[new_idx].append(goal_idx)
                self.edges[goal_idx].append(new_idx)  # Ensure bidirectional connection
                break
        # 路径提取（优先Dijkstra）
        if goal_idx is not None:
            self.shortest_path = self.extract_shortest_path_dijkstra(0, goal_idx)
            # 记录每个节点到goal的最短路径
            for i in range(len(self.nodes)):
                self.all_shortest_paths[i] = self.extract_shortest_path_dijkstra(i, goal_idx)
        else:
            self.shortest_path = []
            self.all_shortest_paths = dict()
        return self.nodes, self.edges, self.shortest_path