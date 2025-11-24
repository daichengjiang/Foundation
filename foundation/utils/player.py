import cv2
import numpy as np
from multiprocessing import Process, Pipe, Array
import ctypes
import time
import matplotlib.pyplot as plt

class DepthViewerProcess(Process):
    """Interactive process for displaying and switching between depth images from multiple environments."""
    
    def __init__(self, img_shape, num_envs):
        super().__init__()
        self.img_shape = img_shape  # (height, width)
        self.num_envs = num_envs
        self._running = True
        
        # Shared memory for depth images
        self.shared_array = Array(ctypes.c_float, num_envs * img_shape[0] * img_shape[1])
        self.shared_imgs = np.frombuffer(self.shared_array.get_obj(), dtype=np.float32)
        self.shared_imgs = self.shared_imgs.reshape((num_envs, img_shape[0], img_shape[1]))
        
        self.parent_conn, self.child_conn = Pipe()
        self.current_env = 0
        self.show_all = False
        self.depth_window = "Depth Viewer"
        self.control_window = "Viewer Controls"
        
        # Control panel parameters
        self.control_height = 200
        self.control_width = 400
        
    def run(self):
        """Main process loop for displaying images."""
        cv2.namedWindow(self.depth_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, self.control_width, self.control_height)
        
        while self._running:
            try:
                # Check for commands from parent
                if self.child_conn.poll():
                    cmd, data = self.child_conn.recv()
                    if cmd == 'update':
                        # Update the shared array with new images
                        with self.shared_array.get_lock():
                            np.copyto(self.shared_imgs, data)
                    elif cmd == 'exit':
                        break
                
                # Get the latest depth images from shared memory
                with self.shared_array.get_lock():
                    depth_imgs = self.shared_imgs.copy()
                
                # Process and display depth image(s)
                self._update_depth_display(depth_imgs)
                
                # Update control panel
                self._update_control_panel()
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                self._handle_input(key)
                
            except Exception as e:
                print(f"Depth viewer error: {e}")
                break
        
        cv2.destroyAllWindows()
    
    def update_images(self, new_images):
        """Update the depth images from main process (non-blocking)"""
        if self.parent_conn.poll():
            self.parent_conn.recv()  # Clear old message if not processed
        self.parent_conn.send(('update', new_images))
    
    def stop(self):
        """Cleanly stop the process."""
        self._running = False
        self.parent_conn.send(('exit', None))
        self.join(timeout=1)
    
    def _update_depth_display(self, depth_imgs):
        """Update the depth image display without any text overlay."""
        if self.show_all:
            # Create a grid of all environments
            cols = int(np.ceil(np.sqrt(self.num_envs)))
            rows = int(np.ceil(self.num_envs / cols))
            
            grid_img = np.zeros((rows * self.img_shape[0] * 2, cols * self.img_shape[1] * 2, 3), dtype=np.uint8)
            
            for i in range(self.num_envs):
                row = i // cols
                col = i % cols
                
                # Normalize and convert to color
                img = depth_imgs[i]
                img = (img * 255 / (img.max() + 1e-6)).astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                img = cv2.resize(img, (self.img_shape[1]*2, self.img_shape[0]*2))
                
                # Highlight current environment
                if i == self.current_env:
                    img = cv2.rectangle(img, (0,0), (self.img_shape[1]*2-1, self.img_shape[0]*2-1), (0,255,0), 2)
                
                # Place in grid
                y_start = row * self.img_shape[0] * 2
                y_end = y_start + self.img_shape[0] * 2
                x_start = col * self.img_shape[1] * 2
                x_end = x_start + self.img_shape[1] * 2
                grid_img[y_start:y_end, x_start:x_end] = img
            
            cv2.imshow(self.depth_window, grid_img)
        else:
            # Show single environment
            img = depth_imgs[self.current_env]
            img = (img * 255 / (img.max() + 1e-6)).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            img = cv2.resize(img, (self.img_shape[1]*4, self.img_shape[0]*4))
            cv2.imshow(self.depth_window, img)
    
    def _update_control_panel(self):
        """Create a separate control panel window with buttons and status."""
        control_img = np.zeros((self.control_height, self.control_width, 3), dtype=np.uint8)
        
        # Add status text
        status_text = f"Viewing Env {self.current_env}/{self.num_envs-1}"
        mode_text = "Mode: Grid View" if self.show_all else "Mode: Single View"
        
        cv2.putText(control_img, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(control_img, mode_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        # Add control instructions
        instructions = [
            "Controls:",
            "N/→ - Next env",
            "P/← - Prev env",
            "A - Toggle view",
            "1-9 - Jump to env",
            "ESC/Q - Quit"
        ]
        
        for i, text in enumerate(instructions):
            y_pos = 90 + i * 20
            cv2.putText(control_img, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        cv2.imshow(self.control_window, control_img)
    
    def _handle_input(self, key):
        """Handle keyboard input for the viewer."""
        if key == ord('n') or key == 83:  # 'n' or right arrow
            self.current_env = (self.current_env + 1) % self.num_envs
        elif key == ord('p') or key == 81:  # 'p' or left arrow
            self.current_env = (self.current_env - 1) % self.num_envs
        elif ord('1') <= key <= ord('9'):
            env_num = key - ord('1')
            if env_num < self.num_envs:
                self.current_env = env_num
        elif key == ord('a'):
            self.show_all = not self.show_all
        elif key == 27 or key == ord('q'):  # ESC or 'q'
            self.stop()

class TerrainVisualizer:
    def __init__(self):
        self.parent_conn, self.child_conn = Pipe()
        self.visualization_process = None
        
    def start(self):
        """Start the visualization process"""
        if self.visualization_process is not None:
            return
            
        self.visualization_process = Process(
            target=self._visualization_loop,
            args=(self.child_conn,),
            daemon=True
        )
        self.visualization_process.start()
        
    def stop(self):
        """Stop the visualization process"""
        if self.visualization_process is not None:
            self.parent_conn.send(('exit', None))
            self.visualization_process.join(timeout=1)
            if self.visualization_process.is_alive():
                self.visualization_process.terminate()
            self.visualization_process = None
            
    def update_data(self, raster_map):
        """Update the visualization data (non-blocking)"""
        if self.visualization_process is None:
            return
            
        # Get categorized positions
        data = raster_map.get_positions_by_category()
        stats = raster_map.get_performance_stats()
        
        # Send only what we need for visualization
        vis_data = {
            'occupied': data['occupied']['positions'],
            'free_reachable': {
                'positions': data['free_reachable']['positions'],
                'distances': data['free_reachable']['distances']
            },
            'free_unreachable': data['free_unreachable']['positions'],
            'stats': stats
        }
        
        # Non-blocking send (will drop previous message if not processed)
        if self.parent_conn.poll():
            self.parent_conn.recv()  # Clear old message
        self.parent_conn.send(('update', vis_data))

    def update_multiple_maps(self, raster_maps):
        """Visualize multiple raster maps in subplots."""
        import matplotlib.pyplot as plt

        num_maps = len(raster_maps)
        cols = min(3, num_maps)
        rows = (num_maps + cols - 1) // cols

        fig = plt.figure(figsize=(6 * cols, 5 * rows))
        for idx, raster_map in enumerate(raster_maps):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            data = raster_map.get_positions_by_category()
            # Occupied
            occ = data['occupied']['positions']
            if len(occ) > 0:
                ax.scatter(occ[:,0], occ[:,1], occ[:,2], c='red', s=5, alpha=0.3, label='Occupied')
            # Reachable
            free = data['free_reachable']['positions']
            dists = data['free_reachable']['distances']
            if len(free) > 0:
                sc = ax.scatter(free[:,0], free[:,1], free[:,2], c=dists, cmap='viridis', s=10, alpha=0.7, label='Reachable')
                plt.colorbar(sc, ax=ax, label='Distance to goal')
            # Unreachable
            un = data['free_unreachable']['positions']
            if len(un) > 0:
                ax.scatter(un[:,0], un[:,1], un[:,2], c='gray', s=5, alpha=0.2, label='Unreachable')
            ax.set_title(f"Map {idx}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        plt.tight_layout()
        plt.show()

    def __del__(self):
        self.stop()
        
    @staticmethod
    def _visualization_loop(conn):
        """Visualization process main loop"""
        viz = MatplotlibVisualizer()
        
        while True:
            if conn.poll():
                cmd, data = conn.recv()
                if cmd == 'exit':
                    break
                elif cmd == 'update':
                    viz.update(data)
            
            viz.refresh()
            time.sleep(0.05)
            
        viz.close()

class BaseVisualizer:
    def update(self, data):
        raise NotImplementedError
        
    def refresh(self):
        pass
        
    def close(self):
        pass

class MatplotlibVisualizer(BaseVisualizer):
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatters = {}

    def clear_plots(self):
        # 清除所有已绘制的 scatter
        for scatter in self.scatters.values():
            try:
                scatter.remove()
            except Exception:
                pass
        self.scatters = {}
        # 清除 colorbar（如果有）
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

    def update(self, data):
        self.clear_plots()
        self.colorbar = None

        # Occupied cells
        if len(data['occupied']) > 0:
            occ = data['occupied']
            self.scatters['occupied'] = self.ax.scatter(
                occ[:,0], occ[:,1], occ[:,2],
                c='red', s=5, label='Occupied', alpha=0.3
            )

        # Reachable free cells
        if len(data['free_reachable']['positions']) > 0:
            free = data['free_reachable']['positions']
            dists = data['free_reachable']['distances']
            self.scatters['reachable'] = self.ax.scatter(
                free[:,0], free[:,1], free[:,2],
                c=dists, cmap='viridis', s=10, 
                label='Reachable', alpha=0.7
            )
            self.colorbar = self.fig.colorbar(self.scatters['reachable'], ax=self.ax, label='Distance to goal')

        # Unreachable free cells
        if len(data['free_unreachable']) > 0:
            un = data['free_unreachable']
            self.scatters['unreachable'] = self.ax.scatter(
                un[:,0], un[:,1], un[:,2],
                c='gray', s=5, label='Unreachable', alpha=0.2
            )

        # Update title with stats
        stats = data['stats']
        title = (f"Terrain Map (Resolution: {stats['resolution']:.2f}m)\n"
                f"Occupied: {stats['occupied_cells']:,} | "
                f"Reachable: {stats['reachable_cells']:,} | "
                f"Free: {stats['free_cells']:,}")
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()

    def update_multiple_maps(self, raster_maps):
        """Visualize multiple raster maps in subplots."""
        import matplotlib.pyplot as plt

        num_maps = len(raster_maps)
        cols = min(3, num_maps)
        rows = (num_maps + cols - 1) // cols

        fig = plt.figure(figsize=(6 * cols, 5 * rows))
        for idx, raster_map in enumerate(raster_maps):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            data = raster_map.get_positions_by_category()
            # Occupied
            occ = data['occupied']['positions']
            if len(occ) > 0:
                ax.scatter(occ[:,0], occ[:,1], occ[:,2], c='red', s=5, alpha=0.3, label='Occupied')
            # Reachable
            free = data['free_reachable']['positions']
            dists = data['free_reachable']['distances']
            if len(free) > 0:
                sc = ax.scatter(free[:,0], free[:,1], free[:,2], c=dists, cmap='viridis', s=10, alpha=0.7, label='Reachable')
                plt.colorbar(sc, ax=ax, label='Distance to goal')
            # Unreachable
            un = data['free_unreachable']['positions']
            if len(un) > 0:
                ax.scatter(un[:,0], un[:,1], un[:,2], c='gray', s=5, alpha=0.2, label='Unreachable')
            ax.set_title(f"Map {idx}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        plt.tight_layout()
        plt.show()

    def refresh(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)