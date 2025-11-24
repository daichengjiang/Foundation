import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from raster import TerrainRasterMap
from player import TerrainVisualizer, MatplotlibVisualizer
import time
def plot_all_raster_maps():
    grid_rows = 1
    grid_cols = 1
    terrain_length = 65
    terrain_width = 20
    too_high = 4.0
    too_low = 0.1
    raster_resolution = 0.1
    maps = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            map_min = [j*terrain_length, i*terrain_width, too_low]
            map_max = [(j+1)*terrain_length, (i+1)*terrain_width, too_high]
            map = TerrainRasterMap(map_min, map_max, raster_resolution, raster_resolution)
            # map.load_from_file(f"RASTER/raster_map_{i}_{j}.npz")
            # map.load_from_file(f"RASTER/raster_map_one.npz")
            map.load_from_file(f"RASTER/raster_map_output_pointcloud.npz")
            maps.append(map)

    for idx, map in enumerate(maps):
        print(f"Reachable ratio for map {idx}: {map.get_reachability_ratio()}")

    visualizer = TerrainVisualizer()
    visualizer.start()
    visualizer.update_data(maps[0])
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting visualizer...")
        visualizer.stop()

if __name__ == "__main__":
    plot_all_raster_maps()