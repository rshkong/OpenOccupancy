import numpy as np
import open3d as o3d
import sys
import os

"""
Usage: 
    conda activate OpenOccupancy-4070
    python tools/show_npy.py <path_to_npy_file>

Example:
    python tools/show_npy.py visualization_results/e005041f659c47e194cd5b18ea6fc346/af84a45530cf448799aefbfd2a7187ad/pred_c.npy
"""

# Color map for the 17 nuScenes classes
color_map = {
    0: [0, 0, 0],          # noise / free
    1: [112, 128, 144],    # barrier
    2: [220, 20, 60],      # bicycle
    3: [255, 127, 80],     # bus
    4: [255, 158, 0],      # car
    5: [233, 150, 70],     # construction_vehicle
    6: [255, 61, 99],      # motorcycle
    7: [0, 0, 230],        # pedestrian
    8: [47, 79, 79],       # traffic_cone
    9: [255, 140, 0],      # trailer
    10: [255, 99, 71],     # truck
    11: [0, 207, 191],     # driveable_surface
    12: [175, 0, 75],      # other_flat
    13: [75, 0, 75],       # sidewalk
    14: [112, 180, 60],    # terrain
    15: [222, 184, 135],   # manmade
    16: [0, 175, 0],       # vegetation
}

def visualize_npy(npy_path):
    if not os.path.exists(npy_path):
        print(f"Error: File {npy_path} does not exist.")
        sys.exit(1)

    print(f"Loading {npy_path}...")
    data = np.load(npy_path)
    
    # coordinates [x, y, z] and labels [class_id]
    coords = data[:, :3]
    labels = data[:, 3].astype(int)
    
    # Filter out free space (class 0 or specific free_id if it's 17)
    # usually 17 is free space in NuScenes occ, let's keep 1-16
    mask = (labels > 0) & (labels < 17)
    coords = coords[mask]
    labels = labels[mask]
    
    colors = np.array([color_map.get(lbl, [128, 128, 128]) for lbl in labels]) / 255.0
    
    # PointCloud representation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Optional: Convert to VoxelGrid for better visualization
    # Voxel size corresponds to the resolution of the occupancy grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0) # Assume 1.0 grid size, adjust if needed

    print("Showing 3D Visualizer... (Press 'Q' or 'Esc' to close)")
    # Use coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([voxel_grid, coord_frame], window_name="OpenOccupancy 3D Viewer")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        visualize_npy(sys.argv[1])
