import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion (x, y, z, w) to rotation matrix"""
    r = Rotation.from_quat(quat)
    return r.as_matrix()


def depth_to_pointcloud(
    rgb_image, depth_image, depth_min, depth_max, intrinsics, camera_pos, camera_quat, subsample_factor=1
):
    """Convert RGB and depth images to 3D point cloud

    Args:
        rgb_image: RGB image (H, W, 3) with values in [0, 1]
        depth_image: Depth image (H, W) with values in [0, 1]
        depth_min: Minimum depth value
        depth_max: Maximum depth value
        intrinsics: Camera intrinsics matrix (3, 3)
        camera_pos: Camera position (x, y, z)
        camera_quat: Camera quaternion (x, y, z, w)
        subsample_factor: Factor to subsample points (for performance)

    Returns:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3)
    """
    # Get image dimensions
    height, width = depth_image.shape

    # Convert normalized depth to actual depth
    actual_depth = depth_image * (depth_max - depth_min) + depth_min

    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Subsample for performance if needed
    if subsample_factor > 1:
        u = u[::subsample_factor, ::subsample_factor]
        v = v[::subsample_factor, ::subsample_factor]
        actual_depth = actual_depth[::subsample_factor, ::subsample_factor]
        rgb_image = rgb_image[::subsample_factor, ::subsample_factor]

    # Flatten arrays
    u = u.flatten()
    v = v.flatten()
    depth = actual_depth.flatten()
    colors = rgb_image.reshape(-1, 3)

    # Remove invalid depth points
    valid_mask = (depth > 0) & (depth < depth_max)
    u = u[valid_mask]
    v = v[valid_mask]
    depth = depth[valid_mask]
    colors = colors[valid_mask]

    # Convert pixel coordinates to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 3D points in camera coordinate system
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    # Stack to get camera coordinates
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # Convert camera coordinates to world coordinates
    # Apply rotation and translation
    rotation_matrix = quaternion_to_rotation_matrix(camera_quat)
    points_world = (rotation_matrix @ points_cam.T).T + camera_pos

    return points_world, colors


def visualize_pointcloud(points, colors, title="Point Cloud Visualization"):
    """Visualize point cloud using plotly

    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) in [0, 1] range
        title: Plot title
    """
    # Convert colors to plotly format (0-255 range)
    colors_plotly = colors
    color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors_plotly]

    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=color_strings,
                ),
                text=[f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in points[:1000]],  # Limit hover text for performance
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=800,
        height=600,
    )

    return fig


def main():
    """Main function to demonstrate point cloud generation and visualization"""
    pointss = []
    colorss = []
    for i in range(1, 10):
        data = np.load(f"tmp_metadata_{i}.npz", allow_pickle=True)
        rgb_image = data["rgb"]
        depth_image = data["depth"]
        depth_min = data["depth_min"]
        depth_max = data["depth_max"]
        intrinsics = data["intrinsics"]
        camera_pos = data["cam_pos"]
        camera_quat = data["cam_quat_ros"]

        camera_quat = camera_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)

        # Generate point cloud
        points, colors = depth_to_pointcloud(
            rgb_image=rgb_image,
            depth_image=depth_image,
            depth_min=depth_min,
            depth_max=depth_max,
            intrinsics=intrinsics,
            camera_pos=camera_pos,
            camera_quat=camera_quat,
            subsample_factor=2,  # Subsample for performance
        )
        pointss.append(points)
        colorss.append(colors)
    points = np.concatenate(pointss, axis=0)
    colors = np.concatenate(colorss, axis=0)

    print(f"Generated {len(points)} points")

    # Visualize point cloud
    fig = visualize_pointcloud(points, colors, "RGB-D Point Cloud")
    fig.show()

    # Print some statistics
    print("\nPoint cloud statistics:")
    print(f"Number of points: {len(points)}")
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")


if __name__ == "__main__":
    main()
