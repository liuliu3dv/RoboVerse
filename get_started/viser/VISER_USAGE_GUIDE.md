# Viser Visualization Toolkit Usage Guide

The Viser visualization toolkit provides interactive 3D visualization and control capabilities for robotics simulations. This guide covers the main features and how to use them effectively.

## Overview

The `ViserVisualizer` class offers several key features:
- **Static Scene Visualization**: Display robots, objects, and environments
- **Dynamic Scene Updates**: Real-time motion and state updates
- **Joint Control**: Interactive robot joint manipulation
- **IK Control**: Inverse kinematics for end-effector positioning
- **Trajectory Playback**: Play and control recorded trajectories
- **Camera Control**: Interactive camera positioning and recording

## Quick Start

### Basic Setup

```python
from get_started.viser_util import ViserVisualizer

# Initialize visualizer
visualizer = ViserVisualizer(port=8080)
visualizer.add_grid()
visualizer.add_frame("/world_frame")

# Access via browser
# Open http://localhost:8080
```

## Core Features

### 1. Camera Controls

**Enable**: `visualizer.enable_camera_controls()`

**Features**:
- Interactive 3D navigation (mouse drag to rotate, scroll to zoom)
- Position sliders (X, Y, Z) for precise camera positioning
- Rotation controls (Yaw, Pitch, Roll buttons)
- Preset views (Top, Side, Front)
- Screenshot capture and video recording
- Live camera view display in GUI

**GUI Workflow**:
1. Use mouse for basic navigation
2. Use position/rotation sliders for precise control
3. Click preset view buttons for standard angles
4. Take screenshots or start/stop video recording
5. Adjust recording FPS as needed

### 2. Joint Control

**Enable**: `visualizer.enable_joint_control()`

**Features**:
- Individual joint sliders for each robot DOF
- Real-time robot pose updates
- Joint limit enforcement
- Multiple robot support
- Reset to initial configurations

**GUI Workflow**:
1. Open "Joint Control" panel
2. Select robot from dropdown
3. Click "Setup Joint Control"
4. Use sliders to control individual joints
5. Click "Reset Joints" to return to initial pose
6. Use "Clear Joint Control" to remove GUI panels

### 3. IK Control

**Enable**: `visualizer.enable_ik_control()`

**Features**:
- Target position control (X, Y, Z sliders)
- Target orientation control (Quaternion W, X, Y, Z sliders)
- Visual target markers (red sphere + RGB axes)
- Real-time IK solving and application
- Integration with joint control

**GUI Workflow**:
1. Open "IK Control" panel
2. Click "Setup IK Control"
3. Adjust target position sliders
4. Adjust target orientation sliders
5. Watch red sphere and RGB axes move to show target
6. Click "Solve & Apply IK" to move robot to target
7. Use "Reset Target" to return to default position

**Visual Indicators**:
- **Red Sphere**: Target end-effector position
- **RGB Axes**: Target orientation (Red=X, Green=Y, Blue=Z)

### 4. Trajectory Playback

**Enable**: `visualizer.enable_trajectory_playback()`

**Features**:
- Load trajectory files (.pkl, .pkl.gz, .json, .yaml)
- Robot and demo selection
- Playback controls (Play, Pause, Stop)
- Timeline scrubbing
- Adjustable playback speed (FPS)

**GUI Workflow**:
1. Load trajectory in Python: `visualizer.load_trajectory('path/to/trajectory.pkl.gz')`
2. Click "Update Robot List" to refresh available robots
3. Select robot and demo index from dropdowns
4. Click "Set Current Trajectory"
5. Use playback controls:
   - **Play**: Start/resume playback
   - **Pause**: Pause playback
   - **Stop**: Stop and reset to beginning
6. Drag timeline slider to seek to specific frames
7. Adjust "Playback FPS" slider to change speed

## Demo Scripts

### 1. Static Scene Visualization

**Purpose**: Basic scene setup and camera control demonstration.

**Run Demo**:
```bash
python get_started/viser_static_scene_demo.py --sim isaaclab
```

**Demonstrates**:
- Basic 3D scene setup with primitive objects
- Robot visualization with initial joint configurations
- Camera controls with preset views

### 2. Dynamic Scene Updates

**Purpose**: Real-time visualization of moving robots and physics simulation.

**Run Demo**:
```bash
python get_started/viser_dynamic_scene_demo.py --sim isaaclab
```

**Demonstrates**:
- Live motion planning and execution visualization
- Real-time IK solving and robot movement
- Automatic camera tracking and video recording

### 3. Joint Control Demo

**Purpose**: Interactive manual control of robot joints.

**Run Demo**:
```bash
python get_started/viser_joint_control_demo.py --sim isaaclab
```

**Demonstrates**:
- Joint control interface usage
- Multiple robot support
- Real-time pose updates

### 4. IK Control Demo

**Purpose**: End-effector positioning using inverse kinematics.

**Run Demo**:
```bash
python get_started/viser_ik_control_demo.py --sim isaaclab
```

**Demonstrates**:
- IK control interface usage
- Visual target markers
- Integration with joint control

### 5. Trajectory Playback Demo

**Purpose**: Loading and playing back recorded robot trajectories.

**Run Demo**:
```bash
python get_started/viser_trajectory_demo.py --sim isaaclab
```

**Demonstrates**:
- Trajectory loading and playback
- Timeline control and seeking
- Playback speed adjustment

## Integration Examples

### Combining Multiple Features
```python
# Setup scene with multiple control modes
visualizer.enable_camera_controls()
visualizer.enable_joint_control()
visualizer.enable_ik_control()
visualizer.enable_trajectory_playback()

# Setup IK solver for robot
visualizer.setup_ik_solver(robot_name, robot_config, env_handler)

# Load and set trajectory
visualizer.load_trajectory('trajectory.pkl.gz')
visualizer.set_current_trajectory('franka', 0)
```

### Custom Scene Setup
```python
# Add custom objects and frames
visualizer.add_grid()
visualizer.add_frame("/world_frame")
visualizer.visualize_scenario_items(objects, object_states)
visualizer.visualize_scenario_items(robots, robot_states)
```