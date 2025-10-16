"""Keyboard object control - Real-time object manipulation with keyboard"""

from __future__ import annotations

import os
from typing import Literal

import pygame
import rootutils
import torch
import tyro

rootutils.setup_root(__file__, pythonpath=True)

from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.math import matrix_from_euler, quat_from_matrix, quat_mul
from metasim.utils.setup_util import get_handler


def save_poses_to_file(states, objects, robots, filename="saved_poses.py"):
    """Save current poses of all objects and robots to a Python file"""
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename.replace(".py", f"_{timestamp}.py")

    with open(filename, "w") as f:
        f.write('"""Saved poses from keyboard control"""\n\n')
        f.write("import torch\n\n")
        f.write("# Saved at: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("poses = {\n")
        f.write('    "objects": {\n')

        # Save objects
        for obj in objects:
            obj_name = obj.name
            if obj_name in states[0]["objects"]:
                obj_state = states[0]["objects"][obj_name]
                pos = obj_state["pos"]
                rot = obj_state["rot"]
                f.write(f'        "{obj_name}": {{\n')
                f.write(f'            "pos": torch.tensor([{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]),\n')
                f.write(f'            "rot": torch.tensor([{rot[0]:.6f}, {rot[1]:.6f}, {rot[2]:.6f}, {rot[3]:.6f}]),\n')
                if obj_state.get("dof_pos"):
                    f.write(f'            "dof_pos": {obj_state["dof_pos"]},\n')
                f.write("        },\n")

        f.write("    },\n")
        f.write('    "robots": {\n')

        # Save robots
        for robot in robots:
            robot_name = robot.name
            if robot_name in states[0]["robots"]:
                robot_state = states[0]["robots"][robot_name]
                pos = robot_state["pos"]
                rot = robot_state["rot"]
                f.write(f'        "{robot_name}": {{\n')
                f.write(f'            "pos": torch.tensor([{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]),\n')
                f.write(f'            "rot": torch.tensor([{rot[0]:.6f}, {rot[1]:.6f}, {rot[2]:.6f}, {rot[3]:.6f}]),\n')
                if "dof_pos" in robot_state:
                    f.write('            "dof_pos": {\n')
                    for joint_name, joint_val in robot_state["dof_pos"].items():
                        f.write(f'                "{joint_name}": {joint_val:.6f},\n')
                    f.write("            },\n")
                f.write("        },\n")

        f.write("    },\n")
        f.write("}\n")

    return filename


class ObjectKeyboardClient:
    """Keyboard client for object and robot control"""

    def __init__(
        self,
        entity_names: list[str],
        entity_types: list[str],
        entity_joints: dict[str, list[str]],  # entity_name -> list of joint names
        width: int = 550,
        height: int = 650,
        title: str = "Object/Robot Control",
    ):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.tiny_font = pygame.font.Font(None, 18)

        self.entity_names = entity_names  # All entities (objects + robots)
        self.entity_types = entity_types  # 'object' or 'robot'
        self.entity_joints = entity_joints  # joint names for each entity
        self.selected_idx = 0

        # Joint control mode
        self.joint_mode = False
        self.selected_joint_idx = 0

        self.instructions = [
            "=== Object/Robot Control ===",
            "",
            "   Movement:",
            "     UP    - Move +X",
            "     DOWN  - Move -X",
            "     LEFT  - Move +Y",
            "     RIGHT - Move -Y",
            "     e     - Move +Z",
            "     d     - Move -Z",
            "",
            "   Rotation:",
            "     q     - Roll +",
            "     w     - Roll -",
            "     a     - Pitch +",
            "     s     - Pitch -",
            "     z     - Yaw +",
            "     x     - Yaw -",
            "",
            "   Joint Control (when in joint mode):",
            "     UP    - Increase angle",
            "     DOWN  - Decrease angle",
            "     LEFT  - Previous joint",
            "     RIGHT - Next joint",
            "",
            "   Control:",
            "     TAB   - Switch entity",
            "     j     - Toggle joint mode",
            "     c     - Save poses",
            "     ESC   - Quit",
        ]

    def update(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def get_selected_entity(self) -> tuple[str, str]:
        """Returns (entity_name, entity_type)"""
        return self.entity_names[self.selected_idx], self.entity_types[self.selected_idx]

    def switch_entity(self):
        """Switch to next entity (TAB key)"""
        self.selected_idx = (self.selected_idx + 1) % len(self.entity_names)
        name, etype = self.get_selected_entity()
        self.selected_joint_idx = 0  # Reset joint selection

        # Auto-exit joint mode if new entity has no joints
        joints = self.entity_joints.get(name, [])
        if self.joint_mode and not joints:
            self.joint_mode = False
            log.info(f"Auto-exited joint mode: {name} has no controllable joints")

        etype_tag = "[R]" if etype == "robot" else "[O]"
        joints = self.entity_joints.get(name, [])
        joint_count = len(joints)

        if joint_count > 0:
            log.info(
                f"Selected: {etype_tag} {name} ({self.selected_idx + 1}/{len(self.entity_names)}) - {joint_count} joints available"
            )
        else:
            log.info(
                f"Selected: {etype_tag} {name} ({self.selected_idx + 1}/{len(self.entity_names)}) - No controllable joints"
            )

    def toggle_joint_mode(self):
        """Toggle joint control mode (J key)"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])

        if not joints:
            log.warning(f"{name} has no controllable joints")
            return

        self.joint_mode = not self.joint_mode
        self.selected_joint_idx = 0

        if self.joint_mode:
            log.info(f"Joint mode ENABLED for {name} ({len(joints)} joints)")
        else:
            log.info("Joint mode DISABLED")

    def get_selected_joint(self) -> str | None:
        """Get currently selected joint name"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])
        if joints and 0 <= self.selected_joint_idx < len(joints):
            return joints[self.selected_joint_idx]
        return None

    def next_joint(self):
        """Select next joint"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])
        if joints:
            self.selected_joint_idx = (self.selected_joint_idx + 1) % len(joints)
            log.info(f"Selected joint: {joints[self.selected_joint_idx]} ({self.selected_joint_idx + 1}/{len(joints)})")

    def prev_joint(self):
        """Select previous joint"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])
        if joints:
            self.selected_joint_idx = (self.selected_joint_idx - 1) % len(joints)
            log.info(f"Selected joint: {joints[self.selected_joint_idx]} ({self.selected_joint_idx + 1}/{len(joints)})")

    def draw_instructions(self):
        self.screen.fill((30, 30, 30))
        y = 25

        # Use unified instruction set
        instructions = self.instructions

        for instruction in instructions:
            if not instruction:
                y += 12
                continue
            color = (100, 200, 255) if "===" in instruction else (200, 200, 200)
            font = self.font if "===" in instruction else self.small_font
            text = font.render(instruction, True, color)
            self.screen.blit(text, (25, y))
            y += 30 if "===" in instruction else 24

        y += 20

        # Show joint info if in joint mode (static info only)
        if self.joint_mode:
            y += 35
            joint_name = self.get_selected_joint()
            name, _ = self.get_selected_entity()
            joints = self.entity_joints.get(name, [])

            if joint_name:
                joint_text = f"Joint: {joint_name} ({self.selected_joint_idx + 1}/{len(joints)})"
                text = self.small_font.render(joint_text, True, (255, 255, 100))
                self.screen.blit(text, (25, y))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


def process_input(dpos: float = 0.01, drot: float = 0.05, dangle: float = 0.02, joint_mode: bool = False):
    """Process keyboard input - returns delta position, delta rotation, joint angle delta, and control flags"""
    keys = pygame.key.get_pressed()

    # Initialize persistent state
    if not hasattr(process_input, "tab_pressed"):
        process_input.tab_pressed = False
        process_input.c_pressed = False
        process_input.j_pressed = False
        process_input.left_pressed = False
        process_input.right_pressed = False

    # In joint mode, arrow keys control joints
    if joint_mode:
        # UP/DOWN for angle adjustment
        dangle_val = dangle * (keys[pygame.K_UP] - keys[pygame.K_DOWN])

        # LEFT/RIGHT for joint selection (single press)
        prev_joint = False
        if keys[pygame.K_LEFT] and not process_input.left_pressed:
            prev_joint = True
            process_input.left_pressed = True
        elif not keys[pygame.K_LEFT]:
            process_input.left_pressed = False

        next_joint = False
        if keys[pygame.K_RIGHT] and not process_input.right_pressed:
            next_joint = True
            process_input.right_pressed = True
        elif not keys[pygame.K_RIGHT]:
            process_input.right_pressed = False

        return None, None, dangle_val, False, False, True, prev_joint, next_joint

    else:
        # Normal mode: position and rotation control
        # Movement (follow teleop convention)
        dx = dpos * (keys[pygame.K_UP] - keys[pygame.K_DOWN])  # UP/DOWN for X
        dy = dpos * (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT])  # LEFT/RIGHT for Y
        dz = dpos * (keys[pygame.K_e] - keys[pygame.K_d])  # e/d for Z

        # Rotation (euler angles: roll, pitch, yaw)
        droll = drot * (keys[pygame.K_q] - keys[pygame.K_w])  # q/w for roll
        dpitch = drot * (keys[pygame.K_a] - keys[pygame.K_s])  # a/s for pitch
        dyaw = drot * (keys[pygame.K_z] - keys[pygame.K_x])  # z/x for yaw

        return [dx, dy, dz], [droll, dpitch, dyaw], 0.0, False, False, False, False, False

    # Common controls (always checked after mode-specific logic)


def process_common_input():
    """Process common input (TAB, J, C, ESC) - separate function"""
    keys = pygame.key.get_pressed()

    if not hasattr(process_common_input, "tab_pressed"):
        process_common_input.tab_pressed = False
        process_common_input.c_pressed = False
        process_common_input.j_pressed = False

    switch = False
    if keys[pygame.K_TAB] and not process_common_input.tab_pressed:
        switch = True
        process_common_input.tab_pressed = True
    elif not keys[pygame.K_TAB]:
        process_common_input.tab_pressed = False

    save_poses = False
    if keys[pygame.K_c] and not process_common_input.c_pressed:
        save_poses = True
        process_common_input.c_pressed = True
    elif not keys[pygame.K_c]:
        process_common_input.c_pressed = False

    toggle_joint = False
    if keys[pygame.K_j] and not process_common_input.j_pressed:
        toggle_joint = True
        process_common_input.j_pressed = True
    elif not keys[pygame.K_j]:
        process_common_input.j_pressed = False

    return switch, save_poses, toggle_joint


if __name__ == "__main__":

    @configclass
    class Args:
        robot: str = "franka"
        sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
        num_envs: int = 1
        headless: bool = False
        step_size: float = 0.01
        rot_size: float = 0.05
        angle_size: float = 0.02  # Joint angle adjustment step size
        print_interval: int = 50

        def __post_init__(self):
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    scenario = ScenarioCfg(
        robots=[args.robot],
        headless=args.headless,
        num_envs=args.num_envs,
        simulator=args.sim,
        cameras=[PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))],
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bbq_sauce",
                scale=(2, 2, 2),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
                enabled_gravity=False,
            ),
            ArticulationObjCfg(
                name="box_base",
                usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            ),
        ],
    )

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)

    init_states = [
        {
            "objects": {
                "cube": {"pos": torch.tensor([0.3, -0.2, 0.05]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
                "sphere": {"pos": torch.tensor([0.4, -0.6, 0.05]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
                "bbq_sauce": {"pos": torch.tensor([0.7, -0.3, 0.14]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
                "box_base": {
                    "pos": torch.tensor([0.5, 0.2, 0.1]),
                    "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                    "dof_pos": {"box_joint": 0.0},
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
            },
        }
    ]
    handler.set_states(init_states * scenario.num_envs)

    # Build entity list: objects first, then robots
    entity_names = [obj.name for obj in scenario.objects] + [robot.name for robot in scenario.robots]
    entity_types = ["object"] * len(scenario.objects) + ["robot"] * len(scenario.robots)

    # Get joint information for each entity
    entity_joints = {}

    # Get joints from objects (articulated objects)
    for obj in scenario.objects:
        if hasattr(obj, "actuators") and obj.actuators:
            entity_joints[obj.name] = list(obj.actuators.keys())
        else:
            # Try to get from init states
            obj_state = init_states[0]["objects"].get(obj.name)
            if obj_state:
                dof_pos = obj_state.get("dof_pos")
                if dof_pos:
                    entity_joints[obj.name] = list(dof_pos.keys())
                else:
                    entity_joints[obj.name] = []
            else:
                entity_joints[obj.name] = []

    # Get joints from robots
    for robot in scenario.robots:
        if hasattr(robot, "actuators") and robot.actuators:
            entity_joints[robot.name] = list(robot.actuators.keys())
        else:
            # Try to get from init states
            robot_state = init_states[0]["robots"].get(robot.name)
            if robot_state:
                dof_pos = robot_state.get("dof_pos")
                if dof_pos:
                    entity_joints[robot.name] = list(dof_pos.keys())
                else:
                    entity_joints[robot.name] = []
            else:
                entity_joints[robot.name] = []

    keyboard_client = ObjectKeyboardClient(entity_names, entity_types, entity_joints)

    log.info("Keyboard Object/Robot Control")
    log.info("Use Arrow keys + e/d for position, q/w/a/s/z/x for rotation")
    log.info("Press J to enter joint control mode")
    log.info("TAB to switch entity, C to save poses, ESC to quit\n")

    os.makedirs("get_started/output", exist_ok=True)

    step = 0
    running = True

    while running:
        running = keyboard_client.update()
        if not running or (keyboard_client.update() and pygame.key.get_pressed()[pygame.K_ESCAPE]):
            break

        # Process common input (TAB, J, C)
        switch, save_poses, toggle_joint = process_common_input()

        if switch:
            keyboard_client.switch_entity()

        if toggle_joint:
            keyboard_client.toggle_joint_mode()

        # Save poses when C is pressed
        if save_poses:
            current_states = handler.get_states(mode="dict")
            saved_file = save_poses_to_file(
                current_states, scenario.objects, scenario.robots, filename="get_started/output/saved_poses.py"
            )
            log.info(f"Poses saved to: {saved_file}")

        # Process mode-specific input
        delta_pos, delta_rot, delta_angle, _, _, in_joint_mode, prev_joint, next_joint = process_input(
            dpos=args.step_size, drot=args.rot_size, dangle=args.angle_size, joint_mode=keyboard_client.joint_mode
        )

        # Handle joint selection
        if prev_joint:
            keyboard_client.prev_joint()
        if next_joint:
            keyboard_client.next_joint()

        # Get current selection for control logic
        selected_name, selected_type = keyboard_client.get_selected_entity()

        keyboard_client.draw_instructions()

        # Handle joint mode updates
        if keyboard_client.joint_mode and abs(delta_angle) > 1e-6:
            joint_name = keyboard_client.get_selected_joint()
            if joint_name:
                states = handler.get_states(mode="dict")
                for env_idx in range(scenario.num_envs):
                    if selected_type == "object" and selected_name in states[env_idx]["objects"]:
                        if "dof_pos" in states[env_idx]["objects"][selected_name]:
                            if joint_name in states[env_idx]["objects"][selected_name]["dof_pos"]:
                                states[env_idx]["objects"][selected_name]["dof_pos"][joint_name] += delta_angle

                    elif selected_type == "robot" and selected_name in states[env_idx]["robots"]:
                        if "dof_pos" in states[env_idx]["robots"][selected_name]:
                            if joint_name in states[env_idx]["robots"][selected_name]["dof_pos"]:
                                states[env_idx]["robots"][selected_name]["dof_pos"][joint_name] += delta_angle

                handler.set_states(states)

        # Handle normal mode updates (position/rotation)
        elif not keyboard_client.joint_mode and delta_pos is not None and delta_rot is not None:
            delta_pos_tensor = torch.tensor(delta_pos)
            delta_rot_tensor = torch.tensor(delta_rot)

            has_input = delta_pos_tensor.abs().sum() > 0 or delta_rot_tensor.abs().sum() > 0

            if has_input:
                states = handler.get_states(mode="dict")
                for env_idx in range(scenario.num_envs):
                    # Check if it's an object or robot
                    if selected_type == "object" and selected_name in states[env_idx]["objects"]:
                        # Update position
                        if delta_pos_tensor.abs().sum() > 0:
                            pos = states[env_idx]["objects"][selected_name]["pos"]
                            new_pos = pos + delta_pos_tensor
                            states[env_idx]["objects"][selected_name]["pos"] = new_pos

                        # Update rotation
                        if delta_rot_tensor.abs().sum() > 0:
                            current_rot = states[env_idx]["objects"][selected_name]["rot"]
                            delta_rot_mat = matrix_from_euler(delta_rot_tensor.unsqueeze(0), "XYZ")
                            delta_quat = quat_from_matrix(delta_rot_mat)[0]
                            new_rot = quat_mul(current_rot.unsqueeze(0), delta_quat.unsqueeze(0))[0]
                            states[env_idx]["objects"][selected_name]["rot"] = new_rot

                    elif selected_type == "robot" and selected_name in states[env_idx]["robots"]:
                        # Update position
                        if delta_pos_tensor.abs().sum() > 0:
                            pos = states[env_idx]["robots"][selected_name]["pos"]
                            new_pos = pos + delta_pos_tensor
                            states[env_idx]["robots"][selected_name]["pos"] = new_pos

                        # Update rotation
                        if delta_rot_tensor.abs().sum() > 0:
                            current_rot = states[env_idx]["robots"][selected_name]["rot"]
                            delta_rot_mat = matrix_from_euler(delta_rot_tensor.unsqueeze(0), "XYZ")
                            delta_quat = quat_from_matrix(delta_rot_mat)[0]
                            new_rot = quat_mul(current_rot.unsqueeze(0), delta_quat.unsqueeze(0))[0]
                            states[env_idx]["robots"][selected_name]["rot"] = new_rot

                handler.set_states(states)

        handler.simulate()

        if step % 10 == 0:
            handler.refresh_render()
        step += 1

    keyboard_client.close()
    handler.close()

    log.info(f"Done (steps: {step})")
