from __future__ import annotations

import math

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
from loguru import logger as log

from metasim.cfg.objects import (
    ArticulationObjCfg,
    BaseObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
    _FileBasedMixin,
)
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import ContactForceSensorCfg
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.state import CameraState, ObjectState, RobotState, SensorState, TensorState


class IsaacgymHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._actions_cache: list[Action] = []
        self._robot_names = [robot.name for robot in self.robots]
        self._robot_init_pos = [robot.default_position for robot in self.robots]
        self._robot_init_quat = [robot.default_orientation for robot in self.robots]
        self._cameras = scenario.cameras
        self._sensors = scenario.sensors
        self.robot_asset_list: list[gymapi.Asset] = []

        self.gym = None
        self.sim = None
        self.viewer = None
        self._enable_viewer_sync: bool = True  # sync viewer flag
        if hasattr(scenario.task, "device"):
            self._device = torch.device(scenario.task.device)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if hasattr(scenario.task, "max_agg_bodies"):
            self._max_agg_bodies = scenario.task.max_agg_bodies
        if hasattr(scenario.task, "max_agg_shapes"):
            self._max_agg_shapes = scenario.task.max_agg_shapes

        self._num_envs: int = scenario.num_envs
        self._episode_length_buf = [0 for _ in range(self.num_envs)]

        # asset related
        self._asset_dict_dict: dict = {}  # dict of object link index dict
        self._articulated_asset_dict_dict: dict = {}  # dict of articulated object link index dict
        self._articulated_joint_dict_dict: dict = {}  # dict of articulated object joint index dict
        self._articulated_dof_prop_dict: dict = {}  # list of articulated object dof properties
        # self._robot_link_dict: dict = {}  # dict of robot link index dict
        # self._robot_joint_dict: dict = {}  # dict of robot joint index dict
        self._joint_info: dict = {}  # dict of joint names of each env
        self._num_joints: int = 0
        self._body_info: dict = {}  # dict of body names of each env
        self._num_bodies: int = 0

        # environment related pointers
        self._envs: list = []
        self._obj_handles: list = []  # 2 dim list: list in list, each list contains object handles of each env
        self._articulated_obj_handles: list = []  # 2 dim list: list in list, each list contains articulated object handles of each env
        self._robot_handles: list = []  # 2 dim list: list of robot handles of each env

        # environment related tensor indices
        self._env_rigid_body_global_indices: list = []  # 2 dim list: list in list, each list contains global indices of each env

        # will update after refresh
        self._root_states: torch.Tensor | None = None
        self._dof_states: torch.Tensor | None = None
        self._rigid_body_states: torch.Tensor | None = None
        self._robot_dof_state: torch.Tensor | None = None
        self._contact_forces: torch.Tensor | None = None

        # control related
        self._robot_num_dof: int = 0  # number of robot dof
        self._obj_num_dof: int = 0  # number of object dof
        self._actions: torch.Tensor | None = None
        self._action_scale: torch.Tensor | None = (
            None  # for configuration: desire_pos = action_scale * action + default_pos
        )
        self._robot_default_dof_pos: torch.Tensor | None = (
            None  # for the configuration: desire_pos = action_scale * action + default_pos
        )
        self._action_offset: bool = False  # for configuration: desire_pos = action_scale * action + default_pos
        self._p_gains: torch.Tensor | None = None  # parameter for PD controller in for pd effort control
        self._d_gains: torch.Tensor | None = None
        self._torque_limits: torch.Tensor | None = None
        self._effort: torch.Tensor | None = None  # output of pd controller, used for effort control
        self._pos_ctrl_dof_dix = []  # joint index in dof state, built-in position control mode
        self._manual_pd_on: bool = False  # turn on maunual pd controller if effort joint exist

    def launch(self) -> None:
        ## IsaacGym Initialization
        self._init_gym()
        self._make_envs()
        self._set_up_camera()
        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)
        self._root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self._dof_states = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self._rigid_body_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self._robot_dof_state = self._dof_states.view(self._num_envs, -1, 2)[:, self._obj_num_dof :]
        self._dof_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim))
        self.num_sensors = len(self._sensors)
        if self.num_sensors > 0:
            self._vec_sensor_tensor = gymtorch.wrap_tensor(self.gym.acquire_force_sensor_tensor(self.sim)).view(
                self.num_envs, self.num_sensors, 6
            )  # shape: (num_envs, num_sensors * 6)
        self._contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))

        # Refresh tensors
        if not self._manual_pd_on:
            self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def _init_gym(self) -> None:
        physics_engine = gymapi.SIM_PHYSX
        self.gym = gymapi.acquire_gym()
        # configure sim
        # TODO move more params into sim_params cfg
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        if self.scenario.sim_params.dt is not None:
            # IsaacGym has a different dt definition than IsaacLab, see https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html#simulation-config
            sim_params.dt = self.scenario.sim_params.dt
        sim_params.substeps = self.scenario.sim_params.substeps
        sim_params.use_gpu_pipeline = self.scenario.sim_params.use_gpu_pipeline
        sim_params.physx.solver_type = self.scenario.sim_params.solver_type
        sim_params.physx.num_position_iterations = self.scenario.sim_params.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.scenario.sim_params.num_velocity_iterations
        sim_params.physx.rest_offset = self.scenario.sim_params.rest_offset
        sim_params.physx.contact_offset = self.scenario.sim_params.contact_offset
        sim_params.physx.friction_offset_threshold = self.scenario.sim_params.friction_offset_threshold
        sim_params.physx.friction_correlation_distance = self.scenario.sim_params.friction_correlation_distance
        sim_params.physx.num_threads = self.scenario.sim_params.num_threads
        sim_params.physx.use_gpu = self.scenario.sim_params.use_gpu
        sim_params.physx.bounce_threshold_velocity = self.scenario.sim_params.bounce_threshold_velocity

        device_id = self._device.index if self._device.type == "cuda" else 0
        compute_device_id = device_id
        graphics_device_id = device_id
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # press 'V' to toggle viewer sync
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            if self.viewer is None:
                raise Exception("Failed to create viewer")

    def _set_up_camera(self) -> None:
        self._depth_tensors = []
        self._rgb_tensors = []
        self._seg_tensors = []
        self._vinv_mats = []
        self._proj_mats = []
        self._camera_handles = []
        self._env_origin = []
        for i_env in range(self.num_envs):
            self._depth_tensors.append([])
            self._rgb_tensors.append([])
            self._seg_tensors.append([])
            self._vinv_mats.append([])
            self._proj_mats.append([])
            self._env_origin.append([])

            origin = self.gym.get_env_origin(self._envs[i_env])
            self._env_origin[i_env] = [origin.x, origin.y, origin.z]
            for cam_cfg in self.cameras:
                camera_props = gymapi.CameraProperties()
                camera_props.width = cam_cfg.width
                camera_props.height = cam_cfg.height
                camera_props.horizontal_fov = cam_cfg.horizontal_fov
                camera_props.near_plane = cam_cfg.clipping_range[0]
                camera_props.far_plane = cam_cfg.clipping_range[1]
                camera_props.enable_tensors = True
                camera_handle = self.gym.create_camera_sensor(self._envs[i_env], camera_props)
                self._camera_handles.append(camera_handle)

                camera_eye = gymapi.Vec3(*cam_cfg.pos)
                camera_lookat = gymapi.Vec3(*cam_cfg.look_at)
                self.gym.set_camera_location(camera_handle, self._envs[i_env], camera_eye, camera_lookat)

                camera_tensor_depth = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_DEPTH
                )
                camera_tensor_rgb = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_COLOR
                )
                camera_tensor_rgb_seg = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[i_env], camera_handle, gymapi.IMAGE_SEGMENTATION
                )
                torch_cam_depth_tensor = gymtorch.wrap_tensor(camera_tensor_depth)
                torch_cam_rgb_tensor = gymtorch.wrap_tensor(camera_tensor_rgb)
                torch_cam_rgb_seg_tensor = gymtorch.wrap_tensor(camera_tensor_rgb_seg)

                cam_vinv = torch.inverse(
                    torch.tensor(self.gym.get_camera_view_matrix(self.sim, self._envs[i_env], camera_handle))
                ).to(self.device)
                cam_proj = torch.tensor(
                    self.gym.get_camera_proj_matrix(self.sim, self._envs[i_env], camera_handle),
                    device=self.device,
                )

                self._depth_tensors[i_env].append(torch_cam_depth_tensor)
                self._rgb_tensors[i_env].append(torch_cam_rgb_tensor)
                self._seg_tensors[i_env].append(torch_cam_rgb_seg_tensor)
                self._vinv_mats[i_env].append(cam_vinv)
                self._proj_mats[i_env].append(cam_proj)

    def _load_object_asset(self, object: BaseObjCfg) -> None:
        asset_root = "."
        if isinstance(object, PrimitiveCubeCfg):
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = object.fix_base_link
            asset_options.disable_gravity = object.disable_gravity
            asset_options.flip_visual_attachments = object.flip_visual_attachments
            asset_options.use_physx_armature = True
            if hasattr(object, "default_density") and object.default_density is not None:
                asset_options.density = object.default_density
            asset = self.gym.create_box(self.sim, object.size[0], object.size[1], object.size[2], asset_options)
        elif isinstance(object, PrimitiveSphereCfg):
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = object.fix_base_link
            asset_options.disable_gravity = object.disable_gravity
            asset_options.flip_visual_attachments = object.flip_visual_attachments
            asset_options.use_physx_armature = True
            if hasattr(object, "default_density") and object.default_density is not None:
                asset_options.density = object.default_density
            asset = self.gym.create_sphere(self.sim, object.radius, asset_options)

        elif isinstance(object, ArticulationObjCfg):
            asset_path = object.mjcf_path if object.isaacgym_read_mjcf else object.urdf_path
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = object.fix_base_link
            asset_options.disable_gravity = object.disable_gravity
            asset_options.flip_visual_attachments = object.flip_visual_attachments
            asset_options.collapse_fixed_joints = object.collapse_fixed_joints
            asset_options.use_physx_armature = True
            if object.override_com:
                asset_options.override_com = True
            if object.override_inertia:
                asset_options.override_inertia = True
            if object.use_mesh_materials:
                asset_options.use_mesh_materials = True
            if object.mesh_normal_mode is not None:
                if object.mesh_normal_mode == "vertex":
                    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                elif object.mesh_normal_mode == "face":
                    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
                else:
                    raise ValueError(f"Invalid mesh_normal_mode: {object.mesh_normal_mode}. Use 'vertex' or 'face'.")
            if object.use_vhacd:
                asset_options.vhacd_enabled = True
                if object.vhacd_resolution is not None:
                    asset_options.vhacd_params.resolution = object.vhacd_resolution
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            if hasattr(object, "default_density") and object.default_density is not None:
                asset_options.density = object.default_density
            asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)
            self._articulated_asset_dict_dict[object.name] = self.gym.get_asset_rigid_body_dict(asset)
            self._articulated_joint_dict_dict[object.name] = self.gym.get_asset_dof_dict(asset)
            self._articulated_dof_prop_dict[object.name] = self.gym.get_asset_dof_properties(asset)
        elif isinstance(object, RigidObjCfg):
            asset_path = object.mjcf_path if object.isaacgym_read_mjcf else object.urdf_path
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = object.fix_base_link
            asset_options.disable_gravity = object.disable_gravity
            asset_options.flip_visual_attachments = object.flip_visual_attachments
            asset_options.use_physx_armature = True
            if object.override_com:
                asset_options.override_com = True
            if object.override_inertia:
                asset_options.override_inertia = True
            if object.use_mesh_materials:
                asset_options.use_mesh_materials = True
            if object.mesh_normal_mode is not None:
                if object.mesh_normal_mode == "vertex":
                    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                elif object.mesh_normal_mode == "face":
                    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
                else:
                    raise ValueError(f"Invalid mesh_normal_mode: {object.mesh_normal_mode}. Use 'vertex' or 'face'.")
            if object.use_vhacd:
                asset_options.vhacd_enabled = True
                if object.vhacd_resolution is not None:
                    asset_options.vhacd_params.resolution = object.vhacd_resolution
            if hasattr(object, "default_density") and object.default_density is not None:
                asset_options.density = object.default_density
            asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)

        asset_link_dict = self.gym.get_asset_rigid_body_dict(asset)
        self._asset_dict_dict[object.name] = asset_link_dict
        self._obj_num_dof += self.gym.get_asset_dof_count(asset)
        return asset

    def _load_robot_assets(self) -> None:
        robot_asset_list = []
        robot_dof_props_list = []
        robot_p_gains = []
        robot_d_gains = []
        robot_torque_limits = []
        _robots_default_dof_pos = []
        self.actuated_root = False  # whether a robot has an actuated root joint
        for robot in self.robots:
            if hasattr(robot, "actuated_root") and robot.actuated_root:
                self.actuated_root = True
            asset_root = "."
            robot_asset_file = robot.mjcf_path if robot.isaacgym_read_mjcf else robot.urdf_path
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = robot.fix_base_link
            asset_options.disable_gravity = not robot.enabled_gravity
            asset_options.flip_visual_attachments = robot.isaacgym_flip_visual_attachments
            asset_options.collapse_fixed_joints = robot.collapse_fixed_joints
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            if robot.use_vhacd:
                asset_options.vhacd_enabled = True
            asset_options.use_physx_armature = True
            # Angular velocity damping for rigid bodies
            if hasattr(robot, "angular_damping") and robot.angular_damping is not None:
                asset_options.angular_damping = robot.angular_damping
            # Linear velocity damping for rigid bodies
            if hasattr(robot, "linear_damping") and robot.linear_damping is not None:
                asset_options.linear_damping = robot.linear_damping
            # Defaults are set to free movement and will be updated based on the configuration in actuator_cfg below.
            asset_options.replace_cylinder_with_capsule = self.scenario.sim_params.replace_cylinder_with_capsule
            robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
            # configure robot dofs
            robot_num_dofs = self.gym.get_asset_dof_count(robot_asset)
            self._robot_num_dof += robot_num_dofs

            self._action_scale = torch.tensor(self.scenario.control.action_scale, device=self.device)
            self._action_offset = self.scenario.control.action_offset

            p_gains = torch.zeros(
                self._num_envs, robot_num_dofs, dtype=torch.float, device=self.device, requires_grad=False
            )
            d_gains = torch.zeros(
                self._num_envs, robot_num_dofs, dtype=torch.float, device=self.device, requires_grad=False
            )
            torque_limits = torch.zeros(
                self._num_envs, robot_num_dofs, dtype=torch.float, device=self.device, requires_grad=False
            )

            robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

            robot_lower_limits = robot_dof_props["lower"]
            robot_upper_limits = robot_dof_props["upper"]
            robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)
            num_actions = 0
            _default_dof_pos = []
            self._manual_pd_on = any(mode == "effort" for mode in robot.control_type.values())

            # set robot tendon properties
            num_tendons = self.gym.get_asset_tendon_count(robot_asset)
            if num_tendons > 0:
                tendon_props = self.gym.get_asset_tendon_properties(robot_asset)

                if hasattr(robot, "tendon_limit_stiffness"):
                    for i in range(num_tendons):
                        tendon_props[i].limit_stiffness = robot.tendon_limit_stiffness
                if hasattr(robot, "tendon_damping"):
                    for i in range(num_tendons):
                        tendon_props[i].damping = robot.tendon_damping

                self.gym.set_asset_tendon_properties(robot_asset, tendon_props)

            dof_names = self.gym.get_asset_dof_names(robot_asset)
            for i, dof_name in enumerate(dof_names):
                # get config
                i_actuator_cfg = robot.actuators[dof_name]
                i_stiffness = i_actuator_cfg.stiffness if i_actuator_cfg.stiffness is not None else 800.0
                i_damping = i_actuator_cfg.damping if i_actuator_cfg.damping is not None else 40.0
                i_control_mode = robot.control_type[dof_name] if dof_name in robot.control_type else "position"

                # task default position from cfg if exist, otherwise use 0.3*(uppper + lower) as default
                if not i_actuator_cfg.is_ee:
                    default_dof_pos_i = (
                        robot.default_joint_positions[dof_name]
                        if dof_name in robot.default_joint_positions
                        else robot_mids[i]
                    )
                    _default_dof_pos.append(default_dof_pos_i)
                # for end effector, always use open as default position
                else:
                    _default_dof_pos.append(robot_upper_limits[i])

                # pd control effort mode
                if i_control_mode == "effort":
                    p_gains[:, i] = i_stiffness
                    d_gains[:, i] = i_damping
                    torque_limit = (
                        i_actuator_cfg.torque_limit
                        if i_actuator_cfg.torque_limit is not None
                        else torch.tensor(robot_dof_props["effort"][i], dtype=torch.float, device=self.device)
                    )
                    torque_limits[:, i] = self.scenario.control.torque_limit_scale * torque_limit
                    if not hasattr(robot, "dof_drive_mode") or robot.dof_drive_mode == "effort":
                        robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                    robot_dof_props["stiffness"][i] = 0.0
                    robot_dof_props["damping"][i] = 0.0

                # built-in position mode
                elif i_control_mode == "position":
                    if not hasattr(robot, "dof_drive_mode") or robot.dof_drive_mode == "position":
                        robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
                    if i_actuator_cfg.stiffness is not None:
                        robot_dof_props["stiffness"][i] = i_actuator_cfg.stiffness
                    if i_actuator_cfg.damping is not None:
                        robot_dof_props["damping"][i] = i_actuator_cfg.damping
                    self._pos_ctrl_dof_dix.append(i + self._obj_num_dof + self._robot_num_dof - robot_num_dofs)
                else:
                    log.error(f"Unknown actuator control mode: {i_control_mode}, only support effort and position")
                    raise ValueError

                if i_actuator_cfg.fully_actuated:
                    num_actions += 1

            # joint_reindex = self.get_joint_reindex(self.robot.name)
            _robots_default_dof_pos.append(torch.tensor(_default_dof_pos, device=self.device).unsqueeze(0))
            self.actions = torch.zeros([self._num_envs, num_actions], device=self.device)
            robot_p_gains.append(p_gains)
            robot_d_gains.append(d_gains)
            robot_torque_limits.append(torque_limits)

            robot_asset_list.append(robot_asset)
            robot_dof_props_list.append(robot_dof_props)

        self._robot_default_dof_pos = torch.cat(_robots_default_dof_pos, dim=-1)  # shape: (1, total_num_robot_dofs)
        self._p_gains = torch.cat(robot_p_gains, dim=-1)  # shape: (num_envs, total_num_robot_dofs)
        self._d_gains = torch.cat(robot_d_gains, dim=-1)  # shape: (num_envs, total_num_robot_dofs)
        self._torque_limits = torch.cat(robot_torque_limits, dim=-1)  # shape: (num_envs, total_num_robot_dofs)

        return robot_asset_list, robot_dof_props_list

    def _load_contact_sensor(self) -> None:
        for sensor in self._sensors:
            sensor: ContactForceSensorCfg
            if sensor.source_link is not None:
                raise NotImplementedError
            robot_name = sensor.base_link if isinstance(sensor.base_link, str) else sensor.base_link[0]
            handle = 0
            if robot_name not in self._robot_names:
                raise ValueError(f"Robot {robot_name} not found in the environment.")
            robot_asset = self.robot_asset_list[self._robot_names.index(robot_name)]
            if isinstance(sensor.base_link, tuple):
                handle = self.gym.find_asset_rigid_body_index(
                    robot_asset,
                    sensor.base_link[1],
                )
            sensor_pose = gymapi.Transform()
            self.gym.create_asset_force_sensor(robot_asset, handle, sensor_pose)

    def _make_envs(
        self,
    ) -> None:
        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = self.scenario.sim_params.env_spacing
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        log.info("Creating %d environments" % self.num_envs)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # get object and robot asset
        obj_assets_list = [self._load_object_asset(obj) for obj in self.objects]
        robot_asset_list, robot_dof_props_list = self._load_robot_assets()
        self.robot_asset_list = robot_asset_list

        self._load_contact_sensor()

        #### Joint Info ####
        for art_obj_name, art_obj_joint_dict in self._articulated_joint_dict_dict.items():
            num_joints = len(art_obj_joint_dict)
            joint_names_ = []
            for joint_i in range(num_joints):
                for joint_name, joint_idx in art_obj_joint_dict.items():
                    if joint_idx == joint_i:
                        joint_names_.append(joint_name)
            assert len(joint_names_) == num_joints
            joint_info_ = {}
            joint_info_["names"] = joint_names_
            joint_info_["local_indices"] = art_obj_joint_dict
            art_obj_joint_dict_global = {k_: v_ + self._num_joints for k_, v_ in art_obj_joint_dict.items()}
            joint_info_["global_indices"] = art_obj_joint_dict_global
            self._num_joints += num_joints
            self._joint_info[art_obj_name] = joint_info_

        # robot
        for robot, robot_asset in zip(self.robots, robot_asset_list):
            robot_link_dict = self.gym.get_asset_dof_dict(robot_asset)
            num_joints = len(robot_link_dict)
            joint_names_ = []
            for joint_i in range(num_joints):
                for joint_name, joint_idx in robot_link_dict.items():
                    if joint_idx == joint_i:
                        joint_names_.append(joint_name)

            assert len(joint_names_) == num_joints
            joint_info_ = {}
            joint_info_["names"] = joint_names_
            joint_info_["local_indices"] = robot_link_dict
            joint_info_["global_indices"] = {k_: v_ + self._num_joints for k_, v_ in robot_link_dict.items()}
            self._joint_info[robot.name] = joint_info_
            self._num_joints += num_joints

        ###################
        #### Body Info ####
        for obj_name, asset_dict in self._asset_dict_dict.items():
            num_bodies = len(asset_dict)
            rigid_body_names = []
            for i in range(num_bodies):
                for rigid_body_name, rigid_body_idx in asset_dict.items():
                    if rigid_body_idx == i:
                        rigid_body_names.append(rigid_body_name)
            assert len(rigid_body_names) == num_bodies
            body_info_ = {}
            body_info_["name"] = rigid_body_names
            body_info_["local_indices"] = asset_dict
            body_info_["global_indices"] = {k_: v_ + self._num_bodies for k_, v_ in asset_dict.items()}
            self._body_info[obj_name] = body_info_
            self._num_bodies += num_bodies

        for robot, robot_asset in zip(self.robots, robot_asset_list):
            robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
            num_bodies = len(robot_link_dict)
            rigid_body_names = []
            for i in range(num_bodies):
                for rigid_body_name, rigid_body_idx in robot_link_dict.items():
                    if rigid_body_idx == i:
                        rigid_body_names.append(rigid_body_name)

            assert len(rigid_body_names) == num_bodies
            rigid_body_info_ = {}
            rigid_body_info_["name"] = rigid_body_names
            rigid_body_info_["local_indices"] = robot_link_dict
            rigid_body_info_["global_indices"] = {k_: v_ + self._num_bodies for k_, v_ in robot_link_dict.items()}
            self._body_info[robot.name] = rigid_body_info_
            self._num_bodies += num_bodies

        #################

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            if hasattr(self, "max_agg_bodies") and hasattr(self, "max_agg_shapes"):
                self.gym.begin_aggregate(env, self.max_agg_bodies, self.max_agg_shapes, True)

            ##  state update  ##
            self._envs.append(env)
            self._obj_handles.append([])
            self._env_rigid_body_global_indices.append({})
            self._articulated_obj_handles.append([])
            ####################

            # carefully set each object
            for obj_i, obj_asset in enumerate(obj_assets_list):
                # add object
                obj_pose = gymapi.Transform()
                obj_pose.p.x = obj_i * 0.2  # place to any position, will update immediately at reset stage
                obj_pose.p.y = 0.0
                obj_pose.p.z = 0.0
                obj_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
                obj_handle = self.gym.create_actor(env, obj_asset, obj_pose, "object", i, 0)
                # print(self.gym.get_actor_rigid_body_properties(env, obj_handle)[0].mass)

                if self.objects[obj_i].friction is not None:
                    shape_props = self.gym.get_actor_rigid_shape_properties(env, obj_handle)
                    for shape_prop in shape_props:
                        shape_prop.friction = self.objects[obj_i].friction
                    self.gym.set_actor_rigid_shape_properties(env, obj_handle, shape_props)

                if isinstance(self.objects[obj_i], _FileBasedMixin):
                    self.gym.set_actor_scale(env, obj_handle, self.objects[obj_i].scale[0])
                elif isinstance(self.objects[obj_i], PrimitiveCubeCfg):
                    color = gymapi.Vec3(
                        self.objects[obj_i].color[0],
                        self.objects[obj_i].color[1],
                        self.objects[obj_i].color[2],
                    )
                    self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                elif isinstance(self.objects[obj_i], PrimitiveSphereCfg):
                    color = gymapi.Vec3(
                        self.objects[obj_i].color[0],
                        self.objects[obj_i].color[1],
                        self.objects[obj_i].color[2],
                    )
                    self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                elif isinstance(self.objects[obj_i], RigidObjCfg):
                    pass
                elif isinstance(self.objects[obj_i], ArticulationObjCfg):
                    self._articulated_obj_handles[-1].append(obj_handle)
                    if self.objects[obj_i].stiffness is not None:
                        for dof_prop in self._articulated_dof_prop_dict[self.objects[obj_i].name]:
                            dof_prop[4] = self.objects[obj_i].stiffness
                    if self.objects[obj_i].damping is not None:
                        for dof_prop in self._articulated_dof_prop_dict[self.objects[obj_i].name]:
                            dof_prop[5] = self.objects[obj_i].damping
                    self.gym.set_actor_dof_properties(env, obj_handle, self._articulated_dof_prop_dict[self.objects[obj_i].name])
                else:
                    log.error("Unknown object type")
                    raise NotImplementedError
                self._obj_handles[-1].append(obj_handle)

                object_rigid_body_indices = {}
                for rigid_body_name, rigid_body_idx in self._asset_dict_dict[self.objects[obj_i].name].items():
                    rigid_body_idx = self.gym.find_actor_rigid_body_index(
                        env, obj_handle, rigid_body_name, gymapi.DOMAIN_SIM
                    )
                    object_rigid_body_indices[rigid_body_name] = rigid_body_idx

                self._env_rigid_body_global_indices[-1][self.objects[obj_i].name] = object_rigid_body_indices

            # carefully set each robot
            env_robot_handles = []
            collision_filter = 1
            for robot, robot_asset, robot_dof_props in zip(self.robots, robot_asset_list, robot_dof_props_list):
                robot_pose = gymapi.Transform()
                robot_pose.p = gymapi.Vec3(*robot.default_position)
                robot_handle = self.gym.create_actor(env, robot_asset, robot_pose, "robot", i, collision_filter)  # TODO
                collision_filter *= 2  # increase collision filter for next robot
                self.gym.enable_actor_dof_force_sensors(env, robot_handle)
                self.gym.set_actor_scale(env, robot_handle, robot.scale[0])
                assert robot.scale[0] == 1.0 and self.robot.scale[1] == 1.0 and robot.scale[2] == 1.0
                env_robot_handles.append(robot_handle)
                self.gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

                if hasattr(robot, "friction") and robot.friction is not None:
                    shape_props = self.gym.get_actor_rigid_shape_properties(env, robot_handle)
                    for shape_prop in shape_props:
                        shape_prop.friction = robot.friction
                    self.gym.set_actor_rigid_shape_properties(env, obj_handle, shape_props)

                robot_rigid_body_indices = {}
                robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
                for rigid_body_name, rigid_body_idx in robot_link_dict.items():
                    rigid_body_idx = self.gym.find_actor_rigid_body_index(
                        env, robot_handle, rigid_body_name, gymapi.DOMAIN_SIM
                    )
                    robot_rigid_body_indices[rigid_body_name] = rigid_body_idx

                self._env_rigid_body_global_indices[-1][robot.name] = robot_rigid_body_indices

            self._robot_handles.append(env_robot_handles)
            if hasattr(self, "max_agg_bodies") and hasattr(self, "max_agg_shapes"):
                self.gym.end_aggregate(env)

        # GET initial state, copy for reset later
        self._initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))
        self.actor_indices = torch.zeros((self.num_envs, len(self.objects) + len(self.robots)), dtype=torch.int32, device=self.device)
        for env_id in range(self.num_envs):
            env_offset = env_id * (len(self.objects) + len(self.robots))
            self.actor_indices[env_id, :] = torch.arange(
                env_offset, env_offset + len(self.objects) + len(self.robots)
            )

        ###### set VEWIER camera ######
        # point camera at middle env
        if not self.headless:  # TODO: update a default viewer
            cam_pos = gymapi.Vec3(1, 1, 1)
            cam_target = gymapi.Vec3(-1, -1, -0.5)
            middle_env = self._envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        ################################

    def _reorder_quat_xyzw_to_wxyz(self, state: torch.Tensor) -> torch.Tensor:
        quat_xyzw = state[..., 3:7]
        quat_wxyz = torch.cat([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], dim=-1)
        return torch.cat([state[..., 0:3], quat_wxyz, state[..., 7:]], dim=-1)

    def _reorder_quat_wxyz_to_xyzw(self, state: torch.Tensor) -> torch.Tensor:
        quat_wxyz = state[..., 3:7]
        quat_xyzw = torch.cat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)
        return torch.cat([state[..., 0:3], quat_xyzw, state[..., 7:]], dim=-1)

    def _get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        object_states = {}
        for obj_id, obj in enumerate(self.objects):
            if isinstance(obj, ArticulationObjCfg):
                joint_ids_reindex = self._get_joint_ids_reindex(obj.name)
                body_ids_reindex = self._get_body_ids_reindex(obj.name)
                root_state = self._root_states.view(self.num_envs, -1, 13)[:, obj_id, :]
                root_state = self._reorder_quat_xyzw_to_wxyz(root_state)
                body_state = self._rigid_body_states.view(self.num_envs, -1, 13)[:, body_ids_reindex, :]
                body_state = self._reorder_quat_xyzw_to_wxyz(body_state)
                state = ObjectState(
                    root_state=root_state,
                    body_names=self.get_body_names(obj.name),
                    body_state=body_state,
                    joint_pos=self._dof_states.view(self.num_envs, -1, 2)[:, joint_ids_reindex, 0],
                    joint_vel=self._dof_states.view(self.num_envs, -1, 2)[:, joint_ids_reindex, 1],
                    joint_force=self._dof_force_tensor.view(self.num_envs, -1)[:, joint_ids_reindex],
                )
            else:
                root_state = self._root_states.view(self.num_envs, -1, 13)[:, obj_id, :]
                root_state = self._reorder_quat_xyzw_to_wxyz(root_state)
                state = ObjectState(
                    root_state=root_state,
                )
            object_states[obj.name] = state

        # FIXME some RL task need joint state as dof_pos - default_dof_pos, not absolute dof_pos. see https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L216 for further details
        robot_states = {}
        for robot_id, robot in enumerate(self.robots):
            joint_ids_reindex = self._get_joint_ids_reindex(robot.name)
            body_ids_reindex = self._get_body_ids_reindex(robot.name)
            root_state = self._root_states.view(self.num_envs, -1, 13)[:, len(self.objects) + robot_id, :]
            root_state = self._reorder_quat_xyzw_to_wxyz(root_state)
            body_state = self._rigid_body_states.view(self.num_envs, -1, 13)[:, body_ids_reindex, :]
            body_state = self._reorder_quat_xyzw_to_wxyz(body_state)

            state = RobotState(
                root_state=root_state,
                body_names=self.get_body_names(robot.name),
                body_state=body_state,
                joint_pos=self._dof_states.view(self.num_envs, -1, 2)[:, joint_ids_reindex, 0],
                joint_vel=self._dof_states.view(self.num_envs, -1, 2)[:, joint_ids_reindex, 1],
                joint_force=self._dof_force_tensor.view(self.num_envs, -1)[:, joint_ids_reindex],
                joint_pos_target=None,  # TODO
                joint_vel_target=None,  # TODO
                joint_effort_target=self._effort if self._manual_pd_on else None,
            )
            # FIXME a temporary solution for accessing net contact forces of robots, it will be moved to
            extra = {
                "contact_forces": self._contact_forces.view(self.num_envs, -1, 3)[:, body_ids_reindex, :],
            }
            state.extra = extra
            robot_states[robot.name] = state

        camera_states = {}
        self.gym.start_access_image_tensors(self.sim)
        for cam_id, cam in enumerate(self.cameras):
            state = CameraState(
                rgb=torch.stack([self._rgb_tensors[env_id][cam_id][..., :3] for env_id in env_ids]),
                depth=torch.stack([self._depth_tensors[env_id][cam_id] for env_id in env_ids]),
            )
            camera_states[cam.name] = state
        self.gym.end_access_image_tensors(self.sim)

        # sensor states
        sensor_states = {}
        for i, sensor in enumerate(self.sensors):
            if isinstance(sensor, ContactForceSensorCfg):
                sensor_states[sensor.name] = SensorState(
                    force=self._vec_sensor_tensor[:, i, :3], torque=self._vec_sensor_tensor[:, i, 3:]
                )
            else:
                raise ValueError(f"Unknown sensor type: {type(sensor)}")
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors=sensor_states)

    @property
    def episode_length_buf(self) -> list[int]:
        return self._episode_length_buf

    ############################################################
    ## Gymnasium main methods
    ############################################################
    def _get_action_array_all(self, actions: list[Action]):
        action_tensor_list = []
        root_force_tensor_list = []
        root_torque_tensor_list = []
        for robot in self.robots:
            action_array_list = []
            root_force_array_list = []
            root_torque_array_list = []
            actuated_root = False
            if hasattr(robot, "actuated_root") and robot.actuated_root:
                actuated_root = True
            for action_data in actions:
                flat_vals = []
                for joint_i, joint_name in enumerate(self._joint_info[robot.name]["names"]):
                    if robot.actuators[joint_name].fully_actuated:
                        flat_vals.append(
                            action_data[robot.name]["dof_pos_target"][joint_name]
                        )  # TODO: support other actions

                    else:
                        flat_vals.append(0.0)  # place holder for under-actuated joints
                action_array = torch.tensor(flat_vals, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_array_list.append(action_array)
                if actuated_root:
                    root_force = torch.tensor(
                        action_data[robot.name]["root_force"], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    root_torque = torch.tensor(
                        action_data[robot.name]["root_torque"], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    root_force_array_list.append(root_force)
                    root_torque_array_list.append(root_torque)
            action_tensor = torch.cat(action_array_list, dim=0)  # shape: (num_envs, robot_num_dof)
            action_tensor_list.append(action_tensor)
            if actuated_root:
                root_force_tensor = torch.cat(root_force_array_list, dim=0).unsqueeze(1)  # shape: (num_envs, 1, 3)
                root_torque_tensor = torch.cat(root_torque_array_list, dim=0).unsqueeze(1)  # shape: (num_envs, 1, 3)
            else:
                root_force_tensor = torch.zeros([self.num_envs, 3], dtype=torch.float32, device=self.device)
                root_torque_tensor = torch.zeros([self.num_envs, 3], dtype=torch.float32, device=self.device)
            root_force_tensor_list.append(root_force_tensor)
            root_torque_tensor_list.append(root_torque_tensor)

        # concatenate all robot action tensors
        action_tensor_all = torch.cat(action_tensor_list, dim=-1)  # shape: (num_envs, total_robot_num_dof)
        root_force_tensor_all = torch.cat(root_force_tensor_list, dim=1)  # shape: (num_envs, robot_root_force_num, 3)
        root_torque_tensor_all = torch.cat(
            root_torque_tensor_list, dim=1
        )  # shape: (num_envs, robot_root_torque_num, 3)
        return action_tensor_all, root_force_tensor_all, root_torque_tensor_all

    def set_dof_targets(self, obj_name: list[str], actions: list[Action] | torch.Tensor):
        self._actions_cache = actions
        action_input = torch.zeros_like(self._dof_states[:, 0])  # shape: (num_envs * total_dof_num)
        applied_force = torch.zeros_like(
            self._rigid_body_states.view(self.num_envs, -1, 13)[:, :, :3]
        )  # shape: (num_envs, total_body_num, 3)
        applied_torque = torch.zeros_like(
            self._rigid_body_states.view(self.num_envs, -1, 13)[:, :, :3]
        )  # shape: (num_envs, total_body_num, 3)
        if isinstance(actions, torch.Tensor):
            # reverse sorted joint indices
            action_array_all = torch.zeros([self.num_envs, self._robot_num_dof], device=self.device)
            action_array_all = actions[:, : self._robot_num_dof]
            if self.actuated_root:
                root_action_dim = actions.shape[1] - self._robot_num_dof
                root_force_array_all = actions[
                    :, self._robot_num_dof : self._robot_num_dof + root_action_dim // 2
                ].reshape(self.num_envs, -1, 3)  # shape: (num_envs, robot_root_force_num, 3)
                root_torque_array_all = actions[
                    :, self._robot_num_dof + root_action_dim // 2 : self._robot_num_dof + root_action_dim
                ].reshape(self.num_envs, -1, 3)  # shape: (num_envs, robot_root_torque_num, 3)

        else:
            action_array_all, root_force_array_all, root_torque_array_all = self._get_action_array_all(
                actions
            )  # shape: (num_envs, total_robot_num_dof)

        assert (
            action_input.shape[0] % self._num_envs == 0
        )  # WARNING: obj dim(env0), robot dim(env0), obj dim(env1), robot dim(env1) ...

        if not hasattr(self, "_robot_dim_index"):
            robot_dim = action_array_all.shape[1]
            chunk_size = action_input.shape[0] // self._num_envs
            self._robot_dim_index = [
                i * chunk_size + offset
                for i in range(self.num_envs)
                for offset in range(chunk_size - robot_dim, chunk_size)
            ]

        if self.actuated_root and not hasattr(self, "_force_body_index"):
            # create a list of robot root body indices for each env
            self._force_body_index = []
            for robot in self.robots:
                if not hasattr(robot, "actuated_root") or not robot.actuated_root:
                    continue
                force_body_index = self._body_info[robot.name]["global_indices"][robot.actuated_link]
                self._force_body_index.append(force_body_index)

        action_input[self._robot_dim_index] = action_array_all.float().to(self.device).reshape(-1)
        if self.actuated_root:
            applied_force[:, self._force_body_index, :] = root_force_array_all.float().to(self.device)
            applied_torque[:, self._force_body_index, :] = root_torque_array_all.float().to(self.device)
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(applied_force),
                gymtorch.unwrap_tensor(applied_torque),
                gymapi.ENV_SPACE,
            )

        # if any effort joint exist, set pd controller's target position for later effort calculation
        if self._manual_pd_on:
            actions_reshape = action_input.view(self._num_envs, self._obj_num_dof + self._robot_num_dof)
            self.actions = actions_reshape[:, self._obj_num_dof :]
            # and set position target for position actuator if any exist
            if len(self._pos_ctrl_dof_dix) > 0:
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_input))

        # directly set position target
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_input))

    def set_actions(self, obj_name: str, actions: torch.Tensor) -> None:
        action_input = torch.zeros_like(self._dof_states[:, 0])

        if not hasattr(self, "_robot_dim_index"):
            chunk_size = action_input.shape[0] // self._num_envs
            robot_dim = actions.shape[1]
            self._robot_dim_index = [
                i * chunk_size + offset
                for i in range(self.num_envs)
                for offset in range(chunk_size - robot_dim, chunk_size)
            ]

        action_input[self._robot_dim_index] = actions.to(self.device).reshape(-1)

        if self._manual_pd_on:
            self.actions = action_input.view(self._num_envs, self._obj_num_dof + self._robot_num_dof)[
                :, self._obj_num_dof :
            ]
            if self._pos_ctrl_dof_dix:
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_input))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_input))

    def refresh_render(self) -> None:
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._render()

    def _simulate_one_physics_step(self, action):
        # for pd control joints by effort api, update torque and step the physics
        if self._manual_pd_on:
            self._apply_pd_control(action)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for position control joints, just step the physics
        else:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

    def _simulate(self) -> None:
        # Step the physics
        for _ in range(self.scenario.decimation):
            self._simulate_one_physics_step(self.actions)

        # Refresh tensors
        if not self._manual_pd_on:
            self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh cameras and
        #         self.gym.step_graphics(self.sim)
        self._render()

    def _render(self) -> None:
        """Listen for keyboard events, step graphics and render the environment"""
        if not self.headless:
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self._enable_viewer_sync = not self._enable_viewer_sync
            if self._enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            else:
                self.gym.poll_viewer_events(self.viewer)
        if self.headless and len(self.cameras) > 0:
            # if headless, we still need to render the cameras
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

    def _compute_effort(self, actions):
        """Compute effort from actions"""
        # scale the actions (generally output from policy)
        action_scaled = self._action_scale * actions
        robot_dof_pos = self._robot_dof_state[..., 0]
        robot_dof_vel = self._robot_dof_state[..., 1]
        if self._action_offset:
            _effort = (
                self._p_gains * (action_scaled + self._robot_default_dof_pos - robot_dof_pos)
                - self._d_gains * robot_dof_vel
            )
        else:
            _effort = self._p_gains * (action_scaled - robot_dof_pos) - self._d_gains * robot_dof_vel
        self._effort = torch.clip(_effort, -self._torque_limits, self._torque_limits)
        effort = self._effort.to(torch.float32)
        return effort

    def _apply_pd_control(self, actions):
        """
        Compute torque using pd controller for effort actuator and set torque.
        """
        effort = self._compute_effort(actions)

        # NOTE: effort passed set_dof_actuation_force_tensor() must have the same dimension as the number of DOFs, even if some DOFs are not actionable.
        if self._obj_num_dof > 0:
            obj_force_placeholder = torch.zeros((self._num_envs, self._obj_num_dof), device=self.device)
            effort = torch.cat((obj_force_placeholder, effort), dim=1)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort))

    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None):
        ## Support setting status only for specified env_ids
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if isinstance(states, list):
            assert len(states) == self.num_envs, (
                f"The length of the state list ({len(states)}) must match the length of num_envs ({self.num_envs})."
            )

            pos_list = []
            rot_list = []
            q_list = []
            states_flat = [{**states[i]["objects"], **states[i]["robots"]} for i in env_ids]

            # Prepare state data for specified env_ids
            env_indices = {env_id: i for i, env_id in enumerate(env_ids)}

            for i in range(self.num_envs):
                if i not in env_indices:
                    continue

                state_idx = env_indices[i]
                state = states_flat[state_idx]

                pos_list_i = []
                rot_list_i = []
                q_list_i = []
                for obj in self.objects:
                    obj_name = obj.name
                    pos = np.array(state[obj_name].get("pos", [0.0, 0.0, 0.0]))
                    rot = np.array(state[obj_name].get("rot", [1.0, 0.0, 0.0, 0.0]))
                    obj_quat = [rot[1], rot[2], rot[3], rot[0]]  # IsaacGym convention

                    pos_list_i.append(pos)
                    rot_list_i.append(obj_quat)
                    if isinstance(obj, ArticulationObjCfg):
                        obj_joint_q = np.zeros(len(self._articulated_joint_dict_dict[obj_name]))
                        articulated_joint_dict = self._articulated_joint_dict_dict[obj_name]
                        for joint_name, joint_idx in articulated_joint_dict.items():
                            if "dof_pos" in state[obj_name]:
                                obj_joint_q[joint_idx] = state[obj_name]["dof_pos"][joint_name]
                            else:
                                log.warning(f"No dof_pos for {joint_name} in {obj_name}")
                                obj_joint_q[joint_idx] = 0.0
                        q_list_i.append(obj_joint_q)

                for robot, robot_asset in zip(self.robots, self.robot_asset_list):
                    robot_joint_dict = self.gym.get_asset_dof_dict(robot_asset)
                    pos_list_i.append(np.array(state[robot.name].get("pos", [0.0, 0.0, 0.0])))
                    rot = np.array(state[robot.name].get("rot", [1.0, 0.0, 0.0, 0.0]))
                    robot_quat = [rot[1], rot[2], rot[3], rot[0]]
                    rot_list_i.append(robot_quat)

                    robot_dof_state_i = np.zeros(robot.num_joints)
                    if "dof_pos" in state[robot.name]:
                        for joint_name, joint_idx in robot_joint_dict.items():
                            robot_dof_state_i[joint_idx] = state[robot.name]["dof_pos"][joint_name]
                    else:
                        for joint_name, joint_idx in robot_joint_dict.items():
                            robot_dof_state_i[joint_idx] = (
                                robot.joint_limits[joint_name][0] + robot.joint_limits[joint_name][1]
                            ) / 2
                    q_list_i.append(robot_dof_state_i)

                pos_list.append(pos_list_i)
                rot_list.append(rot_list_i)
                q_list.append(q_list_i)

            self._set_actor_root_state(pos_list, rot_list, env_ids)
            self._set_actor_joint_state(q_list, env_ids)
        elif isinstance(states, TensorState):
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.int32, device=self.device)
            new_root_states = self._root_states.view(self.num_envs, -1, 13).clone()
            new_dof_states = self._dof_states.view(self.num_envs, -1, 2).clone()
            for obj_id, obj in enumerate(self.objects):
                obj_state = states.objects[obj.name]
                root_state = self._reorder_quat_wxyz_to_xyzw(obj_state.root_state)
                new_root_states[env_ids, obj_id, :] = root_state[env_ids, :].clone()
                if isinstance(obj, ArticulationObjCfg) and len(self._joint_info[obj.name]["names"]) > 0:
                    joint_pos = obj_state.joint_pos
                    global_dof_indices = torch.tensor(list(self._joint_info[obj.name]["global_indices"].values()), dtype=torch.int32, device=self.device)
                    new_dof_states[env_ids_tensor.unsqueeze(1), global_dof_indices.unsqueeze(0), 0] = joint_pos[env_ids, :].clone()
                    new_dof_states[env_ids_tensor.unsqueeze(1), global_dof_indices.unsqueeze(0), 1] = 0.0
            for robot_id, robot in enumerate(self.robots):
                robot_state = states.robots[robot.name]
                root_state = self._reorder_quat_wxyz_to_xyzw(robot_state.root_state)
                new_root_states[env_ids, len(self.objects) + robot_id, :] = root_state[env_ids, :].clone()
                joint_pos = robot_state.joint_pos
                global_dof_indices = torch.tensor(list(self._joint_info[robot.name]["global_indices"].values()), dtype=torch.int32, device=self.device)
                new_dof_states[env_ids_tensor.unsqueeze(1), global_dof_indices.unsqueeze(0), 0] = joint_pos[env_ids, :].clone()
                new_dof_states[env_ids_tensor.unsqueeze(1), global_dof_indices.unsqueeze(0), 1] = 0.0

            new_root_states = new_root_states.view(-1, 13)
            new_dof_states = new_dof_states.view(-1, 2)
            root_reset_actors_indices = self.actor_indices[env_ids, :].view(-1)

            res = self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(new_root_states),
                gymtorch.unwrap_tensor(root_reset_actors_indices),
                len(root_reset_actors_indices),
            )
            assert res

            res = self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(new_dof_states))
            assert res

        else:
            raise Exception("Unsupported state type, must be EnvState or TensorState")


        # Refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # reset all env_id action to default
        self.actions[env_ids] = 0.0

    def _set_actor_root_state(self, position_list, rotation_list, env_ids):
        new_root_states = self._root_states.clone()

        # Only modify the positions and rotations for the specified env_ids
        for i, env_id in enumerate(env_ids):
            env_offset = env_id * (len(self.objects) + len(self.robots))  # objects + robot
            for j in range(len(self.objects) + len(self.robots)):
                actor_idx = env_offset + j
                new_root_states[actor_idx, :3] = torch.tensor(
                    position_list[i][j], dtype=torch.float32, device=self.device
                )
                new_root_states[actor_idx, 3:7] = torch.tensor(
                    rotation_list[i][j], dtype=torch.float32, device=self.device
                )
                new_root_states[actor_idx, 7:13] = torch.zeros(6, dtype=torch.float32, device=self.device)

        # Get the actor indices to update
        actor_indices = []
        for env_id in env_ids:
            env_offset = env_id * (len(self.objects) + len(self.robots))
            actor_indices.extend(range(env_offset, env_offset + len(self.objects) + len(self.robots)))

        # Convert the actor indices to a tensor
        root_reset_actors_indices = torch.tensor(actor_indices, dtype=torch.int32, device=self.device)

        # Use indexed setting to set the root state
        res = self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(new_root_states),
            gymtorch.unwrap_tensor(root_reset_actors_indices),
            len(root_reset_actors_indices),
        )
        assert res

        return

    def _set_actor_joint_state(self, joint_pos_list, env_ids):
        new_dof_states = self._dof_states.clone()

        # Calculate the indices of DOFs in the tensor
        dof_indices = []
        new_dof_pos_values = []

        for i, env_id in enumerate(env_ids):
            # Get the joint positions for this environment
            flat_vals = []
            for obj_joints in joint_pos_list[i]:
                flat_vals.extend(obj_joints)

            # Calculate the indices of DOFs in the global DOF tensor
            dof_start_idx = env_id * self._num_joints
            for j, val in enumerate(flat_vals):
                dof_idx = dof_start_idx + j
                dof_indices.append(dof_idx)
                new_dof_pos_values.append(val)

        # Update the DOF positions for the specified indices
        dof_indices_tensor = torch.tensor(dof_indices, dtype=torch.int64, device=self.device)
        new_dof_pos_tensor = torch.tensor(new_dof_pos_values, dtype=torch.float32, device=self.device)

        # Update the positions and velocities (set velocities to 0)
        new_dof_states[dof_indices_tensor, 0] = new_dof_pos_tensor
        new_dof_states[dof_indices_tensor, 1] = 0.0

        # Apply the updated DOF state
        res = self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(new_dof_states))
        assert res

        return

    def close(self) -> None:
        try:
            self.gym.destroy_sim(self.sim)
            self.gym.destroy_viewer(self.viewer)
            self.gym = None
            self.sim = None
            self.viewer = None
        except Exception as e:
            log.error(f"Error closing IsaacGym environment: {e}")
            pass

    ############################################################
    ## Utils
    ############################################################
    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = list(self._joint_info[obj_name]["names"])
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            body_names = self._body_info[obj_name]["name"]
            if sort:
                body_names.sort()
            return body_names
        else:
            return []

    def _get_body_ids_reindex(self, obj_name: str) -> list[int]:
        return [self._body_info[obj_name]["global_indices"][bn] for bn in self.get_body_names(obj_name)]

    def _get_joint_ids_reindex(self, obj_name: str) -> list[int]:
        return [self._joint_info[obj_name]["global_indices"][jn] for jn in self.get_joint_names(obj_name)]

    def get_body_reindexed_indices_from_substring(self, obj_name, body_names: list[str]) -> torch.tensor:
        """given substring of body name, find all the bodies indices in sorted order."""
        matches = []
        for name in body_names:
            matches.extend([s for s in self._body_info[obj_name]["name"] if name in s])
        index = torch.zeros(len(matches), dtype=torch.int32, device=self.device)
        for i, name in enumerate(matches):
            index[i] = list(self._body_info[obj_name]["local_indices"]).index(name)
        return index

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def default_dof_pos(self) -> torch.tensor:
        joint_reindex = []
        start_idx = 0
        for robot in self.robots:
            robot_joint_reindex = self.get_joint_reindex(robot.name)
            joint_reindex.extend([start_idx + j for j in robot_joint_reindex])
            start_idx += robot.num_joints
        return self._robot_default_dof_pos[:, joint_reindex]

    @property
    def torque_limits(self) -> torch.tensor:
        return self._torque_limits

    @property
    def robot_num_dof(self) -> int:
        return self._robot_num_dof


# TODO: try to align handler API and use GymWrapper instead
IsaacgymEnv: type[EnvWrapper[IsaacgymHandler]] = GymEnvWrapper(IsaacgymHandler)
