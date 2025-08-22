# This naively suites for isaaclab 2.2.0 and isaacsim 5.0.0
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any

import torch
from loguru import logger as log

from metasim.queries.base import BaseQueryType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    BaseArticulationObjCfg,
    BaseObjCfg,
    BaseRigidObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveFrameCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import DictEnvState
from metasim.utils.dict import deep_get
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState

from .material_help import MaterialHelper


class IsaacsimHandler(BaseSimHandler):
    """
    Handler for Isaac Lab simulation environment.
    This class extends BaseSimHandler to provide specific functionality for Isaac Lab.
    """

    def __init__(self, scenario_cfg: ScenarioCfg, optional_queries: list[BaseQueryType] | None = None):
        super().__init__(scenario_cfg, optional_queries)

        # self._actions_cache: list[Action] = []
        self._robot_names = {robot.name for robot in self.robots}
        self._robot_init_pos = {robot.name: robot.default_position for robot in self.robots}
        self._robot_init_quat = {robot.name: robot.default_orientation for robot in self.robots}
        self._cameras = scenario_cfg.cameras

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_envs: int = scenario_cfg.num_envs
        self._episode_length_buf = [0 for _ in range(self.num_envs)]

        self.scenario_cfg = scenario_cfg
        self.dt = self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 0.01
        self._step_counter = 0
        self._is_closed = False
        self.render_interval = 4  # TODO: fix hardcode

        # Initialize material helper
        self.material_helper = MaterialHelper(self)

    def _init_scene(self) -> None:
        """
        Initializes the isaacsim simulation environment.
        """
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        args.enable_cameras = True
        args.headless = self.headless
        app_launcher = AppLauncher(args)
        self.simulation_app = app_launcher.app

        # physics context
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext

        sim_config: SimulationCfg = SimulationCfg(
            device="cuda:0",
            render_interval=self.scenario.decimation,  # TTODO divide into render interval and control decimation
            physx=PhysxCfg(
                bounce_threshold_velocity=self.scenario.sim_params.bounce_threshold_velocity,
                solver_type=self.scenario.sim_params.solver_type,
                max_position_iteration_count=self.scenario.sim_params.num_position_iterations,
                max_velocity_iteration_count=self.scenario.sim_params.num_velocity_iterations,
                friction_correlation_distance=self.scenario.sim_params.friction_correlation_distance,
            ),
        )
        if self.scenario.sim_params.dt is not None:
            sim_config.dt = self.scenario.sim_params.dt

        self.sim: SimulationContext = SimulationContext(sim_config)
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self._num_envs, env_spacing=self.scenario.env_spacing
        )
        self.scene = InteractiveScene(scene_config)

    def _load_robots(self) -> None:
        for robot in self.robots:
            self._add_robot(robot)

    def _load_objects(self) -> None:
        for obj_cfg in self.objects:
            self._add_object(obj_cfg)

    def _load_cameras(self) -> None:
        for camera in self.cameras:
            if isinstance(camera, PinholeCameraCfg):
                self._add_pinhole_camera(camera)
            else:
                raise ValueError(f"Unsupported camera type: {type(camera)}")

    def _update_camera_pose(self) -> None:
        for camera in self.cameras:
            if isinstance(camera, PinholeCameraCfg):
                # set look at position using isaaclab's api
                if camera.mount_to is None:
                    camera_inst = self.scene.sensors[camera.name]
                    position_tensor = torch.tensor(camera.pos, device=self.device, dtype=torch.float32).unsqueeze(0)
                    position_tensor = position_tensor.repeat(self.num_envs, 1)
                    camera_lookat_tensor = torch.tensor(
                        camera.look_at, device=self.device, dtype=torch.float32
                    ).unsqueeze(0)
                    camera_lookat_tensor = camera_lookat_tensor.repeat(self.num_envs, 1)
                    camera_inst.set_world_poses_from_view(position_tensor, camera_lookat_tensor)
                    # log.debug(f"Updated camera {camera.name} pose: pos={camera.pos}, look_at={camera.look_at}")
            else:
                raise ValueError(f"Unsupported camera type: {type(camera)}")

    def launch(self) -> None:
        self._init_scene()
        self._load_robots()
        self._load_cameras()
        self._load_terrain()
        self._load_objects()
        self._load_lights()
        self._load_render_settings()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.sim.reset()
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.scene.reset(indices)

        # Update camera pose after scene reset to avoid being overridden
        self._update_camera_pose()

        # Force another simulation step and camera update to ensure proper initialization
        self.sim.step(render=False)
        self.scene.update(dt=self.dt)
        self._update_camera_pose()

        # Force a render to update camera data after position is set
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()
        for sensor in self.scene.sensors.values():
            sensor.update(dt=0)

    def close(self) -> None:
        log.info("close Isaacsim Handler")
        if not self._is_closed:
            del self.scene
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            self.sim.stop()
            self.sim.clear()
            self._is_closed = True

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    def _set_states(self, states: list[DictEnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states_flat = [states[i]["objects"] | states[i]["robots"] for i in range(self.num_envs)]
        for obj in self.objects + self.robots:
            if obj.name not in states_flat[0]:
                log.warning(f"Missing {obj.name} in states, setting its velocity to zero")
                pos, rot = self._get_pose(obj.name, env_ids=env_ids)
                self._set_object_pose(obj, pos, rot, env_ids=env_ids)
                continue

            if states_flat[0][obj.name].get("pos", None) is None or states_flat[0][obj.name].get("rot", None) is None:
                log.warning(f"No pose found for {obj.name}, setting its velocity to zero")
                pos, rot = self._get_pose(obj.name, env_ids=env_ids)
                self._set_object_pose(obj, pos, rot, env_ids=env_ids)
            else:
                pos = torch.stack([states_flat[env_id][obj.name]["pos"] for env_id in env_ids]).to(self.device)
                rot = torch.stack([states_flat[env_id][obj.name]["rot"] for env_id in env_ids]).to(self.device)
                self._set_object_pose(obj, pos, rot, env_ids=env_ids)

            if isinstance(obj, ArticulationObjCfg):
                if states_flat[0][obj.name].get("dof_pos", None) is None:
                    log.warning(f"No dof_pos found for {obj.name}")
                else:
                    dof_dict = [states_flat[env_id][obj.name]["dof_pos"] for env_id in env_ids]
                    joint_names = self._get_joint_names(obj.name, sort=False)
                    joint_pos = torch.zeros((len(env_ids), len(joint_names)), device=self.device)
                    for i, joint_name in enumerate(joint_names):
                        if joint_name in dof_dict[0]:
                            joint_pos[:, i] = torch.tensor([x[joint_name] for x in dof_dict], device=self.device)
                        else:
                            log.warning(f"Missing {joint_name} in {obj.name}, setting its position to zero")

                    self._set_object_joint_pos(obj, joint_pos, env_ids=env_ids)
                    if obj in self.robots:
                        robot_inst = self.scene.articulations[obj.name]
                        robot_inst.set_joint_position_target(
                            joint_pos, env_ids=torch.tensor(env_ids, device=self.device)
                        )
                        robot_inst.write_data_to_sim()

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Special handling for the first frame to ensure camera is properly positioned
        if self._step_counter == 0:
            self._update_camera_pose()
            # Force render and sensor update for first frame
            if self.sim.has_gui() or self.sim.has_rtx_sensors():
                self.sim.render()
            for sensor in self.scene.sensors.values():
                sensor.update(dt=0)

        object_states = {}
        for obj in self.objects:
            if isinstance(obj, ArticulationObjCfg):
                obj_inst = self.scene.articulations[obj.name]
                joint_reindex = self.get_joint_reindex(obj.name)
                body_reindex = self.get_body_reindex(obj.name)
                root_state = obj_inst.data.root_state_w
                root_state[:, 0:3] -= self.scene.env_origins
                body_state = obj_inst.data.body_state_w[:, body_reindex]
                body_state[:, :, 0:3] -= self.scene.env_origins[:, None, :]
                state = ObjectState(
                    root_state=root_state,
                    body_names=self._get_body_names(obj.name),
                    body_state=body_state,
                    joint_pos=obj_inst.data.joint_pos[:, joint_reindex],
                    joint_vel=obj_inst.data.joint_vel[:, joint_reindex],
                )
            else:
                obj_inst = self.scene.rigid_objects[obj.name]
                root_state = obj_inst.data.root_state_w
                root_state[:, 0:3] -= self.scene.env_origins
                state = ObjectState(
                    root_state=root_state,
                )
            object_states[obj.name] = state

        robot_states = {}
        for obj in self.robots:
            ## TODO: dof_pos_target, dof_vel_target, dof_torque
            obj_inst = self.scene.articulations[obj.name]
            joint_reindex = self.get_joint_reindex(obj.name)
            body_reindex = self.get_body_reindex(obj.name)
            root_state = obj_inst.data.root_state_w
            root_state[:, 0:3] -= self.scene.env_origins
            body_state = obj_inst.data.body_state_w[:, body_reindex]
            body_state[:, :, 0:3] -= self.scene.env_origins[:, None, :]
            state = RobotState(
                root_state=root_state,
                body_names=self._get_body_names(obj.name),
                body_state=body_state,
                joint_pos=obj_inst.data.joint_pos[:, joint_reindex],
                joint_vel=obj_inst.data.joint_vel[:, joint_reindex],
                joint_pos_target=obj_inst.data.joint_pos_target[:, joint_reindex],
                joint_vel_target=obj_inst.data.joint_vel_target[:, joint_reindex],
                joint_effort_target=obj_inst.data.joint_effort_target[:, joint_reindex],
            )
            robot_states[obj.name] = state

        camera_states = {}
        # Force camera sensor update to ensure correct position data
        for sensor in self.scene.sensors.values():
            sensor.update(dt=0)

        for camera in self.cameras:
            camera_inst = self.scene.sensors[camera.name]
            rgb_data = camera_inst.data.output.get("rgb", None)
            depth_data = camera_inst.data.output.get("depth", None)
            instance_seg_data = deep_get(camera_inst.data.output, "instance_segmentation_fast")
            instance_seg_id2label = deep_get(camera_inst.data.info, "instance_segmentation_fast", "idToLabels")
            instance_id_seg_data = deep_get(camera_inst.data.output, "instance_id_segmentation_fast")
            instance_id_seg_id2label = deep_get(camera_inst.data.info, "instance_id_segmentation_fast", "idToLabels")
            if instance_seg_data is not None:
                instance_seg_data = instance_seg_data.squeeze(-1)
            if instance_id_seg_data is not None:
                instance_id_seg_data = instance_id_seg_data.squeeze(-1)
            camera_states[camera.name] = CameraState(
                rgb=rgb_data,
                depth=depth_data,
                instance_seg=instance_seg_data,
                instance_seg_id2label=instance_seg_id2label,
                instance_id_seg=instance_id_seg_data,
                instance_id_seg_id2label=instance_id_seg_id2label,
                pos=camera_inst.data.pos_w,
                quat_world=camera_inst.data.quat_w_world,
                intrinsics=torch.tensor(camera.intrinsics, device=self.device)[None, ...].repeat(self.num_envs, 1, 1),
            )

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states)

    def set_dof_targets(self, actions: torch.Tensor) -> None:
        # TODO: support set torque
        if isinstance(actions, torch.Tensor):
            reverse_reindex = self.get_joint_reindex(self.robots[0].name, inverse=True)
            action_tensor_all = actions[:, reverse_reindex]
        else:
            # Process dictionary-based actions
            action_tensors = []
            for robot in self.robots:
                actuator_names = [k for k, v in robot.actuators.items() if v.fully_actuated]
                action_tensor = torch.zeros((self.num_envs, len(actuator_names)), device=self.device)
                for env_id in range(self.num_envs):
                    for i, actuator_name in enumerate(actuator_names):
                        action_tensor[env_id, i] = torch.tensor(
                            actions[env_id][robot.name]["dof_pos_target"][actuator_name], device=self.device
                        )
                action_tensors.append(action_tensor)
            action_tensor_all = torch.cat(action_tensors, dim=-1)

        # Apply actions to all robots
        start_idx = 0
        for robot in self.robots:
            robot_inst = self.scene.articulations[robot.name]
            actionable_joint_ids = [
                robot_inst.joint_names.index(jn) for jn in robot.actuators if robot.actuators[jn].fully_actuated
            ]
            robot_inst.set_joint_position_target(
                action_tensor_all[:, start_idx : start_idx + len(actionable_joint_ids)],
                joint_ids=actionable_joint_ids,
            )
            start_idx += len(actionable_joint_ids)

    def _simulate(self):
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if self._step_counter % self.render_interval == 0 and is_rendering:
            self.sim.render()
        self.scene.update(dt=self.dt)

        # Ensure camera pose is correct, especially for the first few frames
        if self._step_counter < 5:
            self._update_camera_pose()

        self._step_counter += 1

    def _add_robot(self, robot: ArticulationObjCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=robot.usd_path,
                activate_contact_sensors=True,  # TODO: only activate when contact sensor is added
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=robot.fix_base_link),
            ),
            actuators={
                jn: ImplicitActuatorCfg(
                    joint_names_expr=[jn],
                    stiffness=actuator.stiffness,
                    damping=actuator.damping,
                )
                for jn, actuator in robot.actuators.items()
            },
        )
        cfg.prim_path = f"/World/envs/env_.*/{robot.name}"
        cfg.spawn.usd_path = os.path.abspath(robot.usd_path)
        cfg.spawn.rigid_props.disable_gravity = not robot.enabled_gravity
        cfg.spawn.articulation_props.enabled_self_collisions = robot.enabled_self_collisions
        init_state = ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            joint_pos=robot.default_joint_positions,
            joint_vel={".*": 0.0},
        )
        cfg.init_state = init_state
        for joint_name, actuator in robot.actuators.items():
            cfg.actuators[joint_name].velocity_limit = actuator.velocity_limit
        robot_inst = Articulation(cfg)
        self.scene.articulations[robot.name] = robot_inst

    def _add_object(self, obj: BaseObjCfg) -> None:
        """Add an object to the scene."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

        assert isinstance(obj, BaseObjCfg)

        # Assert that this object should not already exist - if it does, there's a logic bug
        if obj.name in self.scene.rigid_objects or obj.name in self.scene.articulations:
            raise RuntimeError(
                f"Logic error: Attempting to add object '{obj.name}' that already exists in scene. "
                "This indicates a bug in the object update logic."
            )

        prim_path = f"/World/envs/env_.*/{obj.name}"

        ## Articulation object
        if isinstance(obj, ArticulationObjCfg):
            cfg = ArticulationCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=obj.usd_path,
                    scale=obj.scale,
                    activate_contact_sensors=True,  # Enable contact sensors for objects
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        fix_root_link=obj.fix_base_link,
                        enabled_self_collisions=getattr(obj, "enabled_self_collisions", True),
                    ),
                ),
                actuators={},  # Objects typically don't have actuators
            )

            # Set absolute path for USD file
            cfg.spawn.usd_path = os.path.abspath(obj.usd_path)

            # Configure physics properties
            cfg.spawn.rigid_props.disable_gravity = not obj.enabled_gravity

            # Set initial state with default position and orientation
            init_state = ArticulationCfg.InitialStateCfg(
                pos=list(obj.default_position),
                rot=list(obj.default_orientation),
                joint_pos=getattr(obj, "default_joint_positions", {".*": 0.0}),
                joint_vel={".*": 0.0},
            )
            cfg.init_state = init_state

            self.scene.articulations[obj.name] = Articulation(cfg)
            return

        if obj.fix_base_link:
            rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True)
        else:
            rigid_props = sim_utils.RigidBodyPropertiesCfg()
        if obj.collision_enabled:
            collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        else:
            collision_props = None

        ## Primitive object
        if isinstance(obj, PrimitiveCubeCfg):
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshCuboidCfg(
                        size=obj.size,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveSphereCfg):
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshSphereCfg(
                        radius=obj.radius,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveCylinderCfg):
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshCylinderCfg(
                        radius=obj.radius,
                        height=obj.height,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveFrameCfg):
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="metasim/data/quick_start/assets/COMMON/frame/usd/frame.usd",
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True, kinematic_enabled=True
                        ),  # fixed
                        collision_props=None,  # no collision
                        scale=obj.scale,
                    ),
                )
            )
            return

        ## Rigid object
        if isinstance(obj, RigidObjCfg):
            usd_file_cfg = sim_utils.UsdFileCfg(
                usd_path=obj.usd_path,
                rigid_props=rigid_props,
                collision_props=collision_props,
                scale=obj.scale,
            )
            if isinstance(obj, RigidObjCfg):
                self.scene.rigid_objects[obj.name] = RigidObject(
                    RigidObjectCfg(prim_path=prim_path, spawn=usd_file_cfg)
                )
                return

        raise ValueError(f"Unsupported object type: {type(obj)}")

    def _load_terrain(self) -> None:
        # TODO support multiple terrains cfg
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainImporterCfg

        terrain_config = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        terrain_config.num_envs = self.scene.cfg.num_envs
        terrain_config.env_spacing = self.scene.cfg.env_spacing

        self.terrain = terrain_config.class_type(terrain_config)
        self.scene._terrain = self.terrain
        # self.terrain.env_origins = self.terrain.terrain_origins

    def _load_render_settings(self) -> None:
        import carb
        import omni.replicator.core as rep

        # from omni.rtx.settings.core.widgets.pt_widgets import PathTracingSettingsFrame

        rep.settings.set_render_rtx_realtime()  # fix noising rendered images

        settings = carb.settings.get_settings()
        if self.scenario.render.mode == "pathtracing":
            settings.set_string("/rtx/rendermode", "PathTracing")
        elif self.scenario.render.mode == "raytracing":
            settings.set_string("/rtx/rendermode", "RayTracedLighting")
        elif self.scenario.render.mode == "rasterization":
            raise ValueError("Isaaclab does not support rasterization")
        else:
            raise ValueError(f"Unknown render mode: {self.scenario.render.mode}")

        log.info(f"Render mode: {settings.get_as_string('/rtx/rendermode')}")
        log.info(f"Render totalSpp: {settings.get('/rtx/pathtracing/totalSpp')}")
        log.info(f"Render spp: {settings.get('/rtx/pathtracing/spp')}")
        log.info(f"Render adaptiveSampling/enabled: {settings.get('/rtx/pathtracing/adaptiveSampling/enabled')}")
        log.info(f"Render maxBounces: {settings.get('/rtx/pathtracing/maxBounces')}")

    def _load_sensors(self) -> None:
        # TODO move it into query
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
        )
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

    def _load_lights(self) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        from metasim.scenario.lights import (
            CylinderLightCfg,
            DiskLightCfg,
            DistantLightCfg,
            DomeLightCfg,
            SphereLightCfg,
        )

        # Use lights from scenario configuration if available
        if hasattr(self.scenario, "lights") and self.scenario.lights:
            for i, light_cfg in enumerate(self.scenario.lights):
                if isinstance(light_cfg, DistantLightCfg):
                    self._add_distant_light(light_cfg, i)
                elif isinstance(light_cfg, CylinderLightCfg):
                    self._add_cylinder_light(light_cfg, i)
                elif isinstance(light_cfg, DomeLightCfg):
                    self._add_dome_light(light_cfg, i)
                elif isinstance(light_cfg, SphereLightCfg):
                    self._add_sphere_light(light_cfg, i)
                elif isinstance(light_cfg, DiskLightCfg):
                    self._add_disk_light(light_cfg, i)
                else:
                    log.warning(f"Unsupported light type: {type(light_cfg)}, skipping...")
        else:
            # Fallback to default light if no lights are configured
            log.info("No lights configured, using default distant light")
            spawn_light(
                "/World/DefaultLight",
                sim_utils.DistantLightCfg(intensity=2000.0, angle=0.53),  # Increased default intensity
                orientation=(1.0, 0.0, 0.0, 0.0),
                translation=(0, 0, 10),
            )

    def _add_distant_light(self, light_cfg, light_index: int) -> None:
        """Add a distant light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Ensure we have a valid light name
        base_name = light_cfg.name if light_cfg.name else f"distant_light_{light_index}"
        light_name = f"/World/{base_name}"

        # Create Isaac Lab distant light configuration with all supported properties
        # Create Isaac Lab distant light configuration
        isaac_light_cfg_params = {
            "intensity": light_cfg.intensity,
            "angle": getattr(light_cfg, "angle", 0.53),
            "color": light_cfg.color,
            "exposure": getattr(light_cfg, "exposure", 0.0),
            "enable_color_temperature": getattr(light_cfg, "enable_color_temperature", False),
            "color_temperature": getattr(light_cfg, "color_temperature", 6500.0),
        }

        # Create Isaac Lab distant light configuration (normalize not supported in spawn_light)
        isaac_light_cfg = sim_utils.DistantLightCfg(**isaac_light_cfg_params)

        # Use the quaternion from light configuration
        orientation = light_cfg.quat

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=orientation,
            translation=(0, 0, 10),  # Distant lights don't need specific translation
        )

        log.debug(
            f"Added distant light {light_name} with intensity {light_cfg.intensity}, "
            f"angle={getattr(light_cfg, 'angle', 0.53)}°, polar={light_cfg.polar}°, azimuth={light_cfg.azimuth}°"
        )

    def _add_cylinder_light(self, light_cfg, light_index: int) -> None:
        """Add a cylinder light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Ensure we have a valid light name
        base_name = light_cfg.name if light_cfg.name else f"cylinder_light_{light_index}"
        light_name = f"/World/{base_name}"

        # Create Isaac Lab cylinder light configuration with all supported properties
        # Create Isaac Lab cylinder light configuration
        isaac_light_cfg_params = {
            "intensity": light_cfg.intensity,
            "radius": light_cfg.radius,
            "length": light_cfg.length,
            "color": light_cfg.color,
            "exposure": getattr(light_cfg, "exposure", 0.0),
            "enable_color_temperature": getattr(light_cfg, "enable_color_temperature", False),
            "color_temperature": getattr(light_cfg, "color_temperature", 6500.0),
            "treat_as_line": getattr(light_cfg, "treat_as_line", False),
        }

        # Create Isaac Lab cylinder light configuration (normalize not supported in spawn_light)
        isaac_light_cfg = sim_utils.CylinderLightCfg(**isaac_light_cfg_params)

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=light_cfg.rot,
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added cylinder light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius}, length={light_cfg.length}, treat_as_line={getattr(light_cfg, 'treat_as_line', False)}"
        )

    def _add_dome_light(self, light_cfg, light_index: int) -> None:
        """Add a dome light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Ensure we have a valid light name
        base_name = light_cfg.name if light_cfg.name else f"dome_light_{light_index}"
        light_name = f"/World/{base_name}"

        # Create Isaac Lab dome light configuration
        isaac_light_cfg_params = {
            "intensity": light_cfg.intensity,
            "color": light_cfg.color,
            "exposure": getattr(light_cfg, "exposure", 0.0),
            "enable_color_temperature": getattr(light_cfg, "enable_color_temperature", False),
            "color_temperature": getattr(light_cfg, "color_temperature", 6500.0),
            "visible_in_primary_ray": getattr(light_cfg, "visible_in_primary_ray", True),
        }

        # Create Isaac Lab dome light configuration (normalize not supported in spawn_light)
        isaac_light_cfg = sim_utils.DomeLightCfg(**isaac_light_cfg_params)

        # Add texture if specified
        if light_cfg.texture_file is not None:
            isaac_light_cfg.texture_file = light_cfg.texture_file
            isaac_light_cfg.texture_format = getattr(light_cfg, "texture_format", "automatic")

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=(1.0, 0.0, 0.0, 0.0),
            translation=(0, 0, 0),  # Dome lights are typically at origin
        )

        log.debug(f"Added dome light {light_name} with intensity {light_cfg.intensity}")

    def _add_sphere_light(self, light_cfg, light_index: int) -> None:
        """Add a sphere light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Ensure we have a valid light name
        base_name = light_cfg.name if light_cfg.name else f"sphere_light_{light_index}"
        light_name = f"/World/{base_name}"

        # Create Isaac Lab sphere light configuration with all supported properties
        # Create Isaac Lab sphere light configuration
        isaac_light_cfg_params = {
            "intensity": light_cfg.intensity,
            "color": light_cfg.color,
            "radius": light_cfg.radius,
            "exposure": getattr(light_cfg, "exposure", 0.0),
            "enable_color_temperature": getattr(light_cfg, "enable_color_temperature", False),
            "color_temperature": getattr(light_cfg, "color_temperature", 6500.0),
            "treat_as_point": getattr(light_cfg, "treat_as_point", False),
        }

        # Create Isaac Lab sphere light configuration (normalize not supported in spawn_light)
        isaac_light_cfg = sim_utils.SphereLightCfg(**isaac_light_cfg_params)

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=(1.0, 0.0, 0.0, 0.0),
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added sphere light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius}, treat_as_point={getattr(light_cfg, 'treat_as_point', False)} at {light_cfg.pos}"
        )

    def _add_disk_light(self, light_cfg, light_index: int) -> None:
        """Add a disk light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Ensure we have a valid light name
        base_name = light_cfg.name if light_cfg.name else f"disk_light_{light_index}"
        light_name = f"/World/{base_name}"

        # Create Isaac Lab disk light configuration with all supported properties
        # Create Isaac Lab disk light configuration
        isaac_light_cfg_params = {
            "intensity": light_cfg.intensity,
            "color": light_cfg.color,
            "radius": light_cfg.radius,
            "exposure": getattr(light_cfg, "exposure", 0.0),
            "enable_color_temperature": getattr(light_cfg, "enable_color_temperature", False),
            "color_temperature": getattr(light_cfg, "color_temperature", 6500.0),
        }

        # Create Isaac Lab disk light configuration (normalize not supported in spawn_light)
        isaac_light_cfg = sim_utils.DiskLightCfg(**isaac_light_cfg_params)

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=light_cfg.rot,
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added disk light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius} at {light_cfg.pos}"
        )

    # def _load_ground(self) -> None:
    #     import isaaclab.sim as sim_utils
    #     cfg_ground = sim_utils.GroundPlaneCfg(
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    #         color=(1.0,1.0,1.0),
    #     )
    #     cfg_ground.func("/World/ground", cfg_ground)
    # import isaacsim.core.experimental.utils.prim as prim_utils
    # import omni
    # from pxr import Sdf, UsdShade
    # ground_prim = prim_utils.get_prim_at_path("/World/ground")
    # material = UsdShade.MaterialBindingAPI(ground_prim).GetDirectBinding().GetMaterial()
    # shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
    # # Correspond to Shader -> Inputs -> UV -> Texture Tiling (in Isaac Sim 4.2.0)
    # shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set((10,10))

    def _get_pose(
        self, obj_name: str, obj_subpath: str | None = None, env_ids: list[int] | None = None
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if obj_name in self.scene.rigid_objects:
            obj_inst = self.scene.rigid_objects[obj_name]
        elif obj_name in self.scene.articulations:
            obj_inst = self.scene.articulations[obj_name]
        else:
            raise ValueError(f"Object {obj_name} not found")

        if obj_subpath is None:
            pos = obj_inst.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
            rot = obj_inst.data.root_quat_w[env_ids]
        else:
            log.error(f"Subpath {obj_subpath} is not supported in IsaacsimHandler.get_pose")

        assert pos.shape == (len(env_ids), 3)
        assert rot.shape == (len(env_ids), 4)
        return pos, rot

    @property
    def device(self) -> torch.device:
        return self._device

    def _set_object_pose(
        self,
        object: BaseObjCfg,
        position: torch.Tensor,  # (num_envs, 3)
        rotation: torch.Tensor,  # (num_envs, 4)
        env_ids: list[int] | None = None,
    ) -> None:
        """
        Set the pose of an object, set the velocity to zero
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        assert position.shape == (len(env_ids), 3)
        assert rotation.shape == (len(env_ids), 4)

        if isinstance(object, BaseArticulationObjCfg):
            obj_inst = self.scene.articulations[object.name]
        elif isinstance(object, BaseRigidObjCfg):
            obj_inst = self.scene.rigid_objects[object.name]
        else:
            raise ValueError(f"Invalid object type: {type(object)}")

        pose = torch.concat(
            [
                position.to(self.device, dtype=torch.float32) + self.scene.env_origins[env_ids],
                rotation.to(self.device, dtype=torch.float32),
            ],
            dim=-1,
        )
        obj_inst.write_root_pose_to_sim(pose, env_ids=torch.tensor(env_ids, device=self.device))
        obj_inst.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device, dtype=torch.float32),
            env_ids=torch.tensor(env_ids, device=self.device),
        )  # ! critical
        obj_inst.write_data_to_sim()

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = deepcopy(self.scene.articulations[obj_name].joint_names)
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def _set_object_joint_pos(
        self,
        object: BaseObjCfg,
        joint_pos: torch.Tensor,  # (num_envs, num_joints)
        env_ids: list[int] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        assert joint_pos.shape[0] == len(env_ids)
        pos = joint_pos.to(self.device)
        vel = torch.zeros_like(pos)
        obj_inst = self.scene.articulations[object.name]
        obj_inst.write_joint_state_to_sim(pos, vel, env_ids=torch.tensor(env_ids, device=self.device))
        obj_inst.write_data_to_sim()

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            body_names = deepcopy(self.scene.articulations[obj_name].body_names)
            if sort:
                body_names.sort()
            return body_names
        else:
            return []

    def _add_pinhole_camera(self, camera: PinholeCameraCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

        data_type_map = {
            "rgb": "rgb",
            "depth": "depth",
            "instance_seg": "instance_segmentation_fast",
            "instance_id_seg": "instance_id_segmentation_fast",
        }
        if camera.mount_to is None:
            prim_path = f"/World/envs/env_.*/{camera.name}"
            # Use default offset, will be set by set_world_poses_from_view later
            offset = TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world")
        else:
            prim_path = f"/World/envs/env_.*/{camera.mount_to}/{camera.mount_link}/{camera.name}"
            offset = TiledCameraCfg.OffsetCfg(pos=camera.mount_pos, rot=camera.mount_quat, convention="world")

        camera_inst = TiledCamera(
            TiledCameraCfg(
                prim_path=prim_path,
                offset=offset,
                data_types=[data_type_map[dt] for dt in camera.data_types],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=camera.focal_length,
                    focus_distance=camera.focus_distance,
                    horizontal_aperture=camera.horizontal_aperture,
                    clipping_range=camera.clipping_range,
                ),
                width=camera.width,
                height=camera.height,
                colorize_instance_segmentation=False,
                colorize_instance_id_segmentation=False,
            )
        )
        self.scene.sensors[camera.name] = camera_inst
        log.debug(f"Added camera {camera.name} to scene with prim_path: {prim_path}")

    def refresh_render(self) -> None:
        for sensor in self.scene.sensors.values():
            sensor.update(dt=0)
        self.sim.render()

    def update_scene_from_scenario(self, new_scenario_cfg: ScenarioCfg) -> None:
        """
        Update the current scene to match a new scenario configuration.
        This allows dynamic addition/removal of objects, lights, and cameras.
        """
        log.debug("Updating scene from new scenario configuration")

        # 1. Update robots (check if any robot changes, though uncommon)
        self._update_robots_from_config(new_scenario_cfg.robots)

        # 2. Update objects
        self._update_objects_from_config(new_scenario_cfg.objects)

        # 3. Update lights
        self._update_lights_from_config(getattr(new_scenario_cfg, "lights", []))

        # 4. Update cameras
        self._update_cameras_from_config(new_scenario_cfg.cameras)

        # 5. Apply material assignments if they exist
        if hasattr(new_scenario_cfg, "material_assignments") and new_scenario_cfg.material_assignments:
            self.apply_materials(new_scenario_cfg.material_assignments)
            log.debug("Applied material assignments to the scene")

        # 6. Update scenario_cfg reference
        self.scenario_cfg = new_scenario_cfg

        # 6. Force scene update
        self._force_scene_update()

        # Verify scene consistency after update
        self._verify_scene_consistency()

        log.debug("Scene updated successfully from new scenario configuration")

    def _verify_scene_consistency(self) -> None:
        """Verify that the scene state is consistent after updates."""
        try:
            # Check that all objects in scenario_cfg exist in the scene
            # Note: self.objects contains only non-robot objects, robots are separate
            scenario_objects = {obj.name for obj in self.objects}
            scenario_robots = {robot.name for robot in self.robots}
            all_scenario_entities = scenario_objects | scenario_robots

            scene_rigid_objects = set(self.scene.rigid_objects.keys())
            scene_articulations = set(self.scene.articulations.keys())
            scene_objects = scene_rigid_objects | scene_articulations

            # Check objects separately from robots
            missing_objects = scenario_objects - (scene_objects - scenario_robots)
            missing_robots = scenario_robots - scene_objects
            extra_objects = scene_objects - all_scenario_entities

            if missing_objects:
                log.warning(f"Objects in scenario but missing from scene: {missing_objects}")
            if missing_robots:
                log.warning(f"Robots in scenario but missing from scene: {missing_robots}")
            if extra_objects:
                log.debug(f"Extra objects in scene not in scenario: {extra_objects}")

            log.debug(
                f"Scene consistency check: scenario_objects={len(scenario_objects)}, scenario_robots={len(scenario_robots)}, scene_total={len(scene_objects)}"
            )

        except Exception as e:
            log.warning(f"Failed to verify scene consistency: {e}")

    def _update_robots_from_config(self, new_robots: list) -> None:
        """Update robots to match new configuration. Usually robots don't change during randomization."""
        current_robot_names = {robot.name for robot in self.robots}
        new_robot_names = {robot.name for robot in new_robots}

        # For domain randomization, robots typically don't change
        # But we should update robot configurations if they change
        if current_robot_names != new_robot_names:
            log.warning("Robot configuration changed during scene update - this is unusual for domain randomization")

        # Update robot poses/joint positions if they changed
        for robot_cfg in new_robots:
            if robot_cfg.name in current_robot_names:
                self._update_existing_robot(robot_cfg)

        # Update robots list and dependent structures
        self.robots = new_robots

        # Update robot-dependent caches
        self._robot_names = {robot.name for robot in self.robots}
        self._robot_init_pos = {robot.name: robot.default_position for robot in self.robots}
        self._robot_init_quat = {robot.name: robot.default_orientation for robot in self.robots}

        # Rebuild object_dict to include updated robots
        self.object_dict = {obj.name: obj for obj in self.objects + self.robots}

        log.debug(f"Robots updated: {len(new_robots)} robots configured")

    def _update_existing_robot(self, robot_cfg: Any) -> None:
        """Update properties of an existing robot."""
        try:
            robot_name = robot_cfg.name

            # Update position if robot exists and position changed
            if robot_name in self.scene.articulations:
                # Get current pose
                current_pos, current_rot = self._get_pose(robot_name)

                # Update to new default position/orientation
                new_pos = torch.tensor(robot_cfg.default_position, device=self.device).unsqueeze(0)
                new_rot = torch.tensor(robot_cfg.default_orientation, device=self.device).unsqueeze(0)

                # Only update if position/orientation changed significantly
                pos_diff = torch.norm(new_pos - current_pos).item()
                if pos_diff > 0.01:  # 1cm threshold
                    self._set_object_pose(robot_cfg, new_pos, new_rot)
                    log.debug(f"Updated pose for robot {robot_name}")

        except Exception as e:
            log.warning(f"Failed to update existing robot {robot_cfg.name}: {e}")

    def _update_objects_from_config(self, new_objects: list) -> None:
        """Update objects to match new configuration."""
        current_object_names = {obj.name for obj in self.objects}
        new_object_names = {obj.name for obj in new_objects}

        # Remove objects that are no longer needed
        to_remove = current_object_names - new_object_names

        for obj_name in to_remove:
            self._remove_object(obj_name)
            log.debug(f"Successfully removed object: {obj_name}")

        log.debug(f"Removed {len(to_remove)} objects: {to_remove}")

        # Add new objects - these should never conflict since we just removed old ones
        to_add = new_object_names - current_object_names
        for obj_cfg in new_objects:
            if obj_cfg.name in to_add:
                self._add_object(obj_cfg)
                log.debug(f"Successfully added new object: {obj_cfg.name}")

        log.debug(f"Added {len(to_add)} new objects: {to_add}")

        self._force_scene_update()

        # Update existing objects (position, properties, etc.)
        for obj_cfg in new_objects:
            if obj_cfg.name in current_object_names:
                self._update_existing_object(obj_cfg)
                self._force_scene_update()

        # Update the objects list
        self.objects = new_objects

        # Rebuild object_dict to include updated objects
        self.object_dict = {obj.name: obj for obj in self.objects + self.robots}

        log.debug(f"Objects updated: removed {len(to_remove)}, added {len(to_add)}")

    def _update_lights_from_config(self, new_lights: list) -> None:
        """Update lights to match new configuration."""
        # Use the same precise update logic as objects
        current_lights = getattr(self, "lights", [])

        # Ensure all lights have names for consistent tracking (only if they don't have names already)
        for i, light in enumerate(current_lights):
            if not hasattr(light, "name") or not light.name:
                light.name = f"baseline_light_{i}"

        # For new lights, only assign names if they don't have them
        # Don't overwrite existing names as they might come from previous randomizations
        for i, light in enumerate(new_lights):
            if not hasattr(light, "name") or not light.name:
                light.name = f"light_{i}"

        current_light_names = {light.name for light in current_lights}
        new_light_names = {light.name for light in new_lights}

        # Remove lights that are no longer needed
        to_remove = current_light_names - new_light_names
        for light_name in to_remove:
            self._remove_light(light_name)
            log.debug(f"Successfully removed light: {light_name}")

        log.debug(f"Removed {len(to_remove)} lights: {to_remove}")

        # Add new lights - these should never conflict since we just removed old ones
        to_add = new_light_names - current_light_names
        for light_cfg in new_lights:
            if light_cfg.name in to_add:
                # Find the index for this light in new_lights
                light_index = next(i for i, light in enumerate(new_lights) if light.name == light_cfg.name)
                self._add_light(light_cfg, light_index=light_index)
                log.debug(f"Successfully added new light: {light_cfg.name}")

        log.debug(f"Added {len(to_add)} new lights: {to_add}")

        self._force_scene_update()

        # Update existing lights (properties, intensity, etc.)
        for light_cfg in new_lights:
            if light_cfg.name in current_light_names:
                self._update_existing_light(light_cfg)

        # Update the lights list in both scenario_cfg and self
        if hasattr(self.scenario_cfg, "lights"):
            self.scenario_cfg.lights = new_lights
        else:
            self.scenario_cfg.lights = new_lights

        # Update self.lights as well (inherited from base class)
        self.lights = new_lights

        log.debug(f"Lights updated: {len(new_lights)} lights configured")

    def _update_cameras_from_config(self, new_cameras: list) -> None:
        """Update cameras to match new configuration."""
        # For cameras, we mainly update poses since they're usually not added/removed dynamically
        for camera_cfg in new_cameras:
            if camera_cfg.name in self.scene.sensors:
                # Update existing camera pose
                self._update_camera_pose_for_camera(camera_cfg)

        # Update the cameras list
        self.cameras = new_cameras
        log.debug(f"Cameras updated: {len(new_cameras)} cameras configured")

    def apply_materials(self, material_assignments: dict) -> None:
        """Apply material assignments to objects."""
        return self.material_helper.apply_materials(material_assignments)

    def _remove_object(self, obj_name: str) -> None:
        """Remove an object from the scene."""
        try:
            # First, try to remove the USD prim from the stage
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage:
                # Try to find and remove the prim at the expected path
                for env_idx in range(self.num_envs):
                    prim_path = f"/World/envs/env_{env_idx}/{obj_name}"
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        stage.RemovePrim(prim_path)
                        log.debug(f"Removed USD prim: {prim_path}")

            # Then remove from Isaac Lab data structures
            if obj_name in self.scene.rigid_objects:
                del self.scene.rigid_objects[obj_name]
                log.debug(f"Removed rigid object: {obj_name}")
            elif obj_name in self.scene.articulations:
                del self.scene.articulations[obj_name]
                log.debug(f"Removed articulation: {obj_name}")
            else:
                log.warning(f"Object {obj_name} not found in scene for removal")
        except Exception as e:
            log.warning(f"Failed to remove object {obj_name}: {e}")

    def _remove_light(self, light_name: str) -> None:
        """Remove a light from the scene."""
        try:
            # Remove the USD prim from the stage
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage:
                # Try to find and remove the prim at the expected path
                light_path = f"/World/{light_name}"
                prim = stage.GetPrimAtPath(light_path)
                if prim.IsValid():
                    stage.RemovePrim(light_path)
                    log.debug(f"Removed USD light prim: {light_path}")
                else:
                    log.debug(f"Light prim not found at: {light_path}")

        except Exception as e:
            log.warning(f"Failed to remove light {light_name}: {e}")

    def _update_existing_object(self, obj_cfg: Any) -> None:
        """Update properties of an existing object."""
        try:
            obj_name = obj_cfg.name

            # Update position if object exists
            if obj_name in self.scene.rigid_objects or obj_name in self.scene.articulations:
                # Get current pose
                current_pos, current_rot = self._get_pose(obj_name)

                # Update to new default position/orientation
                new_pos = torch.tensor(obj_cfg.default_position, device=self.device).unsqueeze(0)
                new_rot = torch.tensor(obj_cfg.default_orientation, device=self.device).unsqueeze(0)

                # Only update if position/orientation changed significantly
                pos_diff = torch.norm(new_pos - current_pos).item()
                if pos_diff > 0.01:  # 1cm threshold
                    self._set_object_pose(obj_cfg, new_pos, new_rot)
                    log.debug(f"Updated pose for object {obj_name}")

                # Update other properties like color, material if changed
                self._update_object_properties(obj_cfg)

        except Exception as e:
            log.warning(f"Failed to update existing object {obj_cfg.name}: {e}")

    def _update_existing_light(self, light_cfg: Any) -> None:
        """Update properties of an existing light."""
        try:
            light_name = light_cfg.name

            # Update light properties like intensity, color, position
            light_path = f"/World/{light_name}"
            import omni.usd

            stage = omni.usd.get_context().get_stage()

            if stage:
                prim = stage.GetPrimAtPath(light_path)
                if prim.IsValid():
                    # Update light properties based on type
                    self._update_light_properties(light_cfg, prim)
                    log.debug(f"Updated properties for light {light_name}")
                else:
                    log.warning(f"Light prim not found at {light_path} for update")

        except Exception as e:
            log.warning(f"Failed to update existing light {light_cfg.name}: {e}")

    def _update_light_properties(self, light_cfg: Any, prim: Any) -> None:
        """Update visual/physical properties of a light."""
        try:
            from pxr import UsdLux

            # Get the light API based on prim type
            light_api = UsdLux.LightAPI(prim)
            if not light_api:
                log.warning(f"Could not get light API for prim {prim.GetPath()}")
                return

            # Update common light properties
            if hasattr(light_cfg, "intensity"):
                light_api.CreateIntensityAttr(light_cfg.intensity)

            if hasattr(light_cfg, "color"):
                light_api.CreateColorAttr(light_cfg.color)

            if hasattr(light_cfg, "exposure"):
                light_api.CreateExposureAttr(light_cfg.exposure)

            if hasattr(light_cfg, "normalize"):
                try:
                    light_api.CreateNormalizeAttr(bool(light_cfg.normalize))
                except Exception:
                    # Try alternative method with USD attribute
                    try:
                        from pxr import Sdf

                        prim.CreateAttribute("inputs:normalize", Sdf.ValueTypeNames.Bool).Set(bool(light_cfg.normalize))
                    except Exception:
                        # Normalize attribute not supported for this light type, skip silently
                        pass

            # Update color temperature properties
            if hasattr(light_cfg, "enable_color_temperature"):
                light_api.CreateEnableColorTemperatureAttr(bool(light_cfg.enable_color_temperature))
                if light_cfg.enable_color_temperature and hasattr(light_cfg, "color_temperature"):
                    light_api.CreateColorTemperatureAttr(light_cfg.color_temperature)

            # Update type-specific properties
            prim_type = prim.GetTypeName()

            if prim_type == "DistantLight":
                distant_light = UsdLux.DistantLight(prim)
                if hasattr(light_cfg, "angle"):
                    distant_light.CreateAngleAttr(light_cfg.angle)

            elif prim_type == "SphereLight":
                sphere_light = UsdLux.SphereLight(prim)
                if hasattr(light_cfg, "radius"):
                    sphere_light.CreateRadiusAttr(light_cfg.radius)
                if hasattr(light_cfg, "treat_as_point"):
                    sphere_light.CreateTreatAsPointAttr(bool(light_cfg.treat_as_point))

            elif prim_type == "DiskLight":
                disk_light = UsdLux.DiskLight(prim)
                if hasattr(light_cfg, "radius"):
                    disk_light.CreateRadiusAttr(light_cfg.radius)

            elif prim_type == "CylinderLight":
                cylinder_light = UsdLux.CylinderLight(prim)
                if hasattr(light_cfg, "radius"):
                    cylinder_light.CreateRadiusAttr(light_cfg.radius)
                if hasattr(light_cfg, "length"):
                    cylinder_light.CreateLengthAttr(light_cfg.length)
                if hasattr(light_cfg, "treat_as_line"):
                    cylinder_light.CreateTreatAsLineAttr(bool(light_cfg.treat_as_line))

            elif prim_type == "DomeLight":
                dome_light = UsdLux.DomeLight(prim)
                if hasattr(light_cfg, "texture_file") and light_cfg.texture_file:
                    dome_light.CreateTextureFileAttr(light_cfg.texture_file)
                if hasattr(light_cfg, "visible_in_primary_ray"):
                    # This is a special case that doesn't go through inputs:
                    prim.CreateAttribute(
                        "primvars:arnold:visibility:primary",
                        bool(light_cfg.visible_in_primary_ray),
                        variability=Sdf.VariabilityVarying,
                    )

            # Update position for positional lights
            if hasattr(light_cfg, "pos"):
                from pxr import Gf, UsdGeom

                xform = UsdGeom.Xform(prim)
                if xform:
                    # Set translation
                    translate_op = xform.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(*light_cfg.pos))

            # Update rotation for lights that have rotation
            if hasattr(light_cfg, "rot"):
                from pxr import Gf, UsdGeom

                xform = UsdGeom.Xform(prim)
                if xform:
                    # Set rotation (quaternion to rotation)
                    quat = Gf.Quatd(*light_cfg.rot)  # w, x, y, z
                    rotation = quat.GetNormalized().GetInverse().GetMatrix().ExtractRotation()
                    rotate_op = xform.AddRotateXYZOp()
                    rotate_op.Set(rotation.Decompose()[0])  # Extract Euler angles

            log.debug(f"Updated light properties for {light_cfg.name}")

        except Exception as e:
            log.warning(f"Failed to update light properties for {light_cfg.name}: {e}")

    def _update_object_properties(self, obj_cfg: Any) -> None:
        """Update visual and physical properties of an existing object."""
        try:
            obj_name = obj_cfg.name

            # Check if object exists in the scene
            if obj_name not in self.scene.rigid_objects and obj_name not in self.scene.articulations:
                log.debug(f"Object {obj_name} not found in scene for property update")
                return

            # Update visual properties (color) via USD prim attributes
            self._update_object_visual_properties(obj_cfg)

            # Update physical properties (mass, size)
            self._update_object_physical_properties(obj_cfg)

            log.debug(f"Updated properties for object {obj_name}")

        except Exception as e:
            log.warning(f"Failed to update object properties for {obj_cfg.name}: {e}")

    def _update_object_visual_properties(self, obj_cfg: Any) -> None:
        """Update visual properties like color for an existing object."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return

            # Update color if the object has color attribute
            if not (hasattr(obj_cfg, "color") and obj_cfg.color):
                return

            # Find the object's prim in all environments
            for env_idx in range(self.num_envs):
                prim_path = f"/World/envs/env_{env_idx}/{obj_cfg.name}"
                root_prim = stage.GetPrimAtPath(prim_path)

                if not root_prim.IsValid():
                    continue

                # Find all mesh prims under this object (including nested ones)
                mesh_prims = []

                # Check if the root prim itself is a mesh
                if root_prim.IsA(UsdGeom.Mesh):
                    mesh_prims.append(root_prim)

                # Find all mesh descendants
                for descendant_prim in Usd.PrimRange(root_prim):
                    if descendant_prim.IsA(UsdGeom.Mesh):
                        mesh_prims.append(descendant_prim)

                # Update color for all found mesh prims
                for mesh_prim in mesh_prims:
                    self._update_prim_color(mesh_prim, obj_cfg.color)
                    log.debug(f"Updated color for mesh prim {mesh_prim.GetPath()}")

                if not mesh_prims:
                    log.debug(f"No mesh prims found for object {obj_cfg.name} at {prim_path}")

        except Exception as e:
            log.debug(f"Failed to update visual properties for {obj_cfg.name}: {e}")

    def _update_prim_color(self, prim: Any, color: list) -> None:
        """Update the color of a USD prim using displayColor primvar."""
        try:
            from pxr import Gf, UsdGeom

            # Try to update color via displayColor primvar (works for most primitives)
            if prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)

                # Set displayColor primvar
                color_attr = gprim.GetDisplayColorAttr()
                if not color_attr:
                    color_attr = gprim.CreateDisplayColorAttr()

                # Convert to Gf.Vec3f
                color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
                color_attr.Set([color_vec])

                log.debug(f"Updated displayColor for prim {prim.GetPath()}")
                return

            # Fallback: try to find and update material
            self._update_prim_material_color(prim, color)

        except Exception as e:
            log.debug(f"Failed to update color for prim {prim.GetPath()}: {e}")

    def _update_prim_material_color(self, prim: Any, color: list) -> None:
        """Update color by modifying the bound material."""
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdShade

            # Get bound material
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                return

            mat_binding_api = UsdShade.MaterialBindingAPI(prim)
            bound_material = mat_binding_api.GetDirectBinding().GetMaterial()

            if not bound_material:
                return

            # Find the shader
            shader_prim = omni.usd.get_shader_from_material(bound_material, get_prim=True)
            if not shader_prim:
                return

            shader = UsdShade.Shader(shader_prim)
            if not shader:
                return

            # Update diffuseColor input
            diffuse_input = shader.GetInput("diffuseColor")
            if not diffuse_input:
                diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)

            color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            diffuse_input.Set(color_vec)

            log.debug(f"Updated material color for prim {prim.GetPath()}")

        except Exception as e:
            log.debug(f"Failed to update material color for prim {prim.GetPath()}: {e}")

    def _update_object_physical_properties(self, obj_cfg: Any) -> None:
        """Update physical properties like mass and size for an existing object."""
        try:
            obj_name = obj_cfg.name

            # Get the Isaac Lab object instance
            obj_instance = None
            if obj_name in self.scene.rigid_objects:
                obj_instance = self.scene.rigid_objects[obj_name]
            elif obj_name in self.scene.articulations:
                obj_instance = self.scene.articulations[obj_name]
            else:
                return

            # Update mass properties if object has mass attribute
            if hasattr(obj_cfg, "mass"):
                self._update_object_mass(obj_instance, obj_cfg.mass)

            # Update size/scale properties via USD
            self._update_object_size_properties(obj_cfg)

        except Exception as e:
            log.debug(f"Failed to update physical properties for {obj_cfg.name}: {e}")

    def _update_object_mass(self, obj_instance: Any, new_mass: float) -> None:
        """Update mass properties of an object."""
        try:
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return

            # Get the object's prim path
            prim_path = obj_instance.cfg.prim_path

            for env_idx in range(self.num_envs):
                # Replace pattern with specific environment
                env_prim_path = prim_path.replace("env_.*", f"env_{env_idx}")
                prim = stage.GetPrimAtPath(env_prim_path)

                if not prim.IsValid():
                    continue

                # Update mass via PhysX mass API
                if prim.HasAPI(UsdPhysics.MassAPI):
                    mass_api = UsdPhysics.MassAPI(prim)
                    mass_api.GetMassAttr().Set(float(new_mass))
                    log.debug(f"Updated mass to {new_mass} for {env_prim_path}")

        except Exception as e:
            log.debug(f"Failed to update mass: {e}")

    def _update_object_size_properties(self, obj_cfg: Any) -> None:
        """Update size/scale properties via USD attributes."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return

            for env_idx in range(self.num_envs):
                prim_path = f"/World/envs/env_{env_idx}/{obj_cfg.name}"
                root_prim = stage.GetPrimAtPath(prim_path)

                if not root_prim.IsValid():
                    continue

                # For primitives, find the actual mesh prim
                target_prim = root_prim

                # If root is not a mesh/sphere/cylinder/cube, find the mesh child
                if not (
                    root_prim.IsA(UsdGeom.Mesh)
                    or root_prim.IsA(UsdGeom.Sphere)
                    or root_prim.IsA(UsdGeom.Cylinder)
                    or root_prim.IsA(UsdGeom.Cube)
                ):
                    # Find first mesh/sphere/cylinder/cube descendant
                    for descendant_prim in Usd.PrimRange(root_prim):
                        if (
                            descendant_prim.IsA(UsdGeom.Mesh)
                            or descendant_prim.IsA(UsdGeom.Sphere)
                            or descendant_prim.IsA(UsdGeom.Cylinder)
                            or descendant_prim.IsA(UsdGeom.Cube)
                        ):
                            target_prim = descendant_prim
                            break

                # Update size for primitive cubes
                if hasattr(obj_cfg, "size") and hasattr(obj_cfg, "__class__") and "Cube" in obj_cfg.__class__.__name__:
                    self._update_cube_size(target_prim, obj_cfg.size)

                # Update radius for spheres
                elif (
                    hasattr(obj_cfg, "radius")
                    and hasattr(obj_cfg, "__class__")
                    and "Sphere" in obj_cfg.__class__.__name__
                ):
                    self._update_sphere_radius(target_prim, obj_cfg.radius)

                # Update radius and height for cylinders
                elif (
                    hasattr(obj_cfg, "radius")
                    and hasattr(obj_cfg, "height")
                    and hasattr(obj_cfg, "__class__")
                    and "Cylinder" in obj_cfg.__class__.__name__
                ):
                    self._update_cylinder_size(target_prim, obj_cfg.radius, obj_cfg.height)

                # Update scale for file-based objects
                elif hasattr(obj_cfg, "scale"):
                    self._update_object_scale(root_prim, obj_cfg.scale)  # Use root prim for scale

        except Exception as e:
            log.debug(f"Failed to update size properties for {obj_cfg.name}: {e}")

    def _update_cube_size(self, prim: Any, size: list) -> None:
        """Update cube size via USD attributes."""
        try:
            # Isaac Lab creates Mesh, not USD Cube
            # We need to calculate scale factors relative to the original size

            # Get the original size from the prim attributes or calculate from bounding box
            original_size = self._get_original_mesh_size(prim)
            if original_size is None:
                log.debug(f"Could not determine original size for {prim.GetPath()}, skipping size update")
                return

            # Calculate scale factors: new_size / original_size
            scale_factors = [
                float(size[0]) / original_size[0],
                float(size[1]) / original_size[1],
                float(size[2]) / original_size[2],
            ]

            log.debug(f"Updating cube from original size {original_size} to new size {size} with scale {scale_factors}")
            self._update_object_scale(prim, scale_factors)

        except Exception as e:
            log.debug(f"Failed to update cube size: {e}")

    def _get_original_mesh_size(self, prim: Any) -> list | None:
        """Get the original size of a mesh by analyzing its bounding box."""
        try:
            from pxr import Gf, UsdGeom

            # Get mesh points if it's a mesh
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                if points_attr:
                    points = points_attr.Get()
                    if points:
                        # Calculate bounding box
                        min_point = Gf.Vec3f(float("inf"))
                        max_point = Gf.Vec3f(float("-inf"))

                        for point in points:
                            for i in range(3):
                                min_point[i] = min(min_point[i], point[i])
                                max_point[i] = max(max_point[i], point[i])

                        # Calculate size (max - min)
                        size = [
                            float(max_point[0] - min_point[0]),
                            float(max_point[1] - min_point[1]),
                            float(max_point[2] - min_point[2]),
                        ]

                        log.debug(f"Calculated original mesh size: {size} for {prim.GetPath()}")
                        return size

            # Fallback: return None if we can't determine size
            return None

        except Exception as e:
            log.debug(f"Failed to get original mesh size: {e}")
            return None

    def _update_sphere_radius(self, prim: Any, radius: float) -> None:
        """Update sphere radius via USD attributes."""
        try:
            from pxr import UsdGeom

            if prim.IsA(UsdGeom.Sphere):
                sphere = UsdGeom.Sphere(prim)
                sphere.GetRadiusAttr().Set(float(radius))
                log.debug(f"Updated sphere radius to {radius} for {prim.GetPath()}")

        except Exception as e:
            log.debug(f"Failed to update sphere radius: {e}")

    def _update_cylinder_size(self, prim: Any, radius: float, height: float) -> None:
        """Update cylinder size via USD attributes."""
        try:
            from pxr import UsdGeom

            if prim.IsA(UsdGeom.Cylinder):
                cylinder = UsdGeom.Cylinder(prim)
                cylinder.GetRadiusAttr().Set(float(radius))
                cylinder.GetHeightAttr().Set(float(height))
                log.debug(f"Updated cylinder size to radius={radius}, height={height} for {prim.GetPath()}")

        except Exception as e:
            log.debug(f"Failed to update cylinder size: {e}")

    def _update_object_scale(self, prim: Any, scale: tuple | float) -> None:
        """Update object scale via USD transform."""
        try:
            from pxr import Gf, UsdGeom

            if isinstance(scale, (list, tuple)):
                scale_vec = Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2]))
            else:
                scale_vec = Gf.Vec3d(float(scale), float(scale), float(scale))

            # Apply scale via Xform
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)

                # Check if a scale op already exists and reuse it
                existing_scale_ops = [
                    op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale
                ]

                if existing_scale_ops:
                    # Use the existing scale op to maintain precision consistency
                    scale_op = existing_scale_ops[0]
                    scale_op.Set(scale_vec)
                    log.debug(f"Updated existing scale to {scale_vec} for {prim.GetPath()}")
                else:
                    # Add new scale op only if none exists
                    scale_op = xformable.AddScaleOp()
                    scale_op.Set(scale_vec)
                    log.debug(f"Added new scale {scale_vec} for {prim.GetPath()}")

        except Exception as e:
            log.debug(f"Failed to update object scale: {e}")

    def _clear_all_lights(self) -> None:
        """Remove all dynamic lights from the scene."""
        try:
            # Isaac Lab doesn't have a direct way to remove lights, but we can try
            # to delete the prims if they exist
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage:
                # Try to remove all lights that were dynamically added
                light_types = ["DistantLight", "SphereLight", "CylinderLight", "DomeLight", "DiskLight"]
                for light_type in light_types:
                    # Look for lights with our naming pattern
                    for i in range(10):  # reasonable upper bound
                        light_path = f"/World/{light_type}_{i}"
                        prim = stage.GetPrimAtPath(light_path)
                        if prim.IsValid():
                            stage.RemovePrim(light_path)
                            log.debug(f"Removed light: {light_path}")

            log.debug("Cleared existing dynamic lights")
        except Exception as e:
            log.debug(f"Could not clear lights dynamically, using placeholder: {e}")
            # Fallback: just clear the lights list
            if hasattr(self.scenario_cfg, "lights"):
                self.scenario_cfg.lights = []
            self.lights = []

    def _add_light(self, light_cfg: Any, light_index: int | None = None) -> None:
        """Add a light to the scene."""
        # Assert that this light should not already exist - if it does, there's a logic bug
        if hasattr(light_cfg, "name"):
            light_path = f"/World/{light_cfg.name}"
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage and stage.GetPrimAtPath(light_path).IsValid():
                raise RuntimeError(
                    f"Logic error: Attempting to add light '{light_cfg.name}' that already exists in scene. "
                    "This indicates a bug in the light update logic."
                )

        try:
            # Use existing light addition logic but for dynamic addition
            from metasim.scenario.lights import (
                CylinderLightCfg,
                DiskLightCfg,
                DistantLightCfg,
                DomeLightCfg,
                SphereLightCfg,
            )

            # Use provided index or generate one based on current lights
            if light_index is None:
                light_index = len(getattr(self.scenario_cfg, "lights", []))

            if isinstance(light_cfg, DistantLightCfg):
                self._add_distant_light(light_cfg, light_index)
            elif isinstance(light_cfg, CylinderLightCfg):
                self._add_cylinder_light(light_cfg, light_index)
            elif isinstance(light_cfg, DomeLightCfg):
                self._add_dome_light(light_cfg, light_index)
            elif isinstance(light_cfg, SphereLightCfg):
                self._add_sphere_light(light_cfg, light_index)
            elif isinstance(light_cfg, DiskLightCfg):
                self._add_disk_light(light_cfg, light_index)
            else:
                log.warning(f"Unknown light type for dynamic addition: {type(light_cfg)}")

        except Exception as e:
            log.warning(f"Failed to add light dynamically: {e}")

    def _update_camera_pose_for_camera(self, camera_cfg: Any) -> None:
        """Update pose for a specific camera."""
        try:
            if isinstance(camera_cfg, PinholeCameraCfg):
                if camera_cfg.mount_to is None and camera_cfg.name in self.scene.sensors:
                    camera_inst = self.scene.sensors[camera_cfg.name]
                    position_tensor = torch.tensor(camera_cfg.pos, device=self.device, dtype=torch.float32).unsqueeze(0)
                    position_tensor = position_tensor.repeat(self.num_envs, 1)
                    camera_lookat_tensor = torch.tensor(
                        camera_cfg.look_at, device=self.device, dtype=torch.float32
                    ).unsqueeze(0)
                    camera_lookat_tensor = camera_lookat_tensor.repeat(self.num_envs, 1)
                    camera_inst.set_world_poses_from_view(position_tensor, camera_lookat_tensor)
                    log.debug(
                        f"Updated camera {camera_cfg.name} pose: pos={camera_cfg.pos}, look_at={camera_cfg.look_at}"
                    )
        except Exception as e:
            log.warning(f"Failed to update camera {camera_cfg.name}: {e}")

    def _force_scene_update(self) -> None:
        """Force the scene to update after dynamic changes."""
        try:
            # Update camera poses
            self._update_camera_pose()

            self.sim.reset()
            indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            self.scene.reset(indices)
            self.scene.write_data_to_sim()

            # Force a simulation step to apply changes
            self.sim.step(render=False)
            self.scene.update(dt=self.dt)

            # Update sensors
            for sensor in self.scene.sensors.values():
                sensor.update(dt=0)

            log.debug("Forced scene update completed")
        except Exception as e:
            log.warning(f"Failed to force scene update: {e}")
