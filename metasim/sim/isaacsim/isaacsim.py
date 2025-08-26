# This naively suites for isaaclab 2.2.0 and isaacsim 5.0.0
from __future__ import annotations

import argparse
import os
from copy import deepcopy

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

        self.scenario_cfg: ScenarioCfg = scenario_cfg
        self.physics_dt = self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 0.01
        self._step_counter = 0
        self._is_closed = False
        self._render_interval = self.scenario.render_interval

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
            dt=self.physics_dt,
            device=args.device,
            render_interval=self.scenario.decimation,  # TODO divide into render interval and control decimation
            physx=PhysxCfg(
                bounce_threshold_velocity=self.scenario.sim_params.bounce_threshold_velocity,
                solver_type=self.scenario.sim_params.solver_type,
                max_position_iteration_count=self.scenario.sim_params.num_position_iterations,
                max_velocity_iteration_count=self.scenario.sim_params.num_velocity_iterations,
                # TODO: uncomment this
                # friction_correlation_distance=self.scenario.sim_params.friction_correlation_distance,
            ),
        )

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
                    position_tensor = torch.tensor(camera.pos, device=self.device).unsqueeze(0)
                    position_tensor = position_tensor.repeat(self.num_envs, 1)
                    camera_lookat_tensor = torch.tensor(camera.look_at, device=self.device).unsqueeze(0)
                    camera_lookat_tensor = camera_lookat_tensor.repeat(self.num_envs, 1)
                    camera_inst.set_world_poses_from_view(position_tensor, camera_lookat_tensor)
                    # log.debug(f"Updated camera {camera.name} pose: pos={camera.pos}, look_at={camera.look_at}")
            else:
                raise ValueError(f"Unsupported camera type: {type(camera)}")

    def launch(self) -> None:
        self._init_scene()
        self._load_robots()
        self._load_sensors()
        self._load_cameras()
        self._load_terrain()
        self._load_objects()
        self._load_lights()
        # self._load_render_settings()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.sim.reset()
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.scene.reset(indices)

        # Update camera pose after scene reset to avoid being overridden
        self._update_camera_pose()

        # Force another simulation step and camera update to ensure proper initialization
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._update_camera_pose()

        # Force a render to update camera data after position is set
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()
        for sensor in self.scene.sensors.values():
            sensor.update(dt=0)

        if self.optional_queries is None:
            self.optional_queries = {}
        for query_name, query_type in self.optional_queries.items():
            query_type.bind_handler(self)

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

    def _set_states(self, states: list[DictEnvState] | TensorState, env_ids: list[int] | None = None) -> None:
        # if states is list[DictEnvState], iterate over it and set state
        if isinstance(states, list):
            if env_ids is None:
                env_ids = list(range(self.num_envs))
            states_flat = [states[i]["objects"] | states[i]["robots"] for i in range(self.num_envs)]
            for obj in self.objects + self.robots:
                if obj.name not in states_flat[0]:
                    log.warning(f"Missing {obj.name} in states, setting its velocity to zero")
                    pos, rot = self._get_pose(obj.name, env_ids=env_ids)
                    self._set_object_pose(obj, pos, rot, env_ids=env_ids)
                    continue

                if (
                    states_flat[0][obj.name].get("pos", None) is None
                    or states_flat[0][obj.name].get("rot", None) is None
                ):
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

        # if states is TensorState, reindex the tensors and set state
        elif isinstance(states, TensorState):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            elif isinstance(env_ids, list):
                env_ids = torch.tensor(env_ids, device=self.device)

            for _, obj in enumerate(self.objects):
                obj_inst = states.objects[obj.name]
                root_state = obj_inst.root_state.clone()
                root_state[:, :3] += self.scene.env_origins
                obj_inst.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
                obj_inst.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids)
                if isinstance(obj, ArticulationObjCfg):
                    joint_ids_reindex = self.get_joint_reindex(obj.name, inverse=True)
                    obj_inst.write_joint_position_to_sim(
                        states.objects[obj.name].joint_pos[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                    )
                    obj_inst.write_joint_velocity_to_sim(
                        states.objects[obj.name].joint_vel[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                    )

            for _, robot in enumerate(self.robots):
                robot_inst = self.scene.articulations[robot.name]
                root_state = states.robots[robot.name].root_state.clone()
                root_state[:, :3] += self.scene.env_origins
                robot_inst.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
                robot_inst.write_root_velocity_to_sim(
                    states.robots[robot.name].root_state[env_ids, 7:], env_ids=env_ids
                )
                joint_ids_reindex = self.get_joint_reindex(robot.name, inverse=True)
                robot_inst.write_joint_position_to_sim(
                    states.robots[robot.name].joint_pos[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                )
                robot_inst.write_joint_velocity_to_sim(
                    states.robots[robot.name].joint_vel[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                )

        else:
            raise Exception("Unsupported state type, must be DictEnvState or TensorState")

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
        extras = self.get_extra()
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

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
        # start_idx = 0
        for robot in self.robots:
            robot_inst = self.scene.articulations[robot.name]
            # actionable_joint_ids = [
            #     robot_inst.joint_names.index(jn) for jn in robot.actuators if robot.actuators[jn].fully_actuated
            # ]
            # TODO: hard code here, at pos control mode
            robot_inst.set_joint_effort_target(
                action_tensor_all,
                # joint_ids=list(range(self.scenario.task.num_actions)),  #
            )

    def _simulate(self):
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if self._step_counter % self._render_interval == 0 and is_rendering:
            self.sim.render()
        self.scene.update(dt=self.physics_dt)

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
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=robot.fix_base_link,
                    enabled_self_collisions=robot.enabled_self_collisions,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ),
            actuators={
                # jn: ImplicitActuatorCfg(
                jn: ImplicitActuatorCfg(
                    # prim_path
                    joint_names_expr=[jn],
                    # TODO fix this with different mode
                    stiffness=0.0,
                    damping=0,
                    armature=0.01,
                    friction=0.05,
                )
                for jn, actuator in robot.actuators.items()
            },
        )
        cfg.prim_path = f"/World/envs/env_.*/{robot.name}"
        cfg.spawn.usd_path = os.path.abspath(robot.usd_path)
        cfg.spawn.rigid_props.disable_gravity = not robot.enabled_gravity
        init_state = ArticulationCfg.InitialStateCfg(
            # TODO hard code here
            pos=[0.0, 0.0, 0.78],
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
        prim_path = f"/World/envs/env_.*/{obj.name}"

        ## Articulation object
        if isinstance(obj, ArticulationObjCfg):
            self.scene.articulations[obj.name] = Articulation(
                ArticulationCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(usd_path=obj.usd_path, scale=obj.scale),
                    actuators={},
                )
            )
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
        self.terrain.env_origins = self.terrain.terrain_origins

    def _load_render_settings(self) -> None:
        import carb
        import omni.replicator.core as rep

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
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{self.robots[0].name}/.*",
            history_length=3,
            update_period=self.physics_dt * self.scenario.decimation,
            track_air_time=False,
        )
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

    def load_contact_sensor_idx(self) -> None:
        body_names = [
            "pelvis",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
            "waist_yaw_link",
            "waist_roll_link",
            "torso_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",]
        self.body_ids, self.body_names = self.scene.articulations[self.robots[0].name].find_bodies(
            body_names, preserve_order=True
        )

        termination_contact_names = []
        for name in self.robots[0].terminate_contacts_links:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.find_rigid_body_indice(termination_contact_names[i])

        return self.termination_contact_indices



    def find_rigid_body_indice(self, body_name):
        '''
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        '''
        indices, names = self.scene.articulations[self.robots[0].name].find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            log.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else: # multiple bodies found
            log.warning(f"Multiple bodies found for {body_name}.")
            return indices

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

        from isaaclab.assets import AssetBaseCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
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

        light_name = f"/World/DistantLight_{light_index}"

        # Create Isaac Lab distant light configuration
        isaac_light_cfg = sim_utils.DistantLightCfg(
            intensity=light_cfg.intensity,
            angle=0.53,  # Default angle, could be made configurable
            color=light_cfg.color,
        )

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
            f"polar={light_cfg.polar}°, azimuth={light_cfg.azimuth}°"
        )

    def _add_cylinder_light(self, light_cfg, light_index: int) -> None:
        """Add a cylinder light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        light_name = f"/World/CylinderLight_{light_index}"

        # Create Isaac Lab cylinder light configuration
        isaac_light_cfg = sim_utils.CylinderLightCfg(
            intensity=light_cfg.intensity, radius=light_cfg.radius, length=light_cfg.length, color=light_cfg.color
        )

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=light_cfg.rot,
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added cylinder light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius}, length={light_cfg.length}"
        )

    def _add_dome_light(self, light_cfg, light_index: int) -> None:
        """Add a dome light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        light_name = f"/World/DomeLight_{light_index}"

        # Create Isaac Lab dome light configuration
        isaac_light_cfg = sim_utils.DomeLightCfg(
            intensity=light_cfg.intensity,
            color=light_cfg.color,
        )

        # Add texture if specified
        if light_cfg.texture_file is not None:
            isaac_light_cfg.texture_file = light_cfg.texture_file

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

        light_name = f"/World/SphereLight_{light_index}"

        # Create Isaac Lab sphere light configuration
        isaac_light_cfg = sim_utils.SphereLightCfg(
            intensity=light_cfg.intensity,
            color=light_cfg.color,
            radius=light_cfg.radius,
            normalize=light_cfg.normalize,
        )

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=(1.0, 0.0, 0.0, 0.0),
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added sphere light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius} at {light_cfg.pos}"
        )

    def _add_disk_light(self, light_cfg, light_index: int) -> None:
        """Add a disk light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        light_name = f"/World/DiskLight_{light_index}"

        # Create Isaac Lab disk light configuration
        isaac_light_cfg = sim_utils.DiskLightCfg(
            intensity=light_cfg.intensity,
            color=light_cfg.color,
            radius=light_cfg.radius,
            normalize=light_cfg.normalize,
        )

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

    def init_marker_viz(self):
        """Define markers with various different shapes."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        self._marker_viz = VisualizationMarkers(marker_cfg)
        return self._marker_viz

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
                return sorted(body_names)
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

    def _get_body_mass(
        self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> torch.Tensor:
        """
        Get the mass of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            body_name (str, optional): Name of the specific body. If None, returns mass of all bodies
            env_ids (list[int], optional): List of environment ids. If None, returns for all environments

        Returns:
            torch.Tensor: Mass values with shape (num_envs, num_bodies) or (num_envs,) if body_name is specified
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if obj_name in self.scene.articulations:
            obj_inst = self.scene.articulations[obj_name]
            masses = obj_inst.root_physx_view.get_masses()

            if body_name is not None:
                # Get specific body mass
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                return masses[env_ids, body_idx]
            else:
                # Get all body masses
                return masses[env_ids, :]
        elif obj_name in self.scene.rigid_objects:
            obj_inst = self.scene.rigid_objects[obj_name]
            masses = obj_inst.root_physx_view.get_masses()
            return masses[env_ids]
        else:
            raise ValueError(f"Object {obj_name} not found")

    def _set_body_mass(
        self,
        obj_name: str,
        mass: torch.Tensor,
        body_name: str | None = None,
        env_ids: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Set the mass of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            mass (torch.Tensor): Mass values to set
            body_name (str, optional): Name of the specific body. If None, sets mass for all bodies
            env_ids (list[int], optional): List of environment ids. If None, sets for all environments
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if obj_name in self.scene.articulations:
            obj_inst = self.scene.articulations[obj_name]
            masses = obj_inst.root_physx_view.get_masses()

            if body_name is not None:
                # Set specific body mass
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                masses[env_ids, body_idx] = mass.to(device)
            else:
                # Set all body masses
                masses[env_ids, :] = mass.to(device)

            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids, device=device))
        elif obj_name in self.scene.rigid_objects:
            obj_inst = self.scene.rigid_objects[obj_name]
            # TODO: check why obj_inst is cpu
            masses = obj_inst.root_physx_view.get_masses()
            masses[env_ids] = mass.to(masses.device)
            obj_inst.root_physx_view.set_masses(masses, torch.tensor(env_ids, device=device))
        else:
            raise ValueError(f"Object {obj_name} not found")

    def get_body_friction(
        self, obj_name: str, body_name: str | None = None, env_ids: list[int] | None = None
    ) -> torch.Tensor:
        """
        Get the friction coefficient of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            body_name (str, optional): Name of the specific body. If None, returns friction of all bodies
            env_ids (list[int], optional): List of environment ids. If None, returns for all environments

        Returns:
            torch.Tensor: Friction values with shape (num_envs, num_bodies) or (num_envs,) if body_name is specified
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if obj_name in self.scene.articulations:
            obj_inst = self.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            friction = materials[..., 0]  # First component is static friction

            if body_name is not None:
                # Get specific body friction
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                return friction[env_ids, body_idx]
            else:
                # Get all body friction
                return friction[env_ids, :]
        elif obj_name in self.scene.rigid_objects:
            obj_inst = self.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            friction = materials[..., 0]  # First component is static friction
            return friction[env_ids]
        else:
            raise ValueError(f"Object {obj_name} not found")

    def _set_body_friction(
        self,
        obj_name: str,
        friction: torch.Tensor,
        body_name: str | None = None,
        env_ids: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Set the friction coefficient of a specific body or all bodies of an object.

        Args:
            obj_name (str): Name of the object/robot
            friction (torch.Tensor): Friction values to set
            body_name (str, optional): Name of the specific body. If None, sets friction for all bodies
            env_ids (list[int], optional): List of environment ids. If None, sets for all environments
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if obj_name in self.scene.articulations:
            obj_inst = self.scene.articulations[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()

            if body_name is not None:
                # Set specific body friction
                body_names = self._get_body_names(obj_name)
                if body_name not in body_names:
                    raise ValueError(f"Body {body_name} not found in object {obj_name}")
                body_idx = body_names.index(body_name)
                materials[env_ids, body_idx, 0] = friction.to(device)  # Static friction
                materials[env_ids, body_idx, 1] = friction.to(device)  # Dynamic friction
            else:
                # Set all body friction
                materials[env_ids, :, 0] = friction.to(device)  # Static friction
                materials[env_ids, :, 1] = friction.to(device)  # Dynamic friction

            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids, device=device))
        elif obj_name in self.scene.rigid_objects:
            obj_inst = self.scene.rigid_objects[obj_name]
            materials = obj_inst.root_physx_view.get_material_properties()
            materials[env_ids, 0] = friction.to(device)  # Static friction
            materials[env_ids, 1] = friction.to(device)  # Dynamic friction
            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids, device=device))
        else:
            raise ValueError(f"Object {obj_name} not found")
