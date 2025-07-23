from src.utils.simulator.simulator import Simulator
from pathlib import Path
import sys
sys.path.append("/home/charliecheng/caiyi/RoboVerse")

from metasim.sim.isaacgym import IsaacgymHandler
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.randomization import RandomizationCfg
from metasim.utils.setup_util import get_robot

def convert_config_to_scenario(config, num_envs=512, headless=True):
    # 自动推理 robot 名称（优先找带 hand 的）
    robot_name = None
    for key in config.get("asset", {}).keys():
        if "hand" in key.lower():
            robot_name = key
            break
    if robot_name is None:
        raise ValueError("Could not infer robot name from config['asset']")

    # 让 metasim 自动解析 robot config
    robot_cfg = robot_name

    # 构造 object（只取一个即可；你可以扩展成多个）
    objects = []
    if "object" in config.get("asset", {}):
        obj_info = config["asset"]["object"]
        obj = RigidObjCfg(
            name="object",
            urdf_path=obj_info.get("asset_path", ""),
            fix_base_link=obj_info.get("asset_config", {}).get("asset_options", {}).get("fix_base_link", True),
            enabled_gravity=not obj_info.get("asset_config", {}).get("asset_options", {}).get("disable_gravity", False),
        )
        objects.append(obj)

    # 构造 SimParam 和 Control
    sim_params = SimParamCfg(dt=1.0 / config.get("env_hz", 60))
    control = ControlCfg()
    random_cfg = RandomizationCfg()

    # 构造 scenario
    scenario = ScenarioCfg(
        robots=[robot_cfg],
        objects=objects,
        sim="isaacgym",
        headless=headless,
        num_envs=num_envs,
        sim_params=sim_params,
        control=control,
        env_spacing=config.get("env_spacing", 1.0),
        random=random_cfg,
        try_add_table=False,  # 防止与已有物体冲突
    )
    return scenario



class IsaacGymSimulator(Simulator):
    def __init__(self, config, num_envs, device_id, headless):
        super().__init__(config, num_envs, device_id, headless)
        print(config)

        # convert to ScenarioCfg
        scenario = convert_config_to_scenario(config, num_envs=num_envs, headless=headless)

        # create handler
        self.handler = IsaacgymHandler(scenario)
        self.handler.launch()

    def register_asset(self, *args, **kwargs):
        pass  # not needed, assets already registered in scenario

    def create_env(self):
        pass  # not needed, environments created in handler.launch()

    def create_actor(self, *args, **kwargs):
        pass  # not needed

    def prepare_sim(self):
        self.handler.prepare_sim()

    def get_actor_states(self, actor_name):
        return self.handler._get_states(actor_name)

    def set_actor_states(self, actor_name, actor_states):
        self.handler._set_states(actor_name, actor_states)

    def set_actor_actions(self, actor_name, actor_actions):
        self.handler._set_actions(actor_name, actor_actions)

    def disable_gravity(self, actor_name):
        self.handler.disable_gravity(actor_name)

    def enable_gravity(self, actor_name):
        self.handler.enable_gravity(actor_name)

    def close(self):
        self.handler.close()