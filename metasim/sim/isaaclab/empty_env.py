from typing import Sequence

import torch

try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.envs import ViewerCfg
    from omni.isaac.lab.envs.direct_rl_env_cfg import DirectRLEnvCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
    from omni.isaac.lab.utils import configclass
except ModuleNotFoundError:
    from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.utils import configclass


from metasim.types import EnvState

from .utils.custom_direct_rl_env import CustomDirectRLEnv


@configclass
class EmptyEnvCfg(DirectRLEnvCfg):
    decimation: int = 3
    episode_length_s: float = 200 * (1 / 60) * decimation  # 200 steps
    observation_space: int = 0  # no use, but must be initialized
    action_space: int = 0  # no use, but must be initialized
    state_space: int = 0  # no use, but must be initialized

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    sim: SimulationCfg = SimulationCfg(
        device="cpu",  # same as IsaacLab default
        dt=1 / 60,  # same as IsaacLab default
        gravity=(0.0, 0.0, -9.81),  # same as IsaacLab default
        physx=PhysxCfg(),
        use_fabric=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=10.0,
            dynamic_friction=10.0,
            restitution=0.0,
        ),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.05, 1.69),
        lookat=(0.25, 0.05, 0.707),
    )
    # viewer: ViewerCfg = ViewerCfg(
    #     eye=(0, -1, 0.83),
    #     lookat=(0, 0, 0.83),
    # )


class EmptyEnv(CustomDirectRLEnv):
    cfg: EmptyEnvCfg
    init_states: list[EnvState]

    def __init__(self, cfg: EmptyEnvCfg, render_mode: str | None = None, **kwargs):
        self.func = {
            "_setup_scene": kwargs.get("_setup_scene"),
            "_reset_idx": kwargs.get("_reset_idx"),
            "_pre_physics_step": kwargs.get("_pre_physics_step"),
            "_apply_action": kwargs.get("_apply_action"),
            "_get_observations": kwargs.get("_get_observations"),
            "_get_rewards": kwargs.get("_get_rewards"),
            "_get_dones": kwargs.get("_get_dones"),
        }
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.func["_setup_scene"](self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.func["_pre_physics_step"](self, actions)

    def _apply_action(self) -> None:
        self.func["_apply_action"](self)

    def _get_observations(self) -> None:
        return self.func["_get_observations"](self)

    def _get_rewards(self) -> None:
        self.func["_get_rewards"](self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.func["_get_dones"](self)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        self.func["_reset_idx"](self, env_ids)
