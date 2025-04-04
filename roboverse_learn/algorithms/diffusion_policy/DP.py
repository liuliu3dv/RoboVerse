import os
import sys

import dill
import hydra
import torch

from roboverse_learn.algorithms.base_policy import BasePolicy
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.common.pytorch_util import (
    dict_apply,
)
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.env_runner.dp_runner import (
    DPRunner,
)
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.policy.base_image_policy import (
    BaseImagePolicy,
)
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.workspace.robotworkspace import (
    RobotWorkspace,
)


class DiffusionPolicy(BasePolicy):
    def __init__(self, checkpoint_path: str):
        self.policy, self.cfg = get_policy(checkpoint_path=checkpoint_path, output_dir=None, device="cuda:0")
        self.runner = DPRunner(n_obs_steps=self.cfg.n_obs_steps, output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def reset_obs(self):
        self.runner.reset_obs()


def get_policy(checkpoint_path, output_dir, device):
    # load checkpoint
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    # print(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy, cfg
