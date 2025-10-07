#!/usr/bin/env python3
"""
Task Server - 运行真实的任务环境
支持IsaacSim等真实环境
"""

import sys
import os
import torch
import rootutils

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils.setup_util import get_handler

from remote_server import RemoteServer

rootutils.setup_root(__file__, pythonpath=True)


def create_task_env(task_name="stack_cube", num_envs=1, simulator="isaacsim"):
    """创建任务环境"""
    print(f"Creating {simulator} environment for task {task_name}...")

    task_cls = get_task_class(task_name)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))

    scenario = task_cls.scenario.update(
        simulator=simulator,
        cameras=[camera],
        num_envs=num_envs,
        headless=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    print(f"✅ Environment created: {task_name} with {num_envs} envs")
    return env


class TaskEnvWrapper:
    """包装Task环境使其符合RemoteServer接口"""

    def __init__(self, task_env):
        self.task_env = task_env
        self.handler = task_env.handler
        self.last_obs = None
        self.last_reward = None
        self.last_success = None
        self.last_time_out = None

    def step(self, action):
        """执行一步"""
        # 确保action是list格式
        if not isinstance(action, list):
            action = [action]

        # 执行step
        obs, reward, success, time_out, extras = self.task_env.step(action)

        # 保存结果
        self.last_obs = obs
        self.last_reward = reward
        self.last_success = success
        self.last_time_out = time_out

        # 转换为标准格式
        # 注意：这里返回的是tensor，客户端需要处理
        done = (success | time_out).any()
        info = {"success": success, "time_out": time_out, "extras": extras}

        # 返回标准格式: (obs, reward, done, info)
        # reward保持tensor格式，让客户端决定如何处理
        return obs, reward, done, info

    def reset(self):
        """重置环境"""
        obs, info = self.task_env.reset()
        self.last_obs = obs
        return obs

    def get_state(self):
        """获取环境状态"""
        return self.handler.get_states()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task Server")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--task", default="stack_cube", help="Task name")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--simulator", default="isaacsim", help="Simulator type")

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Starting Task Server")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Simulator: {args.simulator}")
    print(f"Num envs: {args.num_envs}")
    print(f"Port: {args.port}")
    print()

    # 创建任务环境
    task_env = create_task_env(task_name=args.task, num_envs=args.num_envs, simulator=args.simulator)

    # 包装环境
    wrapped_env = TaskEnvWrapper(task_env)

    # 创建服务器
    server = RemoteServer(port=args.port)
    server.set_environment(wrapped_env)

    print(f"\n✅ Server ready on port {args.port}")
    print("Waiting for connections...")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n⚠️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        server.stop()
        task_env.close()
        print("✅ Server closed")
