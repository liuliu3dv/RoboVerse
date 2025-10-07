"""
Remote Replay Sync - 同步replay
远程step一步，立即本地set_state并渲染
"""

import os
import sys
import time
import torch
import imageio as iio
import numpy as np
import rootutils
from torchvision.utils import make_grid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import get_traj

from remote_env import RemoteEnv

rootutils.setup_root(__file__, pythonpath=True)


class ObsSaver:
    """保存观察结果"""

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.images = []

    def add(self, state):
        if self.video_path is None:
            return
        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(max(1, rgb_data.shape[0] ** 0.5)))
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            self.images.append(image)
        except Exception as e:
            print(f"Error adding frame: {e}")

    def save(self):
        if self.video_path and self.images:
            print(f"Saving {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


def main():
    """同步remote replay"""

    # 配置
    config = {
        "remote_host": "pabrtxl2.ist.berkeley.edu",
        "ssh_host": "rll_6000_2",
        "python_path": "/datasets/v2p/current/murphy/dev/lab/bin/python",
        "remote_script_path": "/datasets/v2p/current/murphy/dev/RoboVerse/get_started/14_remote_isaac_local_mujoco",
        "task": "stack_cube",
        "robot": "franka",
        "num_envs": 1,
        "simulator": "isaacsim",  # 远程用IsaacSim
    }

    print("=" * 60)
    print("🚀 Remote Replay Sync Example")
    print("=" * 60)
    print(f"Task: {config['task']}")
    print(f"Remote simulator: {config['simulator']}")
    print(f"Local simulator: mujoco")

    # ============================================================
    # 1. 创建本地MuJoCo环境
    # ============================================================
    print("\n[1/5] Creating local MuJoCo environment...")
    task_cls = get_task_class(config["task"])
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))

    scenario = task_cls.scenario.update(
        robots=[config["robot"]],
        simulator="mujoco",
        cameras=[camera],
        num_envs=config["num_envs"],
        headless=False,
    )

    device = torch.device("cpu")
    local_env = task_cls(scenario, device=device)
    print(f"✅ Local MuJoCo environment created")

    # ============================================================
    # 2. 加载真实轨迹
    # ============================================================
    print("\n[2/5] Loading trajectory...")
    traj_filepath = local_env.traj_filepath

    if not os.path.exists(traj_filepath):
        print(f"❌ Trajectory file not found: {traj_filepath}")
        return

    init_states, all_actions, _ = get_traj(traj_filepath, scenario.robots[0], local_env.handler)
    print(f"✅ Loaded trajectory with {len(all_actions[0])} steps")

    # ============================================================
    # 3. 连接远程服务器
    # ============================================================
    print("\n[3/5] Connecting to remote server...")

    remote_env = RemoteEnv(
        remote_host=config["remote_host"],
        port=8888,
        ssh_host=config["ssh_host"],
        use_tunnel=True,
        python_path=config["python_path"],
        remote_script_path=config["remote_script_path"],
    )

    # 启动远程服务器
    if not remote_env.start_remote_server(
        server_script="task_server.py",
        task_name=config["task"],
        num_envs=config["num_envs"],
        simulator=config["simulator"],
        cleanup_old=True,
    ):
        print("❌ Failed to start remote server")
        return

    # 设置SSH隧道
    if not remote_env.setup_tunnel():
        print("❌ Failed to setup tunnel")
        return

    # 连接
    if not remote_env.connect():
        print("❌ Failed to connect")
        return

    print("✅ Connected to remote server")

    # ============================================================
    # 4. 同步replay：远程step → 本地set_state
    # ============================================================
    print("\n[4/5] Running sync replay...")

    os.makedirs("test_output", exist_ok=True)
    saver = ObsSaver(video_path="test_output/remote_replay_sync.mp4")

    # Reset远程环境
    print("Resetting remote environment...")
    remote_obs = remote_env.reset()

    # 获取远程state并同步到本地
    remote_state = remote_env.get_state()
    print (remote_state)
    local_env.handler.set_states(remote_state)
    local_env.handler.refresh_render()
    local_obs = local_env.handler.get_states()
    saver.add(local_obs)

    print(f"✅ Initial state synced")
    print(f"Running {len(all_actions[0])} steps with real trajectory...")

    # 执行轨迹
    max_steps = min(len(all_actions[0]), 100)  # 限制步数

    total_reward = 0
    success_count = 0

    for step in range(max_steps):
        # 获取真实动作（从轨迹中）
        action = all_actions[0][step]

        # 远程执行
        start_time = time.time()
        remote_obs, reward, done, info = remote_env.step(action)
        step_time = time.time() - start_time

        # 处理reward（可能是tensor）
        if hasattr(reward, "item"):
            reward_value = reward.item()
        elif hasattr(reward, "mean"):
            reward_value = reward.mean().item()
        else:
            reward_value = float(reward)

        total_reward += reward_value

        # 处理done（可能是tensor）
        if hasattr(done, "item"):
            done_value = done.item()
        elif hasattr(done, "any"):
            done_value = done.any().item() if hasattr(done.any(), "item") else bool(done.any())
        else:
            done_value = bool(done)

        # 获取远程state
        remote_state = remote_env.get_state()

        # 同步到本地MuJoCo
        local_env.handler.set_states(remote_state)
        local_env.handler.refresh_render()
        local_obs = local_env.handler.get_states()
        saver.add(local_obs)

        # 检查success
        if info and "success" in info:
            success = info["success"]
            if hasattr(success, "any"):
                if success.any():
                    success_count += 1

        if (step + 1) % 10 == 0:
            print(
                f"  Step {step + 1:3d}/{max_steps}: reward={reward_value:.3f}, done={done_value}, "
                f"time={step_time * 1000:.1f}ms, total_reward={total_reward:.3f}"
            )

        if done_value:
            print(f"  → Episode done at step {step + 1}")
            break

        time.sleep(0.01)  # 小延迟以便观察

    print(f"\n✅ Sync replay completed!")
    print(f"   Total steps: {step + 1}")
    print(f"   Total reward: {total_reward:.3f}")
    print(f"   Success count: {success_count}")

    # ============================================================
    # 5. 保存结果
    # ============================================================
    print("\n[5/5] Saving results...")
    saver.save()
    print(f"✅ Video saved to test_output/remote_replay_sync.mp4")

    # 清理
    remote_env.close()
    local_env.close()

    print("\n" + "=" * 60)
    print("🎉 Remote sync replay completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
