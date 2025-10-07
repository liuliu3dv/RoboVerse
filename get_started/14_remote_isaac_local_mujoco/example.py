"""
Example usage of remote environment.
"""

import subprocess
import time

import rootutils
import torch
from remote_env import RemoteEnv
from remote_server import RemoteServer

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils.setup_util import get_handler

rootutils.setup_root(__file__, pythonpath=True)


def create_local_env(task_name="stack_cube", num_envs=1):
    """Create a local environment."""
    task_cls = get_task_class(task_name)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))

    scenario = task_cls.scenario.update(
        simulator="mujoco",  # 或者 "isaacsim"
        cameras=[camera],
        num_envs=num_envs,
        headless=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handler = get_handler(scenario, device)

    return handler


def server_example():
    """Server example - 运行在远程机器上."""
    print("Starting server...")

    # 创建本地环境
    env = create_local_env("stack_cube", num_envs=1)

    # 创建远程服务器并设置环境
    server = RemoteServer(port=8888)
    server.set_environment(env)

    try:
        server.start()
    except KeyboardInterrupt:
        print("Server stopped")
    finally:
        server.stop()


def client_example(remote_host="localhost", port=8888, auto_start_remote=False, task_name="stack_cube", num_envs=1):
    """Client example - 运行在本地机器上."""
    print("Starting client...")

    # 创建远程环境
    env = RemoteEnv(remote_host, port)

    # 自动启动远程服务器
    if auto_start_remote and remote_host != "localhost":
        print(f"Auto-starting remote server on {remote_host}...")
        # 配置ji_12远程路径
        remote_path = "/datasets/v2p/current/murphy/dev/RoboVerse/get_started/14_remote_isaac_local_mujoco"
        ssh_host = "ji_12"  # SSH配置中的Host名称
        if not env.auto_start_remote_server(ssh_host, remote_path, task_name, num_envs):
            print("Failed to auto-start remote server")
            return

    if env.connect():
        try:
            # 重置环境
            obs = env.reset()
            print(f"Reset successful, obs type: {type(obs)}")

            # 运行几步
            for i in range(10):
                action = [0.1, 0.1, 0.1]  # 示例动作
                obs, reward, done, info = env.step(action)
                print(f"Step {i}: reward={reward}, done={done}")

                if done:
                    obs = env.reset()
                    print("Environment reset")

        finally:
            env.close()
    else:
        print("Failed to connect to server")


def test_ji_12_connection():
    """Test connection to ji_12 remote server."""

    # 配置ji_12远程服务器信息
    ssh_host = "ji_12"
    remote_host = "em12.ist.berkeley.edu"
    port = 8888
    remote_path = "/datasets/v2p/current/murphy/dev/RoboVerse/get_started/14_remote_isaac_local_mujoco"

    print("=" * 60)
    print("🚀 Testing Remote Connection to ji_12")
    print("=" * 60)
    print(f"SSH Host: {ssh_host}")
    print(f"Remote Host: {remote_host}")
    print(f"Port: {port}")
    print(f"Remote Path: {remote_path}")
    print()

    try:
        # 步骤1: 测试SSH连接
        print("Step 1: Testing SSH connection...")
        try:
            result = subprocess.run(
                ["ssh", ssh_host, "echo 'SSH connection successful'"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("✅ SSH connection successful")
                print(f"Response: {result.stdout.strip()}")
            else:
                print("❌ SSH connection failed")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ SSH test failed: {e}")
            return False

        print()

        # 步骤2: 检查远程环境
        print("Step 2: Checking remote environment...")
        try:
            result = subprocess.run(
                ["ssh", ssh_host, f"cd {remote_path} && ls -la"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("✅ Remote directory accessible")
                print("Files:")
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        print(f"  {line}")
            else:
                print("❌ Remote directory not accessible")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Remote directory check failed: {e}")
            return False

        print()

        # 步骤3: 启动远程服务器并测试
        print("Step 3: Starting remote server and testing...")

        # 创建远程环境
        env = RemoteEnv(remote_host, port)

        # 启动远程服务器
        if not env.auto_start_remote_server(ssh_host, remote_path, "stack_cube", 1):
            print("❌ Failed to start remote server")
            return False

        print()

        # 步骤4: 连接测试
        print("Step 4: Testing connection to remote server...")
        if not env.connect():
            print("❌ Failed to connect to remote server")
            return False

        print("✅ Connected to remote server")
        print()

        # 步骤5: 功能测试
        print("Step 5: Testing remote environment functionality...")

        # 测试reset
        print("Testing reset...")
        start_time = time.time()
        obs = env.reset()
        reset_time = time.time() - start_time
        print(f"✅ Reset successful in {reset_time * 1000:.1f}ms")
        print(f"Observation type: {type(obs)}")

        # 测试step
        print("Testing step...")
        for i in range(5):
            start_time = time.time()
            action = [0.1, 0.1, 0.1]  # 示例动作
            obs, reward, done, info = env.step(action)
            step_time = time.time() - start_time

            print(f"  Step {i + 1}: reward={reward:.3f}, done={done}, time={step_time * 1000:.1f}ms")

            if done:
                obs = env.reset()
                print("  Environment reset due to done=True")

        print()

        # 步骤6: 性能测试
        print("Step 6: Performance test...")
        start_time = time.time()
        for i in range(10):
            action = [0.1, 0.1, 0.1]
            obs, reward, done, info = env.step(action)
        total_time = time.time() - start_time

        print(f"✅ 10 steps completed in {total_time:.2f}s")
        print(f"Average time per step: {total_time / 10 * 1000:.1f}ms")

        print()
        print("=" * 60)
        print("🎉 All tests passed! Remote environment is working!")
        print("=" * 60)

        env.close()
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Remote Environment Example")
    parser.add_argument("--mode", choices=["server", "client", "test"], default="client", help="Run mode")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--remote_host", help="Remote server host for client mode")
    parser.add_argument("--auto_start_remote", action="store_true", help="Auto-start remote server")
    parser.add_argument("--task", default="stack_cube", help="Task name")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

    # Handle legacy server argument
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        server_example()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_ji_12_connection()
        sys.exit(0 if success else 1)
    else:
        args = parser.parse_args()
        if args.mode == "server":
            server_example()
        elif args.mode == "test":
            success = test_ji_12_connection()
            sys.exit(0 if success else 1)
        else:
            client_example(args.remote_host, args.port, args.auto_start_remote, args.task, args.num_envs)
