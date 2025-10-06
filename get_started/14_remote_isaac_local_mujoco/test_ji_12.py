#!/usr/bin/env python3
"""
Standalone test script for ji_12 remote server - 不依赖metasim
"""

import subprocess
import sys
import time

from remote_env import RemoteEnv


def test_ji_12_connection():
    """Test connection to ji_12 remote server."""

    # 配置ji_12远程服务器信息
    ssh_host = "ji_12"
    remote_host = "em12.ist.berkeley.edu"
    port = 8888
    remote_path = "/home/ghr/RoboVerse/get_started/14_remote_isaac_local_mujoco"

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
        print("Step 3: Starting remote server...")

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
            env.close()
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
        print("\nTesting step...")
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
        print("Step 6: Performance test (10 steps)...")
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
    success = test_ji_12_connection()
    sys.exit(0 if success else 1)
