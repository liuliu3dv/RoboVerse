#!/usr/bin/env python3
"""
Standalone test script for ji_12 remote server - 不依赖metasim
"""

import subprocess
import sys
import time

from remote_env import RemoteEnv


def test_ji_12_connection():
    """Test connection to rll_6000_2 remote server."""

    # 配置rll_6000_2远程服务器信息
    ssh_host = "rll_6000_2"
    remote_host = "pabrtxl2.ist.berkeley.edu"
    port = 8888
    remote_path = "/datasets/v2p/current/murphy/dev/RoboVerse/get_started/14_remote_isaac_local_mujoco"
    python_path = "/datasets/v2p/current/murphy/dev/lab/bin/python"

    print("=" * 60)
    print("🚀 Testing Remote Connection to rll_6000_2")
    print("=" * 60)
    print(f"SSH Host: {ssh_host}")
    print(f"Remote Host: {remote_host}")
    print(f"Port: {port}")
    print(f"Remote Path: {remote_path}")
    print(f"Python Path: {python_path}")
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

        # 先清理旧的服务器进程
        print("Cleaning up old server processes...")
        cleanup_cmd = subprocess.run(
            ["ssh", ssh_host, "pkill -f simple_test_server.py"], capture_output=True, text=True, timeout=5
        )
        time.sleep(1)  # 等待进程清理完成

        # 创建远程环境
        env = RemoteEnv(remote_host, port)

        # 启动远程服务器 - 使用简单测试服务器
        print("Starting fresh server instance...")
        ssh_cmd = ["ssh", ssh_host, f"cd {remote_path} && {python_path} simple_test_server.py > server.log 2>&1 &"]
        subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Waiting for server to start...")
        time.sleep(5)

        # 检查远程服务器日志
        print("\nChecking remote server log...")
        log_check = subprocess.run(
            ["ssh", ssh_host, f"cat {remote_path}/server.log"], capture_output=True, text=True, timeout=5
        )
        if log_check.stdout:
            print(f"Server log output:\n{log_check.stdout}")
        if log_check.stderr:
            print(f"Server log errors:\n{log_check.stderr}")

        print()

        # 步骤4: 设置SSH隧道并连接测试
        print("Step 4: Setting up SSH tunnel and testing connection...")
        print("Creating SSH tunnel: localhost:8888 -> remote:8888...")

        # 创建SSH隧道
        tunnel_cmd = ["ssh", "-N", "-L", f"{port}:localhost:{port}", ssh_host]
        tunnel_process = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Waiting for tunnel to establish...")
        time.sleep(2)

        # 连接到localhost（通过隧道）
        env_local = RemoteEnv("localhost", port)
        if not env_local.connect():
            print("❌ Failed to connect to remote server through tunnel")
            tunnel_process.terminate()
            return False

        print("✅ Connected to remote server")
        print()

        # 步骤5: 功能测试
        print("Step 5: Testing remote environment functionality...")

        # 测试reset
        print("Testing reset...")
        start_time = time.time()
        obs = env_local.reset()
        reset_time = time.time() - start_time
        print(f"✅ Reset successful in {reset_time * 1000:.1f}ms")
        print(f"Observation type: {type(obs)}")

        # 测试step
        print("\nTesting step...")
        for i in range(5):
            start_time = time.time()
            action = [0.1, 0.1, 0.1]  # 示例动作
            obs, reward, done, info = env_local.step(action)
            step_time = time.time() - start_time

            print(f"  Step {i + 1}: reward={reward:.3f}, done={done}, time={step_time * 1000:.1f}ms")

            if done:
                obs = env_local.reset()
                print("  Environment reset due to done=True")

        print()

        # 步骤6: 性能测试
        print("Step 6: Performance test (10 steps)...")
        start_time = time.time()
        for i in range(10):
            action = [0.1, 0.1, 0.1]
            obs, reward, done, info = env_local.step(action)
        total_time = time.time() - start_time

        print(f"✅ 10 steps completed in {total_time:.2f}s")
        print(f"Average time per step: {total_time / 10 * 1000:.1f}ms")

        print()

        # 步骤7: 最终检查远程服务器状态
        print("Step 7: Final remote server status check...")

        # 检查进程是否还在运行
        ps_check = subprocess.run(
            ["ssh", ssh_host, "ps aux | grep simple_test_server | grep -v grep"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if ps_check.stdout:
            print(f"✅ Remote server process is running")
            print(f"Process info: {ps_check.stdout.strip()}")
        else:
            print("⚠️  Remote server process not found")

        # 检查最终日志
        final_log = subprocess.run(
            ["ssh", ssh_host, f"tail -20 {remote_path}/server.log"], capture_output=True, text=True, timeout=5
        )
        if final_log.stdout:
            print(f"\nFinal server log (last 20 lines):\n{final_log.stdout}")

        print()
        print("=" * 60)
        print("🎉 All tests passed! Remote environment is working!")
        print("=" * 60)

        env_local.close()
        tunnel_process.terminate()
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ji_12_connection()
    sys.exit(0 if success else 1)
