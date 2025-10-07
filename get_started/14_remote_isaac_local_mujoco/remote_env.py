"""
Remote environment wrapper - 远程环境包装器
"""

import socket
import subprocess
import time

from simple_protocol import ERROR, GET_STATE, RESET, RESPONSE, STEP, Protocol


class RemoteEnv:
    """Remote environment that looks like a local environment."""

    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.remote_process = None

    def connect(self):
        """Connect to remote server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def step(self, action):
        """Step the environment."""
        if not self.connected:
            raise RuntimeError("Not connected to server")

        try:
            message = {"type": STEP, "action": action}
            Protocol.send(self.socket, message)

            response = Protocol.receive(self.socket)
            if not response:
                raise RuntimeError("No response from server")

            if response.get("type") == RESPONSE:
                return response["data"]
            elif response.get("type") == ERROR:
                raise RuntimeError(f"Server error: {response['error']}")
            else:
                raise RuntimeError(f"Unexpected response: {response}")

        except Exception as e:
            self.connected = False
            raise RuntimeError(f"Step failed: {e}") from e

    def reset(self):
        """Reset the environment."""
        if not self.connected:
            raise RuntimeError("Not connected to server")

        try:
            message = {"type": RESET}
            Protocol.send(self.socket, message)

            response = Protocol.receive(self.socket)
            if not response:
                raise RuntimeError("No response from server")

            if response.get("type") == RESPONSE:
                return response["data"]
            elif response.get("type") == ERROR:
                raise RuntimeError(f"Server error: {response['error']}")
            else:
                raise RuntimeError(f"Unexpected response: {response}")

        except Exception as e:
            self.connected = False
            raise RuntimeError(f"Reset failed: {e}") from e

    def get_state(self):
        """Get environment state."""
        if not self.connected:
            raise RuntimeError("Not connected to server")

        try:
            message = {"type": GET_STATE}
            Protocol.send(self.socket, message)

            response = Protocol.receive(self.socket)
            if not response:
                raise RuntimeError("No response from server")

            if response.get("type") == RESPONSE:
                return response["data"]
            elif response.get("type") == ERROR:
                raise RuntimeError(f"Server error: {response['error']}")
            else:
                raise RuntimeError(f"Unexpected response: {response}")

        except Exception as e:
            self.connected = False
            raise RuntimeError(f"Get state failed: {e}") from e

    def auto_start_remote_server(self, ssh_host, remote_script_path, task_name="stack_cube", num_envs=1):
        """Auto-start remote server via SSH."""
        try:
            print(f"Starting remote server on {ssh_host}...")

            # SSH command to start server
            ssh_cmd = [
                "ssh",
                ssh_host,
                f"source ~/anaconda3/etc/profile.d/conda.sh && "
                f"cd {remote_script_path} && "
                f"conda activate roboverse && "
                f"python example.py server --task {task_name} --num_envs {num_envs}",
            ]

            # Start remote server process
            self.remote_process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Wait for server to start
            print("Waiting for remote server to start...")
            time.sleep(10)  # 增加等待时间让服务器完全启动

            # Check if there are any errors
            import select

            if select.select([self.remote_process.stderr], [], [], 0)[0]:
                error_output = self.remote_process.stderr.read()
                if error_output:
                    print(f"⚠️  Remote server stderr: {error_output}")
                    return False

            print("✅ Remote server started successfully")
            return True

        except Exception as e:
            print(f"Failed to start remote server: {e}")
            return False

    def check_remote_log(self, ssh_host, remote_log_path, lines=20):
        """Check remote server log file."""
        try:
            import subprocess

            result = subprocess.run(
                ["ssh", ssh_host, f"tail -{lines} {remote_log_path}"], capture_output=True, text=True, timeout=5
            )
            if result.stdout:
                return result.stdout
            return None
        except Exception as e:
            print(f"Failed to check remote log: {e}")
            return None

    def check_remote_process(self, ssh_host, process_name="simple_test_server"):
        """Check if remote process is running."""
        try:
            import subprocess

            result = subprocess.run(
                ["ssh", ssh_host, f"ps aux | grep {process_name} | grep -v grep"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.stdout else None
        except Exception as e:
            print(f"Failed to check remote process: {e}")
            return None

    def close(self):
        """Close connection."""
        self.connected = False
        if self.socket:
            self.socket.close()

        # Stop remote server if we started it
        if self.remote_process:
            try:
                self.remote_process.terminate()
                self.remote_process.wait(timeout=5)
            except:
                try:
                    self.remote_process.kill()
                except:
                    pass


# 使用示例
if __name__ == "__main__":
    # 创建远程环境
    env = RemoteEnv("localhost", 8888)

    # 连接
    if env.connect():
        try:
            # 重置环境
            obs = env.reset()
            print(f"Reset: {type(obs)}")

            # 执行几步
            for i in range(5):
                action = [0.1, 0.1]  # 示例动作
                obs, reward, done, info = env.step(action)
                print(f"Step {i}: reward={reward}, done={done}")

                if done:
                    obs = env.reset()
                    print("Environment reset")

        finally:
            env.close()
