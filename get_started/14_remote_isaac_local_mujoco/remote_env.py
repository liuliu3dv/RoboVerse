"""
Remote Environment V2 - 完整的远程环境包装器
支持复杂任务、SSH隧道、日志监控
"""

import socket
import subprocess
import time
from typing import Any, Optional

from simple_protocol import Protocol, STEP, RESET, GET_STATE, RESPONSE, ERROR


class RemoteEnv:
    """
    完整的远程环境接口，像使用本地环境一样使用远程环境

    Features:
    - 自动SSH隧道
    - 远程服务器管理
    - 日志监控
    - 错误处理
    """

    def __init__(
        self,
        remote_host: str,
        port: int = 8888,
        ssh_host: Optional[str] = None,
        use_tunnel: bool = True,
        python_path: str = "python",
        remote_script_path: Optional[str] = None,
    ):
        """
        初始化远程环境

        Args:
            remote_host: 远程服务器地址
            port: 端口号
            ssh_host: SSH配置中的host名称
            use_tunnel: 是否使用SSH隧道
            python_path: 远程Python路径
            remote_script_path: 远程脚本路径
        """
        self.remote_host = remote_host
        self.port = port
        self.ssh_host = ssh_host or remote_host
        self.use_tunnel = use_tunnel
        self.python_path = python_path
        self.remote_script_path = remote_script_path

        self.socket = None
        self.connected = False
        self.remote_process = None
        self.tunnel_process = None

        # 连接到localhost（如果使用隧道）或远程host
        self.connect_host = "localhost" if use_tunnel else remote_host

    def start_remote_server(
        self,
        server_script: str = "simple_test_server.py",
        task_name: str = "stack_cube",
        num_envs: int = 1,
        simulator: str = "isaacsim",
        cleanup_old: bool = True,
    ) -> bool:
        """启动远程服务器"""
        try:
            if cleanup_old:
                print("Cleaning up old server processes...")
                subprocess.run(
                    ["ssh", self.ssh_host, f"pkill -f {server_script}"], check=False, capture_output=True, timeout=5
                )
                time.sleep(1)

            print(f"Starting remote server on {self.ssh_host}...")

            if self.remote_script_path:
                cmd = f"cd {self.remote_script_path} && "
            else:
                cmd = ""

            # 根据server_script类型添加参数
            if "task_server" in server_script:
                cmd += f"{self.python_path} {server_script} --task {task_name} --num_envs {num_envs} --simulator {simulator} > server.log 2>&1 &"
            else:
                cmd += f"{self.python_path} {server_script} > server.log 2>&1 &"

            ssh_cmd = ["ssh", self.ssh_host, cmd]
            self.remote_process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print("Waiting for server to start...")
            time.sleep(10)  # task_server需要更长时间启动

            # 检查日志
            if self.remote_script_path:
                log_path = f"{self.remote_script_path}/server.log"
                log = self.check_remote_log(log_path, lines=10)
                if log and "Traceback" in log:
                    print(f"⚠️  Server log shows errors:\n{log}")
                    return False

            print("✅ Remote server started")
            return True

        except Exception as e:
            print(f"Failed to start remote server: {e}")
            return False

    def setup_tunnel(self) -> bool:
        """设置SSH隧道"""
        if not self.use_tunnel:
            return True

        try:
            print(f"Setting up SSH tunnel: localhost:{self.port} -> {self.ssh_host}:{self.port}")
            tunnel_cmd = ["ssh", "-N", "-L", f"{self.port}:localhost:{self.port}", self.ssh_host]
            self.tunnel_process = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)
            print("✅ SSH tunnel established")
            return True
        except Exception as e:
            print(f"Failed to setup tunnel: {e}")
            return False

    def connect(self) -> bool:
        """连接到远程服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.connect_host, self.port))
            self.connected = True
            print(f"✅ Connected to {self.connect_host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def step(self, action: Any) -> tuple:
        """执行一步"""
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

    def reset(self) -> Any:
        """重置环境"""
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

    def get_state(self) -> Any:
        """获取环境状态"""
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

    def check_remote_log(self, log_path: str, lines: int = 20) -> Optional[str]:
        """检查远程日志"""
        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, f"tail -{lines} {log_path}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout if result.stdout else None
        except Exception as e:
            print(f"Failed to check remote log: {e}")
            return None

    def check_remote_process(self, process_name: str = "server") -> Optional[str]:
        """检查远程进程状态"""
        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, f"ps aux | grep {process_name} | grep -v grep"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.stdout else None
        except Exception as e:
            print(f"Failed to check remote process: {e}")
            return None

    def close(self):
        """关闭连接"""
        print("Closing remote environment...")
        self.connected = False

        if self.socket:
            self.socket.close()

        if self.tunnel_process:
            self.tunnel_process.terminate()
            try:
                self.tunnel_process.wait(timeout=2)
            except:
                self.tunnel_process.kill()

        if self.remote_process:
            try:
                self.remote_process.terminate()
                self.remote_process.wait(timeout=5)
            except:
                try:
                    self.remote_process.kill()
                except:
                    pass

        print("✅ Remote environment closed")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()
