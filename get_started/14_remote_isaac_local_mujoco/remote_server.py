"""
Remote environment server - 远程环境服务器
"""

import argparse
import socket
import threading

from simple_protocol import ERROR, GET_STATE, RESET, RESPONSE, STEP, Protocol


class RemoteServer:
    """Remote server that wraps a local environment."""

    def __init__(self, host="0.0.0.0", port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.env = None  # 外部设置的环境对象

    def set_environment(self, env):
        """Set the environment to wrap."""
        self.env = env

    def start(self):
        """Start the server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)

        self.running = True
        print(f"Remote server started on {self.host}:{self.port}")

        while self.running:
            try:
                client_socket, address = self.socket.accept()
                print(f"Client connected from {address}")

                thread = threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True)
                thread.start()

            except Exception as e:
                if self.running:
                    print(f"Error accepting client: {e}")

    def _handle_client(self, client_socket):
        """Handle client connection."""
        try:
            while self.running:
                message = Protocol.receive(client_socket)
                if not message:
                    break

                response = self._process_message(message)
                if response:
                    Protocol.send(client_socket, response)

        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def _process_message(self, message):
        """Process client message."""
        try:
            msg_type = message.get("type")

            if msg_type == STEP:
                return self._handle_step(message.get("action"))
            elif msg_type == RESET:
                return self._handle_reset()
            elif msg_type == GET_STATE:
                return self._handle_get_state()
            else:
                return {"type": ERROR, "error": f"Unknown message type: {msg_type}"}

        except Exception as e:
            return {"type": ERROR, "error": str(e)}

    def _handle_step(self, action):
        """Handle step request."""
        if not self.env:
            return {"type": ERROR, "error": "No environment set"}

        try:
            result = self.env.step(action)
            return {"type": RESPONSE, "data": result}
        except Exception as e:
            return {"type": ERROR, "error": f"Step failed: {e}"}

    def _handle_reset(self):
        """Handle reset request."""
        if not self.env:
            return {"type": ERROR, "error": "No environment set"}

        try:
            result = self.env.reset()
            return {"type": RESPONSE, "data": result}
        except Exception as e:
            return {"type": ERROR, "error": f"Reset failed: {e}"}

    def _handle_get_state(self):
        """Handle get_state request."""
        if not self.env:
            return {"type": ERROR, "error": "No environment set"}

        try:
            if hasattr(self.env, "get_state"):
                state = self.env.get_state()
                return {"type": RESPONSE, "data": state}
            else:
                return {"type": ERROR, "error": "Environment has no get_state method"}
        except Exception as e:
            return {"type": ERROR, "error": f"Get state failed: {e}"}

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.socket:
            self.socket.close()


def main():
    """Main function for standalone server."""
    parser = argparse.ArgumentParser(description="Remote Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8888, help="Server port")

    args = parser.parse_args()

    # 创建服务器 - 需要外部设置环境
    server = RemoteServer(args.host, args.port)

    try:
        server.start()
    except KeyboardInterrupt:
        print("Server stopped")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
