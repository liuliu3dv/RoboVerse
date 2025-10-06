"""
Simple remote environment protocol.
"""

import pickle
import struct


class Protocol:
    """Simple protocol for remote environment communication."""

    @staticmethod
    def send(socket_conn, message):
        """Send message with length prefix."""
        serialized = pickle.dumps(message)
        length = len(serialized)
        socket_conn.send(struct.pack("!I", length))
        socket_conn.send(serialized)

    @staticmethod
    def receive(socket_conn):
        """Receive message with length prefix."""
        # Get length
        length_data = socket_conn.recv(4)
        if not length_data:
            return None
        length = struct.unpack("!I", length_data)[0]

        # Get data
        data = b""
        while len(data) < length:
            chunk = socket_conn.recv(min(length - len(data), 4096))
            if not chunk:
                return None
            data += chunk

        return pickle.loads(data)


# Message types
STEP = "STEP"
RESET = "RESET"
GET_STATE = "GET_STATE"
RESPONSE = "RESPONSE"
ERROR = "ERROR"
