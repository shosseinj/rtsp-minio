from __future__ import annotations

import socket
from dataclasses import dataclass


def _command(*parts: str) -> bytes:
    encoded = [part.encode("utf-8") for part in parts]
    return (
        f"*{len(encoded)}\r\n".encode()
        + b"".join(
            f"${len(part)}\r\n".encode() + part + b"\r\n"
            for part in encoded
        )
    )


@dataclass
class RedisStreamPublisher:
    host: str
    port: int = 6379
    timeout: float = 2.0

    def publish(self, stream: str, fields: dict[str, object]) -> str:
        forbidden = ("raw", "base64", "cuda", "nvmm", "jpeg_payload")
        for key in fields:
            if any(word in key.lower() for word in forbidden):
                raise ValueError(f"forbidden Redis payload field: {key}")
        args = ["XADD", stream, "*"]
        for key, value in fields.items():
            args.extend((str(key), "" if value is None else str(value)))
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        ) as connection:
            connection.sendall(_command(*args))
            response = connection.recv(4096)
        if response.startswith(b"-"):
            raise ConnectionError(response.decode(errors="replace"))
        if not response.startswith(b"$"):
            raise ConnectionError("unexpected Redis response")
        return response.split(b"\r\n", 2)[1].decode()
