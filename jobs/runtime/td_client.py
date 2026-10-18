"""Client for the cluster's data service, imported by the job. The launcher owns the other end of an inherited
socket (TD_FD); the job gets its batches from it by position and can neither generate them itself nor choose
which rank it speaks for. Everything the job sends here is untrusted input to the launcher."""
import base64
import json
import os
import socket
import threading

import torch

_lock = threading.Lock()
_conn: dict = {}


def _sock() -> socket.socket:
    if "s" not in _conn:
        fd = os.environ.get("TD_FD")
        if fd is None:
            raise RuntimeError("no data service in this process (TD_FD is not set)")
        _conn["s"], _conn["buf"] = socket.socket(fileno=int(fd)), b""
    return _conn["s"]


def _send(msg: dict) -> None:
    _sock().sendall(json.dumps(msg, sort_keys=True).encode() + b"\n")


def _recv() -> dict:
    while b"\n" not in _conn["buf"]:
        chunk = _sock().recv(65536)
        if not chunk:
            raise RuntimeError("data service closed")
        _conn["buf"] += chunk
    line, _conn["buf"] = _conn["buf"].split(b"\n", 1)
    return json.loads(line)


def notify(msg: dict) -> None:
    with _lock:
        _send(msg)


def _tensor(r: dict, key: str) -> torch.Tensor:
    raw = bytearray(base64.b64decode(r[key]))
    return torch.frombuffer(raw, dtype=torch.float32).reshape(r[key + "_shape"]).clone()


def fetch(epoch: int, index: int):
    """The batch at this rank's position (epoch, index): (ids, x, y)."""
    with _lock:
        _send(dict(op="batch", epoch=int(epoch), index=int(index)))
        r = _recv()
    if "error" in r:
        raise RuntimeError(f"data service refused the request: {r['error']}")
    return r["ids"], _tensor(r, "x"), _tensor(r, "y")
