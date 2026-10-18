"""Reads config.yaml: nested `key: value` maps (2-space indent), ints, floats, true/false, null, strings."""
from __future__ import annotations


def _scalar(s: str):
    s = s.strip()
    if s == "{}":
        return {}
    if s in ("true", "false"):
        return s == "true"
    if s in ("null", "~"):
        return None
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s.strip("\"'")


def loads(text: str) -> dict:
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack[-1][0] >= indent:
            stack.pop()
        if val.strip():
            stack[-1][1][key] = _scalar(val)
        else:
            stack[-1][1][key] = child = {}
            stack.append((indent, child))
    return root


def dumps(d: dict, indent: int = 0) -> str:
    out = []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.append(" " * indent + f"{k}:\n" + dumps(v, indent + 2))
        else:
            s = "{}" if v == {} else "null" if v is None else str(v).lower() if isinstance(v, bool) else str(v)
            out.append(" " * indent + f"{k}: {s}\n")
    return "".join(out)


def load(path: str) -> dict:
    with open(path) as f:
        return loads(f.read())
