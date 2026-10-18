"""Pure logic of the Modal Sandbox replay backend (no Modal client or network needed).
Rules enforced: sandbox output is untrusted data (only plain files/dirs, contained in the destination); the image pins
match the repo's pins so the replay runs the same torch/numpy as local verification; only the modal path labels
exec_mode=modal_sandbox; an unknown backend is refused rather than silently running unisolated."""
import io
import re
import tarfile
from pathlib import Path

import pytest

from jobs.dataserver import SaltSchedule
from modal_train import sandbox_backend as sb
from traindoctor_env import reexec

ROOT = Path(__file__).resolve().parent.parent


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for ti, data in members:
            tf.addfile(ti, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def _file(name, data=b"x"):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    return ti, data


def test_safe_extract_accepts_plain_files(tmp_path):
    sb.safe_extract(_tar([_file("protocol/a.txt", b"hi")]), tmp_path)
    assert (tmp_path / "protocol" / "a.txt").read_bytes() == b"hi"


@pytest.mark.parametrize("kind", ["symlink", "escape", "abs", "fifo"])
def test_safe_extract_rejects_untrusted_members(tmp_path, kind):
    if kind == "symlink":
        ti = tarfile.TarInfo("protocol/l"); ti.type = tarfile.SYMTYPE; ti.linkname = "/etc/passwd"; m = [(ti, None)]
    elif kind == "fifo":
        ti = tarfile.TarInfo("protocol/f"); ti.type = tarfile.FIFOTYPE; m = [(ti, None)]
    else:
        m = [_file("../evil" if kind == "escape" else "/tmp/evil")]
    with pytest.raises(ValueError):
        sb.safe_extract(_tar(m), tmp_path)
    assert not (tmp_path.parent / "evil").exists()


def test_tar_dir_drops_links(tmp_path):
    d = tmp_path / "d"; d.mkdir(); (d / "f").write_text("a"); (d / "l").symlink_to("/etc/passwd")
    names = tarfile.open(fileobj=io.BytesIO(sb.tar_dir({"d": d}))).getnames()
    assert "d/f" in names and "d/l" not in names


def test_spec_args_roundtrip():
    a = sb.spec_args(SaltSchedule(b"\x01" * 32, b"\x02" * 32, 5), dict(kill=(1, 3), restart_on_failure=True))
    assert bytes.fromhex(a["base"]) == b"\x01" * 32 and a["boundary"] == 5 and a["kw"]["kill"] == [1, 3]


def test_image_pins_match_repo_pins():
    py = (ROOT / "pyproject.toml").read_text()
    lock = (ROOT / "uv.lock").read_text()
    assert f'"torch=={sb.TORCH_VERSION}"' in py
    assert re.search(rf'name = "numpy"\nversion = "{re.escape(sb.NUMPY_VERSION)}"', lock)


def test_unknown_backend_refused_and_label_constants():
    with pytest.raises(ValueError):
        reexec.Replay.__init__(object.__new__(reexec.Replay), None, None, "docker")
    assert reexec.EXEC_MODAL == sb.EXEC_MODAL == "modal_sandbox"
    assert reexec.FixOutcome.__dataclass_fields__["exec_mode"].default == "sandboxed"   # default never claims modal
