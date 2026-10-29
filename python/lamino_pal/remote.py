"""The GUI steered from another process: a Unix socket per project, served by
the GUI that holds the project's lock. A connection carries one JSON line
`{"cmd", "args"}` and gets one back, `{"ok": true, "result"}` or `{"ok":
false, "error"}`."""

from __future__ import annotations

import fcntl
import hashlib
import json
import socket
import socketserver
import threading
from pathlib import Path
from typing import Any, Callable

from lamino_pal import LaminoPalError
from lamino_pal.dispatch import runtime_dir

# How long the GUI waits for its Qt thread; a caller waits a little longer, to
# hear that the GUI's wait ran out.
TIMEOUT_S = 10.0


class NoGui(LaminoPalError):
    """No GUI is open on the project."""


def _paths(project: Path) -> tuple[Path, Path]:
    """The socket and lock of project.toml `project`."""
    key = hashlib.sha1(str(Path(project).resolve()).encode()).hexdigest()[:12]
    d = runtime_dir()
    return d / f"gui-{key}.sock", d / f"gui-{key}.lock"


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    # A call waiting on a closing GUI's Qt thread does not hold up its close.
    daemon_threads = True
    block_on_close = False


class Server:
    """`handle(cmd, args)` served on project `project`'s socket, a thread per
    call, unless another GUI holds the project's lock: then not `serving`."""

    def __init__(self, project: Path, handle: Callable[[str, dict[str, Any]], Any]):
        self._sock, lock = _paths(project)
        self._server: _UnixServer | None = None
        self._lock = open(lock, "a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            return

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    req = json.loads(self.rfile.readline())
                    reply = {"ok": True, "result": handle(req["cmd"], req.get("args", {}))}
                except Exception as e:
                    err = str(e) if isinstance(e, LaminoPalError) else f"{type(e).__name__}: {e}"
                    reply = {"ok": False, "error": err}
                self.wfile.write(json.dumps(reply).encode() + b"\n")

        # A GUI that died left it.
        self._sock.unlink(missing_ok=True)
        self._server = _UnixServer(str(self._sock), Handler)
        threading.Thread(target=self._server.serve_forever, name="gui-remote", daemon=True).start()

    @property
    def serving(self) -> bool:
        return self._server is not None

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        # While the lock is held: once it goes, the socket is the next GUI's.
        self._sock.unlink(missing_ok=True)
        self._lock.close()
        self._server = None


def call(project: Path, cmd: str, args: dict[str, Any] | None = None) -> Any:
    """Command `cmd` run with `args` by the GUI open on project.toml `project`:
    its result, or its error raised as LaminoPalError; NoGui when none is."""
    sock, _ = _paths(project)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(TIMEOUT_S + 5)
        try:
            s.connect(str(sock))
        except (FileNotFoundError, ConnectionRefusedError) as e:
            raise NoGui(f"no Lamino-Pal GUI is open on {project}") from e
        s.sendall(json.dumps({"cmd": cmd, "args": args or {}}).encode() + b"\n")
        with s.makefile("rb") as f:
            line = f.readline()
    reply = json.loads(line)
    if not reply["ok"]:
        raise LaminoPalError(reply["error"])
    return reply["result"]
