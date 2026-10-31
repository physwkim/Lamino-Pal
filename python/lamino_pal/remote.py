"""The GUI steered from another process: a Unix socket (a named pipe on
Windows) per project, served by the GUI that holds the project's lock. A
connection carries one JSON message `{"cmd", "args"}` and gets one back,
`{"ok": true, "result"}` or `{"ok": false, "error"}`."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import ExitStack
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any, Callable

from lamino_pal import LaminoPalError, locks
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


def _address(sock: Path) -> str:
    # Pipe names are machine-wide: the per-user socket path keeps them apart.
    return "\\\\.\\pipe\\" + sock.as_posix() if os.name == "nt" else str(sock)


class Server:
    """`handle(cmd, args)` served on project `project`'s socket, a thread per
    call, unless another GUI holds the project's lock: then not `serving`."""

    def __init__(self, project: Path, handle: Callable[[str, dict[str, Any]], Any]):
        self._sock, lock = _paths(project)
        self._handle = handle
        self._listener: Listener | None = None
        self._held = ExitStack()
        if self._held.enter_context(locks.locked(lock, wait_s=0)) is None:
            self._held.close()
            return
        # A GUI that died left it.
        self._sock.unlink(missing_ok=True)
        self._listener = Listener(_address(self._sock), backlog=16)
        self._closing = False
        self._accepting = threading.Thread(target=self._accept, name="gui-remote", daemon=True)
        self._accepting.start()

    def _accept(self) -> None:
        while True:
            try:
                conn = self._listener.accept()
            except OSError:
                continue  # a caller that left before it was accepted
            if self._closing:
                conn.close()
                return
            # A call waiting on a closing GUI's Qt thread does not hold up its close.
            threading.Thread(target=self._answer, args=(conn,), daemon=True).start()

    def _answer(self, conn: Connection) -> None:
        with conn:
            try:
                req = json.loads(conn.recv_bytes())
                reply = {"ok": True, "result": self._handle(req["cmd"], req.get("args", {}))}
            except Exception as e:
                err = str(e) if isinstance(e, LaminoPalError) else f"{type(e).__name__}: {e}"
                reply = {"ok": False, "error": err}
            try:
                conn.send_bytes(json.dumps(reply).encode())
            except OSError:
                pass  # the caller gave up

    @property
    def serving(self) -> bool:
        return self._listener is not None

    def close(self) -> None:
        if self._listener is None:
            return
        self._closing = True
        # Closing a listener does not wake its accept; a connection does.
        Client(self._listener.address).close()
        self._accepting.join()
        # Removes the socket, while the lock is held: once it goes, the
        # socket is the next GUI's.
        self._listener.close()
        self._held.close()
        self._listener = None


def call(project: Path, cmd: str, args: dict[str, Any] | None = None) -> Any:
    """Command `cmd` run with `args` by the GUI open on project.toml `project`:
    its result, or its error raised as LaminoPalError; NoGui when none is."""
    sock, _ = _paths(project)
    try:
        conn = Client(_address(sock))
    except (FileNotFoundError, ConnectionRefusedError) as e:
        raise NoGui(f"no Lamino-Pal GUI is open on {project}") from e
    with conn:
        conn.send_bytes(json.dumps({"cmd": cmd, "args": args or {}}).encode())
        if not conn.poll(TIMEOUT_S + 5):
            raise TimeoutError(f"the GUI on {project} did not answer {cmd}")
        reply = json.loads(conn.recv_bytes())
    if not reply["ok"]:
        raise LaminoPalError(reply["error"])
    return reply["result"]
