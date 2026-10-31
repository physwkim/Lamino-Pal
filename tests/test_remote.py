"""The GUI's socket without the GUI: which GUI serves a project, and what a
call gets back."""

import socket

import pytest

from lamino_pal import LaminoPalError, remote


def test_one_gui_serves_a_project_until_it_closes(tmp_path):
    p = tmp_path / "project.toml"
    with pytest.raises(remote.NoGui, match="no Lamino-Pal GUI is open on"):
        remote.call(p, "state")
    first = remote.Server(p, lambda cmd, args: ["first", cmd, args])
    second = remote.Server(p, lambda cmd, args: ["second", cmd, args])
    assert first.serving and not second.serving
    assert remote.call(p, "show_tab", {"tab": "Align"}) == ["first", "show_tab", {"tab": "Align"}]
    first.close()
    with pytest.raises(remote.NoGui):
        remote.call(p, "state")
    third = remote.Server(p, lambda cmd, args: "third")
    try:
        assert third.serving and remote.call(p, "state") == "third"
    finally:
        third.close()


# A Windows GUI serves a named pipe, which goes with its process.
@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix sockets")
def test_a_dead_guis_socket_is_taken_over(tmp_path):
    p = tmp_path / "project.toml"
    sock, _ = remote._paths(p)
    # Bound and never removed, as a GUI that died leaves it.
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(sock))
    s.close()
    with pytest.raises(remote.NoGui):
        remote.call(p, "state")
    server = remote.Server(p, lambda cmd, args: cmd)
    try:
        assert remote.call(p, "state") == "state"
    finally:
        server.close()
    assert not sock.exists()


def test_a_commands_error_reaches_the_caller(tmp_path):
    p = tmp_path / "project.toml"

    def handle(cmd, args):
        if cmd == "refuse":
            raise LaminoPalError("no tab Foo")
        return {}[cmd]

    server = remote.Server(p, handle)
    try:
        with pytest.raises(LaminoPalError, match="^no tab Foo$"):
            remote.call(p, "refuse")
        with pytest.raises(LaminoPalError, match="^KeyError: 'other'$"):
            remote.call(p, "other")
    finally:
        server.close()
