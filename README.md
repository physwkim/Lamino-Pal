# Lamino-Pal

BL7C laminography reconstruction GUI (embedded napari viewer, silx plots) and
MCP server on top of [tomoxide](https://github.com/physwkim/tomoxide) via PyO3.

Design: [docs/DESIGN.md](docs/DESIGN.md). GUI manual (Korean):
[docs/manual/README.md](docs/manual/README.md).

## Install

Linux with an NVIDIA GPU and the CUDA toolkit (`nvcc` on `PATH`), a Rust
toolchain, and micromamba:

```sh
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)                  # micromamba; reopen the shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust

git clone --recurse-submodules https://github.com/physwkim/Lamino-Pal.git
cd Lamino-Pal
micromamba create -f env.yaml
micromamba run -n lamino-pal maturin develop --release   # features: pyproject.toml
```

`maturin develop` installs the package editable into the `lamino-pal`
environment with the commands `lamino-pal-gui`, `lamino-pal-mcp` and
`lamino-pal`. The first `recon.run` builds the tomoxide CLI from
`lib/tomoxide` with cargo into `~/.cache/lamino-pal`, so the clone stays where
it was installed from. After a pull that moves `lib/tomoxide`, run
`git submodule update` and `maturin develop --release` again: `recon.run`
refuses a core built from another tomoxide commit.

## Run

```sh
micromamba run -n lamino-pal lamino-pal-gui --project ~/work/sample/project.toml
```

A `project.toml` that does not exist is created; without `--project` the GUI
asks for one.

## MCP

`lamino-pal-mcp` serves every op as an MCP tool over stdio. `--project` is the
project the tools act on; without it the agent opens or creates one with
`project_open`. With a GUI open on the same project, the `gui_*` tools show
its user a tab, a viewer's zoom or a graph's range.

Claude Code, for every directory:

```sh
claude mcp add lamino-pal --scope user -- \
  micromamba run -n lamino-pal lamino-pal-mcp --project /path/to/project.toml
```

or for this clone only, the repository's `.mcp.json`, with its `--project`
set to your project. `/mcp` in Claude Code shows the server connected.

Codex, `~/.codex/config.toml`:

```toml
[mcp_servers.lamino-pal]
command = "micromamba"
args = ["run", "-n", "lamino-pal", "lamino-pal-mcp",
        "--project", "/path/to/project.toml"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

## Test

```sh
micromamba run -n lamino-pal python -m pytest
cargo nextest run --no-default-features   # Rust modules without libpython
```
