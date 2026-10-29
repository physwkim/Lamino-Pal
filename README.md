# Lamino-Pal

BL7C laminography reconstruction GUI (embedded napari viewer, silx plots) and
MCP server on top of [tomoxide](https://github.com/physwkim/tomoxide) via PyO3.

Design: [docs/DESIGN.md](docs/DESIGN.md).

## Build

```sh
micromamba create -f env.yaml
git submodule update --init
micromamba run -n lamino-pal maturin develop --release   # features: pyproject.toml
micromamba run -n lamino-pal python -m pytest
cargo nextest run --no-default-features   # Rust modules without libpython
```
