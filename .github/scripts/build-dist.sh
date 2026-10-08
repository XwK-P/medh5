#!/usr/bin/env bash
# Build the Python wheel and the `medh5` binary for one target:
#
#     .github/scripts/build-dist.sh TARGET
#
# The wheel goes to dist/ (one abi3 wheel serves CPython 3.10 and later) and the
# binary, archived with its licence and README, to cli/.  CI runs this on every
# platform the project ships --- inside a manylinux_2_28 container on Linux, so
# both run on any glibc from 2.28 --- and the release publishes what CI built.
set -euo pipefail

target="$1"

if [[ -x /opt/python/cp312-cp312/bin/python ]]; then
  # A manylinux container: its own Python and CMake, and a Rust toolchain.
  python=/opt/python/cp312-cp312/bin/python
  "$python" -m pip install --quiet --upgrade maturin cmake
  export PATH="$(dirname "$python"):$HOME/.cargo/bin:$PATH"
  if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
      sh -s -- -y --profile minimal --default-toolchain stable
  fi
  compatibility=(--compatibility manylinux_2_28)
else
  # A hosted runner: Python from setup-python, CMake and rustup preinstalled.
  python=python
  "$python" -m pip install --quiet --upgrade maturin
  rustup toolchain install stable --profile minimal
  rustup default stable
  compatibility=()
fi

# (The `+` form: an empty array is "unbound" to the bash 3.2 macOS ships.)
maturin build --release --locked --out dist --interpreter "$python" ${compatibility[@]+"${compatibility[@]}"}
# The same engine build the wheel just used, so HDF5 is not compiled twice.
cargo build --release --locked -p medh5-cli

exe=target/release/medh5
if [[ -f "$exe.exe" ]]; then
  exe="$exe.exe"
fi
"$exe" --version

version=$("$python" -c 'import tomllib; print(tomllib.load(open("Cargo.toml", "rb"))["workspace"]["package"]["version"])')
name="medh5-$version-$target"
mkdir -p "cli/$name"
cp "$exe" LICENSE "cli/$name/"
cp crates/medh5-cli/README.md "cli/$name/README.md"
cd cli
if [[ "$target" == *windows* ]]; then
  "$python" -m zipfile -c "$name.zip" "$name"
else
  "$python" -m tarfile -c "$name.tar.gz" "$name"
fi
rm -rf "$name"
ls -l . ../dist
