"""Write the Homebrew formula for one release of the ``medh5`` binary.

    python .github/scripts/homebrew_formula.py VERSION SHA256SUMS REPOSITORY

``SHA256SUMS`` is the release's checksum file, ``<sha256>  <archive>`` per
line, as the release workflow writes it.  The formula installs the prebuilt
binary on macOS and Linux, arm64 and x86_64; the release pushes it to the tap.
Nothing here touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

# What the binary is licensed under: the package's own MIT, and the licences of
# the libraries it links statically --- pyproject.toml's `license`, as Homebrew
# spells an SPDX conjunction (THIRD_PARTY_NOTICES lists each library).
LICENSES = ("MIT", "BSD-3-Clause", "BSD-2-Clause", "Zlib", "Apache-2.0", "MIT-0")

# Homebrew's (OS, architecture) blocks and the Rust target built for each.
TARGETS = {
    ("macos", "arm"): "aarch64-apple-darwin",
    ("macos", "intel"): "x86_64-apple-darwin",
    ("linux", "arm"): "aarch64-unknown-linux-gnu",
    ("linux", "intel"): "x86_64-unknown-linux-gnu",
}


def archive(version: str, target: str) -> str:
    """The release asset holding the binary for *target*."""
    return f"medh5-{version}-{target}.tar.gz"


def read_sums(text: str) -> dict[str, str]:
    """``{archive: sha256}`` from a ``sha256sum`` listing."""
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.lstrip("*")] = digest
    return sums


def formula(version: str, sums: dict[str, str], repository: str) -> str:
    """The formula's Ruby source; refuses a release missing a platform."""
    base = f"https://github.com/{repository}/releases/download/v{version}"
    blocks = []
    for os_name in ("macos", "linux"):
        arches = []
        for arch in ("arm", "intel"):
            name = archive(version, TARGETS[(os_name, arch)])
            if name not in sums:
                raise SystemExit(f"SHA256SUMS lists no {name}")
            arches.append(
                f"    on_{arch} do\n"
                f'      url "{base}/{name}"\n'
                f'      sha256 "{sums[name]}"\n'
                f"    end"
            )
        blocks.append(f"  on_{os_name} do\n" + "\n".join(arches) + "\n  end")
    sections = "\n\n".join(blocks)
    licenses = ", ".join(f'"{name}"' for name in LICENSES)
    return f'''class Medh5 < Formula
  desc "Inspect, validate, verify and curate MEDH5 medical imaging files"
  homepage "https://github.com/{repository}"
  version "{version}"
  license all_of: [{licenses}]

{sections}

  def install
    bin.install "medh5"
  end

  test do
    assert_match "medh5 #{{version}}", shell_output("#{{bin}}/medh5 --version")
  end
end
'''


def main(argv: list[str]) -> int:
    version, sums_path, repository = argv
    sums = read_sums(Path(sums_path).read_text(encoding="utf-8"))
    sys.stdout.write(formula(version, sums, repository))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
