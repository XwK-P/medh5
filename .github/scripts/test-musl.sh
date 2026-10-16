#!/usr/bin/env bash
# The musl wheel's test suite, and the static binary's hand-over of the
# converters to it, inside the musllinux image they were built in:
#
#     .github/scripts/test-musl.sh TARGET
#
# CI's `test-linux-musl-*` jobs run this from the checkout, with the build's
# wheels in dist/ and its binary archive in cli/.
set -euo pipefail

target="$1"

/opt/python/cp312-cp312/bin/python -m venv /tmp/venv
# shellcheck disable=SC1091
source /tmp/venv/bin/activate
# The `dev` extra but hdf5plugin, which publishes no musllinux wheel (nor does
# torch): their tests skip here, and run in every other job.  Read from
# pyproject.toml, so the two lists cannot drift apart.
python - > /tmp/dev.txt <<'PY'
import re, tomllib
with open("pyproject.toml", "rb") as f:
    dev = tomllib.load(f)["project"]["optional-dependencies"]["dev"]
print("\n".join(d for d in dev if re.match(r"[\w-]+", d)[0] != "hdf5plugin"))
PY
pip install --quiet "$(ls dist/*-abi3-*.whl)[nifti,dicom]" -r /tmp/dev.txt
python -c "import medh5; print(medh5.__version__, medh5.__file__)"

# The static binary runs here, where there is no glibc, and hands a converter
# to the wheel's Python: only the Python package says this.
mkdir /tmp/cli
tar -xzf cli/medh5-*-"$target".tar.gz -C /tmp/cli
exe=$(echo /tmp/cli/*/medh5)
"$exe" --version
status=0
(cd /tmp && MEDH5_PYTHON=python "$exe" convert from-nifti x.medh5 --image CT 2> bridged.log) || status=$?
cat /tmp/bridged.log
[[ $status == 1 ]]
grep -F -e "--image expects NAME=VALUE" /tmp/bridged.log

(cd /tmp && medh5 conformance build corpus && medh5 conformance run corpus)
# Out of the checkout's `medh5/`, which has no compiled engine and would
# shadow the installed one.
rm -rf medh5
pytest tests/ -v -rs --no-cov -p no:cacheprovider
