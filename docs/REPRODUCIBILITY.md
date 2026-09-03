# Reproducibility

## Supported and locked environment

FlightLab requires Python 3.12 or newer. `uv.lock` is the authoritative lock
for the project and development environment. NumPy is the only runtime
dependency; pytest, Ruff, and SciPy are development dependencies. SciPy is used
only by its explicit independent verification runner.

The final release-candidate package metadata version is `1.0.0`. The project
is licensed under the MIT License. NumPy remains the sole runtime dependency;
SciPy is retained only as a development dependency for its explicit independent
verification runner.

## Clean-environment procedure

From the repository root, create a new temporary environment and cache. The
SQLite output path must also be new because persisted identities are immutable:

```console
clean_dir="$(mktemp -d /tmp/flightlab-clean.XXXXXX)"
UV_CACHE_DIR="$clean_dir/uv-cache" uv venv --python 3.12 "$clean_dir/venv"
UV_PROJECT_ENVIRONMENT="$clean_dir/venv" \
  UV_CACHE_DIR="$clean_dir/uv-cache" \
  uv sync --locked --dev

"$clean_dir/venv/bin/python" -c \
  "import importlib.metadata, flightlab; print(importlib.metadata.version('flightlab')); print(flightlab.__file__)"
"$clean_dir/venv/bin/python" examples/longitudinal_campaign.py \
  "$clean_dir/workflow.sqlite3"
"$clean_dir/venv/bin/pytest" -q tests/test_end_to_end_workflow.py
"$clean_dir/venv/bin/pytest" -q
"$clean_dir/venv/bin/ruff" check
git diff --check
```

The procedure installs the current repository package plus exact locked
development dependencies. A fresh cache requires access to the configured
Python package index. The project source tree is used as the package build
input; the virtual environment, cache, and generated SQLite database remain
outside the repository.

## Verified result

This procedure was executed on 2026-09-03 with CPython 3.12.3. The locked
environment resolved and installed:

```text
flightlab 0.1.0
numpy 2.5.2
scipy 1.18.1
pytest 9.1.1
ruff 0.16.4
```

Verification results:

```text
package import: passed
representative workflow: passed; fresh SQLite database created
focused workflow tests: 4 passed in 1.44s
full test suite: 1631 passed in 11.94s
Ruff: passed
git diff --check: passed
```

The workflow produced campaign `longitudinal-pitch-pole-scale-v1`, persisted
the ordered runs for pole scales `0.8`, `1.0`, and `1.2`, retrieved them through
the campaign bundle boundary, compared IAE/overshoot/settling time, and computed
metric deltas from the `1.0` baseline.

These results establish reproducibility of the current development tree and
locked environment. They are not the v1.0 release audit, an artifact-publication
record, or a physical/certification validation claim.

## Final v1.0.0 artifact validation

The final release-candidate wheel and source distribution were rebuilt on
2026-09-03. The wheel was installed directly into a fresh CPython 3.12.3
environment rather than importing FlightLab from the repository source tree.

The installed artifact reported:

```text
flightlab 1.0.0
numpy 2.5.2
scipy: absent
import source: fresh environment site-packages
license expression: MIT
runtime requirement: numpy>=2.5.2
```

The installed-wheel representative workflow completed successfully, created a
fresh SQLite database, and persisted the three ordered pole-scale runs for
campaign `longitudinal-pitch-pole-scale-v1`.

Final repository verification after the 1.0.0 metadata and MIT license changes:

```text
focused end-to-end workflow tests: 4 passed
focused V&V tests: 188 passed
full test suite: 1631 passed
Ruff: passed
uv lock --check: passed
git diff --check: passed
```

These checks validate reproducible packaging, installation, deterministic
workflow execution, and the documented software verification baseline. They do
not constitute aircraft certification or experimental physical validation.
