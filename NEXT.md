# FlightLab immediate checkpoint

## Current phase

FlightLab v1.0.0 is locally prepared and validated as a release candidate.

The final 1.0.0 package metadata and MIT licensing are in place. The final
wheel and source distribution build successfully, the wheel installs into a
fresh CPython 3.12.3 environment, and the installed package reports FlightLab
1.0.0 with NumPy 2.5.2 as its sole runtime dependency and no SciPy installation.

The representative longitudinal campaign completes from the installed wheel
and persists its three ordered experiment runs to a fresh SQLite database.

Final verification results:

- focused end-to-end workflow tests: 4 passed
- focused V&V tests: 188 passed
- full suite: 1631 passed
- Ruff: passed
- `uv lock --check`: passed
- `git diff --check`: passed

## Exact next step

Review and commit the complete intended v1.0.0 release candidate. After the
release commit is created and a clean tracked worktree is confirmed, obtain
explicit owner authorization before creating the `v1.0.0` tag or pushing or
publishing the release.

Do not create the `v1.0.0` tag, publish, push, or begin post-v1 work without
explicit owner authorization.

## Still explicitly post-v1

- AI/LLM integration and repository-brain/RAG tooling
- OpenFOAM/CFD integration
- neural operators and PhysicsNeMo
- GPU/HPC expansion
- web/frontend work
- unrelated campaign-analysis expansion
- large architectural refactors
- speculative infrastructure
